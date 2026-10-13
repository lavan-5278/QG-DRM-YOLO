"""Error diagnosis for the completed 5k_HRW YOLOv8s experiments.

The script evaluates all selected best.pt checkpoints on the *same* validation
split.  In addition to class AP it reports fixed-threshold error attribution:
missed GT, wrong-class overlap, poor localization, duplicate detections and
background false positives.  This is intentionally a post-hoc, read-only
analysis and never changes checkpoints or training state.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
# Scripts run from ``experiments/`` otherwise resolve the installed Ultralytics
# distribution first, which does not contain this project's custom modules.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class


DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
DEFAULT_OUTPUT = ROOT / "runs/hrw_yolov8s_error_diagnosis_seed42"

MODELS = {
    "YOLOv8s baseline": ROOT / "runs/hrw_standard_baselines_seed42_scratch/native_runs/yolov8s/weights/best.pt",
    "QG-AFPN": ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/native_runs/yolov8s_qgafpn/weights/best.pt",
    "DR-C2f v3": ROOT / "runs/hrw_yolov8s_dr_c2f_v3_seed42_scratch/native_runs/yolov8s_dr_c2f_v3/weights/best.pt",
    "QG-AFPN + DR-C2f v3": ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_drc2fv3_combinations_seed42_scratch/native_runs/yolov8s_qgafpn_drc2fv3/weights/best.pt",
    "PFOM + MLKCM": ROOT / "runs/hrw_yolov8s_pfom_mlkcm_seed42_scratch/native_runs/yolov8s_pfom_mlkcm/weights/best.pt",
    "RC-QG-AFPN + DR-C2f v3": ROOT / "runs/hrw_yolov8s_rcqgafpn_drc2fv3_seed42_scratch/native_runs/yolov8s_rc_qgafpn/weights/best.pt",
}

IOU_LEVELS = np.arange(0.5, 1.0, 0.05)
CONF_ERROR = 0.25


def xywhn_to_xyxy(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalized YOLO labels (cls, cx, cy, w, h) to xyxy pixels."""
    if not len(labels):
        return np.zeros((0, 4), dtype=np.float32)
    x, y, w, h = labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]
    return np.stack(((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height), 1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    inter_y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def greedy_matches(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls, threshold: float):
    """Ultralytics-compatible, IoU-ranked class-aware one-to-one matching."""
    ious = box_iou(pred_boxes, gt_boxes)
    pairs = np.argwhere((pred_cls[:, None] == gt_cls[None, :]) & (ious >= threshold))
    if not len(pairs):
        return [], ious
    pairs = pairs[np.argsort(ious[pairs[:, 0], pairs[:, 1]])[::-1]]
    # Mirrors BaseValidator.match_predictions: first retain the best detection
    # per label, then retain the best label per detection.
    pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
    pairs = pairs[np.argsort(ious[pairs[:, 0], pairs[:, 1]])[::-1]]
    pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return [(int(pi), int(gi), float(ious[pi, gi])) for pi, gi in pairs], ious


def size_bin(area_ratio: float) -> str:
    # Relative-area bins are invariant to original image size.  They are stated
    # explicitly in the report rather than presented as COCO's absolute bins.
    if area_ratio < 0.01:
        return "small (<1%)"
    if area_ratio < 0.05:
        return "medium (1-5%)"
    return "large (>=5%)"


def labels_for(image: Path, images_root: Path, labels_root: Path, width: int, height: int):
    label_path = labels_root / image.relative_to(images_root).with_suffix(".txt")
    if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=float)
    rows = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
    rows = rows[:, :5]
    boxes = xywhn_to_xyxy(rows, width, height)
    areas = rows[:, 3] * rows[:, 4]
    return rows[:, 0].astype(int), boxes, areas


def metric_summary(tp_all, conf_all, pred_cls_all, gt_cls_all, names):
    if not len(gt_cls_all):
        return {"mAP50": 0.0, "mAP50-95": 0.0, "per_class": {}}
    tp, fp, p, r, f1, ap, classes, *_ = ap_per_class(
        np.concatenate(tp_all, 0), np.concatenate(conf_all, 0), np.concatenate(pred_cls_all, 0), np.concatenate(gt_cls_all, 0),
        names=names,
    )
    per_class = {}
    for i, cls in enumerate(classes):
        per_class[names[int(cls)]] = {
            "precision_max_f1": round(float(p[i]), 5),
            "recall_max_f1": round(float(r[i]), 5),
            "AP50": round(float(ap[i, 0]), 5),
            "AP50-95": round(float(ap[i].mean()), 5),
        }
    return {"mAP50": round(float(ap[:, 0].mean()), 5), "mAP50-95": round(float(ap.mean()), 5), "per_class": per_class}


def analyse_model(name: str, checkpoint: Path, images: list[Path], images_root: Path, labels_root: Path, names: dict[int, str], args):
    print(f"\n[{name}] loading {checkpoint}", flush=True)
    model = YOLO(checkpoint)
    all_tp, all_conf, all_pred_cls, all_gt_cls = [], [], [], []
    gt_size = Counter(); size_tp50 = Counter(); size_tp75 = Counter(); gt_error = Counter()
    fp_error = Counter(); tp_ious = []; image_count = 0
    # ``Result.path`` may be only a basename when a Python list is used as the
    # source.  Results preserve source order, so pair them with the original
    # absolute paths rather than reconstructing the label path from Result.path.
    predictions = model.predict(images, stream=True, imgsz=args.imgsz, batch=args.batch, conf=0.001, iou=0.7, max_det=300, device=args.device, verbose=False)
    for image, result in zip(images, predictions):
        height, width = result.orig_shape
        gt_cls, gt_boxes, gt_areas = labels_for(image, images_root, labels_root, width, height)
        pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros((0, 4), np.float32)
        pred_cls = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else np.zeros(0, int)
        pred_conf = result.boxes.conf.cpu().numpy().astype(np.float32) if result.boxes is not None else np.zeros(0, np.float32)

        tp = np.zeros((len(pred_boxes), len(IOU_LEVELS)), dtype=bool)
        for ti, threshold in enumerate(IOU_LEVELS):
            matches, _ = greedy_matches(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls, float(threshold))
            for pi, _, _ in matches:
                tp[pi, ti] = True
        all_tp.append(tp); all_conf.append(pred_conf); all_pred_cls.append(pred_cls); all_gt_cls.append(gt_cls)

        selected = pred_conf >= CONF_ERROR
        p_boxes, p_cls, p_conf = pred_boxes[selected], pred_cls[selected], pred_conf[selected]
        matches50, ious = greedy_matches(p_boxes, p_cls, p_conf, gt_boxes, gt_cls, 0.5)
        matches75, _ = greedy_matches(p_boxes, p_cls, p_conf, gt_boxes, gt_cls, 0.75)
        matched_gt50 = {gi for _, gi, _ in matches50}; matched_gt75 = {gi for _, gi, _ in matches75}
        matched_pred50 = {pi for pi, _, _ in matches50}
        tp_ious.extend(iou for _, _, iou in matches50)

        for gi, area in enumerate(gt_areas):
            group = size_bin(float(area)); gt_size[group] += 1
            if gi in matched_gt50:
                size_tp50[group] += 1
                if gi in matched_gt75:
                    size_tp75[group] += 1
                continue
            any_iou = ious[:, gi] if len(p_boxes) else np.zeros(0)
            if len(any_iou) and np.any((p_cls != gt_cls[gi]) & (any_iou >= 0.5)):
                gt_error["wrong_class"] += 1
            elif len(any_iou) and np.any((p_cls == gt_cls[gi]) & (any_iou >= 0.1)):
                gt_error["poor_localization"] += 1
            else:
                gt_error["missed"] += 1

        for pi in range(len(p_boxes)):
            if pi in matched_pred50:
                continue
            overlap = ious[pi] if len(gt_boxes) else np.zeros(0)
            if len(overlap) and np.any((gt_cls != p_cls[pi]) & (overlap >= 0.5)):
                fp_error["wrong_class"] += 1
            elif len(overlap) and np.any((gt_cls == p_cls[pi]) & (overlap >= 0.5)):
                fp_error["duplicate"] += 1
            elif len(overlap) and np.any((gt_cls == p_cls[pi]) & (overlap >= 0.1)):
                fp_error["poor_localization"] += 1
            else:
                fp_error["background"] += 1
        image_count += 1
        if image_count % 250 == 0:
            print(f"[{name}] {image_count}/{len(images)} images", flush=True)

    summary = metric_summary(all_tp, all_conf, all_pred_cls, all_gt_cls, names)
    summary.update({
        "images": image_count,
        "error_threshold": {"confidence": CONF_ERROR, "iou_match": 0.5, "localization_overlap": 0.1},
        "size_bins": {k: {"gt": int(gt_size[k]), "recall_iou50": round(size_tp50[k] / max(gt_size[k], 1), 5), "recall_iou75": round(size_tp75[k] / max(gt_size[k], 1), 5)} for k in ("small (<1%)", "medium (1-5%)", "large (>=5%)")},
        "gt_error": dict(gt_error),
        "fp_error": dict(fp_error),
        "mean_iou_of_tp50": round(float(np.mean(tp_ious)) if tp_ious else 0.0, 5),
    })
    return summary


def markdown_report(results: dict[str, dict]) -> str:
    lines = [
        "# 5k_HRW YOLOv8s 误差诊断（seed42）", "",
        "所有模型以其 `best.pt` 在同一验证集重新推理。重推 AP 使用置信度≥0.001的预测；错误归因使用置信度≥0.25、IoU=0.50。",
        "“诊断重推 mAP”仅用于本报告内的误差趋势对比；正式 mAP 仍以训练期标准验证结果为准。目标尺度按 GT 相对图像面积划分：small <1%，medium 1–5%，large ≥5%。该报告不替代多 seed 显著性检验。", "",
        "## 总体与定位", "",
        "| 模型 | 诊断重推 mAP50 | 诊断重推 mAP50-95 | TP 平均 IoU | small R@.5/.75 | medium R@.5/.75 | large R@.5/.75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        b = r["size_bins"]
        cell = lambda key: f"{b[key]['recall_iou50']:.3f}/{b[key]['recall_iou75']:.3f}"
        lines.append(f"| {name} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f} | {r['mean_iou_of_tp50']:.4f} | {cell('small (<1%)')} | {cell('medium (1-5%)')} | {cell('large (>=5%)')} |")
    lines += ["", "## GT 漏检归因（IoU=.5，置信度≥.25）", "", "| 模型 | missed | poor localization | wrong class |", "|---|---:|---:|---:|"]
    for name, r in results.items():
        e = r["gt_error"]
        lines.append(f"| {name} | {e.get('missed', 0)} | {e.get('poor_localization', 0)} | {e.get('wrong_class', 0)} |")
    lines += ["", "## FP 归因（IoU=.5，置信度≥.25）", "", "| 模型 | background | poor localization | duplicate | wrong class |", "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        e = r["fp_error"]
        lines.append(f"| {name} | {e.get('background', 0)} | {e.get('poor_localization', 0)} | {e.get('duplicate', 0)} | {e.get('wrong_class', 0)} |")
    lines += ["", "## 分类别 AP", ""]
    classes = next(iter(results.values()))["per_class"].keys()
    lines.append("| 模型 | " + " | ".join(f"{c} AP50/AP50-95" for c in classes) + " |")
    lines.append("|---|" + "---:|" * len(classes))
    for name, r in results.items():
        vals = [f"{r['per_class'].get(c, {}).get('AP50', 0):.4f}/{r['per_class'].get(c, {}).get('AP50-95', 0):.4f}" for c in classes]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--render-existing", action="store_true", help="Render the report from a saved partial JSON without inference.")
    p.add_argument("--core-only", action="store_true", help="Exclude the RC-QG diagnostic-only comparison.")
    args = p.parse_args()
    with DATA_YAML.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data_root = Path(data["path"])
    images_root = data_root / data["val"]
    labels_root = data_root / "labels/val"
    images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not images:
        raise FileNotFoundError(f"No validation images found under {images_root}")
    names = {i: str(n) for i, n in enumerate(data["names"])}
    selected_models = {k: v for k, v in MODELS.items() if not (args.core_only and k.startswith("RC-QG"))}
    missing = [str(p) for p in selected_models.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    partial = args.output / "error_diagnosis.partial.json"
    if args.render_existing:
        if not partial.exists():
            raise FileNotFoundError(partial)
        results = json.loads(partial.read_text(encoding="utf-8"))
        (args.output / "error_diagnosis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / "误差诊断报告.md").write_text(markdown_report(results), encoding="utf-8")
        print(f"Rendered existing evidence: {args.output}", flush=True)
        return
    results = {}
    for name, path in selected_models.items():
        results[name] = analyse_model(name, path, images, images_root, labels_root, names, args)
        # Keep completed-model evidence even if a later custom checkpoint fails
        # to load, and make the lengthy diagnosis safely inspectable.
        (args.output / "error_diagnosis.partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (args.output / "error_diagnosis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "误差诊断报告.md").write_text(markdown_report(results), encoding="utf-8")
    print(f"\nCompleted: {args.output}", flush=True)


if __name__ == "__main__":
    main()
