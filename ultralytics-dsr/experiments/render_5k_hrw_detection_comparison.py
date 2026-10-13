"""Render four diagnostic 5k_HRW validation scenes with GT and three trained detectors."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = ROOT.parents[1] / "5k_HRW_yolo_Dataset"
OUT_DIR = ROOT / "runs/hrw_standard_baselines_seed42_scratch/visualizations"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
CLASS_NAMES = ["hand-raising", "reading", "writing"]
COLORS = [(225, 50, 50), (40, 150, 60), (35, 110, 225)]  # BGR
WEIGHTS = {
    "YOLOv8n": ROOT / "runs/hrw_standard_baselines_seed42_scratch/native_runs/yolov8n/weights/best.pt",
    "QG-DRM-YOLO": ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch/native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail/weights/best.pt",
    "YOLOv8s": ROOT / "runs/hrw_standard_baselines_seed42_scratch/native_runs/yolov8s/weights/best.pt",
}
REPORT_START = "<!-- 5K_HRW_DETECTION_VISUALIZATION_START -->"
REPORT_END = "<!-- 5K_HRW_DETECTION_VISUALIZATION_END -->"


def image_for_label(label: Path) -> Path:
    for suffix in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        candidate = DATA_ROOT / "images/val" / f"{label.stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no image for {label}")


def read_boxes(label: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    records = []
    for line in label.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        cls, cx, cy, bw, bh = map(float, values)
        records.append((int(cls), (cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height))
    return records


def pairwise_ioa(left: tuple, right: tuple) -> float:
    _, ax1, ay1, ax2, ay2 = left
    _, bx1, by1, bx2, by2 = right
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_left = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_right = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(min(area_left, area_right), 1e-6)


def background_edge_density(image: np.ndarray, boxes: list[tuple]) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    mask = np.ones_like(edges, dtype=np.uint8)
    for _, x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (round(x1), round(y1)), (round(x2), round(y2)), 0, -1)
    available = max(int(mask.sum() // 255), 1)
    return float(np.count_nonzero(edges & mask)) / available


def collect_records() -> list[dict]:
    records = []
    for label in sorted((DATA_ROOT / "labels/val").glob("*.txt")):
        image_path = image_for_label(label)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = read_boxes(label, width, height)
        norm_areas = [max(0.0, (x2 - x1) * (y2 - y1) / (width * height)) for _, x1, y1, x2, y2 in boxes]
        records.append({
            "image": image_path,
            "boxes": boxes,
            "classes": [item[0] for item in boxes],
            "norm_areas": norm_areas,
            "max_ioa": max((pairwise_ioa(left, right) for left, right in combinations(boxes, 2)), default=0.0),
            "background_edge_density": background_edge_density(image, boxes),
        })
    if not records:
        raise RuntimeError("no readable 5k_HRW validation images")
    return records


def choose_scenes(records: list[dict]) -> list[tuple[str, str, dict]]:
    selected: set[Path] = set()

    def choose(scorer):
        candidates = sorted((item for item in records if item["image"] not in selected), key=scorer, reverse=True)
        if not candidates:
            raise RuntimeError("insufficient distinct validation images")
        chosen = candidates[0]
        selected.add(chosen["image"])
        return chosen

    # Count very small boxes first, then favour lower median box area and dense scenes.
    small = choose(lambda item: (sum(area < 0.004 for area in item["norm_areas"]), -float(np.median(item["norm_areas"])) if item["norm_areas"] else 0.0, len(item["boxes"])))
    occlusion = choose(lambda item: item["max_ioa"])
    fine = choose(lambda item: (len(set(item["classes"])) * 100 + min(item["classes"].count(cls) for cls in range(3)), len(item["boxes"])))
    background = choose(lambda item: item["background_edge_density"])
    return [
        ("A Small targets at rear", f"{sum(area < 0.004 for area in small['norm_areas'])} boxes below 0.4% image area", small),
        ("B Occlusion", f"max pairwise GT IoA = {occlusion['max_ioa']:.2f}", occlusion),
        ("C Fine-grained behaviors", "hand-raising / reading / writing", fine),
        ("D Complex background", "highest non-target edge density among remaining scenes", background),
    ]


def draw_boxes(image: np.ndarray, boxes: list[tuple], show_confidence: bool) -> np.ndarray:
    canvas = image.copy()
    for cls, x1, y1, x2, y2, *extra in boxes:
        color = COLORS[int(cls) % len(COLORS)]
        start, end = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(canvas, start, end, color, 2, cv2.LINE_AA)
        text = CLASS_NAMES[int(cls)]
        if show_confidence and extra:
            text += f" {float(extra[0]):.2f}"
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        baseline_y = max(height + baseline + 4, start[1])
        cv2.rectangle(canvas, (start[0], baseline_y - height - baseline - 4), (start[0] + width + 4, baseline_y), color, -1)
        cv2.putText(canvas, text, (start[0] + 2, baseline_y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def put_text(image: np.ndarray, text: str, y: int, scale: float = 0.7) -> None:
    cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), 2, cv2.LINE_AA)


def predict(models: dict, image_path: Path) -> dict[str, list[tuple]]:
    results = {}
    for name, model in models.items():
        prediction = model.predict(source=str(image_path), imgsz=640, conf=0.25, iou=0.70, device=0, verbose=False)[0]
        boxes = []
        if prediction.boxes is not None:
            for xyxy, cls, confidence in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.cls.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
                boxes.append((int(cls), float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(confidence)))
        results[name] = boxes
    return results


def render(scenes, models) -> tuple[np.ndarray, dict]:
    panel_w, panel_h = 540, 405
    header_h, row_h, legend_h = 56, 54, 42
    figure = np.full((header_h + len(scenes) * (row_h + panel_h) + legend_h, panel_w * 4, 3), 245, dtype=np.uint8)
    for index, title in enumerate(("Ground Truth", "YOLOv8n", "QG-DRM-YOLO", "YOLOv8s")):
        put_text(figure[:, index * panel_w:(index + 1) * panel_w], title, 36, 0.75)
    manifest = {"threshold": {"confidence": 0.25, "nms_iou": 0.70, "imgsz": 640}, "models": {name: str(path) for name, path in WEIGHTS.items()}, "scenes": []}
    y = header_h
    for title, note, record in scenes:
        put_text(figure[y:y + row_h], f"{title}: {note}  |  {record['image'].name}", 32, 0.58)
        y += row_h
        image = cv2.imread(str(record["image"]))
        predictions = predict(models, record["image"])
        panels = [draw_boxes(image, record["boxes"], False)] + [draw_boxes(image, predictions[name], True) for name in WEIGHTS]
        for index, panel in enumerate(panels):
            height, width = panel.shape[:2]
            scale = min(panel_w / width, panel_h / height)
            resized = cv2.resize(panel, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            target = figure[y:y + panel_h, index * panel_w:(index + 1) * panel_w]
            offset_x, offset_y = (panel_w - resized.shape[1]) // 2, (panel_h - resized.shape[0]) // 2
            target[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
        manifest["scenes"].append({"scenario": title, "note": note, "image": str(record["image"]), "gt_classes": [CLASS_NAMES[cls] for cls in record["classes"]], "max_ioa": record["max_ioa"], "background_edge_density": record["background_edge_density"], "prediction_counts": {name: len(value) for name, value in predictions.items()}})
        y += panel_h
    legend = figure[-legend_h:]
    x = 18
    for index, name in enumerate(CLASS_NAMES):
        cv2.rectangle(legend, (x, 14), (x + 16, 30), COLORS[index], -1)
        cv2.putText(legend, name, (x + 22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        x += 16 + 6 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0] + 22
    return figure, manifest


def update_report(image: Path) -> None:
    relative = image.relative_to(REPORT_PATH.parent).as_posix()
    section = "\n".join([
        REPORT_START, "## 5k_HRW（SCB-Dataset3-S）检测结果可视化", "",
        "四个验证场景覆盖后排小目标、遮挡、三类细粒度行为与复杂背景；列顺序为 Ground Truth、YOLOv8n、QG-DRM-YOLO、YOLOv8s。三种模型均采用 conf=0.25、IoU-NMS=0.70、imgsz=640。QG-DRM-YOLO 对应 QG-AFPN + C4 RGDRC2fV3 + MLKCM + P3DetailRestore。", "",
        f"![5k_HRW 检测结果对比图]({relative})", "", REPORT_END,
    ])
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        prefix, tail = existing.split(REPORT_START, 1)
        _, suffix = tail.split(REPORT_END, 1)
        existing = prefix.rstrip() + "\n\n" + section + suffix
    else:
        existing = existing.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.write_text(existing, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR / "5k_HRW_SCB-Dataset3-S_检测结果对比.png")
    args = parser.parse_args()
    for weight in WEIGHTS.values():
        if not weight.exists():
            raise FileNotFoundError(f"missing trained weight: {weight}")
    from ultralytics import YOLO

    models = {name: YOLO(str(weight)) for name, weight in WEIGHTS.items()}
    figure, manifest = render(choose_scenes(collect_records()), models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), figure, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"failed to write {args.output}")
    manifest["output"] = str(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_report(args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
