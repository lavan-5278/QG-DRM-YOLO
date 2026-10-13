"""Select real QG-DRM-YOLO validation failures and render GT/prediction pairs."""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT.parents[1] / "5k_HRW_yolo_Dataset"
OUT_DIR = ROOT / "runs/hrw_standard_baselines_seed42_scratch/visualizations"
REPORT = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
WEIGHT = ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch/native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail/weights/best.pt"
NAMES = ("hand-raising", "reading", "writing")
START = "<!-- 5K_HRW_FAILURE_GALLERY_START -->"
END = "<!-- 5K_HRW_FAILURE_GALLERY_END -->"


def image_for_label(label: Path) -> Path:
    for suffix in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        path = DATA_ROOT / "images/val" / f"{label.stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(label)


def boxes_from_label(label: Path, shape: tuple[int, int]) -> list[dict]:
    h, w = shape
    result = []
    for line in label.read_text(encoding="utf-8").splitlines():
        cls, cx, cy, bw, bh = map(float, line.split())
        result.append({"cls": int(cls), "xyxy": np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])})
    return result


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(1e-8, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union


def ioa_smaller(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = min(max(1e-8, (a[2] - a[0]) * (a[3] - a[1])), max(1e-8, (b[2] - b[0]) * (b[3] - b[1])))
    return inter / area


def predict(model, image_path: Path) -> list[dict]:
    result = model.predict(str(image_path), imgsz=640, conf=0.25, iou=0.70, device=0, verbose=False)[0]
    if result.boxes is None:
        return []
    return [
        {"cls": int(cls), "conf": float(conf), "xyxy": np.asarray(xyxy, dtype=np.float32)}
        for xyxy, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy())
    ]


def assess(gts: list[dict], predictions: list[dict]) -> dict:
    relations = []
    for gt in gts:
        overlaps = [(iou(gt["xyxy"], prediction["xyxy"]), index, prediction) for index, prediction in enumerate(predictions)]
        best_iou, best_index, best_prediction = max(overlaps, default=(0.0, -1, None), key=lambda item: item[0])
        same = max((item for item in overlaps if item[2]["cls"] == gt["cls"]), default=(0.0, -1, None), key=lambda item: item[0])
        status = "tp" if same[0] >= 0.5 else ("confusion" if best_iou >= 0.3 and best_prediction["cls"] != gt["cls"] else "miss")
        relations.append({"status": status, "best_iou": best_iou, "prediction_index": best_index, "same_iou": same[0]})
    fps = []
    for index, prediction in enumerate(predictions):
        overlap = max((iou(prediction["xyxy"], gt["xyxy"]) for gt in gts), default=0.0)
        if overlap < 0.3:
            fps.append(index)
    return {"relations": relations, "false_positives": fps}


def build_records(model) -> list[dict]:
    records = []
    for label in sorted((DATA_ROOT / "labels/val").glob("*.txt")):
        image_path = image_for_label(label)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gts = boxes_from_label(label, image.shape[:2])
        predictions = predict(model, image_path)
        analysis = assess(gts, predictions)
        areas = np.asarray([(gt["xyxy"][2] - gt["xyxy"][0]) * (gt["xyxy"][3] - gt["xyxy"][1]) / (image.shape[0] * image.shape[1]) for gt in gts])
        max_ioa = max((ioa_smaller(a["xyxy"], b["xyxy"]) for a, b in combinations(gts, 2)), default=0.0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        error_count = sum(item["status"] != "tp" for item in analysis["relations"]) + len(analysis["false_positives"])
        records.append({
            "image": image_path, "gts": gts, "predictions": predictions, "analysis": analysis,
            "areas": areas, "min_area": float(areas.min()), "max_ioa": float(max_ioa),
            "brightness": float(gray.mean()), "error_count": error_count,
        })
    return records


def pick(records: list[dict], used: set[Path], scorer) -> dict:
    candidates = [record for record in records if record["image"] not in used]
    selected = max(candidates, key=scorer)
    used.add(selected["image"])
    return selected


def select_cases(records: list[dict]) -> list[tuple[str, str, str, dict]]:
    used: set[Path] = set()
    cases = []
    occlusion = pick(records, used, lambda r: (r["error_count"] > 0, r["max_ioa"], r["error_count"]))
    cases.append(("A Extreme occlusion", "miss/confusion", "Overlapping people and partial body visibility make instance separation ambiguous.", occlusion))
    small = pick(records, used, lambda r: (sum(rel["status"] != "tp" and r["areas"][i] < 0.004 for i, rel in enumerate(r["analysis"]["relations"])), -r["min_area"], r["error_count"]))
    cases.append(("B Ultra-small target", "miss", "Few pixels remain after stride/downsampling; the action cue is weak.", small))
    dark = pick(records, used, lambda r: (r["error_count"] > 0, -r["brightness"], r["error_count"]))
    cases.append(("C Weak light", "miss/false positive", "Low contrast weakens hand, face and book boundary evidence.", dark))
    bright = pick(records, used, lambda r: (r["error_count"] > 0, r["brightness"], r["error_count"]))
    cases.append(("D Strong light", "false positive/miss", "Highlight saturation and shadow transitions distort appearance cues.", bright))
    fuzzy = pick(records, used, lambda r: (sum(rel["status"] == "confusion" for rel in r["analysis"]["relations"]), r["error_count"], len(r["gts"])))
    cases.append(("E Fuzzy action boundary", "class confusion", "Reading and writing share seated posture and fine-grained hand-object cues.", fuzzy))
    return cases


def text(canvas: np.ndarray, value: str, xy: tuple[int, int], scale: float = 0.45, color=(20, 20, 20)) -> None:
    cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_box(canvas: np.ndarray, item: dict, color: tuple[int, int, int], label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = item["xyxy"].astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    text(canvas, label, (x1 + 2, max(14, y1 - 4)), 0.42, color)


def panels(record: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    image = cv2.imread(str(record["image"]))
    gt_panel = image.copy()
    pred_panel = image.copy()
    failure_notes: list[str] = []
    for index, gt in enumerate(record["gts"]):
        relation = record["analysis"]["relations"][index]
        is_failure = relation["status"] != "tp"
        draw_box(gt_panel, gt, (255, 255, 255) if is_failure else (140, 140, 140), f"GT {NAMES[gt['cls']]}", 2 if is_failure else 1)
        if is_failure:
            color = (245, 60, 200) if relation["status"] == "miss" else (0, 180, 255)
            draw_box(pred_panel, gt, color, f"GT {NAMES[gt['cls']]}: {relation['status']}", 2)
            if relation["status"] == "confusion" and relation["prediction_index"] >= 0:
                prediction = record["predictions"][relation["prediction_index"]]
                draw_box(pred_panel, prediction, (0, 165, 255), f"Pred {NAMES[prediction['cls']]} {prediction['conf']:.2f}", 2)
                failure_notes.append(f"GT {NAMES[gt['cls']]} -> {NAMES[prediction['cls']]}")
            else:
                failure_notes.append(f"Missed {NAMES[gt['cls']]}")
    for index in record["analysis"]["false_positives"]:
        prediction = record["predictions"][index]
        draw_box(pred_panel, prediction, (0, 0, 255), f"FP {NAMES[prediction['cls']]} {prediction['conf']:.2f}", 2)
        failure_notes.append(f"FP {NAMES[prediction['cls']]}")
    return gt_panel, pred_panel, failure_notes or ["No thresholded error selected"]


def paste(destination: np.ndarray, image: np.ndarray) -> None:
    h, w = destination.shape[:2]
    scale = min(w / image.shape[1], h / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    destination[y : y + resized.shape[0], x : x + resized.shape[1]] = resized


def render(cases: list[tuple[str, str, str, dict]]) -> tuple[np.ndarray, dict]:
    panel_w, panel_h, info_h, head_h = 560, 360, 64, 50
    figure = np.full((head_h + len(cases) * (info_h + panel_h), panel_w * 2, 3), 245, dtype=np.uint8)
    text(figure, "Ground Truth (white: focus error; gray: context)", (15, 32), 0.58)
    text(figure[:, panel_w:], "QG-DRM-YOLO prediction diagnosis", (15, 32), 0.58)
    manifest = {"model": str(WEIGHT), "threshold": {"confidence": 0.25, "nms_iou": 0.70, "imgsz": 640}, "cases": []}
    y = head_h
    for title, failure_type, cause, record in cases:
        gt_panel, pred_panel, failures = panels(record)
        header = figure[y : y + info_h]
        text(header, f"{title} | observed: {', '.join(failures[:3])}", (14, 24), 0.48)
        text(header, f"Possible cause: {cause}", (14, 49), 0.40, (65, 65, 65))
        y += info_h
        paste(figure[y : y + panel_h, :panel_w], gt_panel)
        paste(figure[y : y + panel_h, panel_w:], pred_panel)
        manifest["cases"].append({
            "title": title, "failure_type": failure_type, "possible_cause": cause, "image": str(record["image"]),
            "brightness": round(record["brightness"], 3), "min_gt_area_ratio": round(record["min_area"], 6),
            "max_gt_ioa": round(record["max_ioa"], 6), "observed_failures": failures,
        })
        y += panel_h
    return figure, manifest


def update_report(image: Path, manifest: Path) -> None:
    block = "\n".join((
        START,
        "## 5k_HRW 失败案例：漏检、误检与类别混淆",
        "",
        "使用 QG-DRM-YOLO 在验证集上的真实预测，统一设置为 conf=0.25、IoU-NMS=0.70、imgsz=640。左列为 GT，右列仅强调与失败相关的 GT（粉色/橙色）和预测：粉色为漏检 GT、橙色为类别混淆、红色为误检。图下注释给出失败类型与可能原因，属于定性诊断而非因果证明。",
        "",
        f"![5k_HRW 失败案例图]({image.as_posix()})",
        "",
        f"逐例原始属性与阈值记录见 [JSON]({manifest.as_posix()})。",
        END,
    ))
    content = REPORT.read_text(encoding="utf-8")
    if START in content and END in content:
        before, rest = content.split(START, 1)
        _, after = rest.split(END, 1)
        content = before.rstrip() + "\n\n" + block + after
    else:
        content = content.rstrip() + "\n\n" + block + "\n"
    REPORT.write_text(content, encoding="utf-8")


def main() -> None:
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)
    from ultralytics import YOLO

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records(YOLO(str(WEIGHT)))
    figure, manifest = render(select_cases(records))
    output = OUT_DIR / "5k_HRW_QG_DRM_failure_cases.png"
    if not cv2.imwrite(str(output), figure, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(output)
    manifest_path = output.with_suffix(".json")
    manifest["output"] = str(output)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_report(output, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
