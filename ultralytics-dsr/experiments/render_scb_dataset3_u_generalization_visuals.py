"""Render GT and three trained SCB-Dataset3-U detectors on four diagnostic validation scenes."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = ROOT.parent / "data/SCB-Dataset3-U"
OUT_DIR = ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch/visualizations"

CLASS_NAMES = ["hand-raising", "reading", "writing", "using-phone", "head-down", "lying-on-desk"]
COLORS = [(225, 50, 50), (40, 150, 60), (35, 110, 225), (185, 70, 195), (45, 170, 205), (20, 150, 150)]  # BGR
WEIGHTS = {
    "YOLOv8n": ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch/native_runs/yolov8n_scb_u_seed42/weights/best.pt",
    "QG-DRM-YOLO": ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch/native_runs/yolov8s_current_best_topology_scb_u_seed42/weights/best.pt",
    "YOLOv8s": ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch/native_runs/yolov8s_scb_u_seed42/weights/best.pt",
}
# Keep the manuscript column layout independent of model-loading order.
DISPLAY_ORDER = ("YOLOv8n", "YOLOv8s", "QG-DRM-YOLO")


def image_for_label(label: Path) -> Path:
    stem = label.stem
    for suffix in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        candidate = DATA_ROOT / "images/val" / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no image for {label}")


def read_labels(label: Path, width: int, height: int) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = map(float, parts)
        boxes.append((int(cls), (cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def ioa(box_a: tuple, box_b: tuple) -> float:
    _, ax1, ay1, ax2, ay2 = box_a
    _, bx1, by1, bx2, by2 = box_b
    inter_w, inter_h = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a, area_b = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1), max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(min(area_a, area_b), 1e-6)


def inventory() -> list[dict]:
    records = []
    for label in sorted((DATA_ROOT / "labels/val").glob("*.txt")):
        image_path = image_for_label(label)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = read_labels(label, width, height)
        classes = [box[0] for box in boxes]
        max_ioa = max((ioa(left, right) for left, right in combinations(boxes, 2)), default=0.0)
        records.append({"image": image_path, "boxes": boxes, "classes": classes, "max_ioa": max_ioa})
    if not records:
        raise RuntimeError("no readable SCB-Dataset3-U validation records")
    return records


def choose_scenes(records: list[dict]) -> list[tuple[str, str, dict]]:
    selected: set[Path] = set()

    def pick(title: str, scorer):
        candidates = sorted((record for record in records if record["image"] not in selected), key=scorer, reverse=True)
        if not candidates or scorer(candidates[0]) <= 0:
            raise RuntimeError(f"could not find an appropriate scene for {title}")
        record = candidates[0]
        selected.add(record["image"])
        return record

    shared = pick("共享类别", lambda record: sum(cls in {0, 1, 2} for cls in record["classes"]))
    new = pick("新增类别", lambda record: sum(cls in {3, 4, 5} for cls in record["classes"]))
    long_tail = pick("长尾类别", lambda record: sum(cls == 0 for cls in record["classes"]))
    occluded = pick("严重遮挡", lambda record: record["max_ioa"])
    return [
        ("A Shared classes", "reading / writing", shared),
        ("B New classes", "using-phone / head-down / lying-on-desk", new),
        ("C Long-tail class", "hand-raising (rare validation class)", long_tail),
        ("D Severe occlusion", f"max pairwise GT IoA = {occluded['max_ioa']:.2f}", occluded),
    ]


def draw_boxes(image: np.ndarray, boxes: list[tuple], confidence: bool) -> np.ndarray:
    canvas = image.copy()
    for cls, x1, y1, x2, y2, *tail in boxes:
        color = COLORS[int(cls) % len(COLORS)]
        pt1, pt2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(canvas, pt1, pt2, color, 2, cv2.LINE_AA)
        label = CLASS_NAMES[int(cls)]
        if confidence and tail:
            label += f" {float(tail[0]):.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        text_y = max(text_h + baseline + 3, pt1[1])
        cv2.rectangle(canvas, (pt1[0], text_y - text_h - baseline - 4), (pt1[0] + text_w + 4, text_y), color, -1)
        cv2.putText(canvas, label, (pt1[0] + 2, text_y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def place_text(image: np.ndarray, text: str, y: int, scale: float = 0.7, color=(25, 25, 25)) -> None:
    cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def predict(models: dict, image_path: Path) -> dict[str, list[tuple]]:
    outputs = {}
    for name, model in models.items():
        result = model.predict(source=str(image_path), imgsz=640, conf=0.25, iou=0.7, device=0, verbose=False)[0]
        boxes = []
        if result.boxes is not None:
            for xyxy, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                boxes.append((int(cls), float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)))
        outputs[name] = boxes
    return outputs


def make_figure(scenes, models) -> tuple[np.ndarray, dict]:
    panel_w, panel_h = 540, 405
    # Column headings are intentionally omitted so they can be added in the manuscript layout.
    header_h, row_title_h, legend_h = 0, 54, 42
    figure = np.full((header_h + len(scenes) * (row_title_h + panel_h) + legend_h, panel_w * 4, 3), 245, dtype=np.uint8)
    manifest = {"models": {name: str(path) for name, path in WEIGHTS.items()}, "display_order": ["Ground Truth", *DISPLAY_ORDER], "column_titles_omitted": True, "dpi": 300, "scenes": []}
    y = header_h
    for scenario, subtitle, record in scenes:
        scenario_strip = figure[y:y + row_title_h]
        place_text(scenario_strip, f"{scenario}: {subtitle}  |  {record['image'].name}", 32, 0.58)
        y += row_title_h
        image = cv2.imread(str(record["image"]))
        predictions = predict(models, record["image"])
        panels = [draw_boxes(image, record["boxes"], False)] + [draw_boxes(image, predictions[name], True) for name in DISPLAY_ORDER]
        for col, panel in enumerate(panels):
            height, width = panel.shape[:2]
            factor = min(panel_w / width, panel_h / height)
            resized = cv2.resize(panel, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
            cell = figure[y:y + panel_h, col * panel_w:(col + 1) * panel_w]
            offset_x, offset_y = (panel_w - resized.shape[1]) // 2, (panel_h - resized.shape[0]) // 2
            cell[offset_y:offset_y + resized.shape[0], offset_x:offset_x + resized.shape[1]] = resized
        manifest["scenes"].append({"scenario": scenario, "subtitle": subtitle, "image": str(record["image"]), "gt_classes": [CLASS_NAMES[cls] for cls in record["classes"]], "max_ioa": record["max_ioa"], "prediction_counts": {name: len(items) for name, items in predictions.items()}})
        y += panel_h
    legend = figure[-legend_h:]
    x = 18
    for index, name in enumerate(CLASS_NAMES):
        color = COLORS[index]
        cv2.rectangle(legend, (x, 14), (x + 16, 30), color, -1)
        cv2.putText(legend, name, (x + 22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1, cv2.LINE_AA)
        x += 16 + 6 + cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)[0][0] + 22
    return figure, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR / "SCB-Dataset3-U_泛化检测对照.png")
    args = parser.parse_args()
    for path in WEIGHTS.values():
        if not path.exists():
            raise FileNotFoundError(f"missing trained weight: {path}")
    from ultralytics import YOLO

    models = {name: YOLO(str(path)) for name, path in WEIGHTS.items()}
    scenes = choose_scenes(inventory())
    figure, manifest = make_figure(scenes, models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the rendered pixels while embedding 300 DPI metadata for publication workflows.
    Image.fromarray(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)).save(args.output, dpi=(300, 300), compress_level=3)
    manifest["output"] = str(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
