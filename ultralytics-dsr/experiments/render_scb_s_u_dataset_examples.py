"""Render a 3x3 labelled sample sheet for SCB-Dataset3-S and SCB-Dataset3-U."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
S_ROOT = Path(r"D:\YH_Project\DSA-Det\5k_HRW_yolo_Dataset")
U_ROOT = Path(r"D:\YH_Project\DSA-Det\DSA-Det\data\SCB-Dataset3-U")
OUT_DIR = ROOT / "runs/hrw_standard_baselines_seed42_scratch/visualizations"

S_CLASSES = ("hand-raising", "reading", "writing")
U_CLASSES = ("hand-raising", "reading", "writing", "using-phone", "head-down", "lying-on-desk")
COLORS = (
    (30, 185, 255),
    (75, 220, 75),
    (255, 150, 70),
    (220, 90, 210),
    (60, 210, 220),
    (235, 100, 100),
)


@dataclass(frozen=True)
class Sample:
    dataset: str
    root: Path
    class_names: tuple[str, ...]
    target_class: int
    image_path: Path
    labels: np.ndarray


def load_labels(path: Path) -> np.ndarray:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return np.empty((0, 5), dtype=np.float32)
    labels = np.loadtxt(path, dtype=np.float32, ndmin=2)
    return labels.reshape(-1, 5)


def rank_candidate(labels: np.ndarray, target_class: int) -> float:
    target = labels[labels[:, 0].astype(int) == target_class]
    if not len(target):
        return float("-inf")
    target_area = target[:, 3] * target[:, 4]
    other_count = len(labels) - len(target)
    # Favour visible target instances while avoiding panels overwhelmed by annotations.
    return 40.0 * len(target) + 500.0 * float(np.sqrt(target_area).sum()) - 1.5 * other_count


def choose_samples(dataset: str, root: Path, class_names: tuple[str, ...], already_used: set[Path]) -> list[Sample]:
    image_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    candidates: list[tuple[Path, np.ndarray]] = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        labels = load_labels(label_dir / f"{image_path.stem}.txt")
        if len(labels):
            candidates.append((image_path, labels))

    selected: list[Sample] = []
    used_here = set(already_used)
    for target_class in range(len(class_names)):
        ranked = sorted(
            ((rank_candidate(labels, target_class), image_path, labels) for image_path, labels in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        picked = next((item for item in ranked if item[0] > float("-inf") and item[1] not in used_here), None)
        if picked is None:
            raise RuntimeError(f"No usable {dataset} sample found for {class_names[target_class]}.")
        _, image_path, labels = picked
        used_here.add(image_path)
        selected.append(Sample(dataset, root, class_names, target_class, image_path, labels))
    already_used.update(sample.image_path for sample in selected)
    return selected


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    x = max(0, min(x, image.shape[1] - width - 6))
    y = max(height + baseline + 6, y)
    cv2.rectangle(image, (x, y - height - baseline - 6), (x + width + 6, y + 2), color, -1)
    cv2.putText(image, text, (x + 3, y - baseline - 2), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def crop_around_target(source: np.ndarray, target: np.ndarray, aspect_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a context crop and the target box in crop coordinates."""
    height, width = source.shape[:2]
    _, cx, cy, box_w, box_h = target
    cx, cy, box_w, box_h = cx * width, cy * height, box_w * width, box_h * height
    crop_w = max(360.0, box_w * 3.2)
    crop_h = max(260.0, box_h * 3.2)
    if crop_w / crop_h < aspect_ratio:
        crop_w = crop_h * aspect_ratio
    else:
        crop_h = crop_w / aspect_ratio
    crop_w, crop_h = min(crop_w, width), min(crop_h, height)
    x1 = int(round(np.clip(cx - crop_w / 2, 0, width - crop_w)))
    y1 = int(round(np.clip(cy - crop_h / 2, 0, height - crop_h)))
    x2, y2 = int(round(x1 + crop_w)), int(round(y1 + crop_h))
    target_xyxy = np.array((cx - box_w / 2 - x1, cy - box_h / 2 - y1, cx + box_w / 2 - x1, cy + box_h / 2 - y1))
    return source[y1:y2, x1:x2].copy(), target_xyxy


def render_panel(sample: Sample, panel_width: int = 1120, image_height: int = 720, caption_height: int = 110) -> np.ndarray:
    source = cv2.imread(str(sample.image_path))
    if source is None:
        raise RuntimeError(f"Cannot read {sample.image_path}")
    target_instances = sample.labels[sample.labels[:, 0].astype(int) == sample.target_class]
    target = target_instances[np.argmax(target_instances[:, 3] * target_instances[:, 4])]
    cropped, target_xyxy = crop_around_target(source, target, panel_width / image_height)
    cropped_h, cropped_w = cropped.shape[:2]
    resized = cv2.resize(cropped, (panel_width, image_height), interpolation=cv2.INTER_CUBIC)
    scale_x, scale_y = panel_width / cropped_w, image_height / cropped_h
    x1, y1, x2, y2 = target_xyxy
    x1, x2 = int(round(x1 * scale_x)), int(round(x2 * scale_x))
    y1, y2 = int(round(y1 * scale_y)), int(round(y2 * scale_y))
    color = COLORS[sample.target_class]
    cv2.rectangle(resized, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
    draw_label(resized, sample.class_names[sample.target_class], (x1, y1), color)

    panel = np.full((image_height + caption_height, panel_width, 3), 255, dtype=np.uint8)
    panel[:image_height] = resized
    border_color = COLORS[sample.target_class]
    cv2.line(panel, (0, image_height), (panel_width - 1, image_height), border_color, 3, cv2.LINE_AA)
    caption = f"SCB-Dataset3-{sample.dataset}  |  {sample.class_names[sample.target_class]}"
    cv2.putText(panel, caption, (28, image_height + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (35, 35, 35), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    used: set[Path] = set()
    samples = choose_samples("S", S_ROOT, S_CLASSES, used) + choose_samples("U", U_ROOT, U_CLASSES, used)
    panels = [render_panel(sample) for sample in samples]
    panel_h, panel_w = panels[0].shape[:2]
    margin = 30
    canvas = np.full((3 * panel_h + 4 * margin, 3 * panel_w + 4 * margin, 3), 255, dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 3)
        x, y = margin + column * (panel_w + margin), margin + row * (panel_h + margin)
        canvas[y : y + panel_h, x : x + panel_w] = panel

    output = OUT_DIR / "SCB-Dataset3-S_U_数据集样例标注九宫格.png"
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(output, dpi=(300, 300))
    selection = [
        {
            "dataset": sample.dataset,
            "category": sample.class_names[sample.target_class],
            "image": str(sample.image_path),
            "annotation_count": int(len(sample.labels)),
        }
        for sample in samples
    ]
    manifest = output.with_suffix(".json")
    manifest.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest), "size_px": [canvas.shape[1], canvas.shape[0]], "dpi": 300}, ensure_ascii=False))


if __name__ == "__main__":
    main()
