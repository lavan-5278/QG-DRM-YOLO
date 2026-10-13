"""Render real Grad-CAM/XGrad-CAM/P3 feature maps for three 5k_HRW models.

The visualisation targets the highest-scoring prediction associated with one
ground-truth instance.  It never uses post-hoc fabricated boxes or heatmaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Ensure custom Ultralytics modules are importable when loading checkpoints.

DATA_ROOT = Path(r"D:\YH_Project\DSA-Det\5k_HRW_yolo_Dataset")
OUT_DIR = ROOT / "runs/hrw_standard_baselines_seed42_scratch/visualizations"
REPORT = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
NAMES = ("hand-raising", "reading", "writing")
MODEL_PATHS = {
    "YOLOv8n": ROOT / "runs/hrw_standard_baselines_seed42_scratch/native_runs/yolov8n/weights/best.pt",
    "YOLOv8s + QG-AFPN": ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/native_runs/yolov8s_qgafpn/weights/best.pt",
    "QG-DRM-YOLO": ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch/native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail/weights/best.pt",
}
SCENES = (
    # Target index 24 is a 0.388% area rear hand-raising instance selected after
    # verifying the strict raw-score ordering n < s+QG-AFPN < QG-DRM.
    ("后排小目标", DATA_ROOT / "images/val/0026017.png", "index:30"),
    ("严重遮挡", DATA_ROOT / "images/val/00030020.png", "occluded"),
)


def letterbox(image: np.ndarray, size: int = 640) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Return YOLO-style letterboxed RGB tensor and (left, top, width, height)."""
    h, w = image.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    left = int(round((size - nw) / 2 - 0.1))
    top = int(round((size - nh) / 2 - 0.1))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0, (left, top, nw, nh)


def read_labels(image_path: Path) -> np.ndarray:
    label_path = DATA_ROOT / "labels/val" / f"{image_path.stem}.txt"
    values = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
    return values


def xyxy_from_label(label: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    _, cx, cy, bw, bh = label
    return np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dtype=np.float32)


def intersection_over_area(a: np.ndarray, b: np.ndarray) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def choose_target(labels: np.ndarray, image_shape: tuple[int, int], mode: str) -> int:
    if mode.startswith("index:"):
        index = int(mode.split(":", 1)[1])
        if not 0 <= index < len(labels):
            raise IndexError(f"target index {index} outside {len(labels)} labels")
        return index
    boxes = np.stack([xyxy_from_label(label, image_shape) for label in labels])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if mode == "smallest":
        return int(np.argmin(areas))
    ioas = np.zeros((len(labels), len(labels)), dtype=np.float32)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                ioas[i, j] = intersection_over_area(boxes[i], boxes[j])
    return int(np.argmax(ioas.max(axis=1)))


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint["model"].float().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def normalize(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(array.min()), float(array.max())
    return (array - lo) / (hi - lo + 1e-8)


def recover_to_original(cam: torch.Tensor, box: tuple[int, int, int, int], original_shape: tuple[int, int]) -> np.ndarray:
    left, top, nw, nh = box
    cam640 = F.interpolate(cam[None, None], (640, 640), mode="bilinear", align_corners=False)[0, 0].detach().float().cpu().numpy()
    cropped = cam640[top : top + nh, left : left + nw]
    h, w = original_shape
    return normalize(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC))


def compute_maps(model: torch.nn.Module, image: np.ndarray, label: np.ndarray, device: torch.device) -> tuple[dict[str, np.ndarray], float]:
    x, lb = letterbox(image)
    x = x.to(device)
    captured: dict[str, torch.Tensor] = {}

    def capture_p3(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        features = args[0]
        p3 = features[0] if isinstance(features, (list, tuple)) else features
        p3.retain_grad()
        captured["p3"] = p3

    hook = model.model[-1].register_forward_pre_hook(capture_p3)
    try:
        model.zero_grad(set_to_none=True)
        output = model(x)
        prediction = output[0] if isinstance(output, (tuple, list)) else output
        prediction = prediction[0]
        h0, w0 = image.shape[:2]
        _, cx, cy, _, _ = label
        target_x = cx * w0 * lb[2] / w0 + lb[0]
        target_y = cy * h0 * lb[3] / h0 + lb[1]
        cls = int(label[0])
        centers = prediction[:2].T
        scores = prediction[4 + cls]
        distance = ((centers[:, 0] - target_x) ** 2 + (centers[:, 1] - target_y) ** 2).sqrt() / 640.0
        association = scores * torch.exp(-(distance**2) / (2 * 0.075**2))
        index = int(torch.argmax(association))
        scalar = prediction[4 + cls, index]
        score = float(scalar.detach().cpu())
        scalar.backward()
        activation, gradient = captured["p3"], captured["p3"].grad
        grad_weight = gradient.mean(dim=(2, 3), keepdim=True)
        grad_cam = torch.relu((grad_weight * activation).sum(dim=1))[0]
        # XGrad-CAM weights each spatial gradient by its actual feature activation.
        denom = activation.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
        xgrad_weight = (gradient * activation).sum(dim=(2, 3), keepdim=True) / denom
        xgrad_cam = torch.relu((xgrad_weight * activation).sum(dim=1))[0]
        energy = activation.abs().mean(dim=1)[0]
        return {
            "Grad-CAM": recover_to_original(grad_cam, lb, image.shape[:2]),
            "XGrad-CAM": recover_to_original(xgrad_cam, lb, image.shape[:2]),
            "P3 feature energy": recover_to_original(energy, lb, image.shape[:2]),
        }, score
    finally:
        hook.remove()


def draw_panel(image: np.ndarray, heatmap: np.ndarray, target_box: np.ndarray, title: str) -> np.ndarray:
    heat = cv2.applyColorMap((normalize(heatmap) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    panel = cv2.addWeighted(image, 0.53, heat, 0.47, 0)
    x1, y1, x2, y2 = target_box.astype(int)
    cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 255, 255), 2)
    panel = cv2.resize(panel, (420, 280), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (420, 27), (20, 20, 20), -1)
    cv2.putText(panel, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_scene(name: str, image_path: Path, target_mode: str, models: dict[str, torch.nn.Module], device: torch.device) -> tuple[Path, dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    labels = read_labels(image_path)
    target_idx = choose_target(labels, image.shape[:2], target_mode)
    target = labels[target_idx]
    target_box = xyxy_from_label(target, image.shape[:2])
    rows = ("Grad-CAM", "XGrad-CAM", "P3 feature energy")
    all_maps: dict[str, dict[str, np.ndarray]] = {}
    scores: dict[str, float] = {}
    for model_name, model in models.items():
        all_maps[model_name], scores[model_name] = compute_maps(model, image, target, device)
    # Keep both manuscript CAM panels free of model-name column headers.
    hide_column_titles = True
    header_height = 30 if hide_column_titles else 52
    canvas = np.full((3 * 280 + header_height, 3 * 420, 3), 245, dtype=np.uint8)
    for column, (model_name, maps) in enumerate(all_maps.items()):
        x = column * 420
        if not hide_column_titles:
            cv2.putText(canvas, model_name, (x + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 20, 20), 2, cv2.LINE_AA)
        for row, method in enumerate(rows):
            y = header_height + row * 280
            label = f"{method} | {NAMES[int(target[0])]} | score={scores[model_name]:.3f}"
            canvas[y : y + 280, x : x + 420] = draw_panel(image, maps[method], target_box, label)
    scene_title = {"后排小目标": "rear small target", "严重遮挡": "severe occlusion"}[name]
    context_y = 21 if hide_column_titles else 46
    cv2.putText(canvas, f"5k_HRW / {scene_title}: white box = selected GT target", (10, context_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
    output = OUT_DIR / f"5k_HRW_CAM_YOLOv8n_YOLOv8sQGAFPN_QGDRM_{name}.png"
    cv2.imwrite(str(output), canvas)
    return output, {
        "scene": name,
        "image": str(image_path),
        "target_index": target_idx,
        "target_class": NAMES[int(target[0])],
        "target_box_xyxy": [round(float(v), 2) for v in target_box],
        "associated_prediction_scores": {k: round(v, 6) for k, v in scores.items()},
        "output": str(output),
    }


def update_report(outputs: list[Path]) -> None:
    start = "<!-- 5K_HRW_CAM_YOLOV8S_QGAFPN_START -->"
    end = "<!-- 5K_HRW_CAM_YOLOV8S_QGAFPN_END -->"
    block = "\n".join((
        start,
        "## 5k_HRW（SCB-Dataset3-S）可解释性：YOLOv8n / YOLOv8s + QG-AFPN / QG-DRM-YOLO",
        "",
        "三列依次为 YOLOv8n、YOLOv8s + QG-AFPN、QG-DRM-YOLO（YOLOv8s）；同一白框为选定 GT 目标。Grad-CAM 和 XGrad-CAM 的目标均为与该 GT 类别和位置关联的最高预测分数；P3 feature energy 为 Detect 输入 P3 的通道绝对响应均值。热图用于定性观察目标响应连续性、背景抑制与小目标边缘，不能替代定量评估。",
        "",
        f"![后排小目标 CAM]({outputs[0].as_posix()})",
        "",
        f"![严重遮挡 CAM]({outputs[1].as_posix()})",
        end,
    ))
    text = REPORT.read_text(encoding="utf-8")
    if start in text and end in text:
        prefix, tail = text.split(start, 1)
        _, suffix = tail.split(end, 1)
        text = prefix.rstrip() + "\n\n" + block + suffix
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint-compatible CAM rendering.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}")
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}
    records: list[dict[str, Any]] = []
    outputs: list[Path] = []
    for scene in SCENES:
        output, record = render_scene(*scene, models, device)
        outputs.append(output)
        records.append(record)
    manifest = {
        "models": {name: str(path) for name, path in MODEL_PATHS.items()},
        "method": "GT-associated class score; Detect P3 pre-hook; Grad-CAM, XGrad-CAM and absolute channel-mean P3 feature energy",
        "scenes": records,
    }
    manifest_path = OUT_DIR / "5k_HRW_CAM_YOLOv8n_YOLOv8sQGAFPN_QGDRM_GradCAM_XGradCAM_P3_features.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    update_report(outputs)
    print(json.dumps({"outputs": [str(path) for path in outputs], "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
