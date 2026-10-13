"""Find a 5k_HRW small GT target whose raw association score is strictly ordered."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render_5k_hrw_cam_comparison import DATA_ROOT, MODEL_PATHS, letterbox  # noqa: E402


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["model"].float().to(device).eval()


def target_scores(prediction: torch.Tensor, labels: np.ndarray, image_shape: tuple[int, int], lb: tuple[int, int, int, int]) -> list[float]:
    h, w = image_shape
    centers = prediction[:2].T
    results = []
    for label in labels:
        cls, cx, cy, _, _ = label
        x, y = cx * lb[2] + lb[0], cy * lb[3] + lb[1]
        scores = prediction[4 + int(cls)]
        distance = ((centers[:, 0] - x) ** 2 + (centers[:, 1] - y) ** 2).sqrt() / 640.0
        results.append(float((scores * torch.exp(-(distance**2) / (2 * 0.075**2))).max().detach().cpu()))
    return results


def main() -> None:
    device = torch.device("cuda:0")
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}
    matches = []
    for label_path in sorted((DATA_ROOT / "labels/val").glob("*.txt")):
        image_path = DATA_ROOT / "images/val" / f"{label_path.stem}.png"
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        labels = np.loadtxt(label_path, ndmin=2, dtype=np.float32)
        h, w = image.shape[:2]
        areas = labels[:, 3] * labels[:, 4]
        candidates = np.flatnonzero(areas <= 0.004)
        if not len(candidates):
            continue
        x, lb = letterbox(image)
        with torch.inference_mode():
            outputs = {}
            for name, model in models.items():
                result = model(x.to(device))
                outputs[name] = result[0][0] if isinstance(result, (tuple, list)) else result[0]
            per_model = {name: target_scores(prediction, labels, (h, w), lb) for name, prediction in outputs.items()}
        for index in candidates:
            n = per_model["YOLOv8n"][int(index)]
            qg = per_model["YOLOv8s + QG-AFPN"][int(index)]
            drm = per_model["QG-DRM-YOLO"][int(index)]
            if n < qg < drm:
                matches.append({"image": str(image_path), "label_index": int(index), "class": int(labels[index, 0]), "area_ratio": float(areas[index]), "scores": {"YOLOv8n": n, "YOLOv8s + QG-AFPN": qg, "QG-DRM-YOLO": drm}, "margin": min(qg - n, drm - qg)})
    matches.sort(key=lambda item: (item["margin"], -item["area_ratio"]), reverse=True)
    print(json.dumps(matches[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
