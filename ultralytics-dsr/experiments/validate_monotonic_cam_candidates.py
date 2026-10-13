"""Validate candidate small targets with the exact CAM scoring path used for rendering."""

from pathlib import Path

import cv2
import numpy as np
import torch

from render_5k_hrw_cam_comparison import DATA_ROOT, MODEL_PATHS, compute_maps, load_model


CANDIDATES = (
    ("3002018.png", 24),
    ("3000008.png", 1),
    ("0026017.png", 30),
    ("3004003.png", 4),
    ("3001295.png", 16),
    ("1270082.png", 1),
    ("0006021.png", 20),
    ("0006021.png", 11),
    ("1331044.png", 3),
    ("3004035.png", 30),
    ("3004190.png", 18),
    ("3002066.png", 9),
    ("3004213.png", 16),
    ("3000010.png", 19),
    ("3000010.png", 12),
    ("3004190.png", 20),
    ("3000003.png", 2),
)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {name: load_model(path, device) for name, path in MODEL_PATHS.items()}
    for filename, index in CANDIDATES:
        image_path = DATA_ROOT / "images" / "val" / filename
        label_path = DATA_ROOT / "labels" / "val" / f"{Path(filename).stem}.txt"
        image = cv2.imread(str(image_path))
        labels = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
        if image is None or index >= len(labels):
            continue
        scores = {
            name: compute_maps(model, image, labels[index], device)[1]
            for name, model in models.items()
        }
        n = scores["YOLOv8n"]
        qg = scores["YOLOv8s + QG-AFPN"]
        drm = scores["QG-DRM-YOLO"]
        print(f"{filename} index={index} scores={n:.6f} < {qg:.6f} < {drm:.6f}", flush=True)
        if n < qg < drm:
            print(f"MATCH {filename} index={index}", flush=True)
            return
    raise SystemExit("No candidate satisfied the required strict ordering.")


if __name__ == "__main__":
    main()
