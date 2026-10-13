"""Train the packaged QG-DRM-YOLO model with the verified 5k_HRW seed42 protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail.yaml"
DEFAULT_DATA = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
DEFAULT_PROJECT = ROOT / "runs/qg_drm_yolo_seed42_scratch"
DEFAULT_NAME = "yolov8s_qg_drm_yolo"


def verify_structure(model: YOLO) -> None:
    """Fail early if the YAML does not build the exact packaged architecture."""
    layers = model.model.model
    expected = {
        6: "RGDRC2fV3",
        8: "C2f",
        22: "QGAFPN",
        26: "MLKCM",
        27: "MLKCM",
        28: "MLKCM",
        29: "P3DetailRestore",
        30: "Detect",
    }
    actual = {index: layers[index].__class__.__name__ for index in expected}
    if actual != expected:
        raise RuntimeError(f"Unexpected architecture: expected {expected}, got {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model.resolve()))
    verify_structure(model)

    if args.dry_run:
        model.info(verbose=False)
        print("QG-DRM-YOLO structure verification passed.")
        return

    model.train(
        data=str(args.data.resolve()),
        epochs=300,
        batch=8,
        imgsz=640,
        lr0=0.01,
        optimizer="SGD",
        seed=42,
        deterministic=True,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        pretrained=False,
        val=True,
        save=True,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
