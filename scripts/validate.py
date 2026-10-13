"""Validate best.pt using same evaluation size / dataset split."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import qgdrm.ultra  # noqa: F401 - register custom model classes for checkpoint load


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--split', default='val', choices=('val','test'))
    p.add_argument('--batch', type=int, default=8)
    a = p.parse_args()
    result = YOLO(a.weights).val(
        data=str(Path(a.data).expanduser().resolve()), imgsz=640,
        batch=a.batch, device=a.device, split=a.split,
        plots=True, conf=0.001, iou=0.7,  # Ultralytics default AP evaluation settings
    )
    scores = {
        'precision': float(result.box.mp),
        'recall': float(result.box.mr),
        'map50': float(result.box.map50),
        'map50_95': float(result.box.map),
    }
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
