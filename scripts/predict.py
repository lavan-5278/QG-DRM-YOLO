"""Inference with a trained paper-model checkpoint."""
import argparse
from ultralytics import YOLO
import qgdrm.ultra  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--source', required=True, help='image, directory, or video')
    p.add_argument('--device', default='0')
    p.add_argument('--conf', type=float, default=0.25)
    a = p.parse_args()
    YOLO(a.weights).predict(source=a.source, imgsz=640, conf=a.conf,
                            device=a.device, save=True)


if __name__ == '__main__':
    main()
