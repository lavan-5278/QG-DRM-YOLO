"""Main S/U training entrypoint; execute from repository root."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics import __version__ as ultra_version
from qgdrm.ultra import ABLATIONS, trainer_for


def arguments():
    p = argparse.ArgumentParser(description='Train QG-DRM-YOLO from scratch using Ultralytics')
    p.add_argument('--data', required=True, help='YOLO dataset YAML (absolute or relative)')
    p.add_argument('--variant', choices=list(ABLATIONS), default='a4')
    p.add_argument('--device', default='0', help='CUDA device(s), cpu or mps')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--project', default='runs/scb_s')
    p.add_argument('--name', default=None)
    p.add_argument('--resume', default=None, help='Resume via a last.pt file path')
    p.add_argument('--dry-run', action='store_true', help='Print config without actual training')
    return p.parse_args()


def run(args):
    if args.resume:
        # Resume from the full serialized model + optimizer in the checkpoint.
        model = YOLO(args.resume)
        if args.dry_run:
            print(f'Resume checkpoint: {args.resume}')
            return
        model.train(resume=True, device=args.device)
        return

    config = dict(
        data=str(Path(args.data).expanduser().resolve()),
        imgsz=args.imgsz, epochs=args.epochs, patience=100, batch=args.batch,
        optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937,
        weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
        box=7.5, cls=0.5, dfl=1.5,
        mosaic=1.0, close_mosaic=10,
        auto_augment='randaugment', erasing=0.4,
        seed=args.seed, deterministic=True,
        pretrained=False, amp=False,
        workers=args.workers, device=args.device,
        project=args.project, name=args.name or args.variant,
        exist_ok=False, plots=True,
        # Ultralytics defaults for remaining augmentation parameters
        # are version-dependent: pinned version is recorded in README.
    )
    print('Ultralytics version:', ultra_version)
    print('Variant:', args.variant, 'flags:', ABLATIONS[args.variant])
    print(json.dumps(config, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    if not Path(config['data']).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {config['data']}")
    model = YOLO('yolov8n.yaml')  # architecture only, NEVER yolov8n.pt
    model.train(trainer=trainer_for(args.variant), **config)


if __name__ == '__main__':
    run(arguments())
