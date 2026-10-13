"""Train one native YOLOv8 scratch baseline on the SCB-Dataset3-U official split."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

DATA_YAML = ROOT / "ultralytics/cfg/datasets/SCB-Dataset3-U.yaml"
MODEL_YAMLS = {
    "n": ROOT / "ultralytics/cfg/models/v8/yolov8n-scb-u.yaml",
    "s": ROOT / "ultralytics/cfg/models/v8/yolov8s-scb-u.yaml",
}
OUTPUT_ROOT = ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch"

JOBS = {
    "n": {"id": "yolov8n_scb_u_seed42", "name": "YOLOv8n（原生基线）"},
    "s": {"id": "yolov8s_scb_u_seed42", "name": "YOLOv8s（原生基线）"},
}


def read_best_metrics(run_dir: Path) -> dict[str, str]:
    """Return the validation row selected by mAP50-95, with losses for traceability."""
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("results.csv has no validation rows")
    map95_key = next(key for key in rows[0] if "mAP50-95" in key)
    best = max(rows, key=lambda row: float(row[map95_key]))
    keys = {
        "epoch": "epoch",
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map5095": map95_key,
        "train_box": "train/box_loss",
        "train_cls": "train/cls_loss",
        "train_dfl": "train/dfl_loss",
        "val_box": "val/box_loss",
        "val_cls": "val/cls_loss",
        "val_dfl": "val/dfl_loss",
    }
    return {name: best.get(key, "") for name, key in keys.items()}


def build_model(scale: str) -> YOLO:
    """Build a native YOLOv8 n/s detector from YAML, never from pretrained weights."""
    if scale not in JOBS:
        raise ValueError(f"unsupported scale: {scale}")
    model = YOLO(str(MODEL_YAMLS[scale]), task="detect")
    detect = model.model.model[-1]
    if int(getattr(detect, "nc", -1)) != 6:
        raise RuntimeError(f"model class count mismatch: expected 6, got {getattr(detect, 'nc', None)}")
    if list(model.model.stride.cpu().tolist()) != [8.0, 16.0, 32.0]:
        raise RuntimeError(f"unexpected detection strides: {model.model.stride.tolist()}")
    return model


def benchmark_checkpoint(best: Path) -> dict[str, float | int | str]:
    """Measure the final checkpoint under the common 640 FP32 inference protocol."""
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().cuda().eval()
    sample = torch.zeros((1, 3, 640, 640), device="cuda")
    with torch.inference_mode():
        for _ in range(50):
            model(sample)
        torch.cuda.synchronize()
        timings = []
        for _ in range(200):
            start = time.perf_counter()
            model(sample)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "weight_bytes": best.stat().st_size,
        "flops_g": float(get_flops(model, imgsz=640)),
        "inference_ms_b1_640": statistics.mean(timings),
        "inference_protocol": "RTX 4080 SUPER, CUDA, FP32, batch=1, 640x640, 50 warmup + 200 synchronized forward passes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=sorted(JOBS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job = JOBS[args.scale]
    run_dir = OUTPUT_ROOT / "native_runs" / job["id"]
    model = build_model(args.scale)
    if args.dry_run:
        model.info(verbose=False)
        print(json.dumps({"job": job["id"], "classes": 6, "strides": model.model.stride.tolist()}, ensure_ascii=False))
        return

    result = model.train(
        data=str(DATA_YAML),
        epochs=300,
        patience=100,
        batch=8,
        imgsz=640,
        lr0=0.01,
        optimizer="SGD",
        seed=42,
        deterministic=True,
        device=0,
        workers=4,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        pretrained=False,
        val=True,
        save=True,
        plots=True,
        verbose=True,
    )
    final_dir = Path(result.save_dir)
    best = final_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"best checkpoint was not produced: {best}")
    metrics = read_best_metrics(final_dir)
    summary = benchmark_checkpoint(best)
    payload = {
        "dataset": "SCB-Dataset3-U",
        "job": job,
        "status": "completed",
        "protocol": {
            "scratch": True,
            "pretrained": False,
            "seed": 42,
            "batch": 8,
            "optimizer": "SGD",
            "lr0": 0.01,
            "epochs": 300,
            "imgsz": 640,
        },
        "metrics": metrics,
        **summary,
    }
    (final_dir / "native_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
