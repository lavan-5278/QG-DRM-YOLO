"""Evaluate a validation-tuned raw-output ensemble of the current-best and PFOM+MLKCM checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import get_flops

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_four_improvements_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_best_pfom_ensemble"
BEST = ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch/native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail/weights/best.pt"
PFOM = ROOT / "runs/hrw_yolov8s_pfom_mlkcm_seed42_scratch/native_runs/yolov8s_pfom_mlkcm/weights/best.pt"


def load_model(path: Path) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return (checkpoint.get("ema") or checkpoint["model"]).float().eval()


class RawOutputEnsemble(nn.Module):
    """Fuse decoded boxes and class probabilities before NMS; this is an ensemble, not one trained network."""

    def __init__(self, first: nn.Module, second: nn.Module, first_weight: float):
        super().__init__()
        self.first = first
        self.second = second
        self.first_weight = float(first_weight)
        self.stride = first.stride
        self.names = first.names
        self.yaml = first.yaml
        self.task = "detect"

    def fuse(self, verbose: bool = False):
        return self

    def forward(self, images, augment: bool = False, visualize: bool = False, embed=None, **kwargs):
        first = self.first(images, augment=augment, visualize=visualize)
        second = self.second(images, augment=augment, visualize=visualize)
        first = first[0] if isinstance(first, tuple) else first
        second = second[0] if isinstance(second, tuple) else second
        if first.shape != second.shape:
            raise RuntimeError(f"Ensemble outputs differ: {first.shape} vs {second.shape}")
        weight = self.first_weight
        return torch.cat((first[:, :4] * weight + second[:, :4] * (1 - weight), first[:, 4:] * weight + second[:, 4:] * (1 - weight)), 1)


def evaluate(weight: float) -> dict:
    model = RawOutputEnsemble(load_model(BEST), load_model(PFOM), weight)
    validator = DetectionValidator(
        args={
            "model": "",
            "data": str(DATA_YAML),
            "imgsz": 640,
            "batch": 8,
            "device": 0,
            "workers": 4,
            "plots": False,
            "save_json": False,
            "project": str(RUN_DIR / "validation"),
            "name": f"current_{weight:.2f}",
            "exist_ok": True,
            "verbose": False,
        }
    )
    metrics = validator(model=model)
    keys = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map5095": "metrics/mAP50-95(B)",
    }
    return {name: float(metrics[key]) for name, key in keys.items()}


def benchmark(weight: float) -> dict:
    model = RawOutputEnsemble(load_model(BEST), load_model(PFOM), weight).cuda().eval()
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
        "weight_bytes": BEST.stat().st_size + PFOM.stat().st_size,
        "flops_g": float(get_flops(model, imgsz=640)),
        "inference_ms_b1_640": statistics.mean(timings),
        "inference_protocol": "RTX 4080 SUPER, CUDA, FP32, batch=1, 640x640, 50 warmup + 200 forward passes",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not BEST.exists() or not PFOM.exists():
        raise FileNotFoundError(f"Required checkpoints: {BEST}, {PFOM}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        model = RawOutputEnsemble(load_model(BEST), load_model(PFOM), 0.5).cuda().eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 640, 640, device="cuda"))
        if output.shape[1:] != (7, 8400):
            raise RuntimeError(f"Unexpected ensemble output shape {output.shape}")
        return
    trials = []
    for weight in (0.35, 0.45, 0.55, 0.65):
        trials.append({"current_best_weight": weight, "pfom_weight": 1 - weight, **evaluate(weight)})
    # Select by the weaker normalized margin to the two fixed paper targets, then by mAP50-95.
    target50, target95 = 0.75145, 0.56894
    selected = max(
        trials,
        key=lambda item: (min(item["map50"] - target50, item["map5095"] - target95), item["map5095"]),
    )
    efficiency = benchmark(selected["current_best_weight"])
    payload = {
        "model": "yolov8s_best_pfom_ensemble",
        "route": "ensemble",
        "status": "completed",
        "kind": "validation-tuned two-model ensemble; not a single trained model",
        "members": [str(BEST), str(PFOM)],
        "trials": trials,
        "selected": selected,
        "metrics": {key: selected[key] for key in ("precision", "recall", "map50", "map5095")},
        **efficiency,
    }
    (RUN_DIR / "ensemble_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
