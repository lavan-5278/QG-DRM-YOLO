"""Train the established YOLOv8s best topology on the SCB-Dataset3-U official split."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics import YOLO
from train_scb_dataset3_u_yolov8_baseline import benchmark_checkpoint, read_best_metrics

DATA_YAML = ROOT / "ultralytics/cfg/datasets/SCB-Dataset3-U.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail-scb-u.yaml"
OUTPUT_ROOT = ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_current_best_topology_scb_u_seed42"


def verify_structure(model: YOLO) -> None:
    layers = model.model.model
    expected = ["MLKCM", "MLKCM", "MLKCM", "P3DetailRestore"]
    if layers[6].__class__.__name__ != "RGDRC2fV3" or layers[8].__class__.__name__ != "C2f":
        raise RuntimeError("C4 must be RGDRC2fV3 and C5 must remain native C2f")
    if [layers[index].__class__.__name__ for index in (26, 27, 28, 29)] != expected:
        raise RuntimeError("expected MLKCM(P3/P4/P5) followed by P3DetailRestore")
    if int(getattr(layers[-1], "nc", -1)) != 6:
        raise RuntimeError(f"model class count mismatch: expected 6, got {getattr(layers[-1], 'nc', None)}")
    if list(model.model.stride.cpu().tolist()) != [8.0, 16.0, 32.0]:
        raise RuntimeError(f"unexpected detection strides: {model.model.stride.tolist()}")
    # Both residual adapters are designed to be exact identities at initialization.
    with torch.inference_mode():
        c4 = layers[6]
        c4_input = torch.randn(1, c4.base.cv1.conv.in_channels, 40, 40)
        if not torch.equal(c4(c4_input), c4.base(c4_input)):
            raise RuntimeError("C4 reliability-gated residual adapter is not identity at initialization")
        p3 = layers[29]
        p3_input = torch.randn(1, p3.local.conv.in_channels, 80, 80)
        if not torch.equal(p3(p3_input), p3_input):
            raise RuntimeError("P3 detail residual adapter is not identity at initialization")


def adapter_states(best: Path) -> dict[str, dict]:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().eval()
    result = {name: module.adapter_state() for name, module in model.named_modules() if callable(getattr(module, "adapter_state", None))}
    classes = {module.__class__.__name__ for module in model.modules() if callable(getattr(module, "adapter_state", None))}
    if classes != {"RGDRC2fV3", "P3DetailRestore"} or len(result) != 2:
        raise RuntimeError(f"expected one RGDRC2fV3 and one P3DetailRestore, got {classes} / {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model = YOLO(str(MODEL_YAML))
    verify_structure(model)
    if args.dry_run:
        model.info(verbose=False)
        print(json.dumps({"classes": 6, "strides": model.model.stride.tolist()}, ensure_ascii=False))
        return
    result = model.train(
        data=str(DATA_YAML), epochs=300, patience=100, batch=8, imgsz=640, lr0=0.01,
        optimizer="SGD", seed=42, deterministic=True, device=0, workers=4,
        project=str(RUN_DIR.parent), name=RUN_DIR.name, exist_ok=True, pretrained=False,
        val=True, save=True, plots=True, verbose=True,
    )
    final_dir = Path(result.save_dir)
    best = final_dir / "weights/best.pt"
    if not best.exists():
        raise RuntimeError(f"best checkpoint was not produced: {best}")
    metrics = read_best_metrics(final_dir)
    summary = benchmark_checkpoint(best)
    payload = {
        "dataset": "SCB-Dataset3-U", "status": "completed",
        "job": {"id": "yolov8s_current_best_topology_scb_u_seed42", "name": "YOLOv8s 当前最优结构（QG-AFPN + C4 RGDRC2fV3 + MLKCM + P3DetailRestore）"},
        "protocol": {"scratch": True, "pretrained": False, "seed": 42, "batch": 8, "optimizer": "SGD", "lr0": 0.01, "epochs": 300, "imgsz": 640},
        "metrics": metrics, "adapter_states": adapter_states(best), **summary,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (final_dir / "native_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
