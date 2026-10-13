"""Train one of the three single-model routes in the four-route YOLOv8s improvement study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from train_hrw_yolov8s_dr_c2f_v3 import benchmark_checkpoint, read_best_metrics

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_four_improvements_seed42_scratch"
JOBS = {
    "cbqfl": {
        "id": "yolov8s_best_cbqfl",
        "yaml": ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail-cbqfl.yaml",
    },
    "distill": {
        "id": "yolov8s_best_dual_teacher_distill",
        "yaml": ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail-distill.yaml",
    },
    "p2": {
        "id": "yolov8s_best_p2_detect",
        "yaml": ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail-p2.yaml",
    },
}


def verify(job: str, model: YOLO) -> None:
    layers = model.model.model
    if layers[6].__class__.__name__ != "RGDRC2fV3" or layers[29].__class__.__name__ != "P3DetailRestore":
        raise RuntimeError("The current-best C4 RGDRC2fV3/P3DetailRestore structure was not preserved")
    if job == "p2":
        if len(model.model.model[-1].stride) != 4 or list(map(float, model.model.model[-1].stride)) != [4, 8, 16, 32]:
            raise RuntimeError(f"P2 Detect must use strides [4, 8, 16, 32], got {model.model.model[-1].stride}")
    elif len(model.model.model[-1].stride) != 3:
        raise RuntimeError("Non-P2 routes must keep the original three Detect scales")
    if job == "cbqfl" and not model.model.yaml.get("cbqfl", {}).get("enabled"):
        raise RuntimeError("CB-QFL YAML switch is missing")
    if job == "distill" and not model.model.yaml.get("distill", {}).get("enabled"):
        raise RuntimeError("Distillation YAML switch is missing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, choices=JOBS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = JOBS[args.job]
    run_dir = OUTPUT_ROOT / "native_runs" / spec["id"]
    model = YOLO(str(spec["yaml"]))
    verify(args.job, model)
    if args.dry_run:
        model.info(verbose=False)
        return
    result = model.train(
        data=str(DATA_YAML),
        epochs=300,
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
    best = final_dir / "weights/best.pt"
    metrics = read_best_metrics(final_dir)
    summary = benchmark_checkpoint(best)
    payload = {
        "model": spec["id"],
        "route": args.job,
        "status": "completed",
        "protocol": {
            "dataset": "5k_HRW",
            "seed": 42,
            "pretrained": False,
            "batch": 8,
            "optimizer": "SGD",
            "lr0": 0.01,
            "epochs": 300,
            "imgsz": 640,
        },
        "metrics": metrics,
        **summary,
    }
    (final_dir / "native_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
