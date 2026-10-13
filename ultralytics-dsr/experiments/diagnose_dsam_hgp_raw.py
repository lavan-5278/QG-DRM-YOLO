"""Measure raw detector metrics from the paused DSAM/HGP checkpoint.

This diagnostic deliberately bypasses DSA social inference and validation-loss
bookkeeping; it does not alter checkpoint files, queue state, or training.
"""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(r"D:\YH_Project\DSA-Det\dsa_det_ssr_pkg")
sys.path.insert(0, str(PACKAGE_ROOT))

import dsa_det_ssr  # noqa: F401
from ultralytics import YOLO


weights = ROOT / "runs/hrw_yolov8n_dsa_seed42_scratch/native_runs/yolov8n_dsam_hgp/weights/best.pt"
model_yaml = ROOT / "ultralytics/cfg/models/v8/yolov8n-dsam-hgp.yaml"
data_yaml = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"

model = YOLO(str(model_yaml), verbose=False)
model.load(str(weights))
model.model._dsa_initialized = False


def raw_only_loss(batch, preds=None):
    """Avoid recomputing DSA losses; metrics only depend on predictions."""
    return torch.zeros((), device=batch["img"].device), torch.zeros(7, device=batch["img"].device)


model.model.loss = raw_only_loss
metrics = model.val(
    data=str(data_yaml),
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=False,
    save_json=False,
    project=str(ROOT / "runs/hrw_yolov8n_dsa_seed42_scratch/diagnostics"),
    name="raw_detector_best_noloss",
    exist_ok=True,
    verbose=False,
)
print(metrics.results_dict)
