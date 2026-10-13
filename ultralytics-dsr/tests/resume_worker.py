"""Background worker for the official-resume integration test.

Trains DSR-Det on the smoke dataset until killed mid-run by the test, so that
``last.pt`` keeps a non-stripped epoch (optimizer state present) and the official
``model.train(resume=True)`` path can be exercised.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # resolve to the project's vendored ultralytics tree

from ultralytics import YOLO
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/resume_test"

model = YOLO(str(ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"))
cfg = model.model.yaml["dsr"]
cfg["candidate_conf"] = 0.0001
cfg["pre_topk"] = 100
cfg["relation_topk"] = 40
model.train(
    trainer=DSRDetectionTrainer,
    data=str(ROOT / "tests/data/smoke.yaml"),
    epochs=6,
    imgsz=128,
    batch=2,
    device="cpu",
    workers=0,
    project=str(OUT),
    name="a",
    plots=False,
    verbose=False,
    deterministic=True,
)
