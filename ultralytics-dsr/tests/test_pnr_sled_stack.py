"""Contract tests for the PNR-DFL + SLED-Head stacked detector."""

from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import SLEDDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import PNRDetectionLoss


STACK_YAML = "ultralytics/cfg/models/11/yolo11n-pnr-sled.yaml"


def test_pnr_sled_yaml_composes_sled_head_with_pnr_loss():
    """The stacked YAML must retain SLED output tensors while selecting the PNR criterion."""
    assert Path(STACK_YAML).is_file()

    model = YOLO(STACK_YAML).model.eval()
    criterion_model = DetectionModel(STACK_YAML, verbose=False)
    criterion_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert isinstance(model.model[-1], SLEDDetect)
    assert model.yaml["pnr"] == {
        "alpha": 1.0,
        "lambda_pnr": 0.25,
        "scale_floor": 1.0,
        "normalize": True,
        "weight_mode": "joint",
    }
    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 67 for level in raw)
    assert isinstance(criterion_model.init_criterion(), PNRDetectionLoss)


def test_stack_queue_builds_exactly_three_fresh_paired_seeds():
    """The stack needs one new run per seed, not repeated Baseline, PNR, or SLED endpoints."""
    from experiments.pnr_sled_stack_queue import OUTPUT_ROOT, build_jobs

    jobs = build_jobs()

    assert [(job.variant, job.seed) for job in jobs] == [("pnr_sled", 42), ("pnr_sled", 43), ("pnr_sled", 44)]
    assert all(job.yaml == STACK_YAML for job in jobs)
    assert all(job.run_dir == str(OUTPUT_ROOT / "pnr_sled" / f"seed{job.seed}") for job in jobs)


def test_stack_queue_source_uses_the_full_joint_pnr_sled_protocol():
    """A queue drift would invalidate paired attribution even if the combined model itself is valid."""
    from experiments.pnr_sled_stack_queue import Job, _training_source

    source = _training_source(Job("pnr_sled", 43, STACK_YAML, "runs/pnr_sled_stack_3seed/pnr_sled/seed43"))

    assert 'model = YOLO("ultralytics/cfg/models/11/yolo11n-pnr-sled.yaml")' in source
    assert 'data="ultralytics/cfg/datasets/SCB-Dataset3-S.yaml", epochs=300, patience=100' in source
    assert "imgsz=640, batch=16, device=0, workers=8" in source
    assert "seed=43, deterministic=True" in source
    assert 'project="runs/pnr_sled_stack_3seed/pnr_sled", name="seed43", exist_ok=False' in source
