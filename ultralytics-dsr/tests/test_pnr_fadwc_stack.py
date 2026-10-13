"""Regression contracts for the PNR-DFL + FADWC-Head stack."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import FADWC
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import PNRDetectionLoss


STACK_YAML = "ultralytics/cfg/models/11/yolo11n-pnr-fadwc.yaml"


def test_fadwc_preserves_odd_spatial_shape_and_gradients():
    """Frequency routing must retain the Detect feature contract for odd H/W too."""
    module = FADWC(16)
    feature = torch.randn(2, 16, 15, 17, requires_grad=True)

    output = module(feature)

    assert output.shape == feature.shape
    output.mean().backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert module.gamma.grad is not None


def test_pnr_fadwc_yaml_preserves_detect_contract_and_selects_pnr_loss():
    """FADWC must remain a shape-preserving pre-Detect refinement under PNR supervision."""
    model = YOLO(STACK_YAML).model.eval()
    criterion_model = DetectionModel(STACK_YAML, verbose=False)
    criterion_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert all(isinstance(model.model[index], FADWC) for index in (23, 24, 25))
    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 67 for level in raw)
    assert isinstance(criterion_model.init_criterion(), PNRDetectionLoss)


def test_pnr_fadwc_queue_builds_three_fresh_strict_protocol_jobs():
    """The candidate must have three new matched cells and cannot overwrite prior endpoints."""
    from experiments.pnr_fadwc_stack_queue import Job, OUTPUT_ROOT, _training_source, build_jobs

    jobs = build_jobs()
    assert [(job.variant, job.seed) for job in jobs] == [("pnr_fadwc", 42), ("pnr_fadwc", 43), ("pnr_fadwc", 44)]
    assert all(job.run_dir == str(OUTPUT_ROOT / "pnr_fadwc" / f"seed{job.seed}") for job in jobs)

    source = _training_source(Job("pnr_fadwc", 43, STACK_YAML, "runs/pnr_fadwc_stack_3seed/pnr_fadwc/seed43"))
    assert f'model = YOLO("{STACK_YAML}")' in source
    assert 'data="ultralytics/cfg/datasets/SCB-Dataset3-S.yaml", epochs=300, patience=100' in source
    assert "imgsz=640, batch=16, device=0, workers=8" in source
    assert "seed=43, deterministic=True" in source


def test_pnr_fadwc_report_uses_best_epoch_and_paired_sample_std(tmp_path):
    """The report must select validation best rows and report seed-paired uncertainty."""
    from experiments.report_pnr_fadwc_stack import best_metrics, paired_summary

    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.40,0.50,0.60,0.30\n"
        "2,0.50,0.55,0.65,0.40\n"
        "3,0.90,0.60,0.70,0.35\n",
        encoding="utf-8",
    )
    best = best_metrics(csv_path)
    summary = paired_summary(
        {42: {"mAP50-95": 0.50}, 43: {"mAP50-95": 0.52}, 44: {"mAP50-95": 0.54}},
        {42: {"mAP50-95": 0.48}, 43: {"mAP50-95": 0.49}, 44: {"mAP50-95": 0.50}},
    )

    assert best["epoch"] == 2
    assert best["precision"] == 0.50
    assert summary["deltas"] == pytest.approx([0.02, 0.03, 0.04])
    assert summary["mean"] == pytest.approx(0.03)
    assert summary["std"] == pytest.approx(0.01)
