"""Regression contracts for the PNR-DFL + three-scale AFPN stack."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import AFPN
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import PNRDetectionLoss


STACK_YAML = "ultralytics/cfg/models/11/yolo11n-pnr-afpn.yaml"


def test_afpn_preserves_three_scale_shapes_and_gradients():
    """AFPN must refine P3/P4/P5 without changing their contracts to Detect."""
    module = AFPN((16, 32, 64))
    features = [
        torch.randn(2, 16, 16, 16, requires_grad=True),
        torch.randn(2, 32, 8, 8, requires_grad=True),
        torch.randn(2, 64, 4, 4, requires_grad=True),
    ]

    outputs = module(features)
    assert [output.shape for output in outputs] == [feature.shape for feature in features]

    sum(output.mean() for output in outputs).backward()
    assert all(feature.grad is not None and feature.grad.abs().sum() > 0 for feature in features)


def test_pnr_afpn_yaml_preserves_detect_contract_and_selects_pnr_loss():
    """AFPN must be a Neck-only change: normal Detect tensors and PNR criterion remain intact."""
    model = YOLO(STACK_YAML).model.eval()
    criterion_model = DetectionModel(STACK_YAML, verbose=False)
    criterion_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert isinstance(model.model[23], AFPN)
    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 67 for level in raw)
    assert isinstance(criterion_model.init_criterion(), PNRDetectionLoss)


def test_pnr_afpn_queue_builds_three_fresh_strict_protocol_jobs():
    """A stack needs exactly three new matched cells, never overwritten endpoints."""
    from experiments.pnr_afpn_stack_queue import Job, OUTPUT_ROOT, _training_source, build_jobs

    jobs = build_jobs()
    assert [(job.variant, job.seed) for job in jobs] == [("pnr_afpn", 42), ("pnr_afpn", 43), ("pnr_afpn", 44)]
    assert all(job.run_dir == str(OUTPUT_ROOT / "pnr_afpn" / f"seed{job.seed}") for job in jobs)

    source = _training_source(Job("pnr_afpn", 43, STACK_YAML, "runs/pnr_afpn_stack_3seed/pnr_afpn/seed43"))
    assert f'model = YOLO("{STACK_YAML}")' in source
    assert 'data="ultralytics/cfg/datasets/SCB-Dataset3-S.yaml", epochs=300, patience=100' in source
    assert "imgsz=640, batch=16, device=0, workers=8" in source
    assert "seed=43, deterministic=True" in source


def test_pnr_afpn_report_uses_best_epoch_and_paired_sample_std(tmp_path):
    """The final report must use each seed's best mAP50-95 row and seed-paired deltas."""
    from experiments.report_pnr_afpn_stack import best_metrics, paired_summary

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
