"""Regression tests for the PNR-DFL + BCSQ-Head stacked detector."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics import YOLO
import ultralytics.utils.loss as detection_losses
from ultralytics.nn.modules import BCSQDetect
from ultralytics.nn.tasks import DetectionModel


STACK_YAML = "ultralytics/cfg/models/11/yolo11n-pnr-bcsq.yaml"


def test_pnr_bcsq_loss_is_a_registered_hybrid_not_a_silent_bcsq_fallback():
    """A stacked YAML must keep BCSQ quality supervision *and* add PNR to the DFL column."""
    criterion_cls = getattr(detection_losses, "PNRBCSQDetectionLoss", None)
    assert criterion_cls is not None

    head = BCSQDetect(nc=3, ch=(16, 32)).train()
    head.stride = torch.tensor([8.0, 16.0])

    class _TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([head])
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
            self.yaml = {
                "pnr": {
                    "alpha": 1.0,
                    "lambda_pnr": 0.25,
                    "scale_floor": 1.0,
                    "normalize": True,
                    "weight_mode": "joint",
                }
            }

    predictions = head([torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)])
    batch = {
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([0, 2]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.5], [0.4, 0.6, 0.3, 0.3]]),
    }
    loss, items = criterion_cls(_TestModel())(predictions, batch)
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert items.shape == (3,)
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in head.cvq.parameters())


def test_pnr_bcsq_yaml_selects_hybrid_loss_and_standard_prediction_layout():
    """BCSQ's extra training logits must not leak beyond the raw branch, and PNR must not be ignored."""
    model = YOLO(STACK_YAML).model.eval()
    criterion_model = DetectionModel(STACK_YAML, verbose=False)
    criterion_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert isinstance(model.model[-1], BCSQDetect)
    assert isinstance(criterion_model.init_criterion(), detection_losses.PNRBCSQDetectionLoss)
    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 64 + 3 + 3 * 4 for level in raw)


def test_pnr_bcsq_queue_builds_exactly_three_new_paired_seeds():
    """The stack needs three fresh cells, not another baseline or an overwritten endpoint."""
    from experiments.pnr_bcsq_stack_queue import OUTPUT_ROOT, build_jobs

    jobs = build_jobs()

    assert [(job.variant, job.seed) for job in jobs] == [("pnr_bcsq", 42), ("pnr_bcsq", 43), ("pnr_bcsq", 44)]
    assert all(job.yaml == STACK_YAML for job in jobs)
    assert all(job.run_dir == str(OUTPUT_ROOT / "pnr_bcsq" / f"seed{job.seed}") for job in jobs)


def test_pnr_bcsq_queue_source_preserves_the_strict_matched_protocol():
    """Training-protocol drift would make the stacked comparison scientifically uninterpretable."""
    from experiments.pnr_bcsq_stack_queue import Job, _training_source

    source = _training_source(Job("pnr_bcsq", 43, STACK_YAML, "runs/pnr_bcsq_stack_3seed/pnr_bcsq/seed43"))

    assert 'model = YOLO("ultralytics/cfg/models/11/yolo11n-pnr-bcsq.yaml")' in source
    assert 'data="ultralytics/cfg/datasets/SCB-Dataset3-S.yaml", epochs=300, patience=100' in source
    assert "imgsz=640, batch=16, device=0, workers=8" in source
    assert "seed=43, deterministic=True" in source
    assert 'project="runs/pnr_bcsq_stack_3seed/pnr_bcsq", name="seed43", exist_ok=False' in source


def test_pnr_bcsq_report_uses_each_seed_best_map_epoch_and_paired_deltas(tmp_path):
    """A last-row or unpaired report can turn ordinary training variance into a false stacking claim."""
    from experiments.report_pnr_bcsq_stack import best_metrics, paired_summary

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
    assert best["mAP50-95"] == 0.40
    assert summary["deltas"] == pytest.approx([0.02, 0.03, 0.04])
    assert abs(summary["mean"] - 0.03) < 1e-12
    assert summary["std"] == pytest.approx(0.01)
