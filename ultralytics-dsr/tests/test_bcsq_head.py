"""Regression tests for the BCSQ behavior-conditioned side-quality detection head."""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
import ultralytics.utils.loss as detection_losses
from ultralytics.nn.modules import BCSQDetect
from ultralytics.nn.tasks import DetectionModel, parse_model


def test_bcsq_head_is_available_to_the_yolo_model_parser():
    """A BCSQ YAML must resolve its final head through Ultralytics' normal parser."""
    assert "BCSQDetect" in tasks.__dict__


def test_bcsq_training_predictions_include_four_side_logits_per_class():
    """Dropping the side-quality branch must remove 12 logits per anchor for a 3-class head."""
    head = BCSQDetect(nc=3, ch=(16, 32)).train()
    output = head([torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)])

    assert [level.shape[1] for level in output] == [64 + 3 + 3 * 4, 64 + 3 + 3 * 4]


def test_bcsq_inference_returns_standard_layout_with_geometric_side_quality_calibration():
    """Ignoring side qualities would return 0.5 instead of the expected 0.125 class score."""
    head = BCSQDetect(nc=3, ch=(16,)).eval()
    head.stride = torch.tensor([8.0])
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.cvq[0][-1].bias.fill_(math.log(0.25 / 0.75))

        prediction, raw = head([torch.zeros(1, 16, 2, 2)])

    assert raw[0].shape == (1, 64 + 3 + 3 * 4, 2, 2)
    assert prediction.shape == (1, 4 + 3, 4)
    assert torch.allclose(prediction[:, 4:], torch.full((1, 3, 4), 0.125), atol=1e-6)


def test_bcsq_loss_supervises_side_quality_branch_with_finite_gradients():
    """Replacing BCSQ loss with the base loss would leave every cvq parameter without a gradient."""
    criterion_cls = getattr(detection_losses, "BCSQDetectionLoss", None)
    assert criterion_cls is not None

    head = BCSQDetect(nc=3, ch=(16, 32)).train()
    head.stride = torch.tensor([8.0, 16.0])

    class _TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([head])
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    predictions = head([torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)])
    batch = {
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([0, 2]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.4, 0.5], [0.4, 0.6, 0.3, 0.3]]),
    }
    loss, items = criterion_cls(_TestModel())(predictions, batch)
    loss.sum().backward()

    quality_grads = [parameter.grad for parameter in head.cvq.parameters()]
    assert torch.isfinite(loss).all()
    assert items.shape == (3,)
    assert all(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in quality_grads)


def test_parser_injects_all_detect_feature_channels_into_bcsq_head():
    """A parser regression must not construct BCSQ with an empty channel tuple."""
    config = {
        "nc": 3,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [[[0, 1, 2], 1, "BCSQDetect", [3]]],
    }

    model, saved = parse_model(config, ch=3, verbose=False)

    assert isinstance(model[-1], BCSQDetect)
    assert model[-1].nl == 3
    assert saved == [0, 1, 2]


def test_detection_model_selects_bcsq_loss_for_bcsq_head():
    """Using the base criterion here silently ignores BCSQ's quality logits."""
    config = {
        "nc": 3,
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [[[0, 1, 2], 1, "BCSQDetect", [3]]],
    }

    model = DetectionModel(config, verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    assert isinstance(model.init_criterion(), detection_losses.BCSQDetectionLoss)


def test_parser_passes_bcsq_yaml_configuration_to_head_and_loss():
    """Ignoring YAML config would silently train every BCSQ experiment with the same loss weight."""
    config = {
        "nc": 3,
        "bcsq": {"eta": 0.5, "tau": 0.3, "lambda_quality": 0.25, "background_alpha": 0.6},
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [[[0, 1, 2], 1, "BCSQDetect", [3, "bcsq"]]],
    }

    model, _ = parse_model(config, ch=3, verbose=False)
    head = model[-1]

    assert head.quality_eta == 0.5
    assert head.quality_tau == 0.3
    assert head.quality_lambda == 0.25
    assert head.quality_alpha == 0.6


def test_bcsq_model_yaml_builds_and_preserves_standard_inference_contract():
    """A malformed BCSQ YAML would either fail to build or leak side-quality channels to standard NMS."""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-bcsq.yaml").model.eval()

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert isinstance(model.model[-1], BCSQDetect)
    assert prediction.shape[1] == 4 + 3
    assert all(level.shape[1] == 64 + 3 + 3 * 4 for level in raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression coverage requires a CUDA build of PyTorch")
def test_bcsq_quality_targets_match_half_precision_prediction_dtype_under_amp():
    """Writing float quality targets into half logits must not abort CUDA AMP training."""
    head = BCSQDetect(nc=3, ch=(16,)).cuda()
    head.stride = torch.tensor([8.0], device="cuda")

    class _TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([head])
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    criterion = detection_losses.BCSQDetectionLoss(_TestModel())
    loss = criterion._quality_loss(
        pred_quality=torch.zeros(1, 2, 3, 4, device="cuda", dtype=torch.float16),
        pred_bboxes=torch.tensor([[[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 3.0, 3.0]]], device="cuda", dtype=torch.float16),
        target_bboxes=torch.tensor([[[8.0, 8.0, 24.0, 24.0], [8.0, 8.0, 24.0, 24.0]]], device="cuda"),
        target_scores=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], device="cuda"),
        anchor_points=torch.tensor([[2.0, 2.0], [2.0, 2.0]], device="cuda"),
        stride_tensor=torch.tensor([[8.0], [8.0]], device="cuda"),
        fg_mask=torch.tensor([[True, False]], device="cuda"),
    )

    assert torch.isfinite(loss)
