"""Regression tests for Perspective-Normalized Residual DFL."""

import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import PNRDetectionLoss


def _criterion(
    alpha: float = 1.0,
    loss_weight: float = 0.25,
    normalize: bool = True,
    weight_mode: str = "joint",
):
    """Build the loss helper without model wiring; numerical tests exercise only its pure PNR branch."""
    criterion = object.__new__(PNRDetectionLoss)
    criterion.pnr_alpha = alpha
    criterion.pnr_weight = loss_weight
    criterion.pnr_scale_floor = 1.0
    criterion.pnr_normalize = normalize
    criterion.pnr_weight_mode = weight_mode
    criterion.reg_max = 16
    criterion.proj = torch.arange(16, dtype=torch.float)
    return criterion


def test_pnr_weight_prioritizes_small_high_image_target_over_large_low_target():
    """Removing the perspective/scale terms would make these two geometry cases receive equal loss weight."""
    criterion = _criterion()
    boxes_px = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 40.0, 100.0, 100.0]]])

    weights = criterion._perspective_weights(boxes_px, image_height=100.0, image_width=100.0)

    assert weights.shape == (1, 2)
    assert weights[0, 0] > weights[0, 1]
    assert torch.all(weights >= 1.0)


def test_pnr_weight_modes_isolate_uniform_size_and_vertical_geometry():
    """A full-only weight implementation cannot establish which geometry cue caused a gain."""
    boxes_px = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [0.0, 60.0, 60.0, 100.0]]])

    uniform = _criterion(weight_mode="uniform")._perspective_weights(boxes_px, 100.0, 100.0)
    size = _criterion(weight_mode="size")._perspective_weights(boxes_px, 100.0, 100.0)
    vertical = _criterion(weight_mode="vertical")._perspective_weights(boxes_px, 100.0, 100.0)

    assert torch.allclose(uniform, torch.ones_like(uniform))
    assert size[0, 0] > size[0, 1]
    assert vertical[0, 0] > vertical[0, 1]


def test_pnr_raw_residual_exceeds_scale_normalized_residual_for_same_small_target_error():
    """Removing scale division must expose the absolute-distance alternative rather than silently retaining PNR."""
    logits = torch.full((1, 1, 64), -30.0)
    for side in range(4):
        logits[0, 0, side * 16] = 30.0
    args = dict(
        pred_distri=logits,
        target_bboxes_grid=torch.tensor([[[2.0, 2.0, 6.0, 6.0]]]),
        anchor_points=torch.tensor([[4.0, 4.0]]),
        fg_mask=torch.tensor([[True]]),
        target_bboxes_px=torch.tensor([[[16.0, 16.0, 48.0, 48.0]]]),
        image_height=64.0,
        image_width=64.0,
    )

    normalized = _criterion(normalize=True, weight_mode="uniform")._pnr_residual(**args)
    raw = _criterion(normalize=False, weight_mode="uniform")._pnr_residual(**args)

    assert raw > normalized


def test_pnr_residual_is_zero_when_dfl_expected_distances_equal_targets():
    """Using the wrong DFL expectation axis would make an exactly matched four-side distribution nonzero."""
    criterion = _criterion()
    logits = torch.full((1, 1, 64), -30.0)
    for side in range(4):
        logits[0, 0, side * 16 + 2] = 30.0

    residual = criterion._pnr_residual(
        logits,
        target_bboxes_grid=torch.tensor([[[2.0, 2.0, 6.0, 6.0]]]),
        anchor_points=torch.tensor([[4.0, 4.0]]),
        fg_mask=torch.tensor([[True]]),
        target_bboxes_px=torch.tensor([[[16.0, 16.0, 48.0, 48.0]]]),
        image_height=64.0,
        image_width=64.0,
    )

    assert torch.allclose(residual, torch.tensor(0.0), atol=1e-6)


def test_pnr_residual_backpropagates_for_wrong_distribution():
    """Detaching the distribution expectation would leave PNR unable to improve regression logits."""
    criterion = _criterion()
    logits = torch.zeros((1, 1, 64), requires_grad=True)

    residual = criterion._pnr_residual(
        logits,
        target_bboxes_grid=torch.tensor([[[2.0, 2.0, 6.0, 6.0]]]),
        anchor_points=torch.tensor([[4.0, 4.0]]),
        fg_mask=torch.tensor([[True]]),
        target_bboxes_px=torch.tensor([[[16.0, 16.0, 48.0, 48.0]]]),
        image_height=64.0,
        image_width=64.0,
    )
    residual.backward()

    assert torch.isfinite(residual)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_pnr_yaml_uses_standard_detect_shapes_and_selects_pnr_loss():
    """A loss-only PNR YAML must not add channels or silently fall back to the ordinary criterion."""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-pnr-dfl.yaml").model.eval()
    criterion_model = DetectionModel("ultralytics/cfg/models/11/yolo11n-pnr-dfl.yaml", verbose=False)
    criterion_model.args = type("Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5})()

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 67 for level in raw)
    assert isinstance(criterion_model.init_criterion(), PNRDetectionLoss)
