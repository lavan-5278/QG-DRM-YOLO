"""Regression tests for the SERA semantic evidence relation aggregator."""

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import SERA, SERADetect


def test_zero_initialised_sera_is_exact_p3_identity_with_shape_preserved():
    """A nonzero initial fusion gain would alter transferred P3 features before the new evidence gate is trained."""
    module = SERA(ch=(8, 16, 32), sera_cfg={"reduction": 0.25})
    p3 = torch.randn(2, 8, 11, 13)
    p4 = torch.randn(2, 16, 6, 7)
    p5 = torch.randn(2, 32, 3, 4)

    output = module([p3, p4, p5])

    assert output.shape == p3.shape
    assert torch.allclose(output, p3, atol=1e-6)


def test_nonzero_sera_gain_routes_gradients_from_refined_p3_to_both_context_scales():
    """Dropping either P4 or P5 projection would leave that scale without a semantic-evidence learning signal."""
    module = SERA(ch=(8, 16, 32), sera_cfg={"reduction": 0.25})
    module.fusion_gain.data.fill_(1.0)
    p3 = torch.randn(1, 8, 11, 13)
    p4 = torch.randn(1, 16, 6, 7)
    p5 = torch.randn(1, 32, 3, 4)

    module([p3, p4, p5]).mean().backward()

    assert module.to_p4.conv.weight.grad is not None
    assert module.to_p4.conv.weight.grad.abs().sum() > 0
    assert module.to_p5.conv.weight.grad is not None
    assert module.to_p5.conv.weight.grad.abs().sum() > 0


def test_sera_yaml_builds_and_preserves_standard_detect_contract():
    """Incorrect SERA parser channels or Detect sources would fail model construction or change NMS inputs."""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-sera.yaml").model.eval()

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert isinstance(model.model[-1], SERADetect)
    assert isinstance(model.model[-1].sera, SERA)
    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 67 for level in raw)
