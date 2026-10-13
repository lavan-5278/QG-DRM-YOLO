"""Regression tests for the head--behaviour-part complementarity detection head."""

import torch

from ultralytics import YOLO
from ultralytics.nn.modules import HBPC, HBPCDetect


def test_zero_initialised_hbpc_preserves_shape_and_input_exactly():
    """Removing the zero gain would perturb transferred class features before HBPC can learn."""
    module = HBPC(8, {"shift": 1, "reduction": 0.25})
    x = torch.randn(2, 8, 11, 13)

    output = module(x)

    assert output.shape == x.shape
    assert torch.allclose(output, x, atol=1e-6)


def test_enabled_hbpc_routes_gradients_to_head_and_behaviour_paths():
    """Dropping either directional path would prevent complementary upper/lower evidence learning."""
    module = HBPC(8, {"shift": 1, "reduction": 0.25})
    module.gain.data.fill_(1.0)

    module(torch.randn(1, 8, 11, 13)).mean().backward()

    assert module.head_reduce.weight.grad is not None
    assert module.head_reduce.weight.grad.abs().sum() > 0
    assert module.behaviour_reduce.weight.grad is not None
    assert module.behaviour_reduce.weight.grad.abs().sum() > 0


def test_hbpc_yaml_builds_and_preserves_standard_detect_contract():
    """Wrong parser channels or altered Detect output would break YOLO loss/NMS consumers."""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-hbpc.yaml").model.eval()

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert isinstance(model.model[-1], HBPCDetect)
    assert isinstance(model.model[-1].hbpc[0], HBPC)
    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 67 for level in raw)
