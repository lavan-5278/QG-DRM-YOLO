"""Unit tests for the VBR visible upper-body local alignment module."""

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import VBR


@pytest.mark.parametrize("p3_hw", [(20, 20), (40, 64)])
def test_vbr_preserves_p3_shape(p3_hw):
    """VBR must return exactly the P3 shape for every legal P2/P3 pair."""
    height, width = p3_hw
    module = VBR((16, 32), {"samples": 5, "offset_radius": 2.0, "alpha_init": 0.0}).eval()
    p2 = torch.randn(2, 16, height * 2, width * 2)
    p3 = torch.randn(2, 32, height, width)

    assert module([p2, p3]).shape == p3.shape


def test_vbr_zero_gate_is_exact_identity():
    """A zero residual gate makes VBR a bitwise P3 identity before learning."""
    module = VBR((16, 32), {"alpha_init": 0.0}).eval()
    p2 = torch.randn(1, 16, 40, 40)
    p3 = torch.randn(1, 32, 20, 20)

    with torch.no_grad():
        output = module([p2, p3])

    assert torch.equal(output, p3)


def test_vbr_nonzero_gate_backpropagates_to_both_scales():
    """The learned local sampler must receive finite gradients from P2 and P3."""
    module = VBR((16, 32), {"alpha_init": 0.25}).train()
    p2 = torch.randn(2, 16, 40, 40, requires_grad=True)
    p3 = torch.randn(2, 32, 20, 20, requires_grad=True)

    output = module([p2, p3])
    output.square().mean().backward()

    assert not torch.equal(output.detach(), p3.detach())
    assert torch.isfinite(output).all()
    assert p2.grad is not None and torch.isfinite(p2.grad).all()
    assert p3.grad is not None and torch.isfinite(p3.grad).all()
    assert module.offset.weight.grad is not None and torch.isfinite(module.offset.weight.grad).all()
    assert module.weight.weight.grad is not None and torch.isfinite(module.weight.weight.grad).all()


def test_vbr_yaml_builds_and_routes_refined_p3():
    """The parser must route VBR's refined P3, not the original P3, into Detect."""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-vbr.yaml").model

    assert isinstance(model.model[-2], VBR)
    assert model.model[-1].f == [23, 19, 22]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the integration forward test")
def test_vbr_yaml_cuda_forward():
    """The registered VBR graph must run a standard 640x640 CUDA detection forward."""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-vbr.yaml").model.cuda().eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640, device="cuda"))

    assert isinstance(output, tuple)
    assert output[0].shape[0] == 1
