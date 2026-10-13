"""Regression tests for the SLED spatial-layout evidence decoupling head."""

from copy import deepcopy
from types import SimpleNamespace
import warnings

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import SLEDDetect, SLEDLayoutGate
from ultralytics.nn.tasks import DetectionModel, parse_model


def test_zero_initialised_sled_gate_is_identity_and_keeps_spatial_shape():
    """A non-identity initial gate would immediately corrupt transferred YOLO classification features."""
    gate = SLEDLayoutGate(c=16, bands=3, reduction=0.25, scale=1.0)
    feature = torch.randn(2, 16, 11, 13)

    output = gate(feature)

    assert output.shape == feature.shape
    assert torch.allclose(output, feature, atol=1e-6)


def test_sled_detect_preserves_yolo_raw_and_inference_prediction_contracts():
    """Routing layout evidence into regression or changing channels would break the standard detector contract."""
    head = SLEDDetect(nc=3, ch=(16, 32)).train()
    features = [torch.randn(1, 16, 8, 8), torch.randn(1, 32, 4, 4)]

    raw = head([feature.clone() for feature in features])

    assert [level.shape for level in raw] == [(1, 67, 8, 8), (1, 67, 4, 4)]

    head.eval()
    head.stride = torch.tensor([8.0, 16.0])
    prediction, inference_raw = head([feature.clone() for feature in features])

    assert prediction.shape == (1, 7, 80)
    assert [level.shape for level in inference_raw] == [(1, 67, 8, 8), (1, 67, 4, 4)]


def test_sled_layout_gate_receives_gradient_from_classification_logits():
    """Disconnecting the gate from cv3 would leave the new spatial-layout parameters untrainable."""
    head = SLEDDetect(nc=3, ch=(16,)).train()
    raw = head([torch.randn(1, 16, 8, 8)])

    raw[0][:, 64:].mean().backward()

    assert head.sled[0].project.weight.grad is not None
    assert head.sled[0].project.weight.grad.abs().sum() > 0


def test_parser_injects_sled_yaml_configuration_and_keeps_standard_detection_loss():
    """Dropping the parser special case would construct SLED with no feature channels or use the wrong settings."""
    config = {
        "nc": 3,
        "sled": {"bands": 5, "reduction": 0.5, "scale": 0.75},
        "backbone": [
            [-1, 1, "Conv", [16, 3, 2]],
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
        ],
        "head": [[[0, 1, 2], 1, "SLEDDetect", [3, "sled"]]],
    }

    model, saved = parse_model(deepcopy(config), ch=3, verbose=False)
    detection_model = DetectionModel(deepcopy(config), verbose=False)
    detection_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    assert isinstance(model[-1], SLEDDetect)
    assert [gate.bands for gate in model[-1].sled] == [5, 5, 5]
    assert model[-1].sled[0].scale == 0.75
    assert saved == [0, 1, 2]
    assert detection_model.init_criterion().__class__.__name__ == "v8DetectionLoss"


def test_sled_yaml_builds_with_standard_yolo_inference_output():
    """A bad YAML registration must fail before training rather than after a long experiment has started."""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-sled.yaml").model.eval()

    with torch.no_grad():
        prediction, raw = model(torch.zeros(1, 3, 64, 64))

    assert isinstance(model.model[-1], SLEDDetect)
    assert prediction.shape[1] == 7
    assert all(level.shape[1] == 67 for level in raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="deterministic CUDA backward requires CUDA")
def test_sled_gate_backward_has_no_adaptive_pool_determinism_warning():
    """Using adaptive CUDA pooling would make the strict deterministic training protocol emit a warning."""
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    gate = SLEDLayoutGate(c=8, bands=3).cuda()
    feature = torch.randn(1, 8, 11, 13, device="cuda", requires_grad=True)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gate(feature).sum().backward()
        assert not any("adaptive_avg_pool2d_backward_cuda" in str(warning.message) for warning in caught)
    finally:
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)
