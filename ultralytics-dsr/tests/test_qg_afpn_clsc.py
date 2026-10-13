"""Contract tests for the standalone QG-AFPN and CLSC YOLOv8n variants."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import CLSC, CLSCDetect, QGAFPN


ROOT = Path(__file__).resolve().parents[1]


def test_qgafpn_identity_and_shapes():
    """QG-AFPN preserves P3/P4/P5 contracts and starts with neutral modulation."""
    module = QGAFPN((64, 128, 256)).eval()
    features = [torch.randn(1, 64, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20)]
    outputs = module(features)
    assert [x.shape for x in outputs] == [x.shape for x in features]
    assert all(torch.isfinite(x).all() for x in outputs)
    assert all(torch.equal(scale, torch.zeros_like(scale)) for scale in module.modulation_scales)


def test_clsc_identity_routes_and_detect_contract():
    """CLSC provides three class and three localization paths with a Detect-compatible head."""
    adapter = CLSC((64, 128, 256)).eval()
    features = [torch.randn(1, 64, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20)]
    routed = adapter(features)
    assert len(routed) == 6
    assert [x.shape for x in routed[:3]] == [x.shape for x in features]
    assert [x.shape for x in routed[3:]] == [x.shape for x in features]
    head = CLSCDetect(nc=3, ch=(64, 128, 256)).eval()
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    prediction, _ = head(routed)
    assert prediction.shape[1] == 7
    assert torch.isfinite(prediction).all()


@pytest.mark.parametrize("name", ["yolov8n-qg-afpn.yaml", "yolov8n-clsc.yaml"])
def test_custom_model_yaml_builds(name):
    """Both standalone YAML graphs build and emit the ordinary YOLO detection contract."""
    model = YOLO(ROOT / "ultralytics/cfg/models/v8" / name).model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.zeros(1, 3, 640, 640))
    assert prediction.shape[1] == 7
    assert torch.isfinite(prediction).all()
