"""Regression tests for LFM-inspired YOLOv8n backbone blocks."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import LGC, LGRB, LHGB


@pytest.mark.parametrize(
    ("block", "feature", "context"),
    [
        (LGC(c1=16), torch.randn(2, 16, 7, 9), None),
        (LGRB(c1=16, relation_dim=8, max_anchors=6, neighbor_k=3), torch.randn(2, 16, 7, 9), None),
        (LHGB(c1=16, context_channels=12, relation_dim=8, max_anchors=6, neighbor_k=3), torch.randn(2, 16, 7, 9), torch.randn(2, 12, 13, 15)),
    ],
)
def test_lfm_blocks_start_as_finite_shape_preserving_identities(block, feature, context):
    """Zero residual initialization must protect scratch YOLOv8n features at optimization step zero."""
    block.eval()
    output = block(feature) if context is None else block([feature, context])

    assert output.shape == feature.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output, feature)


@pytest.mark.parametrize(
    ("block", "feature", "context"),
    [
        (LGC(c1=16), torch.randn(1, 16, 7, 9), None),
        (LGRB(c1=16, relation_dim=8, max_anchors=80, neighbor_k=80), torch.randn(1, 16, 7, 9), None),
        (LHGB(c1=16, context_channels=12, relation_dim=8, max_anchors=80, neighbor_k=80), torch.randn(1, 16, 7, 9), torch.randn(1, 12, 13, 15)),
    ],
)
def test_lfm_blocks_emit_finite_nonidentity_updates_when_residual_is_enabled(block, feature, context):
    """The local state and sparse relation paths must remain active even when anchor limits exceed map cells."""
    block.eval()
    with torch.no_grad():
        block.gamma.fill_(1.0)
    output = block(feature) if context is None else block([feature, context])

    assert output.shape == feature.shape
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, feature)


@pytest.mark.parametrize("yaml_name", ["yolov8n-lgc.yaml", "yolov8n-lgrb.yaml", "yolov8n-lhgb.yaml"])
def test_lfm_backbone_yaml_builds_a_three_class_detector_with_finite_640_predictions(yaml_name: str):
    """Parser registration and C3/C4/C5 routing must be proven before any 300-epoch training begins."""
    yaml_path = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/v8" / yaml_name
    model = YOLO(yaml_path).model.eval()

    with torch.no_grad():
        prediction, _ = model(torch.zeros(1, 3, 640, 640))

    assert prediction.shape[1] == 7
    assert torch.isfinite(prediction).all()
