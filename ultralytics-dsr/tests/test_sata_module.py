"""Regression tests for the SATA sparse social-anchor backbone adapter."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import DRSATA, RGSATA, SATA


def test_sata_starts_as_a_finite_shape_preserving_identity():
    """Removing neutral residual initialization would perturb scratch backbone features on step zero."""
    adapter = SATA(c1=32, relation_dim=16, max_anchors=8, neighbor_k=3).eval()
    feature = torch.randn(2, 32, 10, 12)

    output = adapter(feature)

    assert output.shape == feature.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output, feature)


def test_sata_writes_a_nonidentity_context_update_after_residual_is_enabled():
    """Dropping anchor-message scatter-back would leave SATA inert even after its residual opens."""
    adapter = SATA(c1=16, relation_dim=8, max_anchors=6, neighbor_k=3).eval()
    feature = torch.randn(1, 16, 7, 9)
    with torch.no_grad():
        adapter.gamma.fill_(1.0)

    output = adapter(feature)

    assert torch.isfinite(output).all()
    assert not torch.allclose(output, feature)


def test_rgsata_uses_reliability_gated_relations_without_changing_the_feature_contract():
    """Removing the reliability-gated scatter-back would make the precision-oriented adapter inert."""
    adapter = RGSATA(c1=16, relation_dim=8, max_anchors=6, neighbor_k=3).eval()
    feature = torch.randn(2, 16, 7, 9)

    identity = adapter(feature)
    with torch.no_grad():
        adapter.gamma.fill_(1.0)
    updated = adapter(feature)

    assert identity.shape == feature.shape
    assert torch.equal(identity, feature)
    assert torch.isfinite(updated).all()
    assert not torch.allclose(updated, feature)


def test_drsata_delivers_finite_directed_context_to_subject_features():
    """Dropping context-to-subject message scattering would leave the dual-role adapter unchanged after activation."""
    adapter = DRSATA(c1=16, relation_dim=8, subject_anchors=4, context_anchors=5, neighbor_k=3).eval()
    feature = torch.randn(1, 16, 7, 9)

    identity = adapter(feature)
    with torch.no_grad():
        adapter.gamma.fill_(1.0)
    updated = adapter(feature)

    assert identity.shape == feature.shape
    assert torch.equal(identity, feature)
    assert torch.isfinite(updated).all()
    assert not torch.allclose(updated, feature)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP regression requires a CUDA device")
def test_drsata_scatter_add_accepts_amp_messages_with_feature_dtype():
    """AMP may promote the relation message path, but scatter_add requires one dtype."""
    adapter = DRSATA(c1=16, relation_dim=8, subject_anchors=4, context_anchors=5, neighbor_k=3).cuda().train()
    feature = torch.randn(2, 16, 9, 11, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = adapter(feature)

    assert output.shape == feature.shape
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("yaml_name", ["yolov8n-sata.yaml", "yolov8n-rgsata.yaml", "yolov8n-drsata.yaml"])
def test_sata_family_yaml_builds_a_three_class_detector_with_finite_640_predictions(yaml_name: str):
    """A missing variant parser registration or bad C4/C5 routing must fail before a 300-epoch run starts."""
    yaml_path = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/v8" / yaml_name
    model = YOLO(yaml_path).model.eval()

    with torch.no_grad():
        prediction, _ = model(torch.zeros(1, 3, 640, 640))

    assert prediction.shape[1] == 7
    assert torch.isfinite(prediction).all()
