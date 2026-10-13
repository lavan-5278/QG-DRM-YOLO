"""Regression coverage for the DSA social branch's AMP precision boundary."""

from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DSA_PACKAGE_ROOT = Path(r"D:\YH_Project\DSA-Det\dsa_det_ssr_pkg")
if str(DSA_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(DSA_PACKAGE_ROOT))

import dsa_det_ssr  # noqa: F401  Registers the custom DSA task.
from ultralytics import YOLO


DSAM_HGP_YAML = PROJECT_ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8n-dsam-hgp.yaml"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP precision contract requires CUDA")
def test_social_branch_keeps_fp32_geometry_when_detector_is_half():
    """FP16 validation must accept FP32 label boxes through the social branch.

    A validation model is converted to FP16 by Ultralytics AMP, but its labels
    remain FP32.  The social RoI, geometry, and LayerNorm path therefore needs
    one explicit FP32 boundary rather than relying on implicit promotion.
    """
    model = YOLO(str(DSAM_HGP_YAML), verbose=False).model.cuda().eval().half()
    p4_channels = model.feat_projector.conv.in_channels
    feature_maps = [
        torch.empty(1, 1, 1, 1, device="cuda", dtype=torch.float16),
        torch.randn(1, p4_channels, 16, 16, device="cuda", dtype=torch.float16),
    ]
    boxes = torch.tensor(
        [[4.0, 4.0, 28.0, 28.0], [32.0, 8.0, 56.0, 36.0]], device="cuda", dtype=torch.float32
    )
    confidence = torch.ones(2, 1, device="cuda", dtype=torch.float32)

    roi_features = model._extract_roi(feature_maps, boxes)
    cls_scores, refined_boxes, enhanced = model._run_social(roi_features, boxes, confidence, (64, 64))

    assert roi_features.dtype == torch.float32
    assert cls_scores.dtype == torch.float32
    assert refined_boxes.dtype == torch.float32
    assert enhanced.dtype == torch.float32
    assert torch.isfinite(cls_scores).all()
    assert torch.isfinite(refined_boxes).all()
