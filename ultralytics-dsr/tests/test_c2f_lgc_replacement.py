"""Regression tests for partial and full C2f-to-LGC backbone ablations."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import C2fLGC, LGCBottleneck


def test_c2f_lgc_replaces_the_last_half_of_internal_units_with_liquid_bottlenecks():
    """C4 has two units, so exactly its final one must be liquid while retaining the first C2f bottleneck."""
    c4 = C2fLGC(c1=16, c2=16, n=2, shortcut=True).eval()
    c5 = C2fLGC(c1=16, c2=16, n=1, shortcut=True).eval()

    assert len(c4.m) == 2
    assert sum(isinstance(unit, LGCBottleneck) for unit in c4.m) == 1
    assert isinstance(c4.m[-1], LGCBottleneck)
    assert len(c5.m) == 1
    assert isinstance(c5.m[0], LGCBottleneck)


def test_c2f_lgc_liquid_branch_is_neutral_then_finite_and_active_when_opened():
    """The ablation must prove the internal LGC residual is present rather than merely renaming C2f."""
    block = C2fLGC(c1=16, c2=16, n=2, shortcut=True).eval()
    feature = torch.randn(1, 16, 9, 11)
    baseline = block(feature)

    with torch.no_grad():
        block.m[-1].liquid.gamma.fill_(1.0)
    updated = block(feature)

    assert baseline.shape == feature.shape
    assert torch.isfinite(baseline).all()
    assert torch.isfinite(updated).all()
    assert not torch.allclose(updated, baseline)


@pytest.mark.parametrize("yaml_name", ["yolov8n-c2f-lgc.yaml", "yolov8n-c2f-to-lgc.yaml"])
def test_c2f_lgc_ablation_yaml_builds_with_finite_640_predictions(yaml_name: str):
    """C4/C5 replacement must preserve all P3/P4/P5 routes into the unchanged YOLOv8n neck."""
    yaml_path = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/v8" / yaml_name
    model = YOLO(yaml_path).model.eval()
    with torch.no_grad():
        prediction, _ = model(torch.zeros(1, 3, 640, 640))

    assert prediction.shape[1] == 7
    assert torch.isfinite(prediction).all()
