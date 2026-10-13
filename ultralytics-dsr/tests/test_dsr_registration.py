"""Phase 1 registration tests: yaml build, output contract, official weights load."""

from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.tasks import yaml_model_load

ROOT = Path(__file__).resolve().parents[1]
DSR_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"
OFFICIAL_YAML = ROOT / "ultralytics/cfg/models/11/yolo11.yaml"
WORKSPACE_WEIGHTS = ROOT.parent / "yolo11n.pt"


def test_dsr_yaml_builds_dsrdetect():
    model = DetectionModel(str(DSR_YAML), nc=3, verbose=False)
    head = model.model[-1]
    assert type(head).__name__ == "DSRDetect"
    assert head.dsr_active is True
    assert head.mdrr is not None and head.refine_head is not None
    assert model.stride.tolist() == [8, 16, 32]


def test_model_info_and_forward_output_contract():
    model = DetectionModel(str(DSR_YAML), nc=3, verbose=False)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    # non-export inference returns (y, raw) tuple, same as official Detect
    assert isinstance(out, tuple) and len(out) == 2
    y = out[0]
    assert y.shape[0] == 1 and y.shape[1] == 7  # 4 + nc
    assert torch.isfinite(y).all()


def test_enabled_false_output_identical_to_official():
    base = DetectionModel(str(OFFICIAL_YAML), nc=3, verbose=False)
    dsr_off = DetectionModel(str(DSR_YAML), nc=3, verbose=False)
    dsr_off.model[-1].dsr_active = False
    dsr_off.load(base)  # load official weights (state dict intersect)
    base.eval()
    dsr_off.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        yb = base(x)
        yd = dsr_off(x)
    assert torch.equal(yb[0], yd[0]), "dsr.enabled=false must produce byte-identical output"


def test_enabled_zero_init_output_equals_base():
    """Untrained DSR head (zero-init) must keep base predictions unchanged."""
    dsr = DetectionModel(str(DSR_YAML), nc=3, verbose=False)
    assert dsr.model[-1].dsr_active is True
    dsr.eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        y_ref = dsr(x)[0]
        # reference: same model with the relation branch bypassed
        dsr.model[-1].dsr_active = False
        y_base = dsr(x)[0]
        dsr.model[-1].dsr_active = True
    assert torch.allclose(y_ref, y_base, atol=1e-5), "zero-init DSR head must not move base predictions"


def test_load_official_yolo11_weights():
    if not WORKSPACE_WEIGHTS.exists():
        pytest.skip("yolo11n.pt not available next to the project")
    from ultralytics.nn.tasks import torch_safe_load

    model = DetectionModel(str(DSR_YAML), nc=3, verbose=False)
    ckpt, _ = torch_safe_load(str(WORKSPACE_WEIGHTS))  # must not raise; base layers transferred
    model.load(ckpt)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.isfinite(model(x)[0]).all()


def test_yaml_scale_resolution():
    d = yaml_model_load(str(DSR_YAML))
    assert d["scale"] == "n"
    d = yaml_model_load(str(ROOT / "ultralytics/cfg/models/11/yolo11s-dsr.yaml"))
    assert d["scale"] == "s"
