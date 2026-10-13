"""P0 fixes: gate ablations (18.5), uncertainty types (11.1), Stage B1/B2 freeze
schedule (14.1) and GICR warm-up (14.1)."""

from pathlib import Path

import pytest
import torch

from ultralytics.models.yolo.dsr.model import DSRDetectionModel
from ultralytics.models.yolo.dsr.loss import DSRDetectionLoss
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer
from ultralytics.nn.modules.dsr import DSRDetect, MDRR

ROOT = Path(__file__).resolve().parents[1]
GATE_MODES = ["fixed", "scalar", "conf", "feature", "uncertainty"]
UNCERTAINTY_TYPES = ["max_prob", "margin", "entropy"]


def _make(**kwargs):
    defaults = dict(
        in_channels=(32, 64, 128),
        nc=3,
        relation_dim=16,
        roi_size=3,
        neighbor_k=2,
        region_grid=(2, 2),
        global_topk=4,
        max_candidates=6,
    )
    defaults.update(kwargs)
    return MDRR(**defaults)


def _inputs(B=2, K=6):
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    x1 = torch.rand(B, K, 1) * 40
    y1 = torch.rand(B, K, 1) * 40
    boxes = torch.cat((x1, y1, x1 + 8 + torch.rand(B, K, 1) * 12, y1 + 8 + torch.rand(B, K, 1) * 12), -1)
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K)
    probs = torch.rand(B, K, 3)
    probs = probs / probs.sum(-1, keepdim=True)
    return feats, boxes, mask, conf, probs


@pytest.mark.parametrize("mode", GATE_MODES)
def test_gate_modes_run(mode):
    """v1.1 18.5: all five gate modes run and stay finite."""
    m = _make(gate_mode=mode, use_uncertainty_gate=False)
    f, aux = _run_with(m)
    assert f.shape == (2, 6, 16)
    assert torch.isfinite(f).all()
    assert m.gate_mode == mode


def _run_with(m, inputs=None):
    feats, boxes, mask, conf, probs = inputs or _inputs()
    return m(feats, boxes, mask, conf, (64, 64), probs)


def test_use_uncertainty_gate_overrides_gate_mode():
    m = _make(gate_mode="scalar", use_uncertainty_gate=True)
    assert m.gate_mode == "uncertainty"
    f, aux = _run_with(m)
    assert torch.isfinite(f).all()


@pytest.mark.parametrize("utype", UNCERTAINTY_TYPES)
def test_uncertainty_types_run(utype):
    m = _make(uncertainty_type=utype)
    f, aux = _run_with(m)
    assert torch.isfinite(f).all()
    assert aux["uncertainty"].shape == (2, 6)
    assert aux["uncertainty"].min() >= 0.0 and aux["uncertainty"].max() <= 1.0


def test_uncertainty_definitions_differ():
    """v1.1 11.1: margin/entropy must actually change the u input vs max_prob."""
    inputs = _inputs()
    u_max = _run_with(_make(uncertainty_type="max_prob"), inputs)[1]["uncertainty"]
    u_margin = _run_with(_make(uncertainty_type="margin"), inputs)[1]["uncertainty"]
    u_entropy = _run_with(_make(uncertainty_type="entropy"), inputs)[1]["uncertainty"]
    assert not torch.allclose(u_max, u_margin)
    assert not torch.allclose(u_max, u_entropy)


def test_dsrdetect_gate_config_switch():
    head = DSRDetect(nc=3, ch=(32, 64, 128))
    head._configure_dsr(
        {"enabled": True, "relation_type": "signed_diff", "relation_dim": 16, "gate_mode": "feature",
         "use_uncertainty_gate": False}
    )
    assert head.mdrr.gate_mode == "feature"
    head._configure_dsr(
        {"enabled": True, "relation_type": "signed_diff", "relation_dim": 16, "gate_mode": "scalar",
         "use_uncertainty_gate": True}
    )
    assert head.mdrr.gate_mode == "uncertainty"


def test_param_scope_mapping():
    scope = DSRDetectionTrainer._param_scope
    assert scope("model.3.conv.weight", 11, 23) == "backbone"
    assert scope("model.13.cv1.conv.weight", 11, 23) == "neck"
    assert scope("model.23.cv2.0.0.conv.weight", 11, 23) == "base_detect"
    assert scope("model.23.dfl.conv.weight", 11, 23) == "base_detect"
    assert scope("model.23.mdrr.w_q.weight", 11, 23) == "dsr"
    assert scope("model.23.refine_head.cls.weight", 11, 23) == "dsr"
    assert scope("model.23.ema_head.cv2.0.0.conv.weight", 11, 23) == "ema"


def test_stage_freeze_scopes():
    """Freeze scopes support Stage B1 and explicit Stage-B2 configurations."""
    model = DSRDetectionModel(str(ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"), nc=3, verbose=False)
    head = model.model[-1]

    DSRDetectionTrainer.apply_freeze(model, ["backbone", "neck", "base_detect"])
    assert not next(model.model[0].parameters()).requires_grad  # backbone
    assert not next(model.model[22].parameters()).requires_grad  # neck
    assert not head.cv2[0][0].conv.weight.requires_grad  # base detect
    assert not head.dfl.conv.weight.requires_grad  # always frozen
    assert head.mdrr.w_q.weight.requires_grad  # DSR branch trains
    assert head.refine_head.cls.weight.requires_grad

    DSRDetectionTrainer.apply_freeze(model, ["backbone"])
    assert next(model.model[22].parameters()).requires_grad  # neck unfrozen
    assert head.cv2[0][0].conv.weight.requires_grad  # base detect unfrozen
    assert not head.dfl.conv.weight.requires_grad  # dfl stays frozen (official behavior)
    assert head.mdrr.w_q.weight.requires_grad
