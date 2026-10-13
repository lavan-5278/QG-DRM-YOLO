"""Phase 2/3: relation_type R0-R7 switching and M-series variants (v1.1 18.3/18.4)."""

import pytest
import torch

from ultralytics.nn.modules.dsr import DSRDetect, MDRR

RELATION_TYPES = ["none", "dot", "concat", "product", "abs_diff", "edgeconv", "signed_diff", "transformer"]
VARIANTS = ["none", "no_geo", "no_gate", "flip", "shared"]


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


def _run(m, B=2, K=6, masked=True):
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    x1 = torch.rand(B, K, 1) * 40
    y1 = torch.rand(B, K, 1) * 40
    boxes = torch.cat((x1, y1, x1 + 8 + torch.rand(B, K, 1) * 12, y1 + 8 + torch.rand(B, K, 1) * 12), -1)
    mask = torch.ones(B, K, dtype=torch.bool)
    if masked:
        mask[0, -1] = False
    conf = torch.rand(B, K)
    return m(feats, boxes, mask, conf, (64, 64))


@pytest.mark.parametrize("rtype", RELATION_TYPES)
def test_relation_type_runs(rtype):
    m = _make(relation_type=rtype)
    f, aux = _run(m, masked=True)
    assert f.shape == (2, 6, 16)
    assert torch.isfinite(f).all()
    for key, value in aux.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key


def test_r0_no_relation_is_identity():
    """R0: individual refinement only -> f_tilde equals the RoI features."""
    m = _make(relation_type="none")
    f, aux = _run(m, masked=False)
    assert torch.equal(f, aux["features"])
    assert not m.is_graph


def test_transformer_padding_is_zeroed():
    """R7: padded candidates must not leak and must be zeroed (no NaN)."""
    m = _make(relation_type="transformer")
    f, aux = _run(m, masked=True)
    assert torch.isfinite(f).all()
    assert f[0, -1].abs().sum() == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_signed_variants_run(variant):
    """M1-M5 variants of signed_diff all run and stay finite."""
    m = _make(relation_type="signed_diff", relation_variant=variant)
    f, aux = _run(m, masked=True)
    assert f.shape == (2, 6, 16)
    assert torch.isfinite(f).all()


def test_no_geo_variant_disables_geometry():
    """M2 (v1.1 18.4): relation_variant 'no_geo' must actually turn geometry off."""
    m = _make(relation_type="signed_diff", relation_variant="no_geo")
    assert m.include_geometry is False
    m2 = _make(relation_type="signed_diff", relation_variant="none", include_geometry=True)
    assert m2.include_geometry is True


def test_scale_ablations_change_graphs():
    """A-series: disabling a scale removes it from graph masks and fusion."""
    m = _make(relation_type="signed_diff", use_global=True, use_region=False, use_neighbor=True)
    _, aux = _run(m, masked=False)
    assert set(aux["graph_masks"].keys()) == {"g", "n"}
    assert aux["scale_weights"].shape == (2, 6, 2)


def test_dsrdetect_relation_type_config_switch():
    """v1.1 18.3: relation operators are switchable through DSRDetect config."""
    head = DSRDetect(nc=3, ch=(32, 64, 128))
    for rtype in RELATION_TYPES:
        head._configure_dsr(
            {
                "enabled": True,
                "relation_type": rtype,
                "relation_dim": 16,
                "relation_topk": 6,
                "pre_topk": 10,
                "candidate_conf": 0.01,
                "global_topk": 4,
                "neighbor_k": 2,
            }
        )
        assert head.mdrr.relation_type == rtype
        assert head.mdrr.is_graph is (rtype not in ("none", "transformer"))
