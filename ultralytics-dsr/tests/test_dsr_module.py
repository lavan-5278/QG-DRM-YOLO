"""MDRR / DSR module unit tests (v1.1 22.1)."""

import torch

from ultralytics.nn.modules.dsr import MDRR


def _make_mdrr(**kwargs):
    defaults = dict(
        in_channels=(32, 64, 128),
        nc=3,
        relation_dim=16,
        roi_size=3,
        neighbor_k=2,
        region_grid=(2, 2),
        global_topk=4,
    )
    defaults.update(kwargs)
    return MDRR(**defaults)


def _boxes(B, K, size=64):
    x1 = torch.rand(B, K, 1) * (size - 20)
    y1 = torch.rand(B, K, 1) * (size - 20)
    x2 = x1 + 8 + torch.rand(B, K, 1) * 12
    y2 = y1 + 8 + torch.rand(B, K, 1) * 12
    return torch.cat((x1, y1, x2, y2), -1)


def test_mdrr_output_dims():
    m = _make_mdrr()
    B, K = 2, 6
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    boxes = _boxes(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K)
    f, aux = m(feats, boxes, mask, conf, (64, 64))
    assert f.shape == (B, K, 16)
    # v2: KNN-only by default -> single scale "n"
    assert set(aux["graph_masks"].keys()) == {"n"}
    assert aux["scale_weights"].shape == (B, K, 1)
    assert aux["gate"].shape == (B, K, 16)
    assert aux["uncertainty"].shape == (B, K)
    for name, msk in aux["graph_masks"].items():
        assert msk.shape == (B, K, K), name
    assert torch.isfinite(f).all()


def test_mdrr_padding_candidates_not_in_graphs():
    m = _make_mdrr()
    B, K = 1, 4
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    boxes = _boxes(B, K)
    mask = torch.tensor([[True, True, True, False]])
    conf = torch.rand(B, K)
    _, aux = m(feats, boxes, mask, conf, (64, 64))
    for name, msk in aux["graph_masks"].items():
        # padded candidate j=3 never appears as a neighbor
        assert not msk[0, :, 3].any(), name
        # padded row i=3 has no edges
        assert not msk[0, 3, :].any(), name


def test_mdrr_graph_masks_no_self_loop():
    m = _make_mdrr()
    B, K = 1, 5
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    boxes = _boxes(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K)
    _, aux = m(feats, boxes, mask, conf, (64, 64))
    for name, msk in aux["graph_masks"].items():
        assert not msk[0].diagonal().any(), name


def test_mdrr_isolated_candidate_no_nan():
    """A valid candidate with no neighbors in a graph must not produce NaN."""
    m = _make_mdrr(use_global=False, use_region=True, use_neighbor=False, region_grid=(2, 2))
    B, K = 1, 3
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    boxes = torch.tensor(
        [[[5, 5, 15, 15], [20, 5, 30, 15], [50, 50, 60, 60]]], dtype=torch.float32
    )  # third candidate sits alone in its region cell
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K)
    f, aux = m(feats, boxes, mask, conf, (64, 64))
    assert torch.isfinite(f).all(), "isolated candidate must not poison MDRR output"
    for key, value in aux.items():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), key


def test_pair_iou_fp16_large_boxes_no_overflow():
    """fp16 area products overflow (>65504) for large boxes; must stay finite."""
    boxes = torch.tensor([[[5.0, 5.0, 600.0, 600.0], [10.0, 10.0, 300.0, 300.0]]])
    iou = MDRR._pair_iou(boxes.half())
    assert torch.isfinite(iou).all()
    assert bool((iou >= 0.0).all() and (iou <= 1.0).all())


def test_uncertainty_selection_gates_relation():
    """v2 5.1: unselected (easy) candidates keep individual features exactly."""
    m = _make_mdrr(neighbor_k=2, gate_mode="scalar", use_uncertainty_gate=False)
    with torch.no_grad():
        m.rho.fill_(1.0)
    B, K = 1, 4
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    boxes = _boxes(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K)
    probs = torch.rand(B, K, 3)
    probs = probs / probs.sum(-1, keepdim=True)
    selection = torch.tensor([[True, False, False, False]])
    f, aux = m(feats, boxes, mask, conf, (64, 64), probs, selection=selection)
    ln = torch.nn.functional.layer_norm
    # unselected rows: residual gated off -> f_tilde == layer_norm(individual feature)
    assert torch.allclose(f[0, 1], ln(aux["features"][0, 1], (16,)), atol=1e-5)
    assert torch.allclose(f[0, 2], ln(aux["features"][0, 2], (16,)), atol=1e-5)
    # selected row receives the relation residual -> differs from individual feature
    assert not torch.allclose(f[0, 0], ln(aux["features"][0, 0], (16,)), atol=1e-3)


def test_composition_prior_selective():
    """v3: composition prior modifies selected candidates only; unselected keep LN(f)."""
    m = _make_mdrr(use_composition=True, neighbor_k=2, gate_mode="scalar", use_uncertainty_gate=False)
    with torch.no_grad():
        m.rho.fill_(1.0)
    B, K = 1, 4
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    boxes = _boxes(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K)
    probs = torch.rand(B, K, 3)
    probs = probs / probs.sum(-1, keepdim=True)
    selection = torch.tensor([[True, False, False, False]])
    f, aux = m(feats, boxes, mask, conf, (64, 64), probs, selection=selection)
    ln = torch.nn.functional.layer_norm
    assert aux["ctx"].shape == (B, K, 16)
    assert torch.isfinite(aux["ctx"]).all()
    # unselected rows: composition residual gated off -> f_tilde == layer_norm(feature)
    assert torch.allclose(f[0, 1], ln(aux["features"][0, 1], (16,)), atol=1e-5)
    assert torch.allclose(f[0, 2], ln(aux["features"][0, 2], (16,)), atol=1e-5)
    # selected row carries the composition residual
    assert not torch.allclose(f[0, 0], ln(aux["features"][0, 0], (16,)), atol=1e-3)


def test_neighbor_reliability_prior_changes_attention():
    """v2 7: reliability prior log(1 - u_j + eps) changes aggregation vs uniform."""
    m_off = _make_mdrr(neighbor_k=2, gate_mode="scalar", use_uncertainty_gate=False, neighbor_reliability=False)
    m_on = _make_mdrr(neighbor_k=2, gate_mode="scalar", use_uncertainty_gate=False, neighbor_reliability=True)
    with torch.no_grad():
        m_off.rho.fill_(1.0)
        m_on.rho.fill_(1.0)
        m_on.load_state_dict(m_off.state_dict())
    B, K = 1, 4
    feats = [torch.randn(B, c, 16, 16) for c in (32, 64, 128)]
    boxes = _boxes(B, K)
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K)
    probs = torch.rand(B, K, 3)
    probs = probs / probs.sum(-1, keepdim=True)
    f_off, _ = m_off(feats, boxes, mask, conf, (64, 64), probs)
    f_on, _ = m_on(feats, boxes, mask, conf, (64, 64), probs)
    assert torch.isfinite(f_on).all()
    assert not torch.allclose(f_on, f_off, atol=1e-5)


def test_signed_message_antisymmetry():
    """v1.1 22.1: the message path W_m(f_i - f_j) flips sign when i,j are swapped."""
    m = _make_mdrr()
    f = torch.randn(2, 16)
    m_ij = m.w_m(f[0] - f[1]) - m.w_m.bias  # affine part (Linear bias is not antisymmetric)
    m_ji = m.w_m(f[1] - f[0]) - m.w_m.bias
    assert torch.allclose(m_ij, -m_ji, atol=1e-6)


def test_mdrr_fp16_no_nan():
    m = _make_mdrr().half().eval()
    B, K = 2, 6
    feats = [torch.randn(B, c, 16, 16).half() for c in (32, 64, 128)]
    boxes = _boxes(B, K).half()
    mask = torch.ones(B, K, dtype=torch.bool)
    conf = torch.rand(B, K).half()
    with torch.no_grad():
        f, aux = m(feats, boxes, mask, conf, (64, 64))
    assert torch.isfinite(f).all()
    assert torch.isfinite(aux["scale_weights"]).all()
