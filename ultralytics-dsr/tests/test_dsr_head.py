"""DSRDetect / RelationRefinementHead unit tests (v1.1 22.1, v2 classification-only)."""

import torch

from ultralytics.nn.modules.dsr import DSRDetect, RelationRefinementHead


def test_relation_head_zero_init_is_identity():
    """v2: W_cls=0 -> refined logits equal base logits (bbox untouched)."""
    head = RelationRefinementHead(16, 3)
    f = torch.randn(2, 5, 16)
    base_logits = torch.randn(2, 5, 3)
    logits = head(f, base_logits)
    assert torch.allclose(logits, base_logits, atol=1e-6)


def test_relation_head_eta_grad_flow():
    """v1.1 12.1: eta_cls=1.0 with zero-init W_cls still allows head gradients."""
    head = RelationRefinementHead(16, 3)
    f = torch.randn(4, 16, requires_grad=True)
    base_logits = torch.randn(4, 3)
    logits = head(f, base_logits)
    target = torch.zeros(4, 3)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert head.cls.weight.grad is not None and head.cls.weight.grad.abs().sum() > 0


def test_selection_mask_preserves_base_logits():
    """v2: candidates not selected by the ambiguity gate keep the YOLO logits."""
    head = RelationRefinementHead(16, 3)
    with torch.no_grad():
        head.cls.weight.normal_()
    f = torch.randn(2, 4, 16)
    base_logits = torch.randn(2, 4, 3)
    selected = torch.tensor([[True, False, True, False], [False, True, False, True]])
    logits = head(f, base_logits, selected=selected)
    assert torch.allclose(logits[~selected], base_logits[~selected], atol=1e-6)  # unselected unchanged
    assert not torch.allclose(logits[selected], base_logits[selected], atol=1e-6)  # selected refined


def test_min_candidates_skips_an_isolated_candidate():
    """The configured candidate lower bound prevents relation refinement with only one object."""
    head = DSRDetect(nc=3, ch=(16, 32, 64), dsr_cfg={"min_candidates": 2, "relation_topk": 4})
    boxes = torch.tensor([[[16.0], [16.0], [8.0], [8.0]]])  # [B, 4, N] cxcywh
    scores = torch.tensor([[[0.9], [0.1], [0.1]]])  # [B, nc, N]

    selected = head._select_candidates(boxes, scores, (64, 64))

    assert not selected["mask"].any()
