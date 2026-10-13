"""DSRDetectionLoss unit tests (v1.1 13.2 / 13.5 / 22.2)."""

from pathlib import Path

import torch

from ultralytics.models.yolo.dsr.loss import DSRDetectionLoss

ROOT = Path(__file__).resolve().parents[1]


def _make_loss():
    from ultralytics.models.yolo.dsr.model import DSRDetectionModel

    model = DSRDetectionModel(
        str(ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"),
        nc=3,
        verbose=False,
    )
    loss = DSRDetectionLoss(model)
    return model, loss


def _batch(images, objs):
    """Build a v8-style training batch. objs: list per image of (cls, cx, cy, w, h)."""
    batch_idx, cls, bboxes = [], [], []
    for bi, objs_i in enumerate(objs):
        for o in objs_i:
            batch_idx.append(bi)
            cls.append(o[0])
            bboxes.append(o[1:])
    return {
        "img": images,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.long),
        "cls": torch.tensor(cls, dtype=torch.long),
        "bboxes": torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
    }


def _dsr_out_for_matching(loss, cand_boxes, refined_logits, f_tilde=None, img_size=(64, 64)):
    """Fabricate dsr_out so _refined_losses can be exercised without a forward pass."""
    B, K = cand_boxes.shape[:2]
    nc = loss.nc
    f = f_tilde if f_tilde is not None else torch.randn(B, K, 128, requires_grad=True)
    rl = refined_logits if refined_logits is not None else torch.randn(B, K, nc, requires_grad=True)
    return {
        "cand_boxes_xyxy": cand_boxes,
        "cand_mask": torch.ones(B, K, dtype=torch.bool),
        "cand_conf": torch.full((B, K), 0.3),
        "refined_logits": rl,
        "base_logits": rl.detach(),
        "f_tilde": f,
        "img_size": img_size,
    }


def test_refined_loss_no_gt():
    _, loss = _make_loss()
    img_size = (64, 64)
    cand_boxes = torch.tensor(
        [[[10, 10, 30, 30], [40, 40, 60, 60], [5, 5, 20, 20], [50, 50, 63, 63]]],
        dtype=torch.float32,
    )
    dsr_out = _dsr_out_for_matching(loss, cand_boxes, None, img_size=img_size)
    batch = _batch(torch.zeros(1, 3, 64, 64), [[]])
    relation, preserve, gicr = loss._refined_losses(None, batch, dsr_out)
    assert torch.isfinite(relation) and torch.isfinite(preserve) and torch.isfinite(gicr)


def test_duplicate_positive_is_ignored_not_background():
    """v1.1 13.2: a second candidate with IoU>=0.5 to the same GT is ignore, not background."""
    _, loss = _make_loss()
    img_size = (64, 64)
    # GT: [20,20,40,40] -> normalized xywh (cx, cy, w, h) = (0.46875, 0.46875, 0.3125, 0.3125)
    batch = _batch(
        torch.zeros(1, 3, 64, 64),
        [[(1, 0.46875, 0.46875, 0.3125, 0.3125)]],
    )
    # candidate 0 best (highest IoU), candidate 1 duplicate (IoU>=0.5, not best), 2/3 background
    cand_boxes = torch.tensor(
        [[[21, 21, 39, 39], [22, 22, 40, 40], [2, 2, 8, 8], [52, 52, 60, 60]]],
        dtype=torch.float32,
    )
    # give the duplicate candidate a strongly positive class-1 logit: as background it
    # would incur a large BCE loss; as ignore it must not.
    rl = torch.randn(1, 4, loss.nc)
    rl[0, 1, 1] = 10.0
    rl = rl.requires_grad_(True)

    dsr_dup = _dsr_out_for_matching(loss, cand_boxes, rl, img_size=img_size)
    ref_cls_dup, _, _ = loss._refined_losses(None, batch, dsr_dup)

    # same setup but candidate 1 moved far away -> background
    cand_boxes_bg = cand_boxes.clone()
    cand_boxes_bg[0, 1] = torch.tensor([2, 2, 8, 8])
    dsr_bg = _dsr_out_for_matching(loss, cand_boxes_bg, rl, img_size=img_size)
    ref_cls_bg, _, _ = loss._refined_losses(None, batch, dsr_bg)

    assert ref_cls_dup.item() < ref_cls_bg.item(), "duplicate positive must be ignored, not trained as background"


def test_preserve_loss_anchors_high_confidence():
    """v2 10: KL preservation anchors high-confidence candidates; ~0 when unchanged."""
    _, loss = _make_loss()
    B, K = 1, 3
    cand_boxes = torch.tensor(
        [[[10, 10, 30, 30], [40, 40, 60, 60], [5, 5, 20, 20]]], dtype=torch.float32
    )
    base = torch.randn(B, K, loss.nc)
    conf = torch.tensor([[0.95, 0.4, 0.2]])
    dsr_out = {
        "cand_boxes_xyxy": cand_boxes,
        "cand_mask": torch.ones(B, K, dtype=torch.bool),
        "cand_conf": conf,
        "base_logits": base,
        "refined_logits": base + torch.tensor(
            [[[0.3, -0.2, 0.1], [1.0, 0.5, -0.8], [0.0, 0.0, 0.0]]]
        ),
        "f_tilde": torch.randn(B, K, 128, requires_grad=True),
        "selection": torch.tensor([[False, True, False]]),
        "img_size": (64, 64),
    }
    batch = _batch(torch.zeros(1, 3, 64, 64), [[]])
    _, preserve, _ = loss._refined_losses(None, batch, dsr_out)
    assert torch.isfinite(preserve)
    assert preserve.item() > 0.0, "moving a confident prediction must incur KL"

    dsr_out["refined_logits"] = base.clone()
    _, preserve_0, _ = loss._refined_losses(None, batch, dsr_out)
    assert preserve_0.item() < 1e-4, "unchanged prediction must have ~0 KL"


def test_preserve_threshold_excludes_low_confidence_candidates():
    """A low-confidence candidate must not contribute to KL preservation when below the threshold."""
    _, loss = _make_loss()
    loss.preserve_threshold = 0.9
    base = torch.tensor([[[2.0, -1.0, -1.0], [1.0, 0.0, -1.0]]])
    dsr_out = {
        "cand_boxes_xyxy": torch.tensor([[[10, 10, 30, 30], [35, 35, 55, 55]]], dtype=torch.float32),
        "cand_mask": torch.ones(1, 2, dtype=torch.bool),
        "cand_conf": torch.tensor([[0.95, 0.4]]),
        "base_logits": base,
        # Only the low-confidence candidate changes.
        "refined_logits": base + torch.tensor([[[0.0, 0.0, 0.0], [-3.0, 2.0, 1.0]]]),
        "f_tilde": torch.randn(1, 2, 128, requires_grad=True),
        "selection": torch.tensor([[False, True]]),
        "img_size": (64, 64),
    }
    batch = _batch(torch.zeros(1, 3, 64, 64), [[]])
    _, preserve, _ = loss._refined_losses(None, batch, dsr_out)
    assert preserve.item() < 1e-6


def test_gicr_finite_and_warmup_zero():
    """v3: GICR is finite and zeroed during warmup epochs."""
    model, loss = _make_loss()
    head = model.model[-1]
    head.train_epoch = 0  # warmup -> gicr must be 0
    B, K = 1, 3
    cand_boxes = torch.tensor(
        [[[10, 10, 30, 30], [40, 40, 60, 60], [5, 5, 20, 20]]], dtype=torch.float32
    )
    base = torch.randn(B, K, loss.nc)
    dsr_out = {
        "cand_boxes_xyxy": cand_boxes,
        "cand_mask": torch.ones(B, K, dtype=torch.bool),
        "cand_conf": torch.tensor([[0.3, 0.3, 0.3]]),
        "base_logits": base,
        "refined_logits": base.clone(),
        "f_tilde": torch.randn(B, K, 128, requires_grad=True),
        "selection": torch.ones(B, K, dtype=torch.bool),
        "img_size": (64, 64),
    }
    batch = _batch(torch.zeros(1, 3, 64, 64), [[(0, 0.5, 0.5, 0.3, 0.3), (1, 0.3, 0.3, 0.2, 0.2)]])
    _, _, gicr = loss._refined_losses(None, batch, dsr_out)
    assert torch.isfinite(gicr)
    assert float(gicr) == 0.0, "GICR must be zeroed during warmup"
    head.train_epoch = 10
    _, _, gicr2 = loss._refined_losses(None, batch, dsr_out)
    assert torch.isfinite(gicr2)


def test_loss_weights_scale_the_refined_terms():
    """Changing lambda_* must change the three DSR loss components returned to the trainer."""
    model, loss = _make_loss()
    head = model.model[-1]
    for cv3 in head.cv3:
        cv3[-1].bias.data.zero_()
    head.candidate_conf = 0.0001
    head.pre_topk = 100
    head.relation_topk = 40
    head.uncertainty_threshold = 0.0
    head.train_epoch = 10
    model.train()

    batch = _batch(
        torch.rand(1, 3, 64, 64),
        [[(0, 0.5, 0.5, 0.3, 0.3), (1, 0.25, 0.25, 0.2, 0.2)]],
    )
    preds = model(batch["img"])
    dsr_out = head.dsr_out
    assert dsr_out and not dsr_out["empty"]

    raw_relation, raw_preserve, raw_gicr = loss._refined_losses(preds, batch, dsr_out)
    loss.lambda_relation = 2.0
    loss.lambda_preserve = 3.0
    loss.lambda_gicr = 4.0
    items, _ = loss(preds, batch, dsr_out)

    expected = torch.stack((raw_relation * 2.0, raw_preserve * 3.0, raw_gicr * 4.0))
    assert torch.allclose(items[3:], expected, atol=1e-6, rtol=1e-5)
