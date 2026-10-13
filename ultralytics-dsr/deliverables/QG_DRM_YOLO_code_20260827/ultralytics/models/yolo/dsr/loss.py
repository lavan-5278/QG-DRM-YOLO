# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DSR-Det v2 loss.

``DSRDetectionLoss`` calls the official ``v8DetectionLoss`` unchanged for the base YOLO
loss and adds the v2 components (see DSR-Det_v2_优化方案):

- relation classification (BCEWithLogits) on *selected* (ambiguous) candidates only;
- preservation loss: KL(stop-grad(P_YOLO) || P_DSR) on high-confidence candidates so
  that correct YOLO predictions are not corrupted by the relation branch;

v2 removes the old box refinement loss and GICR (bbox stays with YOLO).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.dsr import pairwise_iou
from ultralytics.utils.loss import v8DetectionLoss


class DSRDetectionLoss(v8DetectionLoss):
    """Criterion: official YOLO loss + refined classification/box losses + GICR."""

    def __init__(self, model, tal_topk: int = 10):
        super().__init__(model, tal_topk)
        self.model = model
        m = model.model[-1]
        cfg = getattr(m, "dsr_cfg", {}) or {}
        self.lambda_relation = float(cfg.get("lambda_relation", cfg.get("lambda_ref_cls", 1.0)))
        self.lambda_preserve = float(cfg.get("lambda_preserve", 0.5))
        self.preserve_threshold = float(cfg.get("preserve_threshold", 0.9))
        self.uncertainty_threshold = float(cfg.get("uncertainty_threshold", 0.5))
        # v3 GICR: geometry-aware intra-frame contrastive regularization (training
        # regularizer for the joint Stage-B2 fine-tune, not an inference branch).
        self.use_gicr = bool(cfg.get("use_gicr", False))
        self.lambda_gicr = float(cfg.get("lambda_gicr", 0.05))
        self.lambda_gicr_diff = float(cfg.get("lambda_gicr_diff", 0.5))
        self.gicr_warmup_epochs = int(cfg.get("gicr_warmup_epochs", 0))
        self.m_s = float(cfg.get("gicr_m_s", 0.3))
        self.m_d = float(cfg.get("gicr_m_d", 0.8))
        self.tau_g = float(cfg.get("gicr_tau_g", 2.0))
        self.relu = nn.ReLU()

    def __call__(self, preds, batch, dsr_out=None):
        """Return a 6-component loss vector (box, cls, dfl, relation, preserve, gicr)."""
        base_loss, base_items = super().__call__(preds, batch)  # 3-vec, already scaled by batch_size
        if not (dsr_out and dsr_out.get("active") and not dsr_out.get("empty", False)):
            items = torch.cat((base_loss, torch.zeros(3, device=base_loss.device, dtype=base_loss.dtype)))
            items = self._ensure_grad_path(items)
            return items, items.detach()
        bs = preds[0].shape[0] if isinstance(preds, (list, tuple)) else preds.shape[0]
        relation, preserve, gicr = self._refined_losses(preds, batch, dsr_out)
        relation = relation * self.lambda_relation
        preserve = preserve * self.lambda_preserve
        gicr = gicr * self.lambda_gicr
        items = torch.cat(
            (base_loss, (relation * bs).reshape(1), (preserve * bs).reshape(1), (gicr * bs).reshape(1))
        )
        items = self._ensure_grad_path(items)
        return items, items.detach()

    def _ensure_grad_path(self, items: torch.Tensor) -> torch.Tensor:
        """Guarantee a trainable gradient path when base layers are frozen and a
        batch yields no candidates (Stage B1, v1.1 14.1): anchor on MDRR's rho."""
        if items.requires_grad:
            return items
        m = getattr(getattr(self, "model", None), "model", None)
        head = m[-1] if m is not None and len(m) else None
        rho = getattr(getattr(head, "mdrr", None), "rho", None)
        if rho is not None:
            return items + rho * 0.0
        return items

    def _refined_losses(self, preds, batch, dsr_out):
        """Compute v2 relation classification + preservation losses on Top-K candidates."""
        device = self.device
        cand_boxes = dsr_out["cand_boxes_xyxy"]  # [B, K, 4] detached
        cand_mask = dsr_out["cand_mask"]  # [B, K]
        cand_conf = dsr_out["cand_conf"]  # [B, K] detached base confidence
        refined_logits = dsr_out["refined_logits"]  # [B, K, nc]
        f_tilde = dsr_out["f_tilde"]  # [B, K, C]
        selection = dsr_out.get("selection")  # [B, K] ambiguous candidates (v2)
        base_logits = dsr_out["base_logits"]  # [B, K, nc]
        img_h, img_w = dsr_out["img_size"]
        B, K = cand_mask.shape
        nc = self.nc

        if selection is None:
            selection = torch.ones_like(cand_mask)
        # relation loss only touches ambiguous candidates; easy candidates keep YOLO
        active_mask = cand_mask & selection

        # GT targets in input-image pixels (xyxy), same convention as v8DetectionLoss
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        imgsz = torch.tensor((img_h, img_w), device=device, dtype=cand_boxes.dtype)
        targets = self.preprocess(targets, B, imgsz[[1, 0, 1, 0]])
        gt_cls, gt_boxes = targets.split((1, 4), 2)
        mask_gt = gt_boxes.sum(2) > 0

        pos_logits, pos_cls, pos_boxes, pos_feats = [], [], [], []
        bg_logits = []
        for bi in range(B):
            valid = active_mask[bi]
            kv = int(valid.sum())
            if kv == 0:
                continue
            cb = cand_boxes[bi][valid]
            rl = refined_logits[bi][valid]
            ngt = int(mask_gt[bi].sum())
            if ngt == 0:
                bg_logits.append(rl)
                continue
            gb = gt_boxes[bi][:ngt]
            gc = gt_cls[bi][:ngt, 0].long()
            iou = pairwise_iou(cb, gb)  # [kv, ngt]
            max_iou, max_gt = iou.max(dim=1)

            # v1.1 13.2: Top-1 positive per GT; other IoU>=0.5 and 0.4-0.5 -> ignore
            is_pos = torch.zeros(kv, dtype=torch.bool, device=device)
            for g in range(ngt):
                cands = (iou[:, g] >= 0.5).nonzero().squeeze(-1)
                if cands.numel():
                    is_pos[cands[iou[cands, g].argmax()]] = True
            ign = (max_iou >= 0.5) & ~is_pos
            ign |= (max_iou >= 0.4) & (max_iou < 0.5)
            pos_idx = is_pos.nonzero().squeeze(-1)
            bg_idx = (max_iou < 0.4).nonzero().squeeze(-1)
            if pos_idx.numel():
                pos_logits.append(rl[pos_idx])
                pos_cls.append(gc[max_gt[pos_idx]])
                pos_boxes.append(cb[pos_idx])
                pos_feats.append(f_tilde[bi][valid][pos_idx])
            if bg_idx.numel():
                bg_logits.append(rl[bg_idx])

        empty_pos = refined_logits.reshape(-1, nc)[:0]  # keeps a grad path when empty
        pos_logits = torch.cat(pos_logits, 0) if pos_logits else empty_pos
        pos_cls = torch.cat(pos_cls, 0) if pos_cls else refined_logits.reshape(-1)[:0].long()
        bg_logits = torch.cat(bg_logits, 0) if bg_logits else empty_pos
        empty_boxes = cand_boxes.reshape(-1, 4)[:0]
        empty_feats = f_tilde.reshape(-1, f_tilde.shape[-1])[:0]
        pos_boxes = torch.cat(pos_boxes, 0) if pos_boxes else empty_boxes
        pos_feats = torch.cat(pos_feats, 0) if pos_feats else empty_feats

        # v2 relation classification loss: BCE on ambiguous positives + backgrounds
        if pos_logits.numel() == 0 and bg_logits.numel() == 0:
            relation = refined_logits.sum() * 0.0
        else:
            all_logits = torch.cat((pos_logits, bg_logits), 0)
            all_targets = torch.cat(
                (F.one_hot(pos_cls.long(), nc).float(), all_logits.new_zeros(bg_logits.shape[0], nc)), 0
            )
            relation = self.bce(all_logits, all_targets).sum() / max(all_logits.shape[0], 1)

        # v3 preservation loss: confidence-weighted KL(stop-grad(P_YOLO) || P_DSR)
        # on ALL candidates. High-confidence predictions are strongly anchored to
        # YOLO; only low-confidence (ambiguous) candidates are free to change.
        preserve = refined_logits.sum() * 0.0
        preserve_mask = (cand_conf > self.preserve_threshold) & cand_mask
        w_pres = (cand_conf.clamp(min=0.0) ** 2) * preserve_mask.to(cand_conf.dtype)
        if w_pres.sum() > 0:
            p_yolo = base_logits.softmax(-1).detach().clamp(min=1e-9)
            log_p_dsr = refined_logits.log_softmax(-1)
            kl = (p_yolo * (p_yolo.log() - log_p_dsr)).sum(-1)
            preserve = (kl * w_pres).sum() / w_pres.sum()

        # v3 GICR: geometry-aware intra-frame contrastive regularization on the
        # selected positives' RoI features. Same-class pairs compact (margin m_s),
        # diff-class pairs separated (margin m_d), weighted by scale-normalised
        # geometric proximity so the regularizer is local (matches the diagnosis:
        # distance-decaying relational signal).
        gicr = refined_logits.sum() * 0.0
        if self.use_gicr and pos_feats.shape[0] > 1:
            z = F.normalize(pos_feats, dim=-1)
            d_cos = 1.0 - (z @ z.T)
            geo = self._geo_matrix(pos_boxes, (img_w, img_h))
            w_geo = torch.exp(-geo / self.tau_g)
            same = pos_cls.unsqueeze(0) == pos_cls.unsqueeze(1)
            w_same = (w_geo * same).sum()
            w_diff = (w_geo * ~same).sum()
            if w_same > 0:
                l_same = (w_geo * same * self.relu(d_cos - self.m_s)).sum() / w_same
            else:
                l_same = torch.zeros((), device=device)
            if w_diff > 0:
                l_diff = (w_geo * ~same * self.relu(self.m_d - d_cos)).sum() / w_diff
            else:
                l_diff = torch.zeros((), device=device)
            gicr = l_same + self.lambda_gicr_diff * l_diff
            epoch = getattr(self.model.model[-1], "train_epoch", 0)
            if epoch < self.gicr_warmup_epochs:
                gicr = gicr * 0.0

        return relation, preserve, gicr

    @staticmethod
    def _geo_matrix(boxes_xyxy: torch.Tensor, img_wh: tuple[int, int]) -> torch.Tensor:
        """Pairwise scale-aware geometric distance [P, P] (same formula as MDRR)."""
        w_img, h_img = img_wh
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        w = (x2 - x1).clamp(min=1e-3)
        h = (y2 - y1).clamp(min=1e-3)
        u = (x1 + x2) / 2 / max(1, w_img)
        v = (y1 + y2) / 2 / max(1, h_img)
        du = u.unsqueeze(-1) - u.unsqueeze(-2)
        dv = v.unsqueeze(-1) - v.unsqueeze(-2)
        lw = torch.log((w.unsqueeze(-1) / w.unsqueeze(-2).clamp(min=1e-3)).clamp(1e-3, 1e3))
        lh = torch.log((h.unsqueeze(-1) / h.unsqueeze(-2).clamp(min=1e-3)).clamp(1e-3, 1e3))
        return torch.sqrt(du**2 + dv**2 + lw**2 + lh**2 + 1e-8)
