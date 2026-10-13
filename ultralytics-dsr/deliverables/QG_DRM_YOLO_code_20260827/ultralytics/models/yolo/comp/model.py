# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CompositionDetectionModel: YOLO + auxiliary image-level composition head (CAAL).

All training/evaluation still runs through ``model.train()`` / ``model.val()`` /
``model.predict()``. The composition auxiliary loss only exists in training; at
inference ``CompositionDetect`` is byte-identical to the official ``Detect``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from ultralytics.nn.modules.composition import CompositionDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class CompositionLoss(v8DetectionLoss):
    """Official YOLO loss + L_comp (image-level behaviour-presence BCE)."""

    def __init__(self, model, tal_topk: int = 10):
        super().__init__(model, tal_topk)
        cfg = getattr(model.model[-1], "comp_cfg", {}) or {}
        self.lambda_comp = float(cfg.get("lambda_comp", 0.5))
        self.bce_comp = nn.BCEWithLogitsLoss()

    def __call__(self, preds, batch, comp_logits=None):
        """Return a 4-component loss vector (box, cls, dfl, comp)."""
        base_loss, base_items = super().__call__(preds, batch)
        if comp_logits is None:
            comp = base_loss.new_zeros(())
        else:
            bs = comp_logits.shape[0]
            nc = comp_logits.shape[1]
            target = comp_logits.new_zeros(bs, nc)
            for bi in range(bs):
                cls_bi = batch["cls"][batch["batch_idx"] == bi]
                if cls_bi.numel():
                    target[bi, cls_bi.long().unique()] = 1.0
            comp = self.bce_comp(comp_logits, target) * self.lambda_comp
        items = torch.cat((base_loss, comp.reshape(1)))
        return items, items.detach()


class CompositionDetectionModel(DetectionModel):
    """DetectionModel that drives CompositionDetect + CompositionLoss."""

    def __init__(self, cfg="yolo11n-comp.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        head = self.model[-1]
        if isinstance(head, CompositionDetect):
            comp_cfg = self.yaml.get("comp") if isinstance(self.yaml.get("comp"), dict) else {}
            head._configure_comp(comp_cfg)

    def init_criterion(self):
        return CompositionLoss(self)

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        head = self.model[-1]
        comp_logits = getattr(head, "comp_logits", None) if isinstance(head, CompositionDetect) else None
        return self.criterion(preds, batch, comp_logits=comp_logits)
