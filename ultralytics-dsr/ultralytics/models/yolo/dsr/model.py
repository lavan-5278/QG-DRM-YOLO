# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DSR-Det model (Phase 0/1 skeleton).

``DSRDetectionModel`` keeps the official ``DetectionModel`` interface while wiring the
DSR head outputs into ``DSRDetectionLoss`` inside the standard ``loss()`` path. All
training/evaluation still runs through ``model.train()`` / ``model.val()``.
"""

from __future__ import annotations

from types import SimpleNamespace

from ultralytics.nn.modules.dsr import DSRDetect
from ultralytics.nn.tasks import DetectionModel

from .loss import DSRDetectionLoss


class DSRDetectionModel(DetectionModel):
    """DetectionModel subclass that drives the DSRDetect head and DSRDetectionLoss."""

    def __init__(self, cfg="yolo11n-dsr.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg, ch, nc, verbose)
        # Default hyperparameters so the model is usable standalone; the Trainer
        # overwrites `model.args` with the real training config in set_model_attributes.
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        head = self.model[-1]
        if isinstance(head, DSRDetect):
            dsr_cfg = self.yaml.get("dsr") if isinstance(self.yaml.get("dsr"), dict) else {}
            head._configure_dsr(dsr_cfg)

    def init_criterion(self):
        """Initialize the DSRDetectionLoss criterion."""
        return DSRDetectionLoss(self)

    def loss(self, batch, preds=None):
        """Compute base YOLO loss plus refined DSR losses via the head's ``dsr_out``."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        head = self.model[-1]
        dsr_out = getattr(head, "dsr_out", None) if isinstance(head, DSRDetect) else None
        return self.criterion(preds, batch, dsr_out=dsr_out)
