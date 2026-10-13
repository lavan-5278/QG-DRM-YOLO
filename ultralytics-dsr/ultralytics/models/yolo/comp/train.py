# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CompositionTrainer: DetectionTrainer subclass that builds CompositionDetectionModel."""

from __future__ import annotations

from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK

from .model import CompositionDetectionModel


class CompositionTrainer(DetectionTrainer):
    """DetectionTrainer subclass for the composition-aware YOLO variant."""

    LOSS_NAMES = ("box_loss", "cls_loss", "dfl_loss", "comp_loss")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = CompositionDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        from ultralytics.models.yolo.detect.val import DetectionValidator

        self.loss_names = self.LOSS_NAMES
        return DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.LOSS_NAMES]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))
