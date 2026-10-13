"""Behavior-conditioned side-quality detection head."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors

__all__ = "BCSQDetect",


class BCSQDetect(Detect):
    """Detect head with one left/top/right/bottom quality logit for every class."""

    def __init__(self, nc: int = 80, ch: tuple = (), bcsq_cfg: dict | None = None):
        super().__init__(nc, ch)
        c_quality = max(16, min(ch[0], 64))
        self.base_no = self.no
        self.quality_no = self.nc * 4
        self.no = self.base_no + self.quality_no
        cfg = dict(bcsq_cfg or {})
        self.quality_eta = float(cfg.get("eta", 1.0))
        self.quality_tau = float(cfg.get("tau", 0.2))
        self.quality_lambda = float(cfg.get("lambda_quality", 0.5))
        self.quality_alpha = float(cfg.get("background_alpha", 0.75))
        self.quality_gamma = float(cfg.get("background_gamma", 2.0))
        self.cvq = nn.ModuleList(
            nn.Sequential(DWConv(c, c, 3), Conv(c, c_quality, 1), nn.Conv2d(c_quality, self.quality_no, 1)) for c in ch
        )
        for branch in self.cvq:
            nn.init.constant_(branch[-1].bias, 4.59511985013459)  # sigmoid ~= 0.99; stable pre-training ranking

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Return calibrated standard predictions and preserve raw quality logits for training."""
        if self.end2end:
            raise NotImplementedError("BCSQDetect does not support end-to-end heads.")
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i]), self.cvq[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Decode boxes and calibrate each class score with its four side-quality probabilities."""
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (tensor.transpose(0, 1) for tensor in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls, quality = x_cat.split((self.reg_max * 4, self.nc, self.quality_no), 1)
        quality = quality.view(shape[0], self.nc, 4, -1).sigmoid().prod(2).pow(0.25 * self.quality_eta)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid() * quality), 1)
        return x
