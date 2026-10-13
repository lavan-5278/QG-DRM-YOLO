"""Semantic Evidence Relation Aggregator for visible upper-body behaviour cues."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect

__all__ = "SERA", "SERADetect"


class SERA(nn.Module):
    """Inject semantically gated P4/P5 context into P3 while retaining an identity initialization."""

    def __init__(self, ch: tuple[int, int, int], sera_cfg: dict | None = None):
        super().__init__()
        c3, c4, c5 = ch
        cfg = dict(sera_cfg or {})
        hidden = max(8, int(round(c3 * float(cfg.get("reduction", 0.25)))))
        self.to_p4 = Conv(c4, c3, 1)
        self.to_p5 = Conv(c5, c3, 1)
        self.gate = nn.Sequential(Conv(c3 * 2, hidden, 1), DWConv(hidden, hidden, 3), nn.Conv2d(hidden, c3, 1))
        self.fusion_gain = nn.Parameter(torch.zeros(1))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Return P3 plus spatially compatible upsampled context from P4 and P5."""
        p3, p4, p5 = x
        size = p3.shape[-2:]
        context = F.interpolate(self.to_p4(p4), size=size, mode="bilinear", align_corners=False)
        context = context + F.interpolate(self.to_p5(p5), size=size, mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat((p3, context), 1)).sigmoid()
        return p3 + self.fusion_gain * gate * context


class SERADetect(Detect):
    """Standard YOLO Detect head with SERA applied only to its P3 input feature."""

    def __init__(self, nc: int = 80, ch: tuple = (), sera_cfg: dict | None = None):
        super().__init__(nc, ch)
        self.sera = SERA(tuple(ch), sera_cfg)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Refine P3 in-head, preserving the base Detect parameter namespace for pretrained transfer."""
        if self.end2end:
            raise NotImplementedError("SERADetect does not support end-to-end heads.")
        x[0] = self.sera(x)
        return super().forward(x)
