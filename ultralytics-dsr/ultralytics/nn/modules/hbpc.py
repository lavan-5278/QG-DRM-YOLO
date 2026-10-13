"""Head--behaviour-part complementarity classification calibration for YOLO Detect."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import DWConv
from ultralytics.nn.modules.head import Detect

__all__ = "HBPC", "HBPCDetect"


class HBPC(nn.Module):
    """Calibrate one classification feature map from local upper and lower directional evidence."""

    def __init__(self, c: int, hbpc_cfg: dict | None = None):
        super().__init__()
        cfg = dict(hbpc_cfg or {})
        self.shift = int(cfg.get("shift", 1))
        if self.shift < 1:
            raise ValueError(f"shift must be positive, got {self.shift}")
        hidden = max(8, int(round(c * float(cfg.get("reduction", 0.25)))))
        self.scale = float(cfg.get("scale", 1.0))
        self.head_reduce = nn.Conv2d(c, hidden, 1, bias=False)
        self.behaviour_reduce = nn.Conv2d(c, hidden, 1, bias=False)
        self.mixer = nn.Sequential(DWConv(hidden, hidden, 3), nn.SiLU())
        self.project = nn.Conv2d(hidden, c, 1)
        self.gain = nn.Parameter(torch.zeros(1))

    def _directional_context(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return upper and lower neighbouring evidence aligned to every current feature location."""
        height = x.shape[-2]
        head = F.pad(x, (0, 0, self.shift, 0))[:, :, :height, :]
        behaviour = F.pad(x, (0, 0, 0, self.shift))[:, :, self.shift : self.shift + height, :]
        return head, behaviour

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an identity-initialized, complementarity-calibrated class feature."""
        head, behaviour = self._directional_context(x)
        head = self.head_reduce(head)
        behaviour = self.behaviour_reduce(behaviour)
        delta = self.project(self.mixer(head + behaviour + head * behaviour))
        return x * (1.0 + self.gain * self.scale * (delta.sigmoid() - 0.5))


class HBPCDetect(Detect):
    """YOLO Detect head where HBPC affects class logits but not bounding-box regression."""

    def __init__(self, nc: int = 80, ch: tuple = (), hbpc_cfg: dict | None = None):
        super().__init__(nc, ch)
        self.hbpc = nn.ModuleList(HBPC(c, hbpc_cfg) for c in ch)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Keep each box branch original and calibrate only its paired class branch."""
        if self.end2end:
            raise NotImplementedError("HBPCDetect does not support end-to-end heads.")
        for i in range(self.nl):
            box_feature = x[i]
            x[i] = torch.cat((self.cv2[i](box_feature), self.cv3[i](self.hbpc[i](box_feature))), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)
