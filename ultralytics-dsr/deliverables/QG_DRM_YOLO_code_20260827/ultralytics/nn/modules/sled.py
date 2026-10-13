"""Spatial Layout Evidence Decoupling detection head."""

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect

__all__ = "SLEDDetect", "SLEDLayoutGate"


class SLEDLayoutGate(nn.Module):
    """Gate a feature map with compact vertical-layout context while preserving its untrained identity."""

    def __init__(self, c: int, bands: int = 3, reduction: float = 0.25, scale: float = 1.0):
        super().__init__()
        if bands < 1:
            raise ValueError(f"bands must be positive, got {bands}")
        hidden = max(8, int(round(c * reduction)))
        self.bands = int(bands)
        self.scale = float(scale)
        self.reduce = nn.Conv2d(c, hidden, 1, bias=False)
        self.vertical = nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0), groups=hidden, bias=True)
        self.project = nn.Conv2d(hidden, c, 1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return x modulated by a channel-wise gate reconstructed from vertical evidence bands."""
        height, width = x.shape[-2:]
        layout = torch.cat(
            [
                x[:, :, (index * height) // self.bands : min(height, ((index + 1) * height + self.bands - 1) // self.bands), :].mean(
                    dim=(2, 3), keepdim=True
                )
                for index in range(self.bands)
            ],
            dim=2,
        )
        layout = torch.nn.functional.silu(self.vertical(self.reduce(layout)))
        rows = torch.div(torch.arange(height, device=x.device) * self.bands, height, rounding_mode="floor")
        gate = self.project(layout).index_select(2, rows).expand(-1, -1, -1, width)
        return x * (1.0 + self.scale * (gate.sigmoid() - 0.5))


class SLEDDetect(Detect):
    """YOLO detection head that applies SLED only before the classification branch."""

    def __init__(self, nc: int = 80, ch: tuple = (), sled_cfg: dict | None = None):
        super().__init__(nc, ch)
        cfg = dict(sled_cfg or {})
        self.sled = nn.ModuleList(
            SLEDLayoutGate(
                c,
                bands=int(cfg.get("bands", 3)),
                reduction=float(cfg.get("reduction", 0.25)),
                scale=float(cfg.get("scale", 1.0)),
            )
            for c in ch
        )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Use original features for boxes and SLED-gated features for class logits."""
        if self.end2end:
            raise NotImplementedError("SLEDDetect does not support end-to-end heads.")
        for i in range(self.nl):
            box_feature = x[i]
            x[i] = torch.cat((self.cv2[i](box_feature), self.cv3[i](self.sled[i](box_feature))), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)
