# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CompositionHead / CompositionDetect: Composition-Aware Auxiliary Learning (CAAL).

Dataset diagnosis (relation_diagnosis/) showed the strongest relational signal in
classroom behavior detection is the *image-level behaviour composition*: the GT
composition predicts an individual's behaviour at ~0.93 top-1 accuracy vs ~0.82
for the model's own noisy estimate. ``CompositionDetect`` trains the backbone/neck
to encode this signal by adding a tiny auxiliary head that predicts which
behaviours are present in the image (multi-label BCE) during training. At
inference the auxiliary branch is a no-op, so the improved features come for free.

- ``CompositionDetect``: ``Detect`` subclass; training forward snapshots the neck
  features, runs ``CompositionHead`` on the P5 global context and stores
  ``self.comp_logits`` for ``CompositionLoss``. Inference is byte-identical to the
  official ``Detect`` head (``comp.enabled=false`` => identical output).
- ``CompositionHead``: global-avg-pool P5 -> BN + MLP -> per-class presence logits.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect

__all__ = "CompositionDetect", "CompositionHead"


class CompositionHead(nn.Module):
    """Image-level behaviour-composition predictor (auxiliary, training only)."""

    def __init__(self, in_channels: int, nc: int, hidden: int = 256, embed_dim: int = 64):
        super().__init__()
        self.nc = nc
        self.embed_dim = embed_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.backbone_mlp = nn.Sequential(nn.Flatten(), nn.Linear(in_channels, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.embed = nn.Linear(hidden, embed_dim)
        self.cls = nn.Linear(embed_dim, nc)

    def forward(self, p5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (class-presence logits [B, nc], composition embedding [B, C])."""
        h = self.backbone_mlp(self.pool(p5))
        e = self.embed(h)
        return self.cls(e), e


class CompositionDetect(Detect):
    """Detect head extended with an auxiliary composition branch (training only)."""

    def __init__(self, nc: int = 80, ch: tuple = (), comp_cfg: dict | None = None):
        super().__init__(nc, ch)
        self.ch = tuple(ch)
        self.comp_cfg: dict[str, Any] = {}
        self.comp_active = False
        self.comp_logits: torch.Tensor | None = None
        self.comp_bias: nn.Linear | None = None
        self._configure_comp({} if comp_cfg is None else comp_cfg)

    def _configure_comp(self, cfg: dict[str, Any] | None) -> None:
        cfg = dict(cfg or {})
        if cfg == self.comp_cfg and getattr(self, "comp_head", None) is not None:
            return
        self.comp_cfg = cfg
        self.comp_active = bool(cfg.get("enabled", True))
        self.lambda_comp = float(cfg.get("lambda_comp", 0.5))
        self.inject_bias = bool(cfg.get("inject", True))
        if self.comp_active and self.ch:
            self.comp_head = CompositionHead(
                in_channels=int(self.ch[-1]),
                nc=self.nc,
                hidden=int(cfg.get("hidden", 256)),
                embed_dim=int(cfg.get("embed_dim", 64)),
            )
            self.comp_bias = nn.Linear(int(cfg.get("embed_dim", 64)), self.nc)
            nn.init.zeros_(self.comp_bias.weight)
            nn.init.zeros_(self.comp_bias.bias)  # start identical to official Detect
        else:
            self.comp_head = None
            self.comp_bias = None

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Same output contract as ``Detect.forward``; composition branch adds a
        per-image class bias to the classification logits (training + inference)."""
        if self.end2end:
            return self.forward_end2end(x)
        x_in = list(x)  # snapshot neck features BEFORE head convs (P3/P4/P5)
        comp_bias = None
        if self.comp_active and self.comp_head is not None and self.stride.max() > 0:
            logits, emb = self.comp_head(x_in[-1])
            self.comp_logits = logits if self.training else None
            if self.inject_bias and self.comp_bias is not None:
                comp_bias = self.comp_bias(emb)  # [B, nc]
        else:
            self.comp_logits = None
        for i in range(self.nl):
            cls_out = self.cv3[i](x[i])
            if comp_bias is not None:
                cls_out = cls_out + comp_bias.view(cls_out.shape[0], self.nc, 1, 1)
            x[i] = torch.cat((self.cv2[i](x[i]), cls_out), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)
