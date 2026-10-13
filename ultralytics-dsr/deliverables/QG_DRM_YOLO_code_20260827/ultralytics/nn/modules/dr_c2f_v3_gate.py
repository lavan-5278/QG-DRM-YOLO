from __future__ import annotations

import torch
from torch import nn

from .block import C2f
from .dr_c2f_v3 import DRC2fV3


class RGDRC2fV3(nn.Module):
    """Zero-init, reliability-gated DR-C2f v3 residual adapter.

    A native C2f branch preserves the original local representation.  A reduced-width
    DRC2fV3 branch supplies an optional semantic relay, while a learned spatial gate
    selects where that relay is useful.  The residual scale starts at zero, so the
    module is exactly equivalent to its native C2f branch at initialization.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        relay_e: float = 0.25,
    ):
        super().__init__()
        if not 0.0 < relay_e <= 1.0:
            raise ValueError(f"relay_e must be in (0, 1], got {relay_e}.")
        self.base = C2f(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.relay = DRC2fV3(c1, c2, n=n, shortcut=shortcut, g=g, e=relay_e)
        self.reliability = nn.Conv2d(c2, 1, kernel_size=1, stride=1, bias=True)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        self.residual_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        relay = self.relay(x)
        gate = torch.sigmoid(self.reliability(base))
        scale = self.residual_scale.to(dtype=base.dtype)
        return base + scale * gate * (relay - base)

    @torch.no_grad()
    def adapter_state(self) -> dict[str, float | dict[str, list[float]]]:
        return {
            "residual_scale": float(self.residual_scale.detach().cpu()),
            "initial_mean_gate": 0.5,
            "relay_coefficients": self.relay.relay_coefficients(),
        }
