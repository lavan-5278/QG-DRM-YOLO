from __future__ import annotations

import torch
from torch import nn

from .block import Bottleneck
from .conv import Conv


class DRC2fV3(nn.Module):
    """Lite Anchored Delta-Relay C2f: direct relay scalars and ungated state accumulation."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        delta_init: float = 1.0,
        anchor_init: float = 0.25,
        anchor_decay: float = 1.0,
    ):
        super().__init__()
        if n < 1:
            raise ValueError(f"DRC2fV3 requires n >= 1, got n={n}.")
        if not 0.0 < e <= 1.0:
            raise ValueError(f"e must be in (0, 1], got e={e}.")
        if anchor_decay <= 0.0:
            raise ValueError(f"anchor_decay must be > 0, got {anchor_decay}.")
        self.c = int(c2 * e)
        if self.c < 1:
            raise ValueError(f"Hidden channels became {self.c}; check c2={c2}, e={e}.")
        self.n = int(n)
        self.requested_shortcut = bool(shortcut)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(3 * self.c, c2, 1, 1)
        # Shortcut must remain off, otherwise the relay delta collapses to the residual branch.
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=False, g=g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        # Direct parameters deliberately avoid sigmoid, clamp, RMS, and reductions in forward.
        self.delta_gain = nn.Parameter(torch.full((n,), float(delta_init), dtype=torch.float32))
        self.anchor_gain = nn.Parameter(torch.tensor([float(anchor_init) * float(anchor_decay) ** i for i in range(n)], dtype=torch.float32))

    def _relay(self, xb: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        u, state = u0, u0
        for i, block in enumerate(self.m):
            delta = block(u) - u
            u = self.delta_gain[i].to(dtype=u.dtype) * delta + self.anchor_gain[i].to(dtype=u.dtype) * u0
            state = state + u
        return self.cv2(torch.cat((xb, state, u), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb, u0 = self.cv1(x).chunk(2, dim=1)
        return self._relay(xb, u0)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        xb, u0 = self.cv1(x).split((self.c, self.c), dim=1)
        return self._relay(xb, u0)

    @torch.no_grad()
    def relay_coefficients(self) -> dict[str, list[float]]:
        return {"delta_gain": self.delta_gain.detach().cpu().tolist(), "anchor_gain": self.anchor_gain.detach().cpu().tolist()}

    @torch.no_grad()
    def clamp_relay_parameters_(self, delta_range: tuple[float, float] = (0.0, 2.0), anchor_range: tuple[float, float] = (0.0, 1.0)) -> None:
        """Optional post-optimizer-step stability utility; never called by forward or this experiment."""
        self.delta_gain.clamp_(*delta_range)
        self.anchor_gain.clamp_(*anchor_range)
