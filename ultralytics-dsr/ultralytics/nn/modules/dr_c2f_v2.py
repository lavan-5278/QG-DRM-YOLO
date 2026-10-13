from __future__ import annotations

import math

import torch
from torch import nn

from .block import Bottleneck
from .conv import Conv


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    """Convert a probability in (0, 1) to a numerically safe logit."""
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


class DRC2fV2(nn.Module):
    """Anchored Delta-Relay C2f with decoupled work and accumulated state streams."""

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
        state_init: float = 0.5,
        anchor_decay: float = 1.0,
    ):
        super().__init__()
        if n < 1:
            raise ValueError(f"DRC2fV2 requires n >= 1, got {n}.")
        if not 0.0 < e <= 1.0:
            raise ValueError(f"e must be in (0, 1], got {e}.")
        if not 0.0 < delta_init < 2.0:
            raise ValueError(f"delta_init must be in (0, 2), got {delta_init}.")
        if not 0.0 < anchor_init < 1.0:
            raise ValueError(f"anchor_init must be in (0, 1), got {anchor_init}.")
        if not 0.0 < state_init < 1.0:
            raise ValueError(f"state_init must be in (0, 1), got {state_init}.")
        if not 0.0 < anchor_decay <= 1.0:
            raise ValueError(f"anchor_decay must be in (0, 1], got {anchor_decay}.")

        self.c = int(c2 * e)
        if self.c < 1:
            raise ValueError(f"Hidden channels became {self.c}; check c2={c2}, e={e}.")
        self.n = int(n)
        self.requested_shortcut = bool(shortcut)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(3 * self.c, c2, 1, 1)
        # The relay delta must describe a true transformation, so internal residuals stay disabled.
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=False, g=g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        self.delta_logits = nn.Parameter(torch.full((n,), _safe_logit(delta_init / 2.0), dtype=torch.float32))
        anchors = [max(min(anchor_init * anchor_decay**i, 1.0 - 1e-6), 1e-6) for i in range(n)]
        self.anchor_logits = nn.Parameter(torch.tensor([_safe_logit(value) for value in anchors], dtype=torch.float32))
        self.state_logits = nn.Parameter(torch.full((n,), _safe_logit(state_init), dtype=torch.float32))

    def _coefficients(self, i: int, dtype: torch.dtype):
        return (
            (2.0 * torch.sigmoid(self.delta_logits[i])).to(dtype=dtype),
            torch.sigmoid(self.anchor_logits[i]).to(dtype=dtype),
            torch.sigmoid(self.state_logits[i]).to(dtype=dtype),
        )

    def _relay(self, xb: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        u, state = u0, u0
        for i, block in enumerate(self.m):
            delta = block(u) - u
            delta_gain, anchor_gain, state_gain = self._coefficients(i, u.dtype)
            u = delta_gain * delta + anchor_gain * u0
            state = state + state_gain * u
        return self.cv2(torch.cat((xb, state, u), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb, u0 = self.cv1(x).chunk(2, dim=1)
        return self._relay(xb, u0)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        xb, u0 = self.cv1(x).split((self.c, self.c), dim=1)
        return self._relay(xb, u0)

    @torch.no_grad()
    def relay_coefficients(self) -> dict[str, list[float]]:
        """Expose bounded learned relay coefficients for post-training evidence logging."""
        return {
            "delta_gain": (2.0 * torch.sigmoid(self.delta_logits)).cpu().tolist(),
            "anchor_gain": torch.sigmoid(self.anchor_logits).cpu().tolist(),
            "state_gain": torch.sigmoid(self.state_logits).cpu().tolist(),
        }
