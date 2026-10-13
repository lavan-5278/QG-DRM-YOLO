from __future__ import annotations

import torch
from torch import nn

from .block import Bottleneck
from .conv import Conv


class DRC2fV4(nn.Module):
    """Residual-enhanced relay C2f: retain transformed semantics and add a delta correction/state."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, alpha_init: float = 0.25, beta_init: float = 0.5):
        super().__init__()
        if n < 1:
            raise ValueError(f"DRC2fV4 requires n >= 1, got n={n}.")
        if not 0.0 < e <= 1.0:
            raise ValueError(f"e must be in (0, 1], got e={e}.")
        self.c = int(c2 * e)
        if self.c < 1:
            raise ValueError(f"Hidden channels became {self.c}; check c2={c2}, e={e}.")
        self.n = int(n)
        self.requested_shortcut = bool(shortcut)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(3 * self.c, c2, 1, 1)
        # A true transformation delta requires the inner residual shortcut to be off.
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut=False, g=g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # Direct scalars; forward deliberately contains no sigmoid, RMS, division, or reduction.
        self.alpha = nn.Parameter(torch.full((n,), float(alpha_init), dtype=torch.float32))
        self.beta = nn.Parameter(torch.full((n,), float(beta_init), dtype=torch.float32))

    def _relay(self, xb: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        state = torch.zeros_like(u)
        for i, block in enumerate(self.m):
            previous = u
            z = block(previous)
            delta = z - previous
            u = z + self.alpha[i].to(dtype=z.dtype) * delta
            state = state + self.beta[i].to(dtype=z.dtype) * delta
        return self.cv2(torch.cat((xb, u, state), dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb, u = self.cv1(x).chunk(2, dim=1)
        return self._relay(xb, u)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        xb, u = self.cv1(x).split((self.c, self.c), dim=1)
        return self._relay(xb, u)

    @torch.no_grad()
    def relay_coefficients(self) -> dict[str, list[float]]:
        return {"alpha_delta_correction": self.alpha.detach().cpu().tolist(), "beta_delta_state": self.beta.detach().cpu().tolist()}

    @torch.no_grad()
    def clamp_relay_parameters_(self, alpha_range: tuple[float, float] = (-0.5, 1.0), beta_range: tuple[float, float] = (0.0, 1.5)) -> None:
        """Optional post-step stability helper; deliberately unused by this controlled experiment."""
        self.alpha.clamp_(*alpha_range)
        self.beta.clamp_(*beta_range)
