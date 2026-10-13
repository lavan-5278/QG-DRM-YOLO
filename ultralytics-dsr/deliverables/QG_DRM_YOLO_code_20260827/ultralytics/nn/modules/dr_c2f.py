"""Delta-Relay C2f block used by the standalone DR-C2f ablation."""

from __future__ import annotations

import torch
from torch import nn

from .block import Bottleneck
from .conv import Conv


class DRC2f(nn.Module):
    """C2f-compatible delta-relay block with bounded RMS update matching.

    ``shortcut`` is retained only for YAML compatibility. Internal bottlenecks
    always disable it so that the transmitted delta is non-trivial.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        q_min: float = 0.5,
        q_max: float = 2.0,
        eps: float = 1e-6,
        gamma_init: float = 0.5,
        detach_scale: bool = False,
    ):
        super().__init__()
        if n < 1:
            raise ValueError(f"DRC2f requires n >= 1, got {n}.")
        if not (0.0 < e <= 1.0):
            raise ValueError(f"Expansion e must be in (0, 1], got {e}.")
        if not (0.0 < q_min <= q_max):
            raise ValueError(f"Require 0 < q_min <= q_max, got {q_min}, {q_max}.")
        if not (0.0 < gamma_init < 1.0):
            raise ValueError(f"gamma_init must be in (0, 1), got {gamma_init}.")

        self.c = int(c2 * e)
        if self.c < 1:
            raise ValueError(f"Hidden channels became {self.c}; check c2={c2}, e={e}.")
        self.q_min, self.q_max, self.eps = float(q_min), float(q_max), float(eps)
        self.detach_scale = bool(detach_scale)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(3 * self.c, c2, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut=False, g=g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )
        theta0 = torch.logit(torch.tensor(float(gamma_init))).item()
        self.theta = nn.Parameter(torch.full((n,), theta0, dtype=torch.float32))

    def _bounded_rms_match(self, delta: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        delta32, ref32 = delta.float(), ref.float()
        delta_rms = torch.sqrt(delta32.square().mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        ref_rms = torch.sqrt(ref32.square().mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        scale = (ref_rms / delta_rms).clamp(self.q_min, self.q_max)
        if self.detach_scale:
            scale = scale.detach()
        return delta * scale.to(dtype=delta.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb, u = self.cv1(x).chunk(2, dim=1)
        state = u
        for i, block in enumerate(self.m):
            delta = block(u) - u
            u = self._bounded_rms_match(delta, u)
            state = state + torch.sigmoid(self.theta[i]).to(dtype=u.dtype) * u
        return self.cv2(torch.cat((xb, state, u), dim=1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        xb, u = self.cv1(x).split((self.c, self.c), dim=1)
        state = u
        for i, block in enumerate(self.m):
            delta = block(u) - u
            u = self._bounded_rms_match(delta, u)
            state = state + torch.sigmoid(self.theta[i]).to(dtype=u.dtype) * u
        return self.cv2(torch.cat((xb, state, u), dim=1))
