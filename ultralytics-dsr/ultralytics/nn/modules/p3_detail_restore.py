from __future__ import annotations

import torch
from torch import nn

from .conv import Conv


class P3DetailRestore(nn.Module):
    """Zero-init, reliability-gated local residual adapter for the P3 detection feature."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"P3DetailRestore requires an identity channel mapping, got {c1}->{c2}.")
        self.local = Conv(c1, c1, k=3, s=1, g=c1)
        self.project = Conv(c1, c1, k=1, s=1, act=False)
        self.reliability = nn.Conv2d(c1, 1, kernel_size=1, stride=1, bias=True)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        self.residual_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.project(self.local(x))
        gate = torch.sigmoid(self.reliability(x))
        return x + self.residual_scale.to(dtype=x.dtype) * gate * detail

    @torch.no_grad()
    def adapter_state(self) -> dict[str, float]:
        return {"residual_scale": float(self.residual_scale.detach().cpu()), "initial_mean_gate": 0.5}
