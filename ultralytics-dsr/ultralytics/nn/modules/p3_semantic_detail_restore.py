"""P3 local-detail restoration guided by aligned P4 semantic context."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv


class P3SemanticDetailRestore(nn.Module):
    """Zero-init P3 residual adapter conditioned on an aligned P4 feature.

    The P3 input preserves high-resolution localization cues while P4 offers a
    more class-discriminative receptive field.  The residual scale starts at
    zero, so replacing :class:`P3DetailRestore` with this module is an exact
    identity at initialization; this isolates the semantic-context hypothesis
    without disturbing the converged architecture's initial optimization path.
    """

    def __init__(self, c1: int, context_channels: int):
        super().__init__()
        if c1 <= 0 or context_channels <= 0:
            raise ValueError(f"P3SemanticDetailRestore requires positive channels, got {c1}, {context_channels}.")
        self.channels = int(c1)
        self.context_channels = int(context_channels)
        self.local = Conv(c1, c1, k=3, s=1, g=c1)
        self.context = Conv(context_channels, c1, k=1, s=1, act=False)
        self.fuse = Conv(2 * c1, c1, k=1, s=1, act=False)
        self.reliability = nn.Conv2d(2 * c1, 1, kernel_size=1, stride=1, bias=True)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        self.residual_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor | list[torch.Tensor], context: torch.Tensor | None = None) -> torch.Tensor:
        # Ultralytics supplies multi-source YAML layers as a two-element list;
        # retaining the explicit two-tensor form also makes unit tests concise.
        if context is None:
            if not isinstance(x, (list, tuple)) or len(x) != 2:
                raise ValueError("P3SemanticDetailRestore requires [P3, P4] when called with one argument.")
            x, context = x
        if x.ndim != 4 or context.ndim != 4:
            raise ValueError("P3SemanticDetailRestore expects NCHW P3 and P4 tensors.")
        if x.shape[1] != self.channels or context.shape[1] != self.context_channels:
            raise ValueError(
                f"P3SemanticDetailRestore expects {self.channels}-channel P3 and "
                f"{self.context_channels}-channel P4, got {x.shape[1]} and {context.shape[1]}."
            )
        local = self.local(x)
        semantic = self.context(context)
        if semantic.shape[-2:] != x.shape[-2:]:
            semantic = F.interpolate(semantic, size=x.shape[-2:], mode="nearest")
        joint = torch.cat((local, semantic), dim=1)
        detail = self.fuse(joint)
        gate = torch.sigmoid(self.reliability(joint))
        return x + self.residual_scale.to(dtype=x.dtype) * gate * detail

    @torch.no_grad()
    def adapter_state(self) -> dict[str, float]:
        return {
            "residual_scale": float(self.residual_scale.detach().cpu()),
            "initial_mean_gate": 0.5,
            "context_channels": float(self.context_channels),
        }


__all__ = ("P3SemanticDetailRestore",)
