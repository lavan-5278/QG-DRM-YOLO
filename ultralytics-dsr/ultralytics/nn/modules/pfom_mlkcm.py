"""PFOM and MLKCM modules for a YOLOv8s paper-structure reproduction."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    return d * (k - 1) // 2 if p is None else p


class ConvBNAct(nn.Module):
    """Conv-BN-SiLU used for the generic convolutional blocks in the reproduction."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, d: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, d=d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class PFOMLayerNorm2d(nn.Module):
    """LayerNorm across channels for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class GhostConvLite(nn.Module):
    def __init__(self, c1: int, c2: int, ratio: int = 2):
        super().__init__()
        c_primary = max(1, math.ceil(c2 / ratio))
        c_cheap = max(0, c2 - c_primary)
        self.primary = ConvBNAct(c1, c_primary, 1, 1)
        self.cheap = (
            ConvBNAct(c_primary, c_cheap, 5, 1, g=c_primary)
            if c_cheap > 0 and c_cheap % c_primary == 0
            else (
                nn.Sequential(
                    nn.Conv2d(c_primary, c_cheap, 5, 1, 2, groups=math.gcd(c_primary, c_cheap), bias=False),
                    nn.BatchNorm2d(c_cheap),
                    nn.SiLU(inplace=True),
                )
                if c_cheap > 0
                else nn.Identity()
            )
        )
        self.c2, self.c_cheap = c2, c_cheap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        if self.c_cheap == 0:
            return primary[:, : self.c2]
        return torch.cat((primary, self.cheap(primary)), 1)[:, : self.c2]


class GhostModule(nn.Module):
    """Ghost refinement path: Conv -> Ghost -> DWConv -> Ghost -> gated Conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.pre = ConvBNAct(channels, channels, 1, 1)
        self.g1 = GhostConvLite(channels, channels)
        self.dw = ConvBNAct(channels, channels, 3, 1, g=channels)
        self.g2 = GhostConvLite(channels, channels)
        self.post = ConvBNAct(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.pre(x)
        # The supplied reproduction left this multiplicative path unbounded.
        # With six stacked PFOM repeats at P3/P4 it produced an activation/
        # gradient explosion during scratch SGD training.  The paper specifies
        # a multiplicative gate but not its activation, so use a standard
        # bounded sigmoid gate to preserve the mechanism while keeping its
        # gain in [0, 1].
        gate = torch.sigmoid(self.g2(self.dw(self.g1(main))))
        return self.post(main * gate)


class RepeatBlock(nn.Module):
    """The two-branch repeat block used inside PFOM."""

    def __init__(self, channels: int, n: int = 1):
        super().__init__()
        self.left = ConvBNAct(channels, channels, 1, 1)
        self.right_in = ConvBNAct(channels, channels, 1, 1)
        self.ghosts = nn.Sequential(*(GhostModule(channels) for _ in range(max(1, n))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.left(x), self.ghosts(self.right_in(x))), 1)


class PFOM(nn.Module):
    """Progressive Feature Optimization Module with a C2f-compatible signature."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        hidden = max(8, int(c2 * e))
        self.cv_in = ConvBNAct(c1, 2 * hidden, 1, 1)
        self.rb1 = RepeatBlock(hidden, n=n)
        self.rb2 = RepeatBlock(hidden, n=n)
        self.cv3 = ConvBNAct(2 * hidden, hidden, 1, 1)
        self.cv4 = ConvBNAct(2 * hidden, hidden, 1, 1)
        self.cv_out = ConvBNAct(4 * hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1, y2 = self.cv_in(x).chunk(2, 1)
        y3 = self.cv3(self.rb1(y2))
        y4 = self.cv4(self.rb2(y3))
        return self.cv_out(torch.cat((y1, y2, y3, y4), 1))


class DCP(nn.Module):
    """Depthwise convolution path: X * PW(DWD(DW(X)))."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.dw = ConvBNAct(channels, channels, kernel_size, 1, g=channels)
        self.dwd = ConvBNAct(channels, channels, kernel_size, 1, g=channels, d=dilation)
        self.pw = ConvBNAct(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.pw(self.dwd(self.dw(x)))


def _balanced_split_sizes(channels: int, groups: int) -> list[int]:
    base, remainder = divmod(channels, groups)
    return [base + (1 if index < remainder else 0) for index in range(groups)]


class MultiAxisBlock(nn.Module):
    """Axis-wise pooling, sigmoid gating and residual reweighting."""

    def __init__(self, channels: int):
        super().__init__()
        self.pre = ConvBNAct(channels, channels, 1, 1)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, 1, 0, bias=False), nn.BatchNorm2d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        _, _, height, width = z.shape
        pool_h = torch.amax(z, dim=2, keepdim=True).expand(-1, -1, height, -1)
        pool_w = torch.mean(z, dim=3, keepdim=True).expand(-1, -1, -1, width)
        gate = torch.sigmoid(self.fuse(torch.cat((pool_h, pool_w), 1)))
        return x + gate * x


class MLKCM(nn.Module):
    """Three-branch multi-scale large-kernel convolution module.

    The source paper leaves the branch dilation values unspecified; the supplied
    reproduction uses the explicit default ``(1, 2, 3)`` for kernels ``(3, 5, 7)``.
    """

    def __init__(self, c1: int, c2: int | None = None, kernels: Sequence[int] = (3, 5, 7), dilations: Sequence[int] = (1, 2, 3)):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if len(kernels) != 3 or len(dilations) != 3:
            raise ValueError("MLKCM requires exactly three kernel and dilation branches")
        self.in_proj = ConvBNAct(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.norm = PFOMLayerNorm2d(c2)
        self.pw_split = ConvBNAct(c2, c2, 1, 1)
        self.pw_gate = ConvBNAct(c2, c2, 1, 1)
        self.split_sizes = _balanced_split_sizes(c2, 3)
        self.dcp = nn.ModuleList(DCP(channels, int(kernel), int(dilation)) for channels, kernel, dilation in zip(self.split_sizes, kernels, dilations))
        self.pc = ConvBNAct(c2, c2, 1, 1)
        self.mab = MultiAxisBlock(c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        normalized = self.norm(x)
        parts = torch.split(self.pw_split(normalized), self.split_sizes, dim=1)
        multi_scale = torch.cat([module(part) for module, part in zip(self.dcp, parts)], 1)
        fused = self.pc(multi_scale * self.pw_gate(normalized))
        return self.mab(fused)


__all__ = ("PFOM", "MLKCM", "RepeatBlock", "GhostModule", "DCP", "MultiAxisBlock")
