"""YOLOv8 composite modules for the three DSwin integration schemes."""
from __future__ import annotations

import torch
import torch.nn as nn

from .block import Bottleneck
from .conv import Conv
from .dswin import DSwinBlock


class SPPFDSwin(nn.Module):
    """Scheme A: an SPPF followed by one DSwin block."""

    def __init__(self, c1: int, c2: int, k: int = 5, num_heads: int = 8, attn_kernel: int = 5, offset_range_factor: float = 1.0):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1, act=False)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.attn = DSwinBlock(c2, num_heads=num_heads, kernel_size=attn_kernel, offset_range_factor=offset_range_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [self.cv1(x)]
        features.extend(self.m(features[-1]) for _ in range(3))
        return self.attn(self.cv2(torch.cat(features, 1)))


class C2fPostDSwin(nn.Module):
    """Scheme B: regular C2f computation followed by one DSwin block."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, num_heads: int = 8, attn_kernel: int = 5, offset_range_factor: float = 1.0):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = DSwinBlock(c2, num_heads=num_heads, kernel_size=attn_kernel, offset_range_factor=offset_range_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = list(self.cv1(x).chunk(2, 1))
        features.extend(module(features[-1]) for module in self.m)
        return self.attn(self.cv2(torch.cat(features, 1)))


class C2fDSwin(nn.Module):
    """Scheme C: inject one DSwin block after each iterative C2f bottleneck branch."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, num_heads: int = 8, attn_kernel: int = 5, offset_range_factor: float = 1.0):
        super().__init__()
        self.c = int(c2 * e)
        if self.c % num_heads:
            raise ValueError(f"C2fDSwin hidden channels={self.c} must be divisible by num_heads={num_heads}")
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.dswin = nn.ModuleList(DSwinBlock(self.c, num_heads=num_heads, kernel_size=attn_kernel, offset_range_factor=offset_range_factor) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = list(self.cv1(x).chunk(2, 1))
        for bottleneck, attention in zip(self.m, self.dswin):
            features.append(attention(bottleneck(features[-1])))
        return self.cv2(torch.cat(features, 1))
