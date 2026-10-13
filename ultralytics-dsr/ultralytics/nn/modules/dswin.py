"""YOLO-oriented Deformable Sliding Window (DSwin) attention for BCHW tensors."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSwinLayerNorm2d(nn.Module):
    """Channel-wise LayerNorm that preserves a BCHW feature layout."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class DeformableSlidingWindowAttention(nn.Module):
    """Local sliding-window attention with per-head content-conditioned sampling."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kernel_size: int = 5,
        dilation: int = 1,
        offset_range_factor: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_lepe: bool = True,
        rel_pos_bias: bool = True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.offset_range_factor = float(offset_range_factor)
        self.attn_drop = float(attn_drop)
        self.proj_drop = float(proj_drop)

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=True)
        pad = kernel_size // 2
        self.offset_net = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size, 1, pad, groups=self.head_dim, bias=False),
            DSwinLayerNorm2d(self.head_dim),
            nn.GELU(),
            nn.Conv2d(self.head_dim, 2, 1, bias=False),
        )
        k2 = kernel_size * kernel_size
        self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, k2)) if rel_pos_bias else None
        if self.relative_position_bias is not None:
            nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim) if use_lepe else None

    @staticmethod
    def _base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy = torch.zeros(1, device=device, dtype=dtype) if h == 1 else torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xx = torch.zeros(1, device=device, dtype=dtype) if w == 1 else torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((gx, gy), dim=-1).unsqueeze(0)

    def _deform_sample(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q_grouped = q.reshape(batch * self.num_heads, self.head_dim, height, width)
        offset = torch.tanh(self.offset_net(q_grouped))
        step_y = 0.0 if height <= 1 else 2.0 / (height - 1)
        step_x = 0.0 if width <= 1 else 2.0 / (width - 1)
        dy = offset[:, 0] * step_y * self.offset_range_factor
        dx = offset[:, 1] * step_x * self.offset_range_factor
        grid = self._base_grid(height, width, x.device, x.dtype) + torch.stack((dx, dy), dim=-1)
        sampled = F.grid_sample(
            x.reshape(batch * self.num_heads, self.head_dim, height, width),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled.reshape(batch, channels, height, width)

    @staticmethod
    def _shift_from_padded(x_pad: torch.Tensor, pad: int, dy: int, dx: int, height: int, width: int) -> torch.Tensor:
        return x_pad[..., pad + dy : pad + dy + height, pad + dx : pad + dx + width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        q_full = self.q_proj(x)
        sampled = self._deform_sample(x, q_full)
        k_full, v_full = self.k_proj(sampled), self.v_proj(sampled)
        q = q_full.reshape(batch, self.num_heads, self.head_dim, height, width)
        k = k_full.reshape(batch, self.num_heads, self.head_dim, height, width)
        v = v_full.reshape(batch, self.num_heads, self.head_dim, height, width)
        radius, pad = self.kernel_size // 2, (self.kernel_size // 2) * self.dilation
        k_pad, v_pad = F.pad(k, (pad, pad, pad, pad)), F.pad(v, (pad, pad, pad, pad))
        valid_pad = F.pad(torch.ones((1, 1, 1, height, width), device=x.device, dtype=x.dtype), (pad, pad, pad, pad))
        scores, offsets = [], []
        for idx, (iy, ix) in enumerate((iy, ix) for iy in range(-radius, radius + 1) for ix in range(-radius, radius + 1)):
            dy, dx = iy * self.dilation, ix * self.dilation
            offsets.append((dy, dx))
            shifted_k = self._shift_from_padded(k_pad, pad, dy, dx, height, width)
            score = (q * shifted_k).sum(dim=2) * self.scale
            if self.relative_position_bias is not None:
                score = score + self.relative_position_bias[:, idx][None, :, None, None]
            valid = self._shift_from_padded(valid_pad, pad, dy, dx, height, width).squeeze(2)
            scores.append(score.masked_fill(valid == 0, torch.finfo(score.dtype).min))
        attention = F.dropout(torch.stack(scores, dim=-1).softmax(dim=-1), p=self.attn_drop, training=self.training)
        out = torch.zeros_like(q)
        for idx, (dy, dx) in enumerate(offsets):
            shifted_v = self._shift_from_padded(v_pad, pad, dy, dx, height, width)
            out = out + shifted_v * attention[..., idx].unsqueeze(2)
        out = out.reshape(batch, channels, height, width)
        if self.lepe is not None:
            out = out + self.lepe(q_full)
        return F.dropout(self.out_proj(out), p=self.proj_drop, training=self.training)


class DSwinConvFFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.0, drop: float = 0.0):
        super().__init__()
        hidden = max(dim, int(dim * expansion))
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = float(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(self.dw(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        return F.dropout(self.fc2(x), p=self.drop, training=self.training)


class DSwinBlock(nn.Module):
    """Residual pre-norm DSwin block for detection features."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kernel_size: int = 5,
        dilation: int = 1,
        offset_range_factor: float = 1.0,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = DSwinLayerNorm2d(dim)
        self.attn = DeformableSlidingWindowAttention(dim, num_heads, kernel_size, dilation, offset_range_factor, attn_drop, drop)
        self.norm2 = DSwinLayerNorm2d(dim)
        self.ffn = DSwinConvFFN(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class DSwin(nn.Module):
    """Ultralytics parser-friendly DSwin module with a ``(c1, c2, ...)`` signature."""

    def __init__(self, c1: int, c2: int, num_heads: int = 8, kernel_size: int = 5, offset_range_factor: float = 1.0, mlp_ratio: float = 2.0):
        super().__init__()
        self.proj = nn.Identity() if c1 == c2 else nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))
        self.block = DSwinBlock(c2, num_heads, kernel_size, offset_range_factor=offset_range_factor, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))
