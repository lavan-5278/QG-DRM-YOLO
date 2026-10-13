# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visible upper-body region alignment for classroom behavior detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = "VBR",


class VBR(nn.Module):
    """Refine P3 with deformable P2 samples placed on a visible upper-body template."""

    # P2-pixel offsets for left arm/shoulder, head or raised hand, right arm/shoulder,
    # torso/arm, and desk-interaction regions, in (x, y) order.
    template_offsets = ((-1.0, -1.5), (0.0, -2.0), (1.0, -1.5), (-0.8, 0.0), (0.0, 1.0))

    def __init__(self, ch: tuple[int, int], cfg: dict | None = None) -> None:
        """Initialize VBR from P2/P3 channels and its local-sampling configuration."""
        super().__init__()
        if len(ch) != 2:
            raise ValueError(f"VBR expects P2/P3 channel counts, received {len(ch)}")
        c_p2, c_p3 = (int(c) for c in ch)
        if min(c_p2, c_p3) <= 0:
            raise ValueError(f"VBR channel counts must be positive, received {ch}")

        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.samples = int(cfg.get("samples", len(self.template_offsets)))
        if self.samples != len(self.template_offsets):
            raise ValueError(f"VBR requires {len(self.template_offsets)} upper-body samples, received {self.samples}")
        self.offset_radius = float(cfg.get("offset_radius", 2.0))
        if self.offset_radius < 0:
            raise ValueError(f"VBR offset_radius must be non-negative, received {self.offset_radius}")

        self.p2_projection = Conv(c_p2, c_p3, k=1, s=1)
        self.offset = nn.Conv2d(c_p3, 2 * self.samples, 3, padding=1)
        self.weight = nn.Conv2d(c_p3, self.samples, 3, padding=1)
        self.fuse = Conv(2 * c_p3, c_p3, k=1, s=1)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.get("alpha_init", 0.0))))
        self.register_buffer("template", torch.tensor(self.template_offsets, dtype=torch.float32), persistent=False)

        # Start from the declared template and an equal sample mixture. The residual gate,
        # rather than these predictors, guarantees exact P3 identity at alpha=0.
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

    @staticmethod
    def _p3_centers_in_p2(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return P3-cell centers expressed in P2 pixel-index coordinates, shape ``[H, W, 2]``."""
        y = torch.arange(height, device=device, dtype=dtype) * 2.0 + 0.5
        x = torch.arange(width, device=device, dtype=dtype) * 2.0 + 0.5
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)

    def _sample_upper_body_template(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """Sample projected P2 at learned, bounded upper-body template offsets for every P3 cell."""
        batch_size, channels, p2_height, p2_width = p2.shape
        _, _, p3_height, p3_width = p3.shape
        if (p2_height, p2_width) != (2 * p3_height, 2 * p3_width):
            raise ValueError(
                "VBR requires P2 spatial dimensions exactly twice P3, received "
                f"P2={(p2_height, p2_width)}, P3={(p3_height, p3_width)}"
            )

        offsets = self.offset(p3).view(batch_size, self.samples, 2, p3_height, p3_width)
        offsets = offsets.tanh().permute(0, 1, 3, 4, 2) * self.offset_radius
        centers = self._p3_centers_in_p2(p3_height, p3_width, device=p2.device, dtype=p2.dtype)
        template = self.template.to(device=p2.device, dtype=p2.dtype).view(1, self.samples, 1, 1, 2)
        points = centers.view(1, 1, p3_height, p3_width, 2) + template + offsets

        # ``align_corners=False`` maps a pixel-index coordinate x to (2*x+1)/W-1.
        points[..., 0] = (2.0 * points[..., 0] + 1.0) / p2_width - 1.0
        points[..., 1] = (2.0 * points[..., 1] + 1.0) / p2_height - 1.0
        sampling_grid = points.reshape(batch_size * self.samples, p3_height, p3_width, 2)
        repeated_p2 = p2[:, None].expand(-1, self.samples, -1, -1, -1).reshape(
            batch_size * self.samples, channels, p2_height, p2_width
        )
        sampled = F.grid_sample(repeated_p2, sampling_grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(batch_size, self.samples, channels, p3_height, p3_width)
        weights = self.weight(p3).softmax(dim=1).unsqueeze(2)
        return (weights * sampled).sum(dim=1)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Return a P3-shaped refinement from ``[P2, P3]`` without changing external Detect contracts."""
        if len(xs) != 2:
            raise ValueError(f"VBR expects [P2, P3], received {len(xs)} tensors")
        p2, p3 = xs
        if not self.enabled:
            return p3

        detail = self._sample_upper_body_template(self.p2_projection(p2), p3)
        residual = self.fuse(torch.cat((p3, detail), dim=1))
        return p3 + self.alpha.tanh() * residual
