# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Perspective-conditioned feature routing modules for PVR-Q."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = "LatentPerspectiveField", "PVRF"


class LatentPerspectiveField(nn.Module):
    """Predict a scene-conditioned scale field at a requested feature resolution."""

    coordinate_channels = 8  # sin/cos for x/y at two Fourier frequencies

    def __init__(self, c_p5: int, c_ctx: int, c_field: int = 16) -> None:
        """Initialize the latent perspective field.

        Args:
            c_p5: Number of channels in the P5 scene feature.
            c_ctx: Number of channels in the pooled scene context.
            c_field: Number of channels in the returned perspective feature field.
        """
        super().__init__()
        if min(c_p5, c_ctx, c_field) <= 0:
            raise ValueError("LatentPerspectiveField channel counts must be positive")

        self.c_field = int(c_field)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(c_p5, c_ctx),
            nn.SiLU(),
            nn.Linear(c_ctx, c_ctx),
            nn.SiLU(),
        )
        self.field_predictor = nn.Sequential(
            nn.Conv2d(c_ctx + self.coordinate_channels, c_field, 1, bias=False),
            nn.BatchNorm2d(c_field),
            nn.SiLU(),
            nn.Conv2d(c_field, c_field, 3, padding=1, groups=c_field, bias=False),
            nn.BatchNorm2d(c_field),
            nn.SiLU(),
        )
        self.log_scale = nn.Conv2d(c_field, 1, 1)

    @staticmethod
    def _fourier_coordinates(
        height: int, width: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return normalized x/y Fourier features with shape ``[1, 8, H, W]``."""
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        encoded = []
        for frequency in (1.0, 2.0):
            encoded.extend(
                (
                    torch.sin(math.pi * frequency * grid_x),
                    torch.cos(math.pi * frequency * grid_x),
                    torch.sin(math.pi * frequency * grid_y),
                    torch.cos(math.pi * frequency * grid_y),
                )
            )
        return torch.stack(encoded, dim=0).unsqueeze(0)

    def forward(self, p5: torch.Tensor, out_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the perspective feature field and log-scale map at ``out_hw``."""
        height, width = (int(out_hw[0]), int(out_hw[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"out_hw must be positive, received {out_hw}")

        context = self.context(p5)[:, :, None, None].expand(-1, -1, height, width)
        coordinates = self._fourier_coordinates(height, width, device=p5.device, dtype=p5.dtype)
        coordinates = coordinates.expand(p5.shape[0], -1, -1, -1)
        field = self.field_predictor(torch.cat((context, coordinates), dim=1))
        return field, self.log_scale(field)


class PVRF(nn.Module):
    """Refine the P3 feature by perspective-conditioned P2/P3/P4 routing."""

    def __init__(self, ch: tuple[int, int, int, int], cfg: dict) -> None:
        """Initialize PVRF from P2/P3/P4/P5 channel counts and its configuration."""
        super().__init__()
        if len(ch) != 4:
            raise ValueError(f"PVRF expects four input channel counts, received {len(ch)}")

        c_p2, c_p3, c_p4, c_p5 = (int(c) for c in ch)
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.field_channels = int(cfg.get("field_channels", 16))
        context_channels = int(cfg.get("context_channels", self.field_channels))
        gate_hidden_ratio = float(cfg.get("gate_hidden_ratio", 0.5))
        gate_channels = max(1, int(round(c_p3 * gate_hidden_ratio)))

        self.p2_projection = nn.Sequential(
            nn.Conv2d(c_p2, c_p2, 3, stride=2, padding=1, groups=c_p2, bias=False),
            nn.BatchNorm2d(c_p2),
            nn.SiLU(),
            nn.Conv2d(c_p2, c_p3, 1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(),
        )
        self.p3_projection = nn.Sequential(
            nn.Conv2d(c_p3, c_p3, 1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(),
        )
        self.p4_projection = nn.Sequential(
            nn.Conv2d(c_p4, c_p3, 1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(),
        )
        self.lpfield = LatentPerspectiveField(c_p5, context_channels, self.field_channels)
        self.gate = nn.Sequential(
            nn.Conv2d(c_p3 + self.field_channels, gate_channels, 1, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.SiLU(),
            nn.Conv2d(gate_channels, 3, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(c_p3, c_p3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(),
        )
        self.fusion_alpha = nn.Parameter(torch.tensor(float(cfg.get("fusion_alpha_init", 0.0))))
        self.last_aux: dict[str, torch.Tensor] | None = None

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Return a P3-shaped refined feature from ``[P2, P3, P4, P5]``."""
        if len(xs) != 4:
            raise ValueError(f"PVRF expects [P2, P3, P4, P5], received {len(xs)} tensors")
        p2, p3, p4, p5 = xs
        self.last_aux = None
        if not self.enabled:
            return p3

        target_hw = p3.shape[-2:]
        p2_route = self.p2_projection(p2)
        if p2_route.shape[-2:] != target_hw:
            p2_route = F.interpolate(p2_route, size=target_hw, mode="bilinear", align_corners=False)
        p3_route = self.p3_projection(p3)
        p4_route = F.interpolate(
            self.p4_projection(p4), size=target_hw, mode="bilinear", align_corners=False
        )
        field, log_scale = self.lpfield(p5, target_hw)

        gates = self.gate(torch.cat((p3, field), dim=1)).softmax(dim=1)
        routed = gates[:, 0:1] * p2_route + gates[:, 1:2] * p3_route + gates[:, 2:3] * p4_route
        output = p3 + self.fusion_alpha * self.fusion(routed)
        if self.training:
            self.last_aux = {"log_scale": log_scale}
        return output
