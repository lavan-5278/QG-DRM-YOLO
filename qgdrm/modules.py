"""Paper-guided implementations of QG-DRM-YOLO (paper sections 3.1--3.4).

This file deliberately depends only on PyTorch: modules can be tested without Ultralytics.
The original YOLOv8 C2f instance is injected as the stable base of RGDRC2fV3.

The modules are an independent PyTorch implementation of the manuscript's
four proposed feature-processing components.
"""

from __future__ import annotations

from typing import Sequence
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _conv_bn_silu(c1: int, c2: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


class RelayBottleneck(nn.Module):
    """Shortcut-free, two-convolution bottleneck (YOLOv8 Bottleneck semantics)."""

    def __init__(self, channels: int):
        super().__init__()
        self.cv1 = _conv_bn_silu(channels, channels, 3)
        self.cv2 = _conv_bn_silu(channels, channels, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv2(self.cv1(x))


class RGDRC2fV3(nn.Module):
    """Native C2f + two-stage differential relay + reliability gating, eqs. (1)--(7).

    ``base`` must be the existing stride-16 native YOLOv8 C2f. Its input/output
    channels can be queried automatically from the original Ultralytics C2f.
    """

    def __init__(self, base: nn.Module, in_channels: int, out_channels: int, relay_ratio: float = 0.25):
        super().__init__()
        self.base = base
        hidden = round(out_channels * relay_ratio)
        if hidden <= 0:
            raise ValueError("relay_ratio produces zero channels")
        self.hidden = hidden
        self.entry = _conv_bn_silu(in_channels, 2 * hidden, 1)
        self.bottlenecks = nn.ModuleList([RelayBottleneck(hidden), RelayBottleneck(hidden)])
        self.alpha = nn.Parameter(torch.ones(2))
        self.beta = nn.Parameter(torch.zeros(2))
        self.restore = _conv_bn_silu(3 * hidden, out_channels, 1)
        self.reliability = nn.Conv2d(out_channels, 1, 1, bias=True)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        base = self.base(x)
        x_base, u0 = self.entry(x).chunk(2, dim=1)
        u = u0
        accumulated = torch.zeros_like(u)
        for i, bottleneck in enumerate(self.bottlenecks):
            delta = bottleneck(u) - u
            u = self.alpha[i] * delta + self.beta[i] * u0
            accumulated = accumulated + u
        relay = self.restore(torch.cat((x_base, accumulated, u), dim=1))
        gate = torch.sigmoid(self.reliability(base))
        return base + self.residual_scale * gate * (relay - base)


class QGInjection(nn.Module):
    """One P4 or P5 injection, eqs. (10)--(14)."""

    def __init__(self, channels: int):
        super().__init__()
        self.base_gate = nn.Conv2d(2 * channels, channels, 1)
        self.quality = nn.Conv2d(channels, 1, 1)
        self.raw_strength = nn.Parameter(torch.zeros(()))
        self.fuse = _conv_bn_silu(channels, channels, 3)

    @staticmethod
    def _non_affine_chw_layer_norm(x: Tensor) -> Tensor:
        # Eq. (12): each image independently normalized over C,H,W, NOT over only C.
        return F.layer_norm(x, x.shape[1:])

    def forward(self, target: Tensor, source: Tensor) -> Tensor:
        gate = torch.sigmoid(self.base_gate(torch.cat((target, source), dim=1)))
        quality = torch.sigmoid(self.quality(source))
        target_norm = self._non_affine_chw_layer_norm(target)
        source_norm = self._non_affine_chw_layer_norm(source)
        consistency = torch.exp(-(target_norm - source_norm).abs().mean(dim=1, keepdim=True))
        gamma = self.raw_strength.clamp(0.0, 1.0)
        modulation = 1.0 + gamma * (quality * consistency - 1.0)
        return self.fuse(target + gate * modulation * source)


class QGAFPN(nn.Module):
    """P3-initialized progressive P4-then-P5 fusion for each of P3/P4/P5.

    All three source projections and both source injection modules are independent
    for each target scale; 3 targets * 2 injections = 6 learnable scalar strengths.
    """

    def __init__(self, channels: Sequence[int] = (64, 128, 256)):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Expected P3, P4, P5 channel counts")
        self.channels = tuple(channels)
        self.projections = nn.ModuleList(
            [nn.ModuleList([_conv_bn_silu(c_src, c_target, 1) for c_src in channels])
             for c_target in channels]
        )
        self.injections = nn.ModuleList(
            [nn.ModuleList([QGInjection(c_target), QGInjection(c_target)])
             for c_target in channels]
        )

    @staticmethod
    def _align(x: Tensor, size: tuple[int, int]) -> Tensor:
        return x if x.shape[2:] == size else F.interpolate(x, size=size, mode="nearest")

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        if len(features) != 3:
            raise ValueError("QGAFPN requires P3,P4,P5 features")
        out = []
        for t, target in enumerate(features):
            size = target.shape[2:]
            z = self._align(self.projections[t][0](features[0]), size)
            for j, source_index in enumerate((1, 2)):
                src = self._align(self.projections[t][source_index](features[source_index]), size)
                z = self.injections[t][j](z, src)
            out.append(z)
        return out


class ChannelLayerNorm(nn.Module):
    """Non-affine? No: 'channel LayerNorm' is learnable LN over channel at each pixel."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class DepthwiseContextPath(nn.Module):
    """DWConv(k,d=1) -> DWConv(k,d) -> PWConv -> product, eq. (17)."""

    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.dw2 = nn.Conv2d(
            channels, channels, kernel, padding=dilation * (kernel // 2),
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.pointwise(self.dw2(self.dw1(x)))


class MultiAxisBlock(nn.Module):
    """Directional pooled max(H), mean(W), Conv/BN/Sigmoid, MAB residual."""

    def __init__(self, channels: int):
        super().__init__()
        self.pre = nn.Conv2d(channels, channels, 1)
        self.axis_conv = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.axis_bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        v = self.pre(x)
        h, w = v.shape[2:]
        along_h = v.amax(dim=2, keepdim=True).expand(-1, -1, h, -1)
        along_w = v.mean(dim=3, keepdim=True).expand(-1, -1, -1, w)
        axis_gate = torch.sigmoid(self.axis_bn(self.axis_conv(torch.cat((along_h, along_w), dim=1))))
        return x + axis_gate * x


class MLKCM(nn.Module):
    """Three spatial context paths and direction-aware reweighting, eqs. (15)--(20)."""

    def __init__(self, channels: int):
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)
        self.layer_norm = ChannelLayerNorm(channels)
        self.split_proj = nn.Conv2d(channels, channels, 1)
        self.gating_proj = nn.Conv2d(channels, channels, 1)
        # 64->22,21,21; 128->43,43,42; 256->86,85,85
        per = channels // 3
        rem = channels % 3
        self.group_channels = [per + (1 if i < rem else 0) for i in range(3)]
        self.paths = nn.ModuleList(
            [DepthwiseContextPath(n, k, d) for n, (k, d) in zip(
                self.group_channels, ((3, 1), (5, 2), (7, 3)))
            ]
        )
        self.fusion = nn.Conv2d(channels, channels, 1)
        self.multi_axis = MultiAxisBlock(channels)

    def forward(self, x: Tensor) -> Tensor:
        u = self.layer_norm(self.projection(x))
        chunks = torch.split(self.split_proj(u), self.group_channels, dim=1)
        context = torch.cat([path(chunk) for path, chunk in zip(self.paths, chunks)], dim=1)
        v = self.fusion(context * self.gating_proj(u))
        return self.multi_axis(v)


class P3DetailRestore(nn.Module):
    """Only P3: local detail with one-channel spatial gate and zero residual, eqs. (21)--(23)."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.reliability = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.reliability.weight)
        nn.init.zeros_(self.reliability.bias)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: Tensor) -> Tensor:
        local = self.pointwise(self.depthwise(x))
        gate = torch.sigmoid(self.reliability(x))
        return x + self.residual_scale * gate * local


class QGFeatureNeck(nn.Module):
    """After baseline PAN, before native YOLOv8 Detect. Supports progressive ablations."""

    def __init__(self, channels: Sequence[int], qg_afpn: bool, mlkcm: bool, p3dr: bool):
        super().__init__()
        if p3dr and not mlkcm:
            raise ValueError("Paper's progressive setting requires MLKCM before P3DR")
        self.qg_afpn = QGAFPN(channels) if qg_afpn else nn.Identity()
        self.mlkcm = nn.ModuleList([MLKCM(c) for c in channels]) if mlkcm else None
        self.p3_detail = P3DetailRestore(channels[0]) if p3dr else nn.Identity()

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        feats = list(self.qg_afpn(features)) if not isinstance(self.qg_afpn, nn.Identity) else list(features)
        if self.mlkcm is not None:
            feats = [module(x) for module, x in zip(self.mlkcm, feats)]
        feats[0] = self.p3_detail(feats[0])
        return feats
