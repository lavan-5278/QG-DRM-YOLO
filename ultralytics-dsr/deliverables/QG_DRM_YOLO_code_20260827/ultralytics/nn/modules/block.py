# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "AFPN",
    "QGAFPN",
    "RCAQGAFPN",
    "CQGAFPN",
    "RCQGAFPN",
    "DyHead",
    "FADWC",
    "SATA",
    "RGSATA",
    "DRSATA",
    "LGC",
    "LGRB",
    "LHGB",
    "LGCBottleneck",
    "C2fLGC",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
)


class AFPN(nn.Module):
    """Three-scale asymptotic feature pyramid refinement for P3/P4/P5 detector features.

    Each output scale starts from its projected P3 feature and progressively integrates P4 then P5.  The
    projections preserve the original output-channel contracts, while per-source spatial gates let the module
    suppress mismatched context before residual fusion.  Consequently an ordinary Ultralytics ``Detect`` head can
    consume the three returned tensors unchanged.
    """

    def __init__(self, ch: tuple[int, int, int] | list[int]):
        """Initialize a P3/P4/P5 AFPN with output channels matching the input pyramid."""
        super().__init__()
        if len(ch) != 3:
            raise ValueError(f"AFPN expects exactly P3/P4/P5 channels, got {tuple(ch)}")
        self.ch = tuple(int(channel) for channel in ch)
        self.projections = nn.ModuleList(
            nn.ModuleList(Conv(source_channels, target_channels, 1) for source_channels in self.ch)
            for target_channels in self.ch
        )
        self.gates = nn.ModuleList(
            nn.ModuleList(nn.Sequential(nn.Conv2d(target_channels * 2, target_channels, 1), nn.Sigmoid()) for _ in range(2))
            for target_channels in self.ch
        )
        self.fusions = nn.ModuleList(
            nn.ModuleList(Conv(target_channels, target_channels, 3) for _ in range(2)) for target_channels in self.ch
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Progressively fuse aligned P3, P4 and P5 features while retaining their original scale contracts."""
        if len(features) != 3:
            raise ValueError(f"AFPN expects exactly three feature tensors, got {len(features)}")

        outputs = []
        for target_index, target_channels in enumerate(self.ch):
            target_size = features[target_index].shape[-2:]
            fused = self.projections[target_index][0](features[0])
            if fused.shape[-2:] != target_size:
                fused = F.interpolate(fused, size=target_size, mode="nearest")
            for source_index in range(1, 3):
                source = self.projections[target_index][source_index](features[source_index])
                if source.shape[-2:] != target_size:
                    source = F.interpolate(source, size=target_size, mode="nearest")
                gate = self.gates[target_index][source_index - 1](torch.cat((fused, source), dim=1))
                fused = self.fusions[target_index][source_index - 1](fused + gate * source)
            outputs.append(fused)
        return outputs


class QGAFPN(AFPN):
    """Quality-gated AFPN with neutral, learnable source-injection modulation.

    QG-AFPN retains AFPN's progressive P3-to-P5 fusion.  Before each P4/P5 source
    injection, it estimates a one-channel source-quality map and a local agreement
    map between the aligned source and the current fused feature.  A zero-initialized
    scalar keeps this modulation exactly neutral at initialization, so the module
    initially behaves as an ordinary AFPN with the same parameters.
    """

    def __init__(self, ch: tuple[int, int, int] | list[int]):
        """Initialize AFPN-compatible quality projections and neutral modulation scales."""
        super().__init__(ch)
        self.quality = nn.ModuleList(
            nn.ModuleList(nn.Conv2d(target_channels, 1, 1) for _ in range(2)) for target_channels in self.ch
        )
        self.modulation_scales = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in range(6))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse P3/P4/P5 features with learnable, scale-agreement quality modulation."""
        if len(features) != 3:
            raise ValueError(f"QGAFPN expects exactly three feature tensors, got {len(features)}")

        outputs, scale_index = [], 0
        for target_index, target_channels in enumerate(self.ch):
            target_size = features[target_index].shape[-2:]
            fused = self.projections[target_index][0](features[0])
            if fused.shape[-2:] != target_size:
                fused = F.interpolate(fused, size=target_size, mode="nearest")
            for source_index in range(1, 3):
                source = self.projections[target_index][source_index](features[source_index])
                if source.shape[-2:] != target_size:
                    source = F.interpolate(source, size=target_size, mode="nearest")
                base_gate = self.gates[target_index][source_index - 1](torch.cat((fused, source), dim=1))
                fused_norm = F.layer_norm(fused, fused.shape[1:])
                source_norm = F.layer_norm(source, source.shape[1:])
                agreement = torch.exp(-(fused_norm - source_norm).abs().mean(dim=1, keepdim=True))
                quality = torch.sigmoid(self.quality[target_index][source_index - 1](source))
                modulation = 1.0 + torch.tanh(self.modulation_scales[scale_index]) * (quality * agreement - 1.0)
                fused = self.fusions[target_index][source_index - 1](fused + base_gate * modulation * source)
                scale_index += 1
            outputs.append(fused)
        return outputs


class _QGAFPNVariant(QGAFPN):
    """Shared implementation for relay-calibrated and competitive QG-AFPN ablations."""

    def __init__(self, ch: tuple[int, int, int] | list[int], calibrate: bool, competitive: bool, local_agreement: bool):
        super().__init__(ch)
        self.use_calibration = calibrate
        self.use_competition = competitive
        self.use_local_agreement = local_agreement
        if calibrate:
            self.relay_calibrators = nn.ModuleList(
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1, groups=channels))
                for channels in self.ch
            )
            # Zero initialization makes the input contract exactly identical to QG-AFPN at epoch 0.
            self.relay_scales = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in self.ch)
        if competitive:
            self.competition_logits = nn.ModuleList(
                nn.ModuleList(nn.Conv2d(2 * channels, 1, 1) for _ in range(2)) for channels in self.ch
            )
            for logits in self.competition_logits:
                for layer in logits:
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _calibrate_features(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if not self.use_calibration:
            return features
        calibrated = []
        for index, feature in enumerate(features):
            # The bounded channel factor starts at one and can only reweight, never amplify without limit.
            channel_factor = 2.0 * torch.sigmoid(self.relay_calibrators[index](feature))
            scale = torch.tanh(self.relay_scales[index]).to(dtype=feature.dtype)
            calibrated.append(feature * (1.0 + scale * (channel_factor - 1.0)))
        return calibrated

    def _agreement(self, fused: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if not self.use_local_agreement:
            fused_norm = F.layer_norm(fused, fused.shape[1:])
            source_norm = F.layer_norm(source, source.shape[1:])
            return torch.exp(-(fused_norm - source_norm).abs().mean(dim=1, keepdim=True))
        # Per-location channel cosine agreement avoids coupling the quality decision across the entire feature map.
        fused_norm = F.normalize(fused, p=2.0, dim=1, eps=1e-6)
        source_norm = F.normalize(source, p=2.0, dim=1, eps=1e-6)
        return ((fused_norm * source_norm).sum(dim=1, keepdim=True) + 1.0) * 0.5

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != 3:
            raise ValueError(f"{self.__class__.__name__} expects exactly three feature tensors, got {len(features)}")
        features = self._calibrate_features(features)
        outputs, scale_index = [], 0
        for target_index, target_channels in enumerate(self.ch):
            target_size = features[target_index].shape[-2:]
            fused = self.projections[target_index][0](features[0])
            if fused.shape[-2:] != target_size:
                fused = F.interpolate(fused, size=target_size, mode="nearest")
            sources = []
            for source_index in range(1, 3):
                source = self.projections[target_index][source_index](features[source_index])
                if source.shape[-2:] != target_size:
                    source = F.interpolate(source, size=target_size, mode="nearest")
                sources.append(source)
            if self.use_competition:
                logits = torch.cat(
                    [self.competition_logits[target_index][index](torch.cat((fused, source), dim=1)) for index, source in enumerate(sources)],
                    dim=1,
                )
                # 2*softmax begins at one for each source, preserving the existing QG-AFPN initialization.
                source_weights = 2.0 * logits.softmax(dim=1)
            else:
                source_weights = None
            for source_index, source in enumerate(sources, start=1):
                base_gate = self.gates[target_index][source_index - 1](torch.cat((fused, source), dim=1))
                agreement = self._agreement(fused, source)
                quality = torch.sigmoid(self.quality[target_index][source_index - 1](source))
                modulation = 1.0 + torch.tanh(self.modulation_scales[scale_index]) * (quality * agreement - 1.0)
                weight = source_weights[:, source_index - 1 : source_index] if source_weights is not None else 1.0
                fused = self.fusions[target_index][source_index - 1](fused + base_gate * modulation * weight * source)
                scale_index += 1
            outputs.append(fused)
        return outputs


class RCAQGAFPN(_QGAFPNVariant):
    """QG-AFPN with neutral relay-statistic calibration for DR-C2f v3 features."""

    def __init__(self, ch: tuple[int, int, int] | list[int]):
        super().__init__(ch, calibrate=True, competitive=False, local_agreement=False)


class CQGAFPN(_QGAFPNVariant):
    """QG-AFPN with neutral two-source competition at every target scale."""

    def __init__(self, ch: tuple[int, int, int] | list[int]):
        super().__init__(ch, calibrate=False, competitive=True, local_agreement=False)


class RCQGAFPN(_QGAFPNVariant):
    """Relay-calibrated competitive QG-AFPN with local channel agreement."""

    def __init__(self, ch: tuple[int, int, int] | list[int]):
        super().__init__(ch, calibrate=True, competitive=True, local_agreement=True)


class FADWC(nn.Module):
    """Frequency-adaptive dynamic wavelet convolution for a single detector scale.

    A fixed Haar analysis separates low-frequency body context from horizontal, vertical and diagonal detail.
    Per-pixel gates choose between depth-wise high-frequency detail and dilated low-frequency context, while a
    zero-initialized residual scale preserves the original feature contract at initialization.
    """

    def __init__(self, c1: int):
        """Initialize a shape-preserving FADWC layer for ``c1`` input/output channels."""
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"FADWC requires a positive channel count, got {c1}")
        self.c1 = int(c1)
        haar = torch.tensor(
            [[[[1.0, 1.0], [1.0, 1.0]]], [[[1.0, -1.0], [1.0, -1.0]]],
             [[[1.0, 1.0], [-1.0, -1.0]]], [[[1.0, -1.0], [-1.0, 1.0]]]],
            dtype=torch.float32,
        ) / 2.0
        self.register_buffer("haar", haar, persistent=False)
        self.detail = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c1, 1))
        self.context = nn.Sequential(DWConv(c1, c1, 3, d=2), Conv(c1, c1, 1))
        self.gate = nn.Sequential(nn.Conv2d(c1, c1, 1), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return frequency-routed residual features with the same ``B,C,H,W`` shape as ``x``."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"FADWC({self.c1}) expects B,{self.c1},H,W input, got {tuple(x.shape)}")
        # Constant padding keeps the CUDA backward path compatible with the strict deterministic training protocol.
        padded = F.pad(x, (0, 1, 0, 1))
        filters = self.haar.repeat(self.c1, 1, 1, 1)
        wavelets = F.conv2d(padded, filters, groups=self.c1).view(x.shape[0], self.c1, 4, *x.shape[-2:])
        low = wavelets[:, :, 0]
        energy = wavelets[:, :, 1:].abs().mean(dim=2)
        gate = self.gate(energy)
        routed = gate * self.detail(energy) + (1.0 - gate) * self.context(low)
        return x + self.gamma * routed


class SATA(nn.Module):
    """Sparse social-anchor token adapter for high-level backbone feature maps.

    A learned score map selects a small group of salient spatial anchors.  Each
    anchor exchanges messages only with its nearest anchors in normalized image
    coordinates, then the context update is scattered back to the feature map.
    A zero-initialized residual scale makes the adapter an exact identity at
    initialization, preserving the scratch-training feature distribution.
    """

    def __init__(self, c1: int, relation_dim: int = 64, max_anchors: int = 32, neighbor_k: int = 6):
        """Initialize a shape-preserving sparse relation adapter for ``c1`` channels."""
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"SATA requires a positive channel count, got {c1}")
        if relation_dim <= 0 or max_anchors <= 0 or neighbor_k <= 0:
            raise ValueError(
                f"SATA requires positive relation_dim, max_anchors and neighbor_k, got "
                f"{relation_dim}, {max_anchors}, {neighbor_k}"
            )
        self.c1 = int(c1)
        self.max_anchors = int(max_anchors)
        self.neighbor_k = int(neighbor_k)
        self.anchor_score = nn.Conv2d(c1, 1, 1)
        self.query = nn.Linear(c1, relation_dim, bias=False)
        self.key = nn.Linear(c1, relation_dim, bias=False)
        self.value = nn.Linear(c1, relation_dim, bias=False)
        self.coordinate_bias = nn.Sequential(nn.Linear(2, relation_dim), nn.SiLU(), nn.Linear(relation_dim, 1))
        self.output = nn.Linear(relation_dim, c1, bias=False)
        self.diffuse = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject local sparse-anchor context while keeping the ``B,C,H,W`` feature contract."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"SATA({self.c1}) expects B,{self.c1},H,W input, got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        positions = height * width
        anchor_count = min(self.max_anchors, positions)
        neighbor_count = min(self.neighbor_k, anchor_count)

        score = self.anchor_score(x).flatten(1)
        anchor_indices = score.topk(anchor_count, dim=1).indices
        flattened = x.flatten(2).transpose(1, 2)
        gather_index = anchor_indices.unsqueeze(-1).expand(-1, -1, channels)
        anchors = flattened.gather(1, gather_index)

        y = torch.arange(height, device=x.device, dtype=x.dtype)
        x_coordinate = torch.arange(width, device=x.device, dtype=x.dtype)
        yy, xx = torch.meshgrid(y, x_coordinate, indexing="ij")
        coordinate_grid = torch.stack((yy / max(height - 1, 1), xx / max(width - 1, 1)), dim=-1).reshape(positions, 2)
        coordinates = coordinate_grid.unsqueeze(0).expand(batch, -1, -1).gather(
            1, anchor_indices.unsqueeze(-1).expand(-1, -1, 2)
        )

        relative_coordinate = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        squared_distance = relative_coordinate.square().sum(dim=-1)
        neighbor_indices = squared_distance.topk(neighbor_count, dim=-1, largest=False).indices
        neighbor_mask = torch.zeros_like(squared_distance, dtype=torch.bool).scatter_(2, neighbor_indices, True)

        query, key, value = self.query(anchors), self.key(anchors), self.value(anchors)
        logits = query @ key.transpose(-2, -1) / (query.shape[-1] ** 0.5)
        logits = logits + self.coordinate_bias(relative_coordinate).squeeze(-1)
        logits = logits.masked_fill(~neighbor_mask, torch.finfo(logits.dtype).min)
        attention = logits.softmax(dim=-1)
        messages = self.output(attention @ value) * score.gather(1, anchor_indices).sigmoid().unsqueeze(-1)

        scattered = flattened.new_zeros(batch, positions, channels)
        scattered.scatter_add_(1, gather_index, messages)
        spatial_update = scattered.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.gamma * self.diffuse(spatial_update)


class RGSATA(nn.Module):
    """Reliability-gated SATA with semantic-geometric sparse neighbor selection.

    Saliency alone can select isolated background responses.  RG-SATA combines
    it with local feature reliability, then gates both candidate attention and
    source-message injection by that reliability.
    """

    def __init__(self, c1: int, relation_dim: int = 64, max_anchors: int = 32, neighbor_k: int = 6):
        """Initialize a reliability-gated shape-preserving relation adapter."""
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"RGSATA requires a positive channel count, got {c1}")
        if relation_dim <= 0 or max_anchors <= 0 or neighbor_k <= 0:
            raise ValueError(
                f"RGSATA requires positive relation_dim, max_anchors and neighbor_k, got "
                f"{relation_dim}, {max_anchors}, {neighbor_k}"
            )
        self.c1 = int(c1)
        self.max_anchors = int(max_anchors)
        self.neighbor_k = int(neighbor_k)
        self.saliency = nn.Conv2d(c1, 1, 1)
        self.reliability = nn.Sequential(DWConv(c1, c1, 3), nn.Conv2d(c1, 1, 1))
        self.query = nn.Linear(c1, relation_dim, bias=False)
        self.key = nn.Linear(c1, relation_dim, bias=False)
        self.value = nn.Linear(c1, relation_dim, bias=False)
        self.coordinate_bias = nn.Sequential(nn.Linear(2, relation_dim), nn.SiLU(), nn.Linear(relation_dim, 1))
        self.output = nn.Linear(relation_dim, c1, bias=False)
        self.distance_weight = nn.Parameter(torch.zeros(()))
        self.diffuse = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject only locally supported semantic relations with the original feature shape."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"RGSATA({self.c1}) expects B,{self.c1},H,W input, got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        positions = height * width
        anchor_count = min(self.max_anchors, positions)
        neighbor_count = min(self.neighbor_k, anchor_count)

        saliency = self.saliency(x).flatten(1)
        reliability = self.reliability(x).flatten(1).sigmoid()
        selection = saliency.sigmoid() * reliability
        anchor_indices = selection.topk(anchor_count, dim=1).indices
        flattened = x.flatten(2).transpose(1, 2)
        gather_index = anchor_indices.unsqueeze(-1).expand(-1, -1, channels)
        anchors = flattened.gather(1, gather_index)
        anchor_reliability = reliability.gather(1, anchor_indices)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=x.dtype),
            torch.arange(width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinate_grid = torch.stack((yy / max(height - 1, 1), xx / max(width - 1, 1)), dim=-1).reshape(positions, 2)
        coordinates = coordinate_grid.unsqueeze(0).expand(batch, -1, -1).gather(
            1, anchor_indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        relative_coordinate = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        squared_distance = relative_coordinate.square().sum(dim=-1)

        query, key, value = self.query(anchors), self.key(anchors), self.value(anchors)
        semantic_affinity = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        neighbor_affinity = semantic_affinity - F.softplus(self.distance_weight) * squared_distance
        neighbor_indices = neighbor_affinity.topk(neighbor_count, dim=-1).indices
        neighbor_mask = torch.zeros_like(neighbor_affinity, dtype=torch.bool).scatter_(2, neighbor_indices, True)

        logits = query @ key.transpose(-2, -1) / (query.shape[-1] ** 0.5)
        logits = logits + self.coordinate_bias(relative_coordinate).squeeze(-1)
        logits = logits + anchor_reliability.clamp_min(torch.finfo(x.dtype).eps).log().unsqueeze(1)
        logits = logits.masked_fill(~neighbor_mask, torch.finfo(logits.dtype).min)
        attention = logits.softmax(dim=-1)
        messages = self.output(attention @ value) * anchor_reliability.unsqueeze(-1)

        scattered = flattened.new_zeros(batch, positions, channels)
        scattered.scatter_add_(1, gather_index, messages)
        spatial_update = scattered.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.gamma * self.diffuse(spatial_update)


class DRSATA(nn.Module):
    """Dual-role SATA with directed context-to-subject sparse messages."""

    def __init__(
        self, c1: int, relation_dim: int = 64, subject_anchors: int = 16, context_anchors: int = 16, neighbor_k: int = 6
    ):
        """Initialize separate subject and context anchor roles for a ``c1`` feature map."""
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"DRSATA requires a positive channel count, got {c1}")
        if relation_dim <= 0 or subject_anchors <= 0 or context_anchors <= 0 or neighbor_k <= 0:
            raise ValueError(
                f"DRSATA requires positive relation_dim, subject_anchors, context_anchors and neighbor_k, got "
                f"{relation_dim}, {subject_anchors}, {context_anchors}, {neighbor_k}"
            )
        self.c1 = int(c1)
        self.subject_anchors = int(subject_anchors)
        self.context_anchors = int(context_anchors)
        self.neighbor_k = int(neighbor_k)
        self.subject_score = nn.Conv2d(c1, 1, 1)
        self.context_score = nn.Conv2d(c1, 1, 1)
        self.query = nn.Linear(c1, relation_dim, bias=False)
        self.key = nn.Linear(c1, relation_dim, bias=False)
        self.value = nn.Linear(c1, relation_dim, bias=False)
        self.coordinate_bias = nn.Sequential(nn.Linear(2, relation_dim), nn.SiLU(), nn.Linear(relation_dim, 1))
        self.output = nn.Linear(relation_dim, c1, bias=False)
        self.distance_weight = nn.Parameter(torch.zeros(()))
        self.diffuse = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write sparse context-to-subject messages back at subject anchor locations only."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"DRSATA({self.c1}) expects B,{self.c1},H,W input, got {tuple(x.shape)}")
        batch, channels, height, width = x.shape
        positions = height * width
        subject_count = min(self.subject_anchors, positions)
        context_count = min(self.context_anchors, positions)
        neighbor_count = min(self.neighbor_k, context_count)

        subject_scores = self.subject_score(x).flatten(1)
        context_scores = self.context_score(x).flatten(1)
        subject_indices = subject_scores.topk(subject_count, dim=1).indices
        context_indices = context_scores.topk(context_count, dim=1).indices
        flattened = x.flatten(2).transpose(1, 2)
        subject_gather = subject_indices.unsqueeze(-1).expand(-1, -1, channels)
        context_gather = context_indices.unsqueeze(-1).expand(-1, -1, channels)
        subjects = flattened.gather(1, subject_gather)
        contexts = flattened.gather(1, context_gather)
        subject_reliability = subject_scores.gather(1, subject_indices).sigmoid()
        context_reliability = context_scores.gather(1, context_indices).sigmoid()

        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=x.dtype),
            torch.arange(width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinate_grid = torch.stack((yy / max(height - 1, 1), xx / max(width - 1, 1)), dim=-1).reshape(positions, 2)
        coordinate_grid = coordinate_grid.unsqueeze(0).expand(batch, -1, -1)
        subject_coordinates = coordinate_grid.gather(1, subject_indices.unsqueeze(-1).expand(-1, -1, 2))
        context_coordinates = coordinate_grid.gather(1, context_indices.unsqueeze(-1).expand(-1, -1, 2))
        relative_coordinate = subject_coordinates.unsqueeze(2) - context_coordinates.unsqueeze(1)
        squared_distance = relative_coordinate.square().sum(dim=-1)

        query, key, value = self.query(subjects), self.key(contexts), self.value(contexts)
        semantic_affinity = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        neighbor_affinity = semantic_affinity - F.softplus(self.distance_weight) * squared_distance
        neighbor_indices = neighbor_affinity.topk(neighbor_count, dim=-1).indices
        neighbor_mask = torch.zeros_like(neighbor_affinity, dtype=torch.bool).scatter_(2, neighbor_indices, True)

        logits = query @ key.transpose(-2, -1) / (query.shape[-1] ** 0.5)
        logits = logits + self.coordinate_bias(relative_coordinate).squeeze(-1)
        logits = logits + context_reliability.clamp_min(torch.finfo(x.dtype).eps).log().unsqueeze(1)
        logits = logits.masked_fill(~neighbor_mask, torch.finfo(logits.dtype).min)
        attention = logits.softmax(dim=-1)
        messages = self.output(attention @ value) * subject_reliability.unsqueeze(-1)
        # AMP can promote the sparse relation path while the flattened feature map
        # retains its original dtype. scatter_add_ requires source and destination
        # dtypes to match, and this cast remains differentiable for the message path.
        messages = messages.to(dtype=flattened.dtype)

        scattered = flattened.new_zeros(batch, positions, channels)
        scattered.scatter_add_(1, subject_gather, messages)
        spatial_update = scattered.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.gamma * self.diffuse(spatial_update)


class LiquidGatedState(nn.Module):
    """Content-gated short-convolution state used by LFM-inspired visual backbone blocks."""

    def __init__(self, c1: int):
        super().__init__()
        self.c1 = int(c1)
        self.short3 = DWConv(c1, c1, 3)
        self.short5 = DWConv(c1, c1, 5)
        self.gate = nn.Conv2d(c1 * 2, c1, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Return the local liquid state, optionally conditioned by a broadcast stage state."""
        state3, state5 = self.short3(x), self.short5(x)
        logits = self.gate(torch.cat((state3, state5), dim=1))
        if condition is not None:
            if condition.shape != (x.shape[0], self.c1, 1, 1):
                raise ValueError(
                    f"LiquidGatedState({self.c1}) condition expects {(x.shape[0], self.c1, 1, 1)}, "
                    f"got {tuple(condition.shape)}"
                )
            logits = logits + condition
        gate = logits.sigmoid()
        return gate * state3 + (1.0 - gate) * state5


class LGC(nn.Module):
    """Liquid gated convolution: local short-convolution state with a neutral residual start."""

    def __init__(self, c1: int):
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"LGC requires a positive channel count, got {c1}")
        self.c1 = int(c1)
        self.state = LiquidGatedState(c1)
        self.output = Conv(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject gated 3x3/5x5 local state without changing the ``B,C,H,W`` contract."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"LGC({self.c1}) expects B,{self.c1},H,W input, got {tuple(x.shape)}")
        return x + self.gamma * self.output(self.state(x))


class LGRB(nn.Module):
    """Liquid gated relation block with bounded semantic-geometric sparse attention."""

    def __init__(self, c1: int, relation_dim: int = 64, max_anchors: int = 32, neighbor_k: int = 6):
        super().__init__()
        if c1 <= 0:
            raise ValueError(f"LGRB requires a positive channel count, got {c1}")
        if relation_dim <= 0 or max_anchors <= 0 or neighbor_k <= 0:
            raise ValueError(
                f"LGRB requires positive relation_dim, max_anchors and neighbor_k, got "
                f"{relation_dim}, {max_anchors}, {neighbor_k}"
            )
        self.c1 = int(c1)
        self.max_anchors = int(max_anchors)
        self.neighbor_k = int(neighbor_k)
        self.state = LiquidGatedState(c1)
        self.saliency = nn.Conv2d(c1, 1, 1)
        self.reliability = nn.Sequential(DWConv(c1, c1, 3), nn.Conv2d(c1, 1, 1))
        self.condition_reliability = nn.Conv2d(c1, 1, 1, bias=False)
        self.query = nn.Linear(c1, relation_dim, bias=False)
        self.key = nn.Linear(c1, relation_dim, bias=False)
        self.value = nn.Linear(c1, relation_dim, bias=False)
        self.coordinate_bias = nn.Sequential(nn.Linear(2, relation_dim), nn.SiLU(), nn.Linear(relation_dim, 1))
        self.output = nn.Linear(relation_dim, c1, bias=False)
        self.distance_weight = nn.Parameter(torch.zeros(()))
        self.local_output = Conv(c1, c1, 1)
        self.diffuse = nn.Sequential(DWConv(c1, c1, 3), Conv(c1, c1, 1))
        self.fuse = Conv(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        """Fuse local liquid state and reliability-gated sparse relation messages into ``x``."""
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"LGRB({self.c1}) expects B,{self.c1},H,W input, got {tuple(x.shape)}")
        if condition is not None and condition.shape != (x.shape[0], self.c1, 1, 1):
            raise ValueError(
                f"LGRB({self.c1}) condition expects {(x.shape[0], self.c1, 1, 1)}, got {tuple(condition.shape)}"
            )
        batch, channels, height, width = x.shape
        positions = height * width
        anchor_count = min(self.max_anchors, positions)
        neighbor_count = min(self.neighbor_k, anchor_count)
        local_state = self.state(x, condition)
        saliency = self.saliency(local_state).flatten(1)
        reliability_logits = self.reliability(local_state)
        if condition is not None:
            reliability_logits = reliability_logits + self.condition_reliability(condition)
        reliability = reliability_logits.flatten(1).sigmoid()
        anchor_indices = (saliency.sigmoid() * reliability).topk(anchor_count, dim=1).indices
        flattened = local_state.flatten(2).transpose(1, 2)
        gather_index = anchor_indices.unsqueeze(-1).expand(-1, -1, channels)
        anchors = flattened.gather(1, gather_index)
        anchor_reliability = reliability.gather(1, anchor_indices)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=x.dtype),
            torch.arange(width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        coordinate_grid = torch.stack((yy / max(height - 1, 1), xx / max(width - 1, 1)), dim=-1).reshape(positions, 2)
        coordinates = coordinate_grid.unsqueeze(0).expand(batch, -1, -1).gather(
            1, anchor_indices.unsqueeze(-1).expand(-1, -1, 2)
        )
        relative_coordinate = coordinates.unsqueeze(2) - coordinates.unsqueeze(1)
        squared_distance = relative_coordinate.square().sum(dim=-1)

        query, key, value = self.query(anchors), self.key(anchors), self.value(anchors)
        semantic_affinity = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1)
        neighbor_affinity = semantic_affinity - F.softplus(self.distance_weight) * squared_distance
        neighbor_indices = neighbor_affinity.topk(neighbor_count, dim=-1).indices
        neighbor_mask = torch.zeros_like(neighbor_affinity, dtype=torch.bool).scatter_(2, neighbor_indices, True)
        logits = query @ key.transpose(-2, -1) / (query.shape[-1] ** 0.5)
        logits = logits + self.coordinate_bias(relative_coordinate).squeeze(-1)
        logits = logits + anchor_reliability.clamp_min(torch.finfo(x.dtype).eps).log().unsqueeze(1)
        logits = logits.masked_fill(~neighbor_mask, torch.finfo(logits.dtype).min)
        messages = self.output(logits.softmax(dim=-1) @ value) * anchor_reliability.unsqueeze(-1)

        scattered = flattened.new_zeros(batch, positions, channels)
        scattered.scatter_add_(1, gather_index, messages)
        relation_map = scattered.transpose(1, 2).reshape(batch, channels, height, width)
        update = self.fuse(self.local_output(local_state) + self.diffuse(relation_map))
        return x + self.gamma * update


class LHGB(nn.Module):
    """Two-input hierarchical liquid gated block for C4/C5 backbone stages."""

    def __init__(
        self, c1: int, context_channels: int, relation_dim: int = 64, max_anchors: int = 32, neighbor_k: int = 6
    ):
        super().__init__()
        if c1 <= 0 or context_channels <= 0:
            raise ValueError(f"LHGB requires positive current/context channels, got {c1}, {context_channels}")
        self.c1 = int(c1)
        self.context_channels = int(context_channels)
        self.context_project = Conv(context_channels, c1, 1)
        self.relation = LGRB(c1, relation_dim, max_anchors, neighbor_k)

    @property
    def gamma(self) -> nn.Parameter:
        """Expose the only residual scale so initialization and tests remain explicit."""
        return self.relation.gamma

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Condition the current C4/C5 stage on a pooled preceding backbone feature map."""
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("LHGB expects [current_feature, preceding_feature] inputs")
        x, context = inputs
        if x.ndim != 4 or x.shape[1] != self.c1:
            raise ValueError(f"LHGB({self.c1}) current input expects B,{self.c1},H,W, got {tuple(x.shape)}")
        if context.ndim != 4 or context.shape[0] != x.shape[0] or context.shape[1] != self.context_channels:
            raise ValueError(
                f"LHGB({self.c1}) context expects B,{self.context_channels},H,W with batch {x.shape[0]}, got {tuple(context.shape)}"
            )
        stage_state = self.context_project(F.adaptive_avg_pool2d(context, 1))
        return self.relation(x, stage_state)


class DyHead(nn.Module):
    """Lightweight three-attention Dynamic Head for P3/P4/P5 detector features.

    The module follows DyHead's scale-aware, spatial-aware and task-aware decomposition while using only standard,
    deterministic PyTorch operators.  It returns one tensor per input scale with exactly the original channel and
    spatial contracts so an unmodified Ultralytics ``Detect`` layer can consume the outputs.
    """

    def __init__(self, ch: tuple[int, int, int] | list[int]):
        """Initialize dynamic scale fusion and per-level spatial/task attention."""
        super().__init__()
        if len(ch) != 3:
            raise ValueError(f"DyHead expects exactly P3/P4/P5 channels, got {tuple(ch)}")
        self.ch = tuple(int(channel) for channel in ch)
        self.projections = nn.ModuleList(
            nn.ModuleList(Conv(source_channels, target_channels, 1) for source_channels in self.ch)
            for target_channels in self.ch
        )
        self.scale_logits = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel, 1, 1)) for channel in self.ch
        )
        self.spatial = nn.ModuleList(
            nn.Sequential(DWConv(channel, channel, 3), nn.Conv2d(channel, 1, 1), nn.Sigmoid()) for channel in self.ch
        )
        self.task = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channel, max(channel // 4, 4), 1), nn.SiLU(), nn.Conv2d(max(channel // 4, 4), channel, 1), nn.Sigmoid())
            for channel in self.ch
        )
        self.fuse = nn.ModuleList(Conv(channel, channel, 3) for channel in self.ch)
        self.gamma = nn.Parameter(torch.zeros(3))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply dynamic scale, spatial and task attention while preserving all feature shapes."""
        if len(features) != 3:
            raise ValueError(f"DyHead expects exactly three feature tensors, got {len(features)}")
        outputs = []
        for target_index, target_channels in enumerate(self.ch):
            target_size = features[target_index].shape[-2:]
            aligned = []
            logits = []
            for source_index in range(3):
                source = self.projections[target_index][source_index](features[source_index])
                if source.shape[-2:] != target_size:
                    source = F.interpolate(source, size=target_size, mode="nearest")
                aligned.append(source)
                logits.append(self.scale_logits[target_index](source))
            weights = torch.softmax(torch.cat(logits, dim=1), dim=1).unbind(dim=1)
            mixed = sum(weight.unsqueeze(1) * source for weight, source in zip(weights, aligned))
            spatial = self.spatial[target_index](mixed)
            task = self.task[target_index](mixed)
            refined = self.fuse[target_index](mixed * spatial * task)
            outputs.append(features[target_index] + self.gamma[target_index] * refined)
        return outputs


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class LGCBottleneck(nn.Module):
    """C2f branch transform using channel mixing around one liquid gated convolution."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 1.0):
        super().__init__()
        if c1 <= 0 or c2 <= 0:
            raise ValueError(f"LGCBottleneck requires positive channel counts, got {c1}, {c2}")
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.liquid = LGC(c_)
        self.cv2 = Conv(c_, c2, 1, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a C2f branch with liquid local state and preserve optional residual semantics."""
        update = self.cv2(self.liquid(self.cv1(x)))
        return x + update if self.add else update


class C2fLGC(nn.Module):
    """C2f with its final ceil(n/2) internal units replaced by liquid gated bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        if n <= 0:
            raise ValueError(f"C2fLGC requires a positive internal repeat count, got {n}")
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        liquid_count = (n + 1) // 2
        standard_count = n - liquid_count
        modules: list[nn.Module] = []
        for index in range(n):
            if index < standard_count:
                modules.append(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0))
            else:
                modules.append(LGCBottleneck(self.c, self.c, shortcut, g, e=1.0))
        self.m = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retain standard C2f split/concatenate feature reuse around mixed internal transforms."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward using split() for export-compatible C2f behavior."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)
