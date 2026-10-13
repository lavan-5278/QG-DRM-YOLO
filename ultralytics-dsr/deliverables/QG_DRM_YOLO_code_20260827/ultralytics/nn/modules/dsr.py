# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DSR-Det: Differential Social Relation Detection (Phase 2/3).

Ultralytics-native extension described in
《基于 Ultralytics 原生扩展的 DSR-Det 单帧课堂行为检测技术方案》v1.1:

- ``DSRDetect``: ``Detect`` subclass that generates low-threshold candidates, runs the
  MDRR relation-refinement branch, and writes refined results back into the standard
  detection tensor (inference only). Training returns the base per-level predictions
  exactly like the official ``Detect`` head, and exposes ``dsr_out`` for
  ``DSRDetectionLoss``.
- ``MDRR``: Multi-scale Differential Relation Refinement. Multi-scale RoI features,
  classroom/region/neighbor relation graphs, signed-difference aggregation, and an
  uncertainty-aware residual fusion gate (v1.1).
- ``RelationRefinementHead``: zero-initialized residual head (class-logit residual +
  clamped box residual), v1.1 init: ``W_cls=0``, ``W_box=0``, ``rho=0``, ``eta_cls=1``;
- ``relation_type`` R0-R7 and ``relation_variant`` M1-M5 (v1.1 18.3/18.4);
- ``candidate_source: ema`` with a frozen EMA base head (v1.1 14.1).

Phase status (v1.1):
- Phase 0 (baseline lock): pinned ultralytics==8.3.247, see requirements-lock.txt.
- Phase 1 (registration skeleton): DSRDetect/MDRR registration, yolo11n-dsr.yaml,
  ``dsr.enabled=false`` output identity, official weights load, train/val/predict smoke.
- Phase 2-3 (relation operators R0-R7, M-series variants, EMA candidate source):
  implemented and config-driven; GICR loss components are wired in
  ``DSRDetectionLoss`` with v1.1 matching (duplicate positives -> ignore).

All training/evaluation stays on official Ultralytics entry points
(``model.train()`` / ``model.val()`` / ``model.predict()``); no standalone train/val
scripts are part of this project.
"""

from __future__ import annotations

from typing import Any

import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors

try:
    from torchvision.ops import roi_align as _tv_roi_align
except Exception:  # pragma: no cover - torchvision is a hard dependency in this project
    _tv_roi_align = None

__all__ = "DSRDetect", "MDRR", "RelationRefinementHead", "pairwise_iou"


class _TimeIt:
    """Lightweight per-component timing sink (enabled when ``sink`` is not None)."""

    __slots__ = ("sink", "key", "t0")

    def __init__(self, sink: dict[str, list[float]] | None, key: str):
        self.sink = sink
        self.key = key

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.sink is not None:
            self.sink[self.key].append((time.perf_counter() - self.t0) * 1000.0)
        return False


def xyxy2cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert xyxy boxes (..., 4) to cxcywh."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)), -1)


def cxcywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert cxcywh boxes (..., 4) to xyxy."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of xyxy boxes: a [N,4] / b [M,4] -> [N,M]."""
    # fp16 area products overflow (>65504) for boxes larger than ~256x256,
    # turning IoU into NaN. Compute in fp32; callers cast back if needed.
    a = a.float()
    b = b.float()
    a1, a2 = a[..., :2], a[..., 2:]
    b1, b2 = b[..., :2], b[..., 2:]
    area_a = ((a2 - a1).clamp(min=0)[..., 0] * (a2 - a1).clamp(min=0)[..., 1])
    area_b = ((b2 - b1).clamp(min=0)[..., 0] * (b2 - b1).clamp(min=0)[..., 1])
    lt = torch.maximum(a1.unsqueeze(-2), b1.unsqueeze(-3))
    rb = torch.minimum(a2.unsqueeze(-2), b2.unsqueeze(-3))
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a.unsqueeze(-1) + area_b.unsqueeze(-2) - inter + 1e-7
    return inter / union


class MDRR(nn.Module):
    """Multi-scale Differential Relation Refinement (Phase 2/3).

    v1.1 components implemented here:
    - multi-scale RoIAlign (P3/P4/P5) -> per-scale projection -> LayerNorm fusion;
    - classroom (``global_topk`` conf subset), region (``region_grid``) and neighbor
      (scale-aware KNN) relation masks, config-gated via use_global/use_region/use_neighbor;
    - relation operators switchable via ``relation_type`` (v1.1 18.3 R0-R7):
      ``none``, ``dot``, ``concat``, ``product``, ``abs_diff``, ``edgeconv``,
      ``signed_diff`` (main), ``transformer``;
    - MDRR internal ablations via ``relation_variant`` (v1.1 18.4 M1-M5):
      ``none``, ``no_geo``, ``no_gate``, ``flip``, ``shared``;
    - uncertainty-aware residual gate (``rho`` init 0, gate bias init negative).
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (),
        nc: int = 3,
        relation_dim: int = 128,
        roi_size: int = 3,
        neighbor_k: int = 5,
        region_grid: tuple[int, int] = (2, 2),
        global_topk: int = 30,
        uncertainty_type: str = "margin",
        lambda_s: float = 1.0,
        max_candidates: int = 100,
        relation_type: str = "signed_diff",
        relation_variant: str = "none",
        include_geometry: bool = True,
        use_uncertainty_gate: bool = True,
        gate_mode: str = "uncertainty",
        use_global: bool = False,  # v2: KNN-only, global/region removed
        use_region: bool = False,
        use_neighbor: bool = True,
        uncertainty_threshold: float = 0.5,
        neighbor_reliability: bool = True,
        use_composition: bool = False,  # model-level default on via yaml (v3)
        composition_proto: str = "conf",
    ):
        super().__init__()
        assert _tv_roi_align is not None, "torchvision.ops.roi_align is required"
        self.nc = nc
        self.relation_dim = int(relation_dim)
        self.roi_size = int(roi_size)
        self.neighbor_k = int(neighbor_k)
        self.region_grid = tuple(int(x) for x in region_grid)
        self.global_topk = max(1, int(global_topk))
        self.uncertainty_type = uncertainty_type
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.neighbor_reliability = bool(neighbor_reliability)
        self.use_composition = bool(use_composition)
        self.composition_proto = str(composition_proto)
        self.lambda_s = float(lambda_s)
        self.max_candidates = int(max_candidates)
        self.timings: dict[str, list[float]] | None = None
        self.relation_type = str(relation_type)
        self.relation_variant = str(relation_variant)
        self.include_geometry = bool(include_geometry)
        # M2 ablation (v1.1 18.4): relation_variant "no_geo" must disable the
        # geometric edge encoding even when include_geometry is left at its default.
        if self.relation_variant == "no_geo":
            self.include_geometry = False
        # v1.1 18.5 gate ablations: fixed | scalar | conf | feature | uncertainty
        self.gate_mode = "uncertainty" if use_uncertainty_gate else str(gate_mode)
        self.use_uncertainty_gate = bool(use_uncertainty_gate)
        self.scale_names = [n for n, on in (("g", use_global), ("r", use_region), ("n", use_neighbor)) if on]
        C = self.relation_dim

        # v3 composition prior (dataset diagnosis): image-level class composition is
        # the strongest relational signal. A conf-weighted group prototype + soft
        # class histogram is projected and added to *selected* candidates only.
        if self.use_composition:
            self.ctx_proj = nn.Linear(C + self.nc, C)

        # Multi-scale RoI features -> per-scale projection -> concat fusion
        self.scale_proj = nn.ModuleList(nn.Linear(c, C) for c in in_channels)
        self.fuse = nn.Sequential(
            nn.Linear(C * max(1, len(in_channels)), C),
            nn.LayerNorm(C),
        )

        self.is_graph = self.relation_type not in ("none", "transformer") and len(self.scale_names) > 0

        if self.relation_type == "transformer":  # R7: generic attention baseline
            layer = nn.TransformerEncoderLayer(
                d_model=C, nhead=4, dim_feedforward=2 * C, batch_first=True, activation="gelu", norm_first=True
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=2)
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_candidates, C))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            return

        if self.is_graph:
            # Query/key projections shared by dot & signed (v1.1 10.1)
            self.w_q = nn.Linear(C, C)
            self.w_k = nn.Linear(C, C)
            if self.relation_type == "dot":  # R1: value projection
                self.w_v = nn.Linear(C, C)
            if self.relation_type in ("concat", "edgeconv"):  # R2/R5 pair projection
                self.pair_proj = nn.Linear(2 * C, C)

            # Edge encoding (+ optional geometric joint encoding)
            if self.relation_type in ("concat", "product", "abs_diff", "edgeconv", "signed_diff"):
                edge_in = C + (5 if self.include_geometry else 0)
                self.edge_enc = nn.Sequential(nn.Linear(edge_in, C), nn.SiLU(), nn.Linear(C, C), nn.SiLU())
                self.edge_s = nn.Linear(C, 1)  # scalar edge weight
                self.edge_v = nn.Linear(C, C)  # channel gate (M3 ablation: ignored)
                self.w_m = nn.Linear(C, C)  # message projection

            # Learned scale fusion (only over enabled scales)
            self.scale_w = nn.Linear(C * (1 + len(self.scale_names)), len(self.scale_names))

            # Residual fusion gate (v1.1 11.x / 18.5). rho=0 keeps stable init.
            self.rho = nn.Parameter(torch.zeros(1))
            if self.gate_mode in ("feature", "uncertainty"):
                gate_in = C * 2 + (1 if self.gate_mode == "uncertainty" else 0)
                self.gate = nn.Linear(gate_in, C)
                nn.init.constant_(self.gate.bias, -3.0)  # keep fusion small initially

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        features: list[torch.Tensor],
        boxes: torch.Tensor,
        mask: torch.Tensor,
        base_conf: torch.Tensor,
        img_size: tuple[int, int],
        probs: torch.Tensor | None = None,
        selection: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run MDRR on padded per-image candidates.

        Args:
            features: multi-scale neck features [B, C_l, H_l, W_l] (P3/P4/P5).
            boxes: candidate boxes [B, K, 4] xyxy in input-image pixels (detached).
            mask: valid-candidate mask [B, K] (bool).
            base_conf: base confidence [B, K] (detached).
            probs: per-class probabilities [B, K, nc] (detached, optional; needed for
                margin/entropy uncertainty).
            selection: ambiguous-candidate mask [B, K] (v2): only these candidates
                receive the relation residual; easy candidates keep individual
                features (YOLO result preserved).
            img_size: (H, W) of the padded input image.

        Returns:
            (f_tilde [B, K, C], aux dict with scale weights / gate / uncertainty / features).
        """
        with _TimeIt(self.timings, "mdrr_total"):
            f = self._roi_features(features, boxes, mask, img_size)
        u = self._uncertainty(base_conf, probs)  # [B, K]
        if selection is None:
            selection = torch.ones_like(u, dtype=torch.bool)
        sel = selection.unsqueeze(-1).to(f.dtype)  # [B, K, 1]
        B, K = boxes.shape[:2]

        if self.relation_type == "transformer":  # R7: generic attention baseline
            h = self._transformer(f, mask)
            aux = {
                "scale_weights": f.new_zeros(B, K, max(1, len(self.scale_names))),
                "gate": f.new_zeros(B, K, self.relation_dim),
                "uncertainty": u,
                "features": f,
                "relations": f.new_zeros(B, K, self.relation_dim),
                "graph_masks": {},
                "ctx": f.new_zeros(B, K, self.relation_dim),
            }
            return h, aux

        if not self.is_graph:  # R0: individual refinement only (no cross-object relation)
            aux = {
                "scale_weights": f.new_zeros(B, K, max(1, len(self.scale_names))),
                "gate": f.new_zeros(B, K, self.relation_dim),
                "uncertainty": u,
                "features": f,
                "relations": f.new_zeros(B, K, self.relation_dim),
                "graph_masks": {},
                "ctx": f.new_zeros(B, K, self.relation_dim),
            }
            return f, aux

        with _TimeIt(self.timings, "graphs"):
            masks = self._build_graphs(boxes, mask, img_size)
        with _TimeIt(self.timings, "aggregate"):
            reliability = (1.0 - u) if self.neighbor_reliability else None
            rel = self._aggregate(f, boxes, mask, masks, img_size, reliability=reliability)
        rel_list = [rel[n] for n in self.scale_names]
        r_all = torch.stack(rel_list, dim=-2)  # [B, K, S, C]
        beta = F.softmax(self.scale_w(torch.cat((f, *rel_list), -1)), dim=-1)  # [B, K, S]
        r_fused = (beta.unsqueeze(-1) * r_all).sum(dim=-2)  # [B, K, C]
        C = self.relation_dim
        gamma = None
        with _TimeIt(self.timings, "gate"):
            if self.gate_mode == "fixed":  # v1.1 18.5: no learned residual control
                f_tilde = F.layer_norm(f + sel * r_fused, (C,))
            elif self.gate_mode == "scalar":  # single learnable scale rho
                f_tilde = F.layer_norm(f + sel * self.rho * r_fused, (C,))
            elif self.gate_mode == "conf":  # direct confidence multiplier (1 - c_i)
                f_tilde = F.layer_norm(f + sel * u.unsqueeze(-1) * r_fused, (C,))
            elif self.gate_mode == "feature":  # feature gate without uncertainty input
                gamma = torch.sigmoid(self.gate(torch.cat((f, r_fused), -1)))
                f_tilde = F.layer_norm(f + sel * self.rho * gamma * r_fused, (C,))
            else:  # uncertainty-aware gate (default)
                gamma = torch.sigmoid(self.gate(torch.cat((f, r_fused, u.unsqueeze(-1)), -1)))
                f_tilde = F.layer_norm(f + sel * self.rho * gamma * r_fused, (C,))
        if self.use_composition:
            with _TimeIt(self.timings, "composition"):
                ctx = self._composition_context(f, probs, base_conf, mask, sel)
                f_tilde = f_tilde + ctx
        else:
            ctx = f.new_zeros(B, K, self.relation_dim)
        aux = {
            "scale_weights": beta,
            "gate": gamma if gamma is not None else f.new_zeros(B, K, self.relation_dim),
            "uncertainty": u,
            "features": f,
            "relations": r_fused,
            "graph_masks": masks,
            "ctx": ctx,
        }
        return f_tilde, aux

    # ------------------------------------------------------------- composition
    def _composition_context(
        self,
        f: torch.Tensor,
        probs: torch.Tensor | None,
        base_conf: torch.Tensor,
        mask: torch.Tensor,
        sel: torch.Tensor,
    ) -> torch.Tensor:
        """v3 composition prior: deviation from the conf-weighted group prototype
        plus the soft class histogram of all valid candidates.

        Returns ctx [B, K, C], masked by ``sel`` so only selected (ambiguous)
        candidates receive the composition residual (easy candidates keep YOLO).
        """
        B, K, C = f.shape
        counts = mask.sum(dim=1).clamp(min=1)  # [B]
        mf = mask.to(f.dtype)
        if probs is None:
            h = f.new_zeros(B, self.nc)
        else:
            mp = mask.unsqueeze(-1).to(probs.dtype)
            h = (probs * mp).sum(dim=1) / counts.unsqueeze(-1).clamp(min=1)  # [B, nc]
        if self.composition_proto == "conf":
            w = (base_conf * mask.to(base_conf.dtype)).clamp(min=0)
        else:  # uniform
            w = mf
        p = (f * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp(min=1e-6)  # [B, C]
        dev = f - p.unsqueeze(1)  # [B, K, C]
        h_b = h.unsqueeze(1).expand(B, K, self.nc)
        ctx = self.ctx_proj(torch.cat((dev, h_b), -1)) * sel.to(f.dtype)
        return ctx

    # ---------------------------------------------------------- multi-scale RoI
    def _roi_features(
        self,
        features: list[torch.Tensor],
        boxes: torch.Tensor,
        mask: torch.Tensor,
        img_size: tuple[int, int],
    ) -> torch.Tensor:
        B, K = boxes.shape[:2]
        device, dtype = boxes.device, boxes.dtype
        img_h = img_size[0]
        outs = []
        for lvl, feat in enumerate(features):
            h_l, w_l = feat.shape[-2:]
            stride = img_h / h_l if img_h else 1.0
            max_x = torch.full_like(boxes[..., 0], w_l * stride - 1)
            max_y = torch.full_like(boxes[..., 1], h_l * stride - 1)
            x1 = boxes[..., 0].clamp(max=max_x).clamp(min=0)
            y1 = boxes[..., 1].clamp(max=max_y).clamp(min=0)
            x2 = boxes[..., 2].clamp(min=x1 + 1).clamp(max=max_x + 1)
            y2 = boxes[..., 3].clamp(min=y1 + 1).clamp(max=max_y + 1)
            bi = torch.arange(B, device=device).view(-1, 1).expand(B, K).reshape(-1).float()
            rois = torch.stack((bi, x1.reshape(-1), y1.reshape(-1), x2.reshape(-1), y2.reshape(-1)), -1)
            rois = rois.to(dtype=feat.dtype)
            pooled = _tv_roi_align(
                feat,
                rois,
                output_size=(self.roi_size, self.roi_size),
                spatial_scale=1.0 / stride,
                aligned=True,
                # Fixed 2x2 sampling per bin (Detectron2 convention). With the
                # default sampling_ratio=-1 (adaptive), torchvision 0.18's
                # deterministic (pure-Python) roi_align path materializes
                # [K, C, PH, PW, H_l, W_l] tensors and OOMs at imgsz=640.
                sampling_ratio=2,
            )  # [B*K, C_l, roi, roi]
            pooled = F.adaptive_avg_pool2d(pooled, 1).flatten(1)  # [B*K, C_l]
            proj = self.scale_proj[lvl](pooled).view(B, K, self.relation_dim)
            outs.append(proj)
        f = self.fuse(torch.cat(outs, -1))
        f = f * mask.unsqueeze(-1).to(f.dtype)
        return f

    # ------------------------------------------------------------------ graphs
    def _build_graphs(
        self, boxes: torch.Tensor, mask: torch.Tensor, img_size: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        B, K = boxes.shape[:2]
        device = boxes.device
        eye = torch.eye(K, device=device, dtype=torch.bool).view(1, K, K)
        valid = (mask.unsqueeze(-1) & mask.unsqueeze(-2)) & ~eye  # [B, K, K]
        masks: dict[str, torch.Tensor] = {}

        if "g" in self.scale_names:  # classroom graph: high-confidence subset
            idx = torch.arange(K, device=device).view(1, K, 1).expand(B, K, K)
            masks["g"] = valid & (idx < self.global_topk)

        if "r" in self.scale_names:  # region graph: fixed grid over normalized centers
            x1, y1, x2, y2 = boxes.unbind(-1)
            cx = (x1 + x2) / 2 / max(1, img_size[1])
            cy = (y1 + y2) / 2 / max(1, img_size[0])
            gr, gc = self.region_grid
            zone = (cy.clamp(0, 1 - 1e-6) * gr).long() * gc + (cx.clamp(0, 1 - 1e-6) * gc).long()
            masks["r"] = valid & (zone.unsqueeze(-1) == zone.unsqueeze(-2))

        if "n" in self.scale_names:  # neighbor graph: scale-aware KNN
            d = self._geo_distance(boxes, img_size)  # [B, K, K]
            k = min(self.neighbor_k, K - 1)
            if k >= 1:
                d_masked = d.masked_fill(~valid, float("inf"))
                _, topk_idx = d_masked.topk(k, dim=-1, largest=False)
                n_mask = torch.zeros_like(valid)
                n_mask.scatter_(-1, topk_idx, True)
                masks["n"] = n_mask & valid
            else:
                masks["n"] = torch.zeros_like(valid)

        return masks

    def _geo_distance(self, boxes: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
        """Scale-aware geometric distance (v1.1 9.3; not camera-calibrated)."""
        boxes = boxes.float()
        x1, y1, x2, y2 = boxes.unbind(-1)
        w = (x2 - x1).clamp(min=1e-3)
        h = (y2 - y1).clamp(min=1e-3)
        u = (x1 + x2) / 2 / max(1, img_size[1])
        v = (y1 + y2) / 2 / max(1, img_size[0])
        du = u.unsqueeze(-1) - u.unsqueeze(-2)
        dv = v.unsqueeze(-1) - v.unsqueeze(-2)
        lw = torch.log((w.unsqueeze(-1) / w.unsqueeze(-2).clamp(min=1e-3)).clamp(1e-3, 1e3))
        lh = torch.log((h.unsqueeze(-1) / h.unsqueeze(-2).clamp(min=1e-3)).clamp(1e-3, 1e3))
        return torch.sqrt(du**2 + dv**2 + (self.lambda_s * lw) ** 2 + (self.lambda_s * lh) ** 2 + 1e-8)

    # -------------------------------------------------------------- aggregation
    def _aggregate(
        self,
        f: torch.Tensor,
        boxes: torch.Tensor,
        mask: torch.Tensor,
        masks: dict[str, torch.Tensor],
        img_size: tuple[int, int],
        reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, K, C = f.shape
        op = self.relation_type
        q = self.w_q(f)
        k = self.w_k(f)

        if op == "dot":  # R1: scaled dot-product attention (no difference, no geometry)
            s = (q @ k.transpose(-1, -2)) / math.sqrt(C)  # [B, K, K]
            v = self.w_v(f)  # [B, K, C]
            msg = v.unsqueeze(1).expand(B, K, K, C)  # value_j
        else:
            fi = f.unsqueeze(2).expand(B, K, K, C)
            fj = f.unsqueeze(1).expand(B, K, K, C)
            if op == "concat":  # R2: Relation Network style
                pair = self.pair_proj(torch.cat((fi, fj), -1))
            elif op == "edgeconv":  # R5: center + neighbor difference
                pair = self.pair_proj(torch.cat((fi, fj - fi), -1))
            elif op == "product":  # R3: element-wise product
                pair = fi * fj
            elif op == "abs_diff":  # R4: direction-free difference
                pair = (fi - fj).abs()
            elif op == "signed_diff":  # R6: signed difference (main method)
                diff = q.unsqueeze(2) - k.unsqueeze(1)  # [B, K, K, C]
                if self.relation_variant == "flip":  # M4
                    diff = -diff
                if self.relation_variant == "shared":  # M5: single projection space
                    msg_pair = diff
                else:
                    msg_pair = fi - fj
                pair = diff
            else:  # pragma: no cover - guarded by is_graph in __init__
                raise ValueError(f"unsupported relation_type for graph aggregation: {op}")

            geom = self._geometry(boxes, img_size) if self.include_geometry else None
            e_in = torch.cat((pair, geom.to(pair.dtype)), -1) if geom is not None else pair
            e = self.edge_enc(e_in)
            s = self.edge_s(e).squeeze(-1)  # [B, K, K]
            if self.relation_variant == "no_gate":  # M3: disable channel gating
                v = torch.ones_like(pair)
            else:
                v = torch.sigmoid(self.edge_v(e))  # channel gate [B, K, K, C]
            msg_base = msg_pair if op == "signed_diff" and self.relation_variant != "shared" else pair
            msg = v * self.w_m(msg_base)  # [B, K, K, C]

        out = {}
        for name, msk in masks.items():
            if not msk.any():
                out[name] = torch.zeros_like(f)
                continue
            s_m = s.masked_fill(~msk, float("-inf"))
            if reliability is not None:
                # v2 7: alpha_ij = softmax(s_ij + log(q_j + eps)), q_j = 1 - u_j.
                # Prefer confident, behavior-clear neighbors over ambiguous ones.
                log_q = torch.log(reliability.clamp(min=0.0) + 1e-6)  # [B, K]
                s_m = s_m + log_q.unsqueeze(-2)
            alpha = F.softmax(s_m, dim=-1)
            # A row whose entire neighbor set is masked (e.g. an isolated
            # candidate in a region cell, or a padded row) is all -inf, which
            # softmax turns into NaN. Zero those rows out instead.
            alpha = torch.nan_to_num(alpha, nan=0.0).masked_fill(~msk, 0.0)
            out[name] = (alpha.unsqueeze(-1) * msg).sum(dim=2)
        return out

    def _transformer(self, f: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """R7: two-layer Transformer Encoder over candidates (batch_first, masked)."""
        B, K, C = f.shape
        h = f + self.pos_embed[:, :K]
        h = self.transformer(h, src_key_padding_mask=~mask)
        return torch.where(mask.unsqueeze(-1), h, torch.zeros_like(h))

    def _geometry(self, boxes: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
        """Geometric edge encoding [dx_norm, dy_norm, log(w_i/w_j), log(h_i/h_j), IoU]."""
        boxes = boxes.float()  # fp16 area/IoU overflows for large classroom boxes
        x1, y1, x2, y2 = boxes.unbind(-1)
        w = (x2 - x1).clamp(min=1e-6)
        h = (y2 - y1).clamp(min=1e-6)
        u = (x1 + x2) / 2 / max(1, img_size[1])
        v = (y1 + y2) / 2 / max(1, img_size[0])
        du = u.unsqueeze(-1) - u.unsqueeze(-2)
        dv = v.unsqueeze(-1) - v.unsqueeze(-2)
        lw = torch.log((w.unsqueeze(-1) / w.unsqueeze(-2).clamp(min=1e-6)).clamp(1e-4, 1e4))
        lh = torch.log((h.unsqueeze(-1) / h.unsqueeze(-2).clamp(min=1e-6)).clamp(1e-4, 1e4))
        iou = self._pair_iou(boxes)
        return torch.stack((du, dv, lw, lh, iou), -1)

    @staticmethod
    def _pair_iou(boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes.float()  # fp16 overflow for boxes with area > 65504
        x1, y1, x2, y2 = boxes.unbind(-1)
        area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)  # [B, K]
        inter_w = (
            torch.minimum(x2.unsqueeze(-1), x2.unsqueeze(-2))
            - torch.maximum(x1.unsqueeze(-1), x1.unsqueeze(-2))
        ).clamp(min=0)
        inter_h = (
            torch.minimum(y2.unsqueeze(-1), y2.unsqueeze(-2))
            - torch.maximum(y1.unsqueeze(-1), y1.unsqueeze(-2))
        ).clamp(min=0)
        inter = inter_w * inter_h
        union = area.unsqueeze(-1) + area.unsqueeze(-2) - inter + 1e-6
        return inter / union

    # --------------------------------------------------------------- uncertainty
    def _uncertainty(self, base_conf: torch.Tensor, probs: torch.Tensor | None = None) -> torch.Tensor:
        """Uncertainty input u (v1.1 11.1): max_prob | margin | entropy."""
        if self.uncertainty_type == "entropy" and probs is not None:
            p = probs.clamp(1e-6, 1.0)
            h = -(p * p.log()).sum(-1) / math.log(max(2, self.nc))
            return h.clamp(0.0, 1.0)
        if self.uncertainty_type == "margin" and probs is not None and self.nc >= 2:
            top2 = probs.topk(2, dim=-1).values
            margin = (top2[..., 0] - top2[..., 1]).clamp(0.0, 1.0)
            return (1.0 - margin).clamp(0.0, 1.0)
        return 1.0 - base_conf  # max_prob (default) / fallback


class RelationRefinementHead(nn.Module):
    """Classification-only residual head (DSR-Det v2).

    v2 keeps the YOLO bounding box untouched (social relations carry little
    spatial evidence) and only corrects behavior classification:
    ``logits = base_logits + W_cls(f_tilde) * eta_cls``, zero-initialized so the
    refined prediction equals the base prediction at start. When ``selected`` is
    provided, only ambiguous candidates receive the residual (easy candidates keep
    the raw YOLO logits).
    """

    def __init__(self, relation_dim: int, nc: int, cls_init_eta: float = 1.0):
        super().__init__()
        self.cls = nn.Linear(relation_dim, nc)
        self.eta_cls = nn.Parameter(torch.tensor(float(cls_init_eta)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Zero-init classification residual: refined logits equal base logits."""
        nn.init.zeros_(self.cls.weight)
        nn.init.zeros_(self.cls.bias)

    def forward(
        self,
        f_tilde: torch.Tensor,
        base_logits: torch.Tensor,
        selected: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return refined classification logits (bbox untouched in v2)."""
        dlogit = self.cls(f_tilde) * self.eta_cls
        if selected is not None:
            dlogit = dlogit * selected.unsqueeze(-1).to(dlogit.dtype)
        if u is not None:
            # v3: scale the residual by uncertainty (u in [0.5, 1] for selected),
            # so moderately-ambiguous candidates receive only a partial correction.
            dlogit = dlogit * u.unsqueeze(-1).to(dlogit.dtype)
        return base_logits + dlogit


class DSRDetect(Detect):
    """Detect head extended with candidate generation + MDRR relation refinement.

    Output contract (identical to official ``Detect``):
    - training: returns the list of per-level prediction tensors (base loss unchanged);
      DSR-specific tensors are exposed via ``self.dsr_out`` for ``DSRDetectionLoss``.
    - inference (non-export): returns ``(y_refined, x)`` where ``y_refined`` keeps the
      standard [B, 4+nc, N] cxcywh+sigmoid layout after Top-K write-back.
    - export: falls back to base decode (export support is a later phase).
    """

    def __init__(self, nc: int = 80, ch: tuple = (), dsr_cfg: dict | None = None):
        super().__init__(nc, ch)
        self.ch = tuple(ch)
        self.dsr_cfg: dict[str, Any] = {}
        self.dsr_active = False
        self.dsr_out: dict[str, Any] | None = None
        self.mdrr: MDRR | None = None
        self.refine_head: RelationRefinementHead | None = None
        self._configure_dsr({} if dsr_cfg is None else dsr_cfg)

    # ------------------------------------------------------------- configuration
    def _configure_dsr(self, cfg: dict[str, Any] | None) -> None:
        """Apply the ``dsr:`` YAML block (idempotent)."""
        cfg = dict(cfg or {})
        if cfg == self.dsr_cfg and self.mdrr is not None:
            return
        self.dsr_cfg = cfg
        relation_dim = int(cfg.get("relation_dim", 128))
        self.pre_topk = int(cfg.get("pre_topk", 300))
        self.relation_topk = int(cfg.get("relation_topk", 100))
        self.candidate_conf = float(cfg.get("candidate_conf", 0.03))
        self.candidate_iou = float(cfg.get("candidate_iou", 0.70))
        self.min_candidates = int(cfg.get("min_candidates", 2))
        self.use_global = bool(cfg.get("use_global", False))
        self.use_region = bool(cfg.get("use_region", False))
        self.use_neighbor = bool(cfg.get("use_neighbor", True))
        self.use_composition = bool(cfg.get("use_composition", True))
        self.composition_proto = str(cfg.get("composition_proto", "conf"))
        # selective composition blending (diagnosis-driven, default off):
        # for ambiguous candidates that disagree with a confident image-level
        # composition prior, blend the base score toward the prior.
        self.composition_blend = float(cfg.get("composition_blend", 0.0))
        self.blend_prior_threshold = float(cfg.get("blend_prior_threshold", 0.5))
        self.use_uncertainty_gate = bool(cfg.get("use_uncertainty_gate", True))
        self.relation_type = str(cfg.get("relation_type", "signed_diff"))
        self.relation_variant = str(cfg.get("relation_variant", "none"))
        self.include_geometry = bool(cfg.get("include_geometry", True))
        self.gate_mode = str(cfg.get("gate_mode", "uncertainty"))
        # v2 selective refinement: only ambiguous candidates (u > threshold) enter
        # the relation branch; confident candidates keep the YOLO result, and the
        # preservation loss anchors them (v2 5.1 / 10).
        self.uncertainty_threshold = float(cfg.get("uncertainty_threshold", 0.5))
        self.preserve_threshold = float(cfg.get("preserve_threshold", 0.9))
        self.track_uncertainty = bool(cfg.get("track_uncertainty", True))
        self.candidate_source = str(cfg.get("candidate_source", "current"))
        self.ema_decay = float(cfg.get("ema_decay", 0.999))
        self.track_stability = bool(cfg.get("track_stability", True))
        self.train_epoch = 0
        self.dsr_active = bool(cfg.get("enabled", True))
        self.timings: dict[str, list[float]] | None = None
        self.mdrr = MDRR(
            in_channels=self.ch,
            nc=self.nc,
            relation_dim=relation_dim,
            roi_size=int(cfg.get("roi_size", 3)),
            neighbor_k=int(cfg.get("neighbor_k", 5)),
            region_grid=tuple(cfg.get("region_grid", (2, 2))),
            global_topk=int(cfg.get("global_topk", 30)),
            uncertainty_type=str(cfg.get("uncertainty_type", "margin")),
            lambda_s=float(cfg.get("lambda_s", 1.0)),
            max_candidates=self.relation_topk,
            relation_type=self.relation_type,
            relation_variant=self.relation_variant,
            include_geometry=self.include_geometry,
            use_uncertainty_gate=self.use_uncertainty_gate,
            gate_mode=self.gate_mode,
            use_global=self.use_global,
            use_region=self.use_region,
            use_neighbor=self.use_neighbor,
            uncertainty_threshold=self.uncertainty_threshold,
            neighbor_reliability=bool(cfg.get("neighbor_reliability", True)),
            use_composition=self.use_composition,
            composition_proto=self.composition_proto,
        )
        self.refine_head = RelationRefinementHead(
            relation_dim, self.nc, cls_init_eta=float(cfg.get("eta_cls_init", 1.0))
        )
        self.ema_head: nn.Module | None = None
        self.ema_step = 0
        self.uncertainty_samples: list[torch.Tensor] = []
        self.conf_samples: list[torch.Tensor] = []

    # ----------------------------------------------------- EMA candidate source
    def _ensure_ema_head(self) -> nn.Module:
        """Create (lazily) a frozen EMA copy of the base prediction head (cv2/cv3/dfl)."""
        if self.ema_head is None:
            self.ema_head = nn.Module()
            self.ema_head.cv2 = copy.deepcopy(self.cv2)
            self.ema_head.cv3 = copy.deepcopy(self.cv3)
            self.ema_head.dfl = copy.deepcopy(self.dfl)
            self.ema_head.eval()
            for p in self.ema_head.parameters():
                p.requires_grad_(False)
        return self.ema_head

    def ema_reset(self) -> None:
        """Rebuild the EMA head from the current weights (call after loading weights)."""
        self.ema_head = None
        self.ema_step = 0

    def ema_update(self) -> None:
        """Polyak-average the base prediction head into the EMA copy (v1.1 14.1)."""
        self.ema_step += 1
        decay = min(self.ema_decay, (1 + self.ema_step) / (10 + self.ema_step))
        ema = self._ensure_ema_head()
        cur = dict(self.named_parameters())
        with torch.no_grad():
            for name, ema_p in ema.named_parameters():
                if name in cur:
                    ema_p.data.mul_(decay).add_(cur[name].data, alpha=1 - decay)

    @staticmethod
    def stability_compare(boxes_cur: torch.Tensor, boxes_ref: torch.Tensor) -> float:
        """Mean max-IoU of current candidate boxes against reference candidate boxes."""
        if boxes_cur is None or boxes_ref is None or boxes_cur.numel() == 0 or boxes_ref.numel() == 0:
            return 0.0
        iou = pairwise_iou(boxes_cur, boxes_ref)  # [N, M]
        return float(iou.max(dim=1).values.mean().item())

    # -------------------------------------------------------- uncertainty audit
    @staticmethod
    def _push_capped(samples: list[torch.Tensor], values: torch.Tensor, cap: int = 20000) -> None:
        """Append samples, keeping the most recent window under ``cap`` elements."""
        samples.append(values)
        total = sum(x.numel() for x in samples)
        while total > cap and len(samples) > 1:
            samples.pop(0)
            total = sum(x.numel() for x in samples)

    def uncertainty_reset_epoch(self) -> None:
        """Clear per-epoch uncertainty samples (called at each epoch start)."""
        self.uncertainty_samples = []
        self.conf_samples = []

    def uncertainty_audit(self) -> dict[str, float] | None:
        """u-distribution audit (v1.1 7.1): report u>0.9 ratio for the current epoch."""
        if not self.uncertainty_samples:
            return None
        # AMP (fp16) stores fp16 samples; quantile requires float32/64.
        u = torch.cat(self.uncertainty_samples).float()
        c = torch.cat(self.conf_samples).float()
        q = torch.quantile(u, torch.tensor([0.5, 0.9], dtype=u.dtype))
        return {
            "n": int(u.numel()),
            "u_mean": round(float(u.mean()), 4),
            "u_p50": round(float(q[0]), 4),
            "u_p90": round(float(q[1]), 4),
            "ratio_u_gt_0.9": round(float((u > 0.9).float().mean()), 4),
            "conf_mean": round(float(c.mean()), 4),
        }

    # -------------------------------------------------------------------- forward
    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Same output contract as ``Detect.forward``, plus DSR refinement."""
        if self.end2end:
            return self.forward_end2end(x)
        x_in = list(x)  # snapshot neck features BEFORE head convs (P3/P4/P5)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            self.dsr_out = self._train_branch(x, x_in) if self.dsr_active else None
            return x
        with _TimeIt(self.timings, "base_detect"):
            y = self._inference(x)
        self.base_y = y.detach()  # base decode before DSR write-back (validator analysis)
        self.neck_feats = x_in  # pre-conv neck features for validator oracle analysis (19.5)
        self.last_aux = None
        self.last_cand_mask = None
        if self.dsr_active and not self.export:
            with _TimeIt(self.timings, "dsr_total"):
                y = self._inference_refine(y, x_in)
            return (y, x)
        return y if self.export else (y, x)

    # -------------------------------------------------------------- train branch
    def _train_branch(self, feats: list[torch.Tensor], x_in: list[torch.Tensor]) -> dict[str, Any] | None:
        """Decode candidates from base predictions and run MDRR (training path).

        Returns a dict consumed by ``DSRDetectionLoss``. Candidate selection and box
        coordinates are detached (stop-gradient, v1.1 7.1); RoI features keep
        gradients so Stage B2 joint fine-tuning can reach the neck/backbone.
        """
        if self.stride.max() <= 0:  # stride-computation pass in DetectionModel.__init__
            return None
        B = feats[0].shape[0]
        pred = torch.cat([xi.view(B, self.no, -1) for xi in feats], 2)  # [B, no, N]
        box_dist, cls = pred.split((self.reg_max * 4, self.nc), 1)
        anchors, stride_tensor = make_anchors(feats, self.stride, 0.5)
        strides = stride_tensor.t()  # [1, N] per-anchor stride, same layout as Detect._inference
        boxes_cxcywh = self.decode_bboxes(self.dfl(box_dist), anchors.t().unsqueeze(0)) * strides  # [B,4,N]
        scores = cls.sigmoid()  # [B, nc, N]
        img_h = feats[0].shape[2] * self.stride[0].item()
        img_w = feats[0].shape[3] * self.stride[0].item()
        if self.candidate_source == "ema":  # v1.1 14.1: stable candidates from EMA head
            ema = self._ensure_ema_head()
            with torch.no_grad():
                ema_raw = [torch.cat((ema.cv2[i](x_in[i]), ema.cv3[i](x_in[i])), 1) for i in range(self.nl)]
                ema_pred = torch.cat([xi.view(B, self.no, -1) for xi in ema_raw], 2)
                eb, ec = ema_pred.split((self.reg_max * 4, self.nc), 1)
                ema_boxes = self.decode_bboxes(ema.dfl(eb), anchors.t().unsqueeze(0)) * strides
                ema_scores = ec.sigmoid()
            sel = self._select_candidates(ema_boxes, ema_scores, (int(img_h), int(img_w)))
        else:
            with _TimeIt(self.timings, "candidate"):
                sel = self._select_candidates(boxes_cxcywh, scores, (int(img_h), int(img_w)))
        cand_boxes = sel["boxes_xyxy"].detach()  # [B, K, 4]
        cand_mask = sel["mask"]  # [B, K]
        cand_conf = sel["conf"].detach()  # [B, K]
        cand_idx = sel["idx"]  # [B, K]

        K = cand_boxes.shape[1]
        logits = cls.permute(0, 2, 1)  # [B, N, nc] (keeps gradient -> base head)
        base_logits = logits.gather(1, cand_idx.unsqueeze(-1).expand(-1, -1, self.nc))  # [B, K, nc]
        probs = scores.permute(0, 2, 1).gather(
            1, cand_idx.unsqueeze(-1).expand(-1, -1, self.nc)
        ).detach()  # [B, K, nc] for margin/entropy uncertainty

        if not cand_mask.any():
            return {"active": True, "empty": True}

        # v2 selective refinement: ambiguous candidates (u > threshold) get the
        # relation residual; easy candidates keep the YOLO logits.
        u = self.mdrr._uncertainty(cand_conf, probs)
        selection = (u > self.uncertainty_threshold) & cand_mask  # [B, K]
        with _TimeIt(self.timings, "mdrr"):
            f_tilde, aux = self.mdrr(
                x_in, cand_boxes, cand_mask, cand_conf, (img_h, img_w), probs, selection=selection
            )
        if self.track_uncertainty and "uncertainty" in aux:
            self._push_capped(self.uncertainty_samples, aux["uncertainty"][cand_mask].detach().cpu().flatten())
            self._push_capped(self.conf_samples, cand_conf[cand_mask].detach().cpu().flatten())
        with _TimeIt(self.timings, "refine_head"):
            refined_logits = self.refine_head(f_tilde, base_logits, selected=selection, u=u)
        self.dsr_out = {
            "active": True,
            "empty": False,
            "cand_boxes_xyxy": cand_boxes,
            "cand_mask": cand_mask,
            "cand_conf": cand_conf,
            "cand_idx": cand_idx,
            "base_logits": base_logits,
            "refined_logits": refined_logits,
            "f_tilde": f_tilde,
            "selection": selection,
            "img_size": (img_h, img_w),
            "aux": aux,
        }
        return self.dsr_out

    # ------------------------------------------------------------ inference path
    def _inference_refine(self, y: torch.Tensor, x_in: list[torch.Tensor]) -> torch.Tensor:
        """Refine Top-K candidates and write back into the standard tensor (inference)."""
        boxes_cxcywh = y[:, :4]  # [B, 4, N]
        scores = y[:, 4:]  # [B, nc, N]
        B = y.shape[0]
        img_h = int(x_in[0].shape[2] * self.stride[0].item())
        img_w = int(x_in[0].shape[3] * self.stride[0].item())
        with _TimeIt(self.timings, "candidate"):
            if self.candidate_source == "ema":
                sel = self.candidates_from_features(x_in, (img_h, img_w), source="ema")
            else:
                sel = self._select_candidates(boxes_cxcywh, scores, (img_h, img_w))
        if not sel["mask"].any():
            return y
        cand_boxes = sel["boxes_xyxy"]  # detached
        cand_mask = sel["mask"]
        self.last_cand_mask = cand_mask
        cand_conf = sel["conf"]
        cand_idx = sel["idx"]
        K = cand_boxes.shape[1]
        base_logits = torch.logit(scores.permute(0, 2, 1).gather(1, cand_idx.unsqueeze(-1).expand(-1, -1, self.nc)).clamp(1e-4, 1 - 1e-4))
        base_probs = scores.permute(0, 2, 1).gather(1, cand_idx.unsqueeze(-1).expand(-1, -1, self.nc))
        # v2: only ambiguous candidates receive the relation residual; boxes stay YOLO.
        u = self.mdrr._uncertainty(cand_conf, base_probs)
        selection = (u > self.uncertainty_threshold) & cand_mask
        with _TimeIt(self.timings, "mdrr"):
            f_tilde, aux = self.mdrr(
                x_in, cand_boxes, cand_mask, cand_conf, (img_h, img_w), base_probs, selection=selection
            )
        self.last_aux = aux  # scale weights / gate / uncertainty for visualization (19.5)
        with _TimeIt(self.timings, "refine_head"):
            refined_logits = self.refine_head(f_tilde, base_logits, selected=selection, u=u)
        refined_scores = refined_logits.sigmoid()
        # v3 score calibration: keep the base confidence level, only change the
        # class mix. Unselected candidates keep base scores exactly (scale ~= 1);
        # selected candidates get their max score pinned to the base confidence so
        # refinement cannot inflate false-positive scores.
        ref_max = refined_scores.max(dim=-1).values.clamp(min=1e-6)
        scale = (cand_conf / ref_max).clamp(max=3.0, min=0.0)
        refined_scores = (refined_scores * scale.unsqueeze(-1)).clamp(max=1.0)
        if getattr(self, "composition_blend", 0.0) > 0.0:
            # selective prior blend: only when the base is ambiguous, the prior is
            # confident (image dominated by one behavior) and they disagree.
            valid = cand_mask.to(base_probs.dtype)
            h = (base_probs * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
            prior_arg = h.argmax(dim=-1).unsqueeze(-1)  # [B, 1]
            base_arg = base_probs.argmax(dim=-1)
            prior_max = h.max(dim=-1).values.unsqueeze(-1)  # [B, 1]
            gate = (
                (u > self.uncertainty_threshold)
                & (prior_arg != base_arg)
                & (prior_max >= getattr(self, "blend_prior_threshold", 0.5))
                & cand_mask
            )
            w = (u * self.composition_blend).clamp(max=1.0)
            blended = (1 - w.unsqueeze(-1)) * base_probs + w.unsqueeze(-1) * h.unsqueeze(1).expand_as(base_probs)
            blended = blended / blended.sum(-1, keepdim=True).clamp(min=1e-9)
            refined_scores = torch.where(gate.unsqueeze(-1), blended, refined_scores)
        with _TimeIt(self.timings, "writeback"):
            y_ref = y.clone()
            for bi in range(B):
                k = int(cand_mask[bi].sum().item())
                if k == 0:
                    continue
                idx_bi = cand_idx[bi, :k]
                y_ref[bi, 4:, idx_bi] = refined_scores[bi, :k].t()  # v2: classification only
        return y_ref

    # ------------------------------------------------------------ candidates
    def _select_candidates(
        self,
        boxes_cxcywh: torch.Tensor,
        scores: torch.Tensor,
        img_size: tuple[int, int],
        conf_thr: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Low-threshold candidate selection: conf filter -> pre-Top-K -> light dedup -> K."""
        B = boxes_cxcywh.shape[0]
        device = boxes_cxcywh.device
        boxes_xyxy = cxcywh2xyxy(boxes_cxcywh.permute(0, 2, 1))  # [B, N, 4]
        conf, _ = scores.max(dim=1)  # [B, N]
        thr = self.candidate_conf if conf_thr is None else float(conf_thr)
        K = self.relation_topk
        out_boxes = torch.zeros(B, K, 4, device=device, dtype=boxes_xyxy.dtype)
        out_mask = torch.zeros(B, K, dtype=torch.bool, device=device)
        out_conf = torch.zeros(B, K, device=device, dtype=boxes_xyxy.dtype)
        out_idx = torch.zeros(B, K, dtype=torch.long, device=device)
        for bi in range(B):
            m = conf[bi] >= thr
            idx = torch.nonzero(m).squeeze(-1)
            if idx.numel() == 0:
                continue
            order = conf[bi][idx].argsort(descending=True)[: self.pre_topk]
            idx = idx[order]
            cand_b = boxes_xyxy[bi][idx]
            cand_c = conf[bi][idx]
            # Greedy IoU dedup. Exactly equivalent to the sequential per-candidate
            # check (keep j iff max IoU vs every kept candidate < threshold), but
            # vectorized: viol[j] tracks that running max; each newly kept candidate
            # updates viol with its own IoU column (one vectorized maximum per step).
            iou_mat = pairwise_iou(cand_b, cand_b).detach()  # [n, n], fp32
            n = idx.numel()
            viol = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
            keep = []
            for j in range(n):
                if viol[j] < self.candidate_iou:
                    keep.append(j)
                    torch.maximum(viol, iou_mat[:, j], out=viol)
            idx = idx[keep][:K]
            k = idx.numel()
            if k < self.min_candidates:
                continue
            out_boxes[bi, :k] = boxes_xyxy[bi][idx]
            out_conf[bi, :k] = conf[bi][idx]
            out_idx[bi, :k] = idx
            out_mask[bi, :k] = True
        return {"boxes_xyxy": out_boxes, "mask": out_mask, "conf": out_conf, "idx": out_idx}

    def candidates_from_features(
        self,
        x_in: list[torch.Tensor],
        img_size: tuple[int, int],
        conf_thr: float | None = None,
        source: str = "current",
    ) -> dict[str, torch.Tensor]:
        """Decode + select candidates from neck features using the current or EMA head.

        Used by the stability probe (v1.1 14.1) so that ``candidate_source: ema``
        measures drift of the EMA candidate set, not the raw base head's.
        """
        B = x_in[0].shape[0]
        anchors, stride_tensor = make_anchors(x_in, self.stride, 0.5)
        strides = stride_tensor.t()
        with torch.no_grad():
            if source == "ema":
                ema = self._ensure_ema_head()
                raw = [torch.cat((ema.cv2[i](x_in[i]), ema.cv3[i](x_in[i])), 1) for i in range(self.nl)]
                pred = torch.cat([xi.view(B, self.no, -1) for xi in raw], 2)
                box_dist, cls = pred.split((self.reg_max * 4, self.nc), 1)
                boxes = self.decode_bboxes(ema.dfl(box_dist), anchors.t().unsqueeze(0)) * strides
            else:
                raw = [torch.cat((self.cv2[i](x_in[i]), self.cv3[i](x_in[i])), 1) for i in range(self.nl)]
                pred = torch.cat([xi.view(B, self.no, -1) for xi in raw], 2)
                box_dist, cls = pred.split((self.reg_max * 4, self.nc), 1)
                boxes = self.decode_bboxes(self.dfl(box_dist), anchors.t().unsqueeze(0)) * strides
            scores = cls.sigmoid()
        return self._select_candidates(boxes, scores, img_size, conf_thr=conf_thr)
