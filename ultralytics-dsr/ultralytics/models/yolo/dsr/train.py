# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DSR-Det trainer (Phase 2/3).

All training runs through the official ``model.train()`` entry point with this
``DetectionTrainer`` subclass; no standalone training script exists in the project.

Phase 2/3 additions (v1.1 14.1 / 18.3):
- ``candidate_source: ema``: polyak-average the base prediction head into a frozen
  EMA copy used for candidate generation (stable-candidate control group);
- candidate stability probe: every ``stability_every`` epochs, candidates are
  extracted on a fixed validation batch and compared (mean max-IoU) against the
  reference probe (``stability_ref_epoch``), logged and saved to
  ``candidate_stability.json`` in the run directory.
"""

from __future__ import annotations

import json
from copy import copy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.modules.dsr import DSRDetect, pairwise_iou
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .model import DSRDetectionModel
from .val import DSRDetectionValidator


class DSRDetectionTrainer(DetectionTrainer):
    """DetectionTrainer subclass that builds DSRDetectionModel and logs refined losses."""

    LOSS_NAMES = ("box_loss", "cls_loss", "dfl_loss", "relation_loss", "preserve_loss", "gicr_loss")

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Register stage-schedule and candidate-stability hooks (v1.1 14.1)."""
        super().__init__(cfg, overrides, _callbacks)
        self._stab_ref = None
        self._stab_history: list[dict] = []
        self.add_callback("on_train_epoch_start", self._on_train_epoch_start_stage)
        self.add_callback("on_train_epoch_end", self._on_train_epoch_end_candidate_stability)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a DSRDetectionModel (official interface, v1.1 16.3)."""
        model = DSRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        head = model.model[-1]
        if isinstance(head, DSRDetect):
            head.ema_reset()  # EMA copy must start from the loaded weights
        return model

    def get_validator(self):
        """Official mAP via DetectionValidator; DSRDetectionValidator only when the
        model yaml requests it (``dsr.validator: true``), keeping the default path
        byte-for-byte identical to the official validator."""
        model_cfg = getattr(getattr(self, "model", None), "yaml", {}) or {}
        use_dsr_val = bool(model_cfg.get("dsr", {}).get("validator", False))
        self.loss_names = self.LOSS_NAMES
        validator_cls = DSRDetectionValidator if use_dsr_val else yolo.detect.DetectionValidator
        return validator_cls(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Label the 6-component loss vector (3 base + 3 refined)."""
        keys = [f"{prefix}/{x}" for x in self.LOSS_NAMES]
        if loss_items is None:
            return keys
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))

    def optimizer_step(self):
        """Official optimizer step, then update the DSR EMA head (if enabled)."""
        super().optimizer_step()
        head = unwrap_model(self.model).model[-1]
        if isinstance(head, DSRDetect) and head.candidate_source == "ema":
            head.ema_update()

    def _model_train(self):
        """v3: keep frozen base BatchNorm layers in eval mode.

        requires_grad=False alone does NOT stop BatchNorm from updating running
        statistics during forward, which silently drifts the frozen base detector
        (measured: -3.3 mAP after 20 epochs). Frozen BN must stay in eval().
        """
        super()._model_train()
        self.keep_frozen_bn_eval(self.model)

    # ------------------------------------------------- Stage B1/B2 schedule
    @staticmethod
    def _param_scope(name: str, n_backbone: int, head_idx: int) -> str:
        """Map a parameter name to backbone | neck | base_detect | dsr | ema | other."""
        parts = name.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            return "other"
        idx = int(parts[1])
        if idx < n_backbone:
            return "backbone"
        if idx < head_idx:
            return "neck"
        if idx == head_idx:
            if parts[2] in ("mdrr", "refine_head"):
                return "dsr"
            if parts[2] == "ema_head":
                return "ema"
            if parts[2] in ("cv2", "cv3", "dfl"):
                return "base_detect"
        return "other"

    @staticmethod
    def apply_freeze(model, scopes, always_frozen=(".dfl", ".ema_head")):
        """Set requires_grad per parameter scope; ``always_frozen`` substrings stay frozen."""
        yaml_cfg = getattr(model, "yaml", {}) or {}
        n_backbone = len(yaml_cfg["backbone"]) if isinstance(yaml_cfg.get("backbone"), list) else 11
        head_idx = len(model.model) - 1
        freeze = set(scopes or [])
        for name, p in model.named_parameters():
            scope = DSRDetectionTrainer._param_scope(name, n_backbone, head_idx)
            frozen = scope in freeze or any(s in name for s in always_frozen)
            p.requires_grad_(not frozen)

    @staticmethod
    def keep_frozen_bn_eval(model):
        """Force BatchNorm layers with frozen parameters into eval mode.

        Without this, ``requires_grad=False`` still lets BN update running stats
        during training forwards, silently degrading a frozen base detector.
        """
        for _, m in unwrap_model(model).named_modules():
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()

    def _scale_lr(self, scale: float) -> None:
        """Scale optimizer learning rates (B2 uses 1/10-1/20 of Stage B1 LR, v1.1 14.1)."""
        for group in self.optimizer.param_groups:
            group["initial_lr"] = group["initial_lr"] * scale
            group["lr"] = group["lr"] * scale

    def _on_train_epoch_start_stage(self, trainer):
        """Apply the Stage B1/B2 freeze + LR schedule at epoch boundaries."""
        model = unwrap_model(self.model)
        head = model.model[-1]
        if not isinstance(head, DSRDetect):
            return
        cfg = head.dsr_cfg
        b1 = max(0, int(cfg.get("stage_b1_epochs", 0)))
        epoch = self.epoch
        head.train_epoch = epoch
        if head.track_uncertainty:
            head.uncertainty_reset_epoch()
        if b1 > 0:
            if epoch == 0:
                scopes = cfg.get("stage_b1_freeze") or ["backbone", "neck", "base_detect"]
                DSRDetectionTrainer.apply_freeze(model, scopes)
                LOGGER.info(f"Stage B1: freezing {scopes} (epoch {epoch})")
            elif epoch == b1:
                # v3: the base detector stays frozen for the whole Stage B so the
                # joint loss can never degrade it; DSR only refines on top.
                scopes = cfg.get("stage_b2_freeze") or ["backbone", "neck", "base_detect"]
                DSRDetectionTrainer.apply_freeze(model, scopes)
                scale = float(cfg.get("stage_b2_lr_scale", 0.1))
                self._scale_lr(scale)
                LOGGER.info(f"Stage B2: freezing {scopes}, LR scaled by {scale} (epoch {epoch})")

    # --------------------------------------------------- candidate stability
    def _stab_probe(self):
        """Extract base-head Top-K candidates on the first validation batch(es).

        Analysis only (v1.1 14.1): the DSR write-back is temporarily disabled so the
        probe measures drift of the base candidate set across epochs.
        """
        head = unwrap_model(self.model).model[-1]
        if not isinstance(head, DSRDetect) or not self.test_loader:
            return None
        n_batches = max(1, int(head.dsr_cfg.get("stability_probe_batches", 1)))
        probe: list[torch.Tensor] = []
        try:
            loader_iter = iter(self.test_loader)
            for _ in range(n_batches):
                batch = self.preprocess_batch(next(loader_iter))
                was_training = self.model.training
                self.model.eval()
                was_active = head.dsr_active
                head.dsr_active = False  # extract base candidates, not refined ones
                try:
                    with torch.no_grad():
                        preds = unwrap_model(self.model)(batch["img"])
                finally:
                    head.dsr_active = was_active
                    self.model.train(was_training)
                y = preds[0] if isinstance(preds, (list, tuple)) else preds
                H, W = batch["img"].shape[2], batch["img"].shape[3]
                if head.candidate_source == "ema" and getattr(head, "neck_feats", None) is not None:
                    # EMA-candidate stability (v1.1 14.1): candidates from the EMA head
                    sel = head.candidates_from_features(head.neck_feats, (int(H), int(W)), source="ema")
                else:
                    sel = head._select_candidates(y[:, :4], y[:, 4:], (int(H), int(W)))
                for bi in range(y.shape[0]):
                    probe.append(sel["boxes_xyxy"][bi][sel["mask"][bi]].detach().cpu())
        except StopIteration:
            pass
        return probe if probe else None

    def _on_train_epoch_end_candidate_stability(self, trainer):
        """Epoch-end hook: uncertainty audit + candidate-stability probe."""
        if RANK not in {-1, 0}:
            return
        head = unwrap_model(self.model).model[-1]
        if not isinstance(head, DSRDetect):
            return
        cfg = head.dsr_cfg
        epoch = self.epoch
        # u-distribution audit (v1.1 7.1), written every epoch
        if head.track_uncertainty and self.save_dir:
            audit = head.uncertainty_audit()
            if audit:
                try:
                    (self.save_dir / "uncertainty_audit.json").write_text(
                        json.dumps({**{"epoch": epoch}, **audit}, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                except Exception:
                    pass
        if not head.track_stability:
            return
        ref_epoch = int(cfg.get("stability_ref_epoch", 0))
        every = max(1, int(cfg.get("stability_every", 5)))
        if epoch % every != 0 and epoch != ref_epoch:
            return
        current = self._stab_probe()
        if current is None:
            return
        if self._stab_ref is None:
            self._stab_ref = current
            iou = 1.0
        else:
            ious = []
            for cur, ref in zip(current, self._stab_ref):
                if cur.numel() == 0 or ref.numel() == 0:
                    ious.append(1.0 if cur.numel() == 0 and ref.numel() == 0 else 0.0)
                else:
                    ious.append(pairwise_iou(cur, ref).max(dim=1).values.mean().item())
            iou = float(np.mean(ious)) if ious else 1.0
        self._stab_history.append({"epoch": epoch, "candidate_iou": round(iou, 4)})
        LOGGER.info(f"candidate stability IoU vs epoch {ref_epoch}: {iou:.4f} (epoch {epoch})")
        if self.save_dir:
            try:
                (self.save_dir / "candidate_stability.json").write_text(
                    json.dumps(self._stab_history, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
