# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DSR-Det validator (Phase 5, v1.1 7.2 / 19.x / 20.3).

Overall mAP is always computed by the official ``DetectionValidator`` parent.
This subclass additionally produces, all inside ``model.val()``:

- difficulty-scene strata (v1.1 19.1-19.4): occlusion (IoA), density, size and
  per-class AP / confusion matrix, each with AP50 / AP50-95 / Recall / missed /
  misclassified counts and image-level bootstrap 95% CI (20.3);
- candidate recall protocol (v1.1 7.2): val-set ``Recall_proposal@0.5`` swept over
  ``scenario_conf_sweep`` (0.01/0.03/0.05/0.10) with the >=95% acceptance note;
- oracle candidate analysis (v1.1 19.5): GT boxes as candidates through MDRR +
  refinement head, compared against predicted-candidate mAP.

Results are written to ``dsr_scenario.json`` in the run directory. No standalone
evaluation script is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.modules.dsr import DSRDetect, cxcywh2xyxy, xyxy2cxcywh
from ultralytics.utils import RANK, nms
from ultralytics.utils.metrics import DetMetrics, ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression


class DSRDetectionValidator(DetectionValidator):
    """Official mAP via parent + DSR scenario statistics (strata/recall/oracle)."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.dsr_records: list[dict[str, Any]] = []
        self.last_base_y: torch.Tensor | None = None
        self.last_x_in: list[torch.Tensor] | None = None
        self._dsr_model = None
        self.oracle_metrics: DetMetrics | None = None
        self._oracle_collected = False

    # ------------------------------------------------------------- model access
    def init_metrics(self, model) -> None:
        """Store the raw model (for the DSR head) and reset scenario state."""
        super().init_metrics(model)
        self._dsr_model = model
        self.dsr_records = []
        self.last_base_y = None
        self.last_x_in = None
        self.oracle_metrics = DetMetrics(names=self.metrics.names)
        self._oracle_collected = False

    def _dsr_head(self) -> DSRDetect | None:
        """Resolve the DSRDetect head through DDP / AutoBackend wrappers."""

        def candidates(x):
            seen = set()
            stack = [x]
            while stack:
                c = stack.pop()
                if c is None or id(c) in seen:
                    continue
                seen.add(id(c))
                yield c
                for attr in ("model", "module"):
                    y = getattr(c, attr, None)
                    if y is not None:
                        stack.append(y)

        for c in candidates(getattr(self, "_dsr_model", None)):
            seq = getattr(c, "model", None)
            if isinstance(seq, torch.nn.Sequential) and len(seq) and isinstance(seq[-1], DSRDetect):
                return seq[-1]
        return None

    # ------------------------------------------------------------ postprocess
    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Run official NMS; remember the base decode + neck features for analysis."""
        self.last_base_y = None
        self.last_x_in = None
        head = self._dsr_head()
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            y, x_raw = preds
            self.last_x_in = getattr(head, "neck_feats", None) or x_raw
            if head is not None and getattr(head, "base_y", None) is not None:
                self.last_base_y = head.base_y
            preds = y
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    # ------------------------------------------------------------- update loop
    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Official mAP update + DSR scenario collection (strata/recall/oracle)."""
        super().update_metrics(preds, batch)
        if RANK in {-1, 0}:
            self._collect_scenario(preds, batch)

    def _collect_scenario(self, preds, batch) -> None:
        """Collect per-image records, candidate recall and oracle predictions."""
        head = self._dsr_head()
        cfg = head.dsr_cfg if head is not None else {}
        sweep = [float(x) for x in cfg.get("scenario_conf_sweep", [0.01, 0.03, 0.05, 0.10])]
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            gt_cls, gt_boxes = pbatch["cls"], pbatch["bboxes"]
            h, w = pbatch["imgsz"]
            n_gt = gt_boxes.shape[0]
            ioa = torch.zeros(n_gt, device=gt_boxes.device) if n_gt else gt_boxes.new_zeros(0)
            area_ratio = gt_boxes.new_zeros(0)
            if n_gt:
                ioa = self._gt_ioa(gt_boxes)
                wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=0)
                area_ratio = (wh[:, 0] * wh[:, 1]) / max(1, h * w)
            det_conf = predn["conf"] if predn["conf"].numel() else predn["conf"].new_zeros(0)
            det_cls = predn["cls"] if predn["cls"].numel() else predn["cls"].new_zeros(0)
            det_boxes = predn["bboxes"] if predn["bboxes"].numel() else predn["bboxes"].new_zeros(0, 4)
            # relation statistics from the current batch's MDRR aux (19.5)
            relation = None
            aux = getattr(head, "last_aux", None)
            cm = getattr(head, "last_cand_mask", None)
            if aux is not None and cm is not None and cm[si].any():
                scale_names = getattr(head.mdrr, "scale_names", ["g", "r", "n"])
                sw = aux["scale_weights"][si][cm[si]].mean(dim=0)
                relation = {
                    "scale_weights": {name: round(float(v), 4) for name, v in zip(scale_names, sw.tolist())},
                    "gate_mean": round(float(aux["gate"][si][cm[si]].mean()), 4),
                    "uncertainty_mean": round(float(aux["uncertainty"][si][cm[si]].mean()), 4),
                }
            # base (pre-refinement) detections for DSR error attribution (19.5)
            base_dets = None
            if self.last_base_y is not None:
                with torch.no_grad():
                    b = non_max_suppression(
                        self.last_base_y[si : si + 1],
                        self.args.conf,
                        self.args.iou,
                        max_det=self.args.max_det,
                    )[0]
                if b.shape[0]:
                    base_dets = {
                        "boxes": b[:, :4].cpu().numpy(),
                        "conf": b[:, 4].cpu().numpy(),
                        "cls": b[:, 5].cpu().numpy(),
                    }
            rec = {
                "gt_cls": gt_cls.cpu().numpy(),
                "gt_boxes": gt_boxes.cpu().numpy(),
                "gt_ioa": ioa.cpu().numpy(),
                "gt_area_ratio": area_ratio.cpu().numpy(),
                "n_gt": n_gt,
                "det_conf": det_conf.cpu().numpy(),
                "det_cls": det_cls.cpu().numpy(),
                "det_boxes": det_boxes.cpu().numpy(),
                "imgsz": (int(h), int(w)),
                "im_file": str(pbatch["im_file"]),
                "recall_sweep": {},
                "relation": relation,
                "base_dets": base_dets,
            }
            if n_gt and head is not None and self.last_base_y is not None:
                for c in sweep:
                    sel = head._select_candidates(
                        self.last_base_y[:, :4], self.last_base_y[:, 4:], (int(h), int(w)), conf_thr=c
                    )
                    cand = sel["boxes_xyxy"][si][sel["mask"][si]]
                    covered = 0
                    if cand.shape[0]:
                        iou = box_iou(gt_boxes, cand)
                        covered = int((iou.max(dim=1).values >= 0.5).sum())
                    rec["recall_sweep"][str(c)] = {"covered_gt": covered, "n_gt": n_gt}
            self.dsr_records.append(rec)

        # Oracle: GT boxes as candidates through MDRR + refinement head (19.5)
        if (
            head is not None
            and cfg.get("oracle", True)
            and self.last_base_y is not None
            and self.last_x_in is not None
        ):
            self._collect_oracle(head, batch)

    # ------------------------------------------------------------------ oracle
    def _collect_oracle(self, head: DSRDetect, batch) -> None:
        """Build GT-candidate predictions and accumulate oracle metrics."""
        yb = self.last_base_y  # [B, 4+nc, N] cxcywh + sigmoid scores
        B, N = yb.shape[0], yb.shape[1]
        device = yb.device
        nc = self.nc
        max_gt = 1
        gt_per_image = []
        for si in range(B):
            pbatch = self._prepare_batch(si, batch)
            gt_per_image.append(pbatch["bboxes"])
            max_gt = max(max_gt, pbatch["bboxes"].shape[0])
        boxes = yb.new_zeros(B, max_gt, 4)
        mask = torch.zeros(B, max_gt, dtype=torch.bool, device=device)
        pos = torch.zeros(B, max_gt, dtype=torch.long, device=device)
        base_xyxy = cxcywh2xyxy(yb[:, :4].permute(0, 2, 1))  # [B, N, 4]
        for si in range(B):
            gt = gt_per_image[si]
            k = gt.shape[0]
            if k == 0:
                continue
            boxes[si, :k] = gt
            mask[si, :k] = True
            iou = box_iou(gt, base_xyxy[si])  # [k, N]
            order = iou.max(dim=1).values.argsort(descending=True)
            used = torch.zeros(N, dtype=torch.bool, device=device)
            for g in order:
                cand = torch.nonzero(~used).squeeze(-1)
                if cand.numel() == 0:
                    break
                best = iou[g, cand].argmax()
                p = cand[best]
                used[p] = True
                pos[si, g] = p
        if not mask.any():
            return
        base_probs = yb[:, 4:].permute(0, 2, 1).gather(1, pos.unsqueeze(-1).expand(-1, -1, nc)).clamp(1e-4, 1 - 1e-4)
        base_logits = torch.logit(base_probs)
        base_conf = base_probs.max(dim=-1).values
        h, w = batch["img"].shape[2:]
        with torch.no_grad():
            f_tilde, aux = head.mdrr(self.last_x_in, boxes, mask, base_conf, (int(h), int(w)), base_probs)
            # v2: classification-only residual; oracle diagnoses relation effectiveness
            # on GT candidates, so all GT boxes pass through the refinement branch.
            refined_logits = head.refine_head(f_tilde, base_logits)
        refined_scores = refined_logits.sigmoid()
        oracle_y = yb.clone()
        for si in range(B):
            k = int(mask[si].sum())
            if k == 0:
                continue
            p = pos[si, :k]
            oracle_y[si, 4:, p] = refined_scores[si, :k].t()  # v2: classification only
        outputs = nms.non_max_suppression(
            oracle_y,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        for si, x in enumerate(outputs):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred({"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]})
            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.oracle_metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
        self._oracle_collected = True

    @staticmethod
    def _gt_ioa(gt_boxes: torch.Tensor) -> torch.Tensor:
        """Per-GT max overlap ratio |b_i ∩ b_j| / |b_i| over other GT boxes (19.1)."""
        x1, y1, x2, y2 = gt_boxes.unbind(1)
        area = ((x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)).clamp(min=1e-6)
        inter_w = (torch.minimum(x2[:, None], x2[None, :]) - torch.maximum(x1[:, None], x1[None, :])).clamp(min=0)
        inter_h = (torch.minimum(y2[:, None], y2[None, :]) - torch.maximum(y1[:, None], y1[None, :])).clamp(min=0)
        inter = inter_w * inter_h
        ioa = inter / area[:, None]
        ioa.fill_diagonal_(0)
        return ioa.max(dim=1).values

    # --------------------------------------------------------------- finalize
    def finalize_metrics(self) -> None:
        """Official finalize + DSR scenario report."""
        super().finalize_metrics()
        if not self.dsr_records or RANK not in {-1, 0}:
            return
        if self.oracle_metrics is not None and self._oracle_collected:
            self.oracle_metrics.process(save_dir=self.save_dir or Path("."))
        report = self._build_scenario_report()
        if self.save_dir:
            try:
                (self.save_dir / "dsr_scenario.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                (self.save_dir / "per_image_metrics.json").write_text(
                    json.dumps(self._per_image_metrics(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass

    def _per_image_metrics(self) -> dict[str, Any]:
        """Per-image AP50-95 / Recall@0.5 for paired image-level bootstrap (20.3)."""
        rows = []
        for rec in self.dsr_records:
            gt_cls, gt_boxes = rec["gt_cls"], rec["gt_boxes"]
            det_conf, det_cls, det_boxes = rec["det_conf"], rec["det_cls"], rec["det_boxes"]
            n_gt = int(rec["n_gt"])
            if n_gt == 0 or det_boxes.shape[0] == 0:
                rows.append(
                    {"im_file": rec["im_file"], "n_gt": n_gt, "ap50_95": 0.0, "recall05": 0.0 if n_gt else 1.0}
                )
                continue
            iou = box_iou(torch.from_numpy(gt_boxes), torch.from_numpy(det_boxes))
            correct_class = gt_cls[:, None] == det_cls[None, :]
            tp = self.match_predictions(
                torch.from_numpy(det_cls), torch.from_numpy(gt_cls), iou
            ).cpu().numpy()
            matched = ((iou.cpu().numpy() * correct_class) >= 0.5).any(axis=1)
            ap = ap_per_class(tp, det_conf, det_cls, gt_cls, names=self.metrics.names)[5]
            rows.append(
                {
                    "im_file": rec["im_file"],
                    "n_gt": n_gt,
                    "ap50_95": round(float(ap.mean()), 5),
                    "recall05": round(float(matched.sum()) / n_gt, 5),
                }
            )
        return {
            "note": "per-image AP50-95/Recall@0.5 from the official matching; "
            "paired image-level bootstrap in scripts/summarize_runs.py --compare",
            "images": rows,
        }

    def _build_scenario_report(self) -> dict[str, Any]:
        """Assemble the full dsr_scenario.json report."""
        head = self._dsr_head()
        cfg = head.dsr_cfg if head is not None else {}
        names = self.metrics.names
        areas = np.concatenate([r["gt_area_ratio"] for r in self.dsr_records if r["gt_area_ratio"].size])
        q1, q2 = (np.quantile(areas, [1 / 3, 2 / 3]).tolist() if areas.size else [0.0, 1.0])
        report: dict[str, Any] = {
            "images_seen": self.seen,
            "names": names,
            "size_quantiles": [round(float(q1), 5), round(float(q2), 5)],
            "note_size": "size strata use val-set area-ratio quantiles",
        }

        # Candidate recall protocol (7.2)
        recall = {}
        for r in self.dsr_records:
            for c, v in r["recall_sweep"].items():
                entry = recall.setdefault(c, {"covered_gt": 0, "n_gt": 0})
                entry["covered_gt"] += v["covered_gt"]
                entry["n_gt"] += v["n_gt"]
        report["candidate_recall"] = {
            c: {
                "recall_proposal@0.5": round(v["covered_gt"] / v["n_gt"], 4) if v["n_gt"] else None,
                "covered_gt": v["covered_gt"],
                "n_gt": v["n_gt"],
            }
            for c, v in recall.items()
        }
        report["candidate_recall_note"] = "acceptance gate: >= 0.95 at the chosen candidate_conf (v1.1 7.2)"

        # Strata (19.1-19.4)
        occlusion = {
            "无遮挡": lambda m: m < 0.10,
            "轻微遮挡": lambda m: (m >= 0.10) & (m < 0.30),
            "中等遮挡": lambda m: (m >= 0.30) & (m < 0.50),
            "严重遮挡": lambda m: m >= 0.50,
        }
        density = {
            "低密度": lambda m, n: n <= 10,
            "中密度": lambda m, n: (10 < n) & (n <= 30),
            "高密度": lambda m, n: n > 30,
        }
        size = {
            "小目标": lambda m, q1=q1, q2=q2: m < q1,
            "中目标": lambda m, q1=q1, q2=q2: (m >= q1) & (m < q2),
            "大目标": lambda m, q1=q1, q2=q2: m >= q2,
        }
        strata: dict[str, Any] = {"all": self._stratum_metrics(lambda r: np.ones(r["n_gt"], dtype=bool))}
        strata["occlusion"] = {k: self._stratum_metrics(lambda r, f=f: f(r["gt_ioa"])) for k, f in occlusion.items()}
        strata["density"] = {
            k: self._stratum_metrics(lambda r, f=f: np.ones(r["n_gt"], dtype=bool) if f(None, r["n_gt"]) else np.zeros(r["n_gt"], dtype=bool))
            for k, f in density.items()
        }
        strata["size"] = {k: self._stratum_metrics(lambda r, f=f: f(r["gt_area_ratio"])) for k, f in size.items()}
        report["strata"] = strata

        # Relation effectiveness (19.5): scale weights / gate / uncertainty
        rel_rows = [r["relation"] for r in self.dsr_records if r["relation"] is not None]
        if rel_rows:
            scale_names = list(rel_rows[0]["scale_weights"].keys())
            relation_stats: dict[str, Any] = {
                "scale_weights_mean": {
                    name: round(float(np.mean([r["scale_weights"][name] for r in rel_rows])), 4)
                    for name in scale_names
                },
                "gate_mean": round(float(np.mean([r["gate_mean"] for r in rel_rows])), 4),
                "uncertainty_mean": round(float(np.mean([r["uncertainty_mean"] for r in rel_rows])), 4),
                "n_images_with_candidates": len(rel_rows),
            }
            dens: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
            for rec in self.dsr_records:
                if rec["relation"] is None:
                    continue
                n = rec["n_gt"]
                bucket = "low" if n <= 10 else ("mid" if n <= 30 else "high")
                dens[bucket].append(rec["relation"]["gate_mean"])
            relation_stats["gate_mean_by_density"] = {
                k: round(float(np.mean(v)), 4) for k, v in dens.items() if v
            }
            report["relation_stats"] = relation_stats
        else:
            report["relation_stats"] = {"note": "no candidates found during validation"}

        # DSR error attribution: base vs refined (19.5)
        report["base_vs_refined"] = self._base_vs_refined()
        report["recall_relation"] = {
            "final_recall@0.5": report["strata"].get("all", {}).get("Recall@0.5"),
            "note": "candidate_recall (7.2) bounds achievable recall; final recall comes from the refined detector",
        }

        # Oracle (19.5)
        orr = self.oracle_metrics.results_dict if self.oracle_metrics is not None else {}
        report["oracle"] = {
            "note": "GT boxes as candidates through MDRR + refinement head; predicted-candidate mAP is the official result",
            "mAP50-95": orr.get("metrics/mAP50-95(B)"),
            "mAP50": orr.get("metrics/mAP50(B)"),
            "recall": orr.get("metrics/recall(B)"),
            "precision": orr.get("metrics/precision(B)"),
        }

        # Confusion matrix (19.4)
        try:
            cm = np.asarray(self.confusion_matrix.matrix)
            if cm.size:
                cmn = cm / cm.sum(1, keepdims=True).clip(min=1e-9)
                report["confusion_matrix_normalized"] = cmn.round(4).tolist()
        except Exception:
            pass

        bootstrap_iters = int(cfg.get("bootstrap_iters", 200))
        report["bootstrap"] = {
            "iters": bootstrap_iters if (not self.training and cfg.get("scenario_bootstrap", True)) else 0,
            "note": "image-level 95% CI; run standalone model.val() to enable",
        }
        return report

    def _base_vs_refined(self) -> dict[str, Any]:
        """v2 Fix/Harm/NetGain attribution between base and refined detections (v2 13)."""
        out = {"n_gt": 0, "n_base_tp": 0, "n_ref_tp": 0,
               "n_fix": 0, "n_harm": 0, "n_kept": 0, "n_added": 0, "n_removed": 0}
        for rec in self.dsr_records:
            gt_cls, gt_boxes = rec["gt_cls"], rec["gt_boxes"]
            det_boxes, det_cls = rec["det_boxes"], rec["det_cls"]
            base = rec["base_dets"]
            if rec["n_gt"] == 0:
                continue
            n = rec["n_gt"]
            out["n_gt"] += n
            matched_ref = np.zeros(n, dtype=bool)
            if det_boxes.shape[0]:
                iou = box_iou(torch.from_numpy(gt_boxes), torch.from_numpy(det_boxes))
                correct = gt_cls[:, None] == det_cls[None, :]
                tp = self.match_predictions(torch.from_numpy(det_cls), torch.from_numpy(gt_cls), iou)[:, 0]
                out["n_ref_tp"] += int(tp.sum())
                matched = ((iou.cpu().numpy() * correct) >= 0.5).any(axis=1)
                matched_ref = matched
            matched_base = np.zeros(n, dtype=bool)
            if base is not None and base["boxes"].shape[0]:
                iou_b = box_iou(torch.from_numpy(gt_boxes), torch.from_numpy(base["boxes"]))
                tp_b = self.match_predictions(torch.from_numpy(base["cls"]), torch.from_numpy(gt_cls), iou_b)[:, 0]
                out["n_base_tp"] += int(tp_b.sum())
                matched_b = ((iou_b.cpu().numpy() * (gt_cls[:, None] == base["cls"][None, :])) >= 0.5).any(axis=1)
                matched_base = matched_b
                if det_boxes.shape[0]:
                    pair = box_iou(torch.from_numpy(base["boxes"]), torch.from_numpy(det_boxes)) >= 0.5
                    out["n_kept"] += int(pair.any(dim=0).sum())
                    out["n_added"] += det_boxes.shape[0] - int(pair.any(dim=0).sum())
                    out["n_removed"] += base["boxes"].shape[0] - int(pair.any(dim=1).sum())
            out["n_fix"] += int((matched_ref & ~matched_base).sum())
            out["n_harm"] += int((matched_base & ~matched_ref).sum())
        out["correction_rate"] = round(out["n_fix"] / max(out["n_gt"], 1), 4)
        out["harm_rate"] = round(out["n_harm"] / max(out["n_gt"], 1), 4)
        out["net_gain"] = out["n_fix"] - out["n_harm"]
        out["note"] = ("fix = GT missed by base but matched by refined; "
                       "harm = GT matched by base but missed by refined; net_gain = fix - harm")
        return out

    def _stratum_metrics(self, mask_fn) -> dict[str, Any]:
        """Per-stratum AP50/AP50-95/Recall/missed/misclassified + bootstrap CI."""
        tp_all, conf_all, pred_all, tgt_all = [], [], [], []
        n_gt = n_match = n_wrong = n_images = 0
        rows = []
        for rec in self.dsr_records:
            mask = mask_fn(rec)
            if mask.sum() == 0:
                continue
            n_images += 1
            gt_cls = rec["gt_cls"][mask]
            gt_boxes = rec["gt_boxes"][mask]
            det_conf = rec["det_conf"]
            det_cls = rec["det_cls"]
            det_boxes = rec["det_boxes"]
            img_n_gt = int(mask.sum())
            n_gt += img_n_gt
            img_match = img_wrong = 0
            if det_boxes.shape[0] and gt_boxes.shape[0]:
                gt_t = torch.from_numpy(gt_boxes)
                det_t = torch.from_numpy(det_boxes)
                iou = box_iou(gt_t, det_t)
                correct_class = torch.from_numpy(gt_cls[:, None] == det_cls[None, :])
                tp = self.match_predictions(
                    torch.from_numpy(det_cls), torch.from_numpy(gt_cls), iou
                ).cpu().numpy()
                matched_gt = ((iou * correct_class) >= 0.5).any(dim=1)
                wrong_gt = ((iou * ~correct_class) >= 0.5).any(dim=1) & ~matched_gt
                img_match = int(matched_gt.sum())
                img_wrong = int(wrong_gt.sum())
            else:
                tp = np.zeros((det_boxes.shape[0], self.niou), dtype=bool)
            n_match += img_match
            n_wrong += img_wrong
            tp_all.append(tp)
            conf_all.append(det_conf)
            pred_all.append(det_cls)
            tgt_all.append(gt_cls)
            rows.append(
                {
                    "tp": tp,
                    "conf": det_conf,
                    "pred": det_cls,
                    "tgt": gt_cls,
                    "n_gt": img_n_gt,
                    "n_match": img_match,
                }
            )
        out: dict[str, Any] = {"n_gt": n_gt, "n_images": n_images}
        if n_gt == 0:
            return out
        tp_c = np.concatenate(tp_all, 0)
        conf_c = np.concatenate(conf_all, 0)
        pred_c = np.concatenate(pred_all, 0)
        tgt_c = np.concatenate(tgt_all, 0)
        ap, unique_classes = ap_per_class(tp_c, conf_c, pred_c, tgt_c, names=self.metrics.names)[5:7]
        out["AP50-95"] = round(float(ap.mean()), 4)
        out["AP50"] = round(float(ap[:, 0].mean()), 4)
        out["Recall@0.5"] = round(n_match / n_gt, 4)
        out["missed"] = int(n_gt - n_match)
        out["misclassified"] = int(n_wrong)
        out["per_class_AP50-95"] = {
            str(self.metrics.names.get(int(c), int(c))): round(float(ap[i].mean()), 4)
            for i, c in enumerate(unique_classes)
        }
        # image-level bootstrap 95% CI (20.3)
        head = self._dsr_head()
        cfg = head.dsr_cfg if head is not None else {}
        iters = int(cfg.get("bootstrap_iters", 200))
        if iters > 0 and not self.training and cfg.get("scenario_bootstrap", True) and len(rows) >= 8:
            out["bootstrap_CI_AP50-95"], out["bootstrap_CI_recall"] = self._bootstrap_ci(rows, iters)
        return out

    def _bootstrap_ci(self, rows, iters):
        """Image-level bootstrap 95% CI for mAP50-95 and Recall@0.5."""
        rng = np.random.default_rng(0)
        maps, recalls = [], []
        n_rows = len(rows)
        for _ in range(iters):
            idx = rng.integers(0, n_rows, size=n_rows)
            tp = np.concatenate([rows[i]["tp"] for i in idx], 0)
            conf = np.concatenate([rows[i]["conf"] for i in idx], 0)
            pred = np.concatenate([rows[i]["pred"] for i in idx], 0)
            tgt = np.concatenate([rows[i]["tgt"] for i in idx], 0)
            n_gt = sum(rows[i]["n_gt"] for i in idx)
            n_match = sum(rows[i]["n_match"] for i in idx)
            if tp.shape[0] == 0 or n_gt == 0:
                maps.append(0.0)
                recalls.append(0.0 if n_gt else 1.0)
                continue
            ap = ap_per_class(tp, conf, pred, tgt, names=self.metrics.names)[5]
            maps.append(float(ap.mean()))
            recalls.append(n_match / n_gt)
        ci = lambda xs: [round(float(np.percentile(xs, 2.5)), 4), round(float(np.percentile(xs, 97.5)), 4)]
        return ci(maps), ci(recalls)
