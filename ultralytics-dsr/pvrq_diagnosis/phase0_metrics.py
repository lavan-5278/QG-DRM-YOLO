"""Deterministic metrics for the falsifiable PVR-Q Phase 0 gate.

Ground-truth localization records and post-NMS detection records are kept
separate on purpose.  Unmatched GTs participate in localization diagnostics,
whereas calibration and ranking contain detections only, including every FP.
"""

from __future__ import annotations

from typing import Any

import numpy as np


IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def _boxes(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float64)
    if values.size == 0:
        return values.reshape(0, 4)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if not np.isfinite(values).all():
        raise ValueError("boxes must be finite")
    return values


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise IoU of ``[N,4]`` and ``[M,4]`` xyxy boxes."""
    a_values, b_values = _boxes(a), _boxes(b)
    if len(a_values) == 0 or len(b_values) == 0:
        return np.empty((len(a_values), len(b_values)), dtype=np.float64)
    left_top = np.maximum(a_values[:, None, :2], b_values[None, :, :2])
    right_bottom = np.minimum(a_values[:, None, 2:], b_values[None, :, 2:])
    intersection_wh = np.clip(right_bottom - left_top, 0.0, None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = np.prod(np.clip(a_values[:, 2:] - a_values[:, :2], 0.0, None), axis=1)
    area_b = np.prod(np.clip(b_values[:, 2:] - b_values[:, :2], 0.0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def compute_competition(
    boxes: np.ndarray, radius: float = 2.0, beta: float = 1.0, n_ref: int = 4
) -> np.ndarray:
    """Return ``1-exp(-(max IoA + beta*nearby/n_ref))`` for every GT box."""
    values = _boxes(boxes)
    if radius < 0 or n_ref <= 0:
        raise ValueError("radius must be non-negative and n_ref must be positive")
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    widths_heights = np.clip(values[:, 2:] - values[:, :2], 0.0, None)
    areas = widths_heights[:, 0] * widths_heights[:, 1]
    left_top = np.maximum(values[:, None, :2], values[None, :, :2])
    right_bottom = np.minimum(values[:, None, 2:], values[None, :, 2:])
    overlap_wh = np.clip(right_bottom - left_top, 0.0, None)
    intersections = overlap_wh[..., 0] * overlap_wh[..., 1]
    ioa = np.divide(intersections, areas[:, None], out=np.zeros_like(intersections), where=areas[:, None] > 0)
    np.fill_diagonal(ioa, 0.0)
    centres = (values[:, :2] + values[:, 2:]) / 2.0
    distances = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    nearby = distances < (radius * np.sqrt(areas))[:, None]
    np.fill_diagonal(nearby, False)
    raw = ioa.max(axis=1) + beta * nearby.sum(axis=1) / n_ref
    result = 1.0 - np.exp(-np.clip(raw, 0.0, 700.0))
    return np.minimum(result, np.nextafter(1.0, 0.0))


def assign_detection_quality(
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign every detection an IoU quality using confidence-ordered matching.

    Matching is class-aware and one-to-one.  A prediction that cannot claim an
    overlapping same-class GT remains in the returned detection list with
    ``quality_iou=0``.  Unmatched GTs appear only in the GT-side return value;
    they are never manufactured into zero-confidence calibration samples.
    """
    gt_boxes, pred_boxes = _boxes(gt_xyxy), _boxes(pred_xyxy)
    gt_classes = np.asarray(gt_cls).reshape(-1)
    pred_classes = np.asarray(pred_cls).reshape(-1)
    confidences = np.asarray(pred_conf, dtype=np.float64).reshape(-1)
    if len(gt_classes) != len(gt_boxes) or len(pred_classes) != len(pred_boxes) or len(confidences) != len(pred_boxes):
        raise ValueError("class and confidence arrays must align with their boxes")
    if not np.isfinite(confidences).all():
        raise ValueError("prediction confidences must be finite")

    gt_matches = [
        {"gt_index": index, "matched_pred_index": None, "matched_iou": 0.0, "matched_conf": 0.0}
        for index in range(len(gt_boxes))
    ]
    detections = [
        {"pred_index": index, "matched_gt_index": None, "quality_iou": 0.0}
        for index in range(len(pred_boxes))
    ]
    available = np.ones(len(gt_boxes), dtype=bool)
    ious = box_iou_xyxy(pred_boxes, gt_boxes)
    order = sorted(range(len(pred_boxes)), key=lambda index: (-confidences[index], index))
    for pred_index in order:
        candidates = np.flatnonzero(available & (gt_classes == pred_classes[pred_index]))
        if candidates.size == 0:
            continue
        candidate_ious = ious[pred_index, candidates]
        position = int(np.argmax(candidate_ious))
        quality = float(candidate_ious[position])
        if quality <= 0.0:
            continue
        gt_index = int(candidates[position])
        available[gt_index] = False
        gt_matches[gt_index] = {
            "gt_index": gt_index,
            "matched_pred_index": pred_index,
            "matched_iou": quality,
            "matched_conf": float(confidences[pred_index]),
        }
        detections[pred_index] = {
            "pred_index": pred_index,
            "matched_gt_index": gt_index,
            "quality_iou": quality,
        }
    return gt_matches, detections


def match_instances(
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    pred_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
) -> list[dict[str, Any]]:
    """Backward-compatible GT-side view of :func:`assign_detection_quality`."""
    return assign_detection_quality(gt_xyxy, gt_cls, pred_xyxy, pred_cls, pred_conf)[0]


def bootstrap_mean_diff(
    focus: np.ndarray, reference: np.ndarray, n_boot: int = 2000, seed: int = 3407
) -> dict[str, float]:
    """Bootstrap mean(focus)-mean(reference); retained for small unit diagnostics."""
    focus_values = np.asarray(focus, dtype=np.float64).reshape(-1)
    reference_values = np.asarray(reference, dtype=np.float64).reshape(-1)
    if len(focus_values) == 0 or len(reference_values) == 0 or n_boot <= 0:
        raise ValueError("focus, reference, and n_boot must all be non-empty/positive")
    if not np.isfinite(focus_values).all() or not np.isfinite(reference_values).all():
        raise ValueError("bootstrap inputs must be finite")
    rng = np.random.default_rng(seed)
    focus_indices = rng.integers(0, len(focus_values), size=(n_boot, len(focus_values)))
    reference_indices = rng.integers(0, len(reference_values), size=(n_boot, len(reference_values)))
    differences = focus_values[focus_indices].mean(axis=1) - reference_values[reference_indices].mean(axis=1)
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"mean": float(focus_values.mean() - reference_values.mean()), "ci95_low": float(low), "ci95_high": float(high)}


def _average_precision_101(quality: np.ndarray, score: np.ndarray, gt_count: int, threshold: float) -> float:
    """Return a fixed-match 101-point AP proxy including every prediction."""
    if gt_count <= 0 or len(score) == 0:
        return 0.0
    # Stable sort preserves record order for exact score ties.
    order = np.argsort(-score, kind="mergesort")
    correct = (quality[order] >= threshold).astype(np.float64)
    tp = np.cumsum(correct)
    fp = np.cumsum(1.0 - correct)
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, 1.0)
    precision = np.concatenate(([1.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    samples = np.interp(np.linspace(0.0, 1.0, 101), recall, precision)
    return float(np.trapz(samples, np.linspace(0.0, 1.0, 101)))


def fixed_match_ranking_summary(detections: list[dict[str, Any]], gt_count: int) -> dict[str, Any]:
    """Measure confidence ranking and perfect-IoU reranking on fixed matches.

    This is deliberately named a ranking proxy, not detector AP: one-to-one
    matches are frozen before reranking.  It isolates the maximum headroom of a
    score that perfectly predicts the assigned localization quality.
    """
    quality = np.asarray([float(row["quality_iou"]) for row in detections], dtype=np.float64)
    confidence = np.asarray([float(row["conf"]) for row in detections], dtype=np.float64)
    confidence_scores, oracle_scores = {}, {}
    for threshold in IOU_THRESHOLDS:
        confidence_scores[threshold] = _average_precision_101(quality, confidence, gt_count, threshold)
        # Quality is the oracle signal; an infinitesimal confidence term only
        # makes equal-quality ordering deterministic without changing ranks.
        oracle = quality + confidence * 1e-12
        oracle_scores[threshold] = _average_precision_101(quality, oracle, gt_count, threshold)
    baseline = float(np.mean(list(confidence_scores.values())))
    oracle = float(np.mean(list(oracle_scores.values())))
    return {
        "metric": "fixed_match_AP_101 (diagnostic ranking proxy, not detector AP)",
        "n_predictions": len(detections),
        "n_unmatched_predictions": int(np.sum(quality == 0.0)),
        "gt_count": int(gt_count),
        "confidence_ap50": confidence_scores[0.5],
        "confidence_ap75": confidence_scores[0.75],
        "confidence_ap50_95": baseline,
        "oracle_ap50": oracle_scores[0.5],
        "oracle_ap75": oracle_scores[0.75],
        "oracle_ap50_95": oracle,
        "oracle_gain_ap50_95": oracle - baseline,
    }


def calibration_summary(records: list[dict[str, Any]], n_bins: int = 10) -> dict[str, Any]:
    """Summarize confidence-to-quality calibration over detections only."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    confidences = np.asarray([float(row.get("conf", row.get("matched_conf", 0.0))) for row in records])
    qualities = np.asarray([float(row.get("quality_iou", row.get("matched_iou", 0.0))) for row in records])
    if len(confidences) == 0:
        return {"n_predictions": 0, "brier": 0.0, "ece": 0.0, "bins": []}
    bins, ece = [], 0.0
    for index in range(n_bins):
        lo, hi = index / n_bins, (index + 1) / n_bins
        mask = (confidences >= lo) & ((confidences <= hi) if index == n_bins - 1 else (confidences < hi))
        count = int(mask.sum())
        mean_conf = float(confidences[mask].mean()) if count else 0.0
        mean_quality = float(qualities[mask].mean()) if count else 0.0
        if count:
            ece += count / len(confidences) * abs(mean_conf - mean_quality)
        bins.append({"lo": float(lo), "hi": float(hi), "n": count, "mean_conf": mean_conf, "mean_quality": mean_quality})
    return {
        "n_predictions": len(records),
        "n_unmatched_predictions": int(np.sum(qualities == 0.0)),
        "mean_conf_minus_quality": float(np.mean(confidences - qualities)),
        # Compatibility alias; this still contains detection rows only.
        "mean_conf_minus_iou": float(np.mean(confidences - qualities)),
        "brier": float(np.mean((confidences - qualities) ** 2)),
        "ece": float(ece),
        "bins": bins,
    }


def _matched_score_at_iou(records: list[dict[str, Any]], threshold: float) -> float:
    if not records:
        return 0.0
    ordered = sorted(records, key=lambda row: -float(row.get("matched_conf", 0.0)))
    correct = np.asarray([float(row.get("matched_iou", 0.0)) >= threshold for row in ordered], dtype=np.float64)
    tp, fp = np.cumsum(correct), np.cumsum(1.0 - correct)
    recall, precision = tp / len(records), tp / np.maximum(tp + fp, 1.0)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.mean([envelope[recall >= point].max() if np.any(recall >= point) else 0.0 for point in np.linspace(0, 1, 11)]))


def compute_group_matched_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    """Legacy per-GT score used only for localization strata, never for G0."""
    values = {threshold: _matched_score_at_iou(records, threshold) for threshold in IOU_THRESHOLDS}
    return {
        "matched_score50": values[0.5],
        "matched_score75": values[0.75],
        "matched_score50_95": float(np.mean(list(values.values()))),
    }


def _finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return 0.0

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranked, sorted_values = np.empty(len(values), dtype=np.float64), values[order]
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1
            ranked[order[start:end]] = (start + 1 + end) / 2.0
            start = end
        return ranked

    a, b = ranks(left), ranks(right)
    a, b = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    return 0.0 if denominator == 0 else float(np.sum(a * b) / denominator)


def _subset_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "mean_matched_iou": float(np.mean([row["matched_iou"] for row in records])) if records else 0.0,
        **compute_group_matched_scores(records),
    }


def _in_bin(values: np.ndarray, lo: float, hi: float, final: bool) -> np.ndarray:
    return (values >= lo) & ((values <= hi) if final else (values < hi))


def gate_g0(
    iou_diff: dict[str, float],
    oracle_gain: dict[str, float],
    vertical_area_spearman: float,
    min_oracle_gain: float = 0.0,
) -> dict[str, object]:
    """Choose full PVR-Q, localization-only PVRF, or stop."""
    localization_deficit = float(iou_diff["ci95_high"]) < 0.0
    quality_headroom = float(oracle_gain["ci95_low"]) > float(min_oracle_gain)
    lpfield = abs(float(vertical_area_spearman)) >= 0.20
    if not localization_deficit:
        decision = "STOP"
    elif quality_headroom:
        decision = "PROCEED_PVRQ"
    else:
        decision = "PROCEED_PVRF_ONLY"
    return {
        "decision": decision,
        "lpfield": lpfield,
        "localization_deficit": localization_deficit,
        "quality_headroom": quality_headroom,
    }


def _cluster_bootstrap(
    gt_records: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    focus: np.ndarray,
    image_ids: list[str],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Jointly image-cluster bootstrap localization deficit and reranking gain."""
    if n_boot <= 0 or not image_ids:
        raise ValueError("cluster bootstrap requires positive n_boot and image IDs")
    gt_groups = {image_id: [] for image_id in image_ids}
    det_groups = {image_id: [] for image_id in image_ids}
    for index, row in enumerate(gt_records):
        gt_groups[str(row["image_id"])].append(index)
    for row in detections:
        det_groups[str(row["image_id"])].append(row)
    ious = np.asarray([float(row["matched_iou"]) for row in gt_records])
    reference = ~focus
    observed_diff = float(ious[focus].mean() - ious[reference].mean())
    observed_gain = fixed_match_ranking_summary(detections, len(gt_records))["oracle_gain_ap50_95"]
    rng = np.random.default_rng(seed)
    differences, gains, attempts = [], [], 0
    while len(differences) < n_boot:
        attempts += 1
        if attempts > n_boot * 100:
            raise ValueError("unable to draw bootstrap samples containing both localization groups")
        sampled_ids = [image_ids[index] for index in rng.integers(0, len(image_ids), size=len(image_ids))]
        gt_indices = [index for image_id in sampled_ids for index in gt_groups[image_id]]
        if not gt_indices:
            continue
        selected_focus = focus[gt_indices]
        if not np.any(selected_focus) or np.all(selected_focus):
            continue
        selected_iou = ious[gt_indices]
        differences.append(float(selected_iou[selected_focus].mean() - selected_iou[~selected_focus].mean()))
        selected_detections = [row for image_id in sampled_ids for row in det_groups[image_id]]
        gains.append(fixed_match_ranking_summary(selected_detections, len(gt_indices))["oracle_gain_ap50_95"])
    return (
        {"mean": observed_diff, "ci95_low": float(np.percentile(differences, 2.5)), "ci95_high": float(np.percentile(differences, 97.5))},
        {"mean": float(observed_gain), "ci95_low": float(np.percentile(gains, 2.5)), "ci95_high": float(np.percentile(gains, 97.5))},
    )


def build_phase0_summary(
    gt_records: list[dict[str, Any]],
    detections: list[dict[str, Any]] | None = None,
    image_ids: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 3407,
    min_oracle_gain: float = 0.0,
) -> dict[str, Any]:
    """Build image-bootstrap evidence from separate GT and detection populations."""
    if not gt_records:
        raise ValueError("Phase 0 summary requires at least one valid GT record")
    # Legacy input is accepted for old callers, but new evidence must pass an
    # explicit detection list so unmatched GTs cannot enter calibration.
    if detections is None:
        detections = [
            {
                "image_id": row["image_id"],
                "pred_index": row.get("matched_pred_index", index),
                "conf": row.get("matched_conf", 0.0),
                "quality_iou": row.get("matched_iou", 0.0),
            }
            for index, row in enumerate(gt_records)
            if row.get("matched_pred_index") is not None
        ]
    if image_ids is None:
        image_ids = sorted({str(row["image_id"]) for row in gt_records + detections})
    y = np.asarray([float(row["y_center_norm"]) for row in gt_records])
    log_area = np.asarray([float(row["log_area_norm"]) for row in gt_records])
    area = np.asarray([float(row["area_norm"]) for row in gt_records])
    competition = np.asarray([float(row["competition"]) for row in gt_records])
    iou = np.asarray([float(row["matched_iou"]) for row in gt_records])

    y_edges = [0.0, 0.25, 0.50, 0.75, 1.00]
    area_edges = [float(np.min(log_area)), *map(float, np.quantile(log_area, [1 / 3, 2 / 3])), float(np.max(log_area))]
    cells = []
    for y_index in range(4):
        row_cells = []
        y_mask = _in_bin(y, y_edges[y_index], y_edges[y_index + 1], y_index == 3)
        for area_index in range(3):
            area_mask = _in_bin(log_area, area_edges[area_index], area_edges[area_index + 1], area_index == 2)
            row_cells.append(_subset_metrics([row for row, keep in zip(gt_records, y_mask & area_mask) if keep]))
        cells.append(row_cells)

    competition_edges = list(map(float, np.quantile(competition, [0.0, 1 / 3, 2 / 3, 1.0])))
    competition_bins = []
    for index in range(3):
        mask = _in_bin(competition, competition_edges[index], competition_edges[index + 1], index == 2)
        competition_bins.append({"lo": competition_edges[index], "hi": competition_edges[index + 1], **_subset_metrics([row for row, keep in zip(gt_records, mask) if keep])})

    area_q33, competition_q50 = float(np.quantile(area, 1 / 3)), float(np.quantile(competition, 1 / 2))
    focus = (area <= area_q33) & (competition >= competition_q50)
    if np.any(focus) and np.any(~focus):
        iou_diff, oracle_gain = _cluster_bootstrap(gt_records, detections, focus, image_ids, n_boot, seed)
    else:
        iou_diff = oracle_gain = {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    vertical_area = _finite_spearman(y, log_area)
    quality = np.asarray([float(row["quality_iou"]) for row in detections])
    confidence = np.asarray([float(row["conf"]) for row in detections])
    ranking = fixed_match_ranking_summary(detections, len(gt_records))
    correlations = {
        "y_center_vs_log_area_spearman": vertical_area,
        "competition_vs_one_minus_iou_spearman": _finite_spearman(competition, 1.0 - iou),
        "detection_confidence_vs_quality_spearman": _finite_spearman(confidence, quality),
    }
    return {
        "population": {"image_count": len(image_ids), "gt_count": len(gt_records), "prediction_count": len(detections)},
        "vertical_area": {"y_edges": y_edges, "log_area_edges": area_edges, "cells": cells},
        "competition": {
            "edges": competition_edges,
            "bins": competition_bins,
            "focus_definition": "area_norm <= q33(area_norm) AND competition >= q50(competition)",
            "focus_thresholds": {"area_q33": area_q33, "competition_q50": competition_q50},
            "focus_vs_reference_iou": iou_diff,
        },
        "quality": {"calibration": calibration_summary(detections), "ranking": ranking, "oracle_gain_bootstrap": oracle_gain},
        "correlations": correlations,
        "g0": gate_g0(iou_diff, oracle_gain, vertical_area, min_oracle_gain),
    }
