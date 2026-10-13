"""Ultralytics-faithful baseline inference and immutable Phase 0 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset, get_hash

from .phase0_metrics import (
    IOU_THRESHOLDS,
    assign_detection_quality,
    build_phase0_summary,
    compute_competition,
)


RAW_RECORD_FILES = ("images.jsonl", "ground_truth.jsonl", "predictions.jsonl")
DERIVED_RECORD_FILES = ("analysis_ground_truth.jsonl", "detection_quality.jsonl")


def write_instance_records(records: list[dict], path: Path) -> None:
    """Write stable UTF-8 JSONL records."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _validated_samples(data: dict[str, Any], imgsz: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the validation set through the same cache/verification path as YOLO.

    ``YOLODataset`` calls ``verify_image_label`` when its cache is stale and
    reads the same cache when current.  Consequently corrupt image/label pairs
    are excluded and duplicate rows are removed exactly as in validation.
    """
    dataset = YOLODataset(
        img_path=data["val"],
        imgsz=imgsz,
        augment=False,
        cache=False,
        rect=False,
        batch_size=1,
        stride=32,
        data=data,
        task="detect",
        prefix="Phase0 val: ",
    )
    samples = list(dataset.labels)
    # ``dataset.im_files`` has already been filtered to valid samples. Re-read
    # the source list so the provenance hash also covers corrupt/excluded
    # images, matching the population used by YOLO's cache hash.
    source_im_files = dataset.get_img_files(dataset.img_path)
    provenance = {
        "loader": "ultralytics.data.dataset.YOLODataset",
        "verification": "verify_image_label/cache semantics",
        "dataset_source_hash": get_hash(dataset.label_files + source_im_files),
        "source_image_count": len(source_im_files),
        "valid_image_count": len(samples),
        "valid_gt_count": int(sum(len(sample["cls"]) for sample in samples)),
    }
    return samples, provenance


def _labels_xyxy(bboxes: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    values = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if len(values) == 0:
        return np.empty((0, 4), dtype=np.float32)
    cx, cy, width, height = (values[:, index] for index in range(4))
    return np.column_stack(
        ((cx - width / 2) * image_w, (cy - height / 2) * image_h,
         (cx + width / 2) * image_w, (cy + height / 2) * image_h)
    ).astype(np.float32)


def collect_phase0_records(
    weights: Path, data_yaml: Path, imgsz: int, device: str, conf: float, iou: float, max_det: int,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Collect immutable valid-image, GT, and all post-NMS prediction rows."""
    weights, data_yaml = Path(weights).resolve(), Path(data_yaml).resolve()
    data = check_det_dataset(str(data_yaml))
    samples, dataset_provenance = _validated_samples(data, imgsz)
    model = YOLO(str(weights))
    images, ground_truth, predictions = [], [], []

    basenames = Counter(Path(sample["im_file"]).name for sample in samples)
    for sample in samples:
        image_path = Path(sample["im_file"]).resolve()
        image_id = image_path.name if basenames[image_path.name] == 1 else str(image_path)
        image_h, image_w = map(int, sample["shape"])
        bboxes = np.asarray(sample["bboxes"], dtype=np.float32).reshape(-1, 4)
        gt_cls = np.asarray(sample["cls"], dtype=np.float32).reshape(-1).astype(np.int64)
        gt_boxes = _labels_xyxy(bboxes, image_h, image_w)
        competition = compute_competition(gt_boxes)
        images.append({"image_id": image_id, "im_file": str(image_path), "shape": [image_h, image_w]})
        for index, (box, cls, normalized_box) in enumerate(zip(gt_boxes, gt_cls, bboxes)):
            cx, cy, width, height = map(float, normalized_box)
            area = width * height
            ground_truth.append(
                {
                    "image_id": image_id,
                    "gt_index": int(index),
                    "gt_cls": int(cls),
                    "xyxy": [float(value) for value in box],
                    "x_center_norm": cx,
                    "y_center_norm": cy,
                    "area_norm": area,
                    "log_area_norm": float(np.log(max(area, 1e-12))),
                    "competition": float(competition[index]),
                }
            )

        result = model.predict(
            source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        pred_boxes = result.boxes.xyxy.detach().cpu().numpy()
        pred_cls = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        pred_conf = result.boxes.conf.detach().cpu().numpy()
        for index, (box, cls, confidence) in enumerate(zip(pred_boxes, pred_cls, pred_conf)):
            predictions.append(
                {
                    "image_id": image_id,
                    "pred_index": int(index),
                    "pred_cls": int(cls),
                    "conf": float(confidence),
                    "xyxy": [float(value) for value in box],
                }
            )

    manifest = {
        "schema_version": 2,
        "weights": str(weights),
        "data_yaml": str(data_yaml),
        "image_count": len(images),
        "gt_count": len(ground_truth),
        "prediction_count": len(predictions),
        "imgsz": int(imgsz),
        "device": str(device),
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "dataset": dataset_provenance,
        "utc_timestamp": datetime.now(UTC).isoformat(),
    }
    return images, ground_truth, predictions, manifest


def derive_analysis_records(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive GT localization and detection-quality rows without mutating raw input."""
    gt_groups, pred_groups = defaultdict(list), defaultdict(list)
    for row in ground_truth:
        gt_groups[str(row["image_id"])].append(row)
    for row in predictions:
        pred_groups[str(row["image_id"])].append(row)
    image_ids = sorted(set(gt_groups) | set(pred_groups))
    analyzed_gt, detections = [], []
    for image_id in image_ids:
        gt_rows = sorted(gt_groups[image_id], key=lambda row: int(row["gt_index"]))
        pred_rows = sorted(pred_groups[image_id], key=lambda row: int(row["pred_index"]))
        gt_boxes = np.asarray([row["xyxy"] for row in gt_rows], dtype=np.float32).reshape(-1, 4)
        gt_cls = np.asarray([row["gt_cls"] for row in gt_rows], dtype=np.int64)
        pred_boxes = np.asarray([row["xyxy"] for row in pred_rows], dtype=np.float32).reshape(-1, 4)
        pred_cls = np.asarray([row["pred_cls"] for row in pred_rows], dtype=np.int64)
        pred_conf = np.asarray([row["conf"] for row in pred_rows], dtype=np.float64)
        gt_matches, qualities = assign_detection_quality(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)
        for raw, match in zip(gt_rows, gt_matches):
            analyzed_gt.append({**raw, **match})
        for raw, quality in zip(pred_rows, qualities):
            detections.append({**raw, **quality})
    return analyzed_gt, detections


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analysis_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: _sha256(root / name) for name in ("phase0_metrics.py", "phase0_runner.py", "run_phase0.py")}


def _analysis_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "min_oracle_gain": float(args.min_oracle_gain),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "focus_definition": "area_norm <= q33(area_norm) AND competition >= q50(competition)",
        "matching": "class-aware confidence-ordered one-to-one; unmatched predictions quality=0",
        "ranking_metric": "fixed-match AP101 oracle IoU reranking proxy",
    }


def _reuse_matches(manifest: dict[str, Any], args: argparse.Namespace) -> bool:
    expected = {
        "weights": str(Path(args.weights).resolve()),
        "data_yaml": str(Path(args.data).resolve()),
        "imgsz": int(args.imgsz),
        "device": str(args.device),
        "conf": float(args.conf),
        "iou": float(args.iou),
        "max_det": int(args.max_det),
    }
    if manifest.get("schema_version") != 2 or not all(manifest.get(key) == value for key, value in expected.items()):
        return False
    if manifest.get("weights_sha256") != _sha256(Path(args.weights)):
        return False
    if manifest.get("data_yaml_sha256") != _sha256(Path(args.data)):
        return False
    return manifest.get("analysis_source_sha256") == _analysis_source_hashes()


def _environment_disclosure(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    actual = str(args.device)
    return {
        "requested_device": str(args.device),
        "actual_device": actual,
        "cuda_available": cuda_available,
        "cpu_fallback": actual.lower() == "cpu" and not cuda_available,
        "cpu_fallback_reason": (
            "CUDA was unavailable in this runtime; the same inference settings were executed on CPU."
            if actual.lower() == "cpu" and not cuda_available else None
        ),
        "torch_version": str(torch.__version__),
    }


def _unknown_environment_disclosure() -> dict[str, Any]:
    return {
        "requested_device": "unknown", "actual_device": "unknown", "cuda_available": None,
        "cpu_fallback": None, "cpu_fallback_reason": "Not recorded in the manifest.", "torch_version": "unknown",
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_phase0_outputs(summary: dict[str, Any], output_dir: Path, environment: dict[str, Any]) -> None:
    """Write the report, machine-readable summary, and deterministic figures."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    summary_path = output_dir / "phase0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    iou_diff = summary["competition"]["focus_vs_reference_iou"]
    gain = summary["quality"]["oracle_gain_bootstrap"]
    corr = summary["correlations"]["y_center_vs_log_area_spearman"]
    g0 = summary["g0"]
    report = f"""# PVR-Q Phase 0 diagnostic report (repaired)

## G0 rule

The gate uses two image-cluster bootstrap quantities: (1) small/competitive GT focus-minus-reference matched IoU, and (2) the gain from perfectly reranking **all post-NMS detections** by their class-aware one-to-one assigned IoU quality. Unmatched predictions have quality 0; unmatched GTs are not calibration samples.

- A localization deficit exists when the IoU-difference CI upper bound is below 0.
- Quality-head headroom exists when the oracle-gain CI lower bound exceeds the declared minimum.
- With both: `PROCEED_PVRQ`; with localization deficit only: `PROCEED_PVRF_ONLY`; without a localization deficit: `STOP`.
- LPF eligibility is reported separately using `|Spearman(y, log-area)| >= 0.20`.

## Result

- Decision: **{g0['decision']}**
- Localization deficit: **{g0['localization_deficit']}**
- Quality-head headroom: **{g0['quality_headroom']}**
- Latent perspective field eligible: **{g0['lpfield']}**
- Focus-minus-reference IoU: {iou_diff['mean']:.6f} (95% CI [{iou_diff['ci95_low']:.6f}, {iou_diff['ci95_high']:.6f}])
- Oracle fixed-match AP50-95 ranking gain: {gain['mean']:.6f} (95% CI [{gain['ci95_low']:.6f}, {gain['ci95_high']:.6f}])
- Spearman(y-center, log-area): {corr:.6f}
- Population: {summary['population']['image_count']} valid images, {summary['population']['gt_count']} GT, {summary['population']['prediction_count']} post-NMS predictions
- Inference device: `{environment['actual_device']}`; CUDA available: `{environment['cuda_available']}`
- CPU fallback: `{environment['cpu_fallback']}`{f" — {environment['cpu_fallback_reason']}" if environment['cpu_fallback_reason'] else ""}

The oracle metric freezes confidence-ordered matches and is a diagnostic ranking proxy, not detector AP or a claim of achievable mAP gain. G0 authorizes implementation experiments; only controlled training can establish an actual improvement.
"""
    (output_dir / "phase0_report.md").write_text(report, encoding="utf-8")

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    heatmap = np.asarray([[cell["mean_matched_iou"] for cell in row] for row in summary["vertical_area"]["cells"]])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    image = axis.imshow(heatmap, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    axis.set(xlabel="log-area tertile", ylabel="vertical y-center bin")
    axis.set_xticks(range(3), ["small", "middle", "large"])
    axis.set_yticks(range(4), ["0-.25", ".25-.50", ".50-.75", ".75-1.0"])
    for row_index in range(heatmap.shape[0]):
        for col_index in range(heatmap.shape[1]):
            axis.text(col_index, row_index, f"{heatmap[row_index, col_index]:.3f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="mean matched IoU")
    figure.tight_layout(); figure.savefig(figures / "vertical_area_heatmap.png", dpi=180); plt.close(figure)

    bins = summary["competition"]["bins"]
    positions, width = np.arange(len(bins)), 0.36
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(positions - width / 2, [entry["mean_matched_iou"] for entry in bins], width, label="mean IoU")
    axis.bar(positions + width / 2, [entry["matched_score75"] for entry in bins], width, label="matched score@0.75")
    axis.set_xticks(positions, [f"T{index + 1}" for index in range(len(bins))]); axis.set_ylim(0, 1)
    axis.set(xlabel="competition tertile", ylabel="metric"); axis.legend(); figure.tight_layout()
    figure.savefig(figures / "competition_metrics.png", dpi=180); plt.close(figure)

    calibration_bins = [row for row in summary["quality"]["calibration"]["bins"] if row["n"]]
    figure, axis = plt.subplots(figsize=(5.5, 5.0)); axis.plot([0, 1], [0, 1], "k--", label="ideal")
    if calibration_bins:
        axis.plot([row["mean_conf"] for row in calibration_bins], [row["mean_quality"] for row in calibration_bins], "o-", label="baseline")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean confidence", ylabel="mean assigned IoU quality")
    axis.legend(); figure.tight_layout(); figure.savefig(figures / "quality_calibration.png", dpi=180); plt.close(figure)


def run_phase0(args: argparse.Namespace) -> Path:
    """Collect/reuse raw records, derive analysis, and emit provenance-complete evidence."""
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if args.reuse_records:
        if not manifest_path.is_file() or any(not (output_dir / name).is_file() for name in RAW_RECORD_FILES):
            raise FileNotFoundError("--reuse-records requires a schema-v2 manifest and all raw JSONL files")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _reuse_matches(manifest, args):
            raise ValueError("Refusing to reuse records: provenance or inference arguments do not match")
        for name in RAW_RECORD_FILES:
            if manifest.get("raw_records_sha256", {}).get(name) != _sha256(output_dir / name):
                raise ValueError(f"Refusing to reuse records: {name} SHA-256 mismatch")
        images = _read_records(output_dir / "images.jsonl")
        ground_truth = _read_records(output_dir / "ground_truth.jsonl")
        predictions = _read_records(output_dir / "predictions.jsonl")
        environment = manifest.get("execution_environment", _unknown_environment_disclosure())
        manifest.setdefault("execution_environment", environment)
    else:
        images, ground_truth, predictions, manifest = collect_phase0_records(
            args.weights, args.data, args.imgsz, args.device, args.conf, args.iou, args.max_det
        )
        manifest["schema_version"] = 2
        for name, rows in zip(RAW_RECORD_FILES, (images, ground_truth, predictions)):
            write_instance_records(rows, output_dir / name)
        manifest["raw_records_sha256"] = {name: _sha256(output_dir / name) for name in RAW_RECORD_FILES}
        manifest["weights_sha256"] = _sha256(Path(args.weights))
        manifest["data_yaml_sha256"] = _sha256(Path(args.data))
        manifest["analysis_source_sha256"] = _analysis_source_hashes()
        environment = _environment_disclosure(args)
        manifest["execution_environment"] = environment

    analyzed_gt, detections = derive_analysis_records(ground_truth, predictions)
    for name, rows in zip(DERIVED_RECORD_FILES, (analyzed_gt, detections)):
        write_instance_records(rows, output_dir / name)
    analysis_parameters = _analysis_parameters(args)
    manifest["analysis_parameters"] = analysis_parameters
    manifest["derived_records_sha256"] = {name: _sha256(output_dir / name) for name in DERIVED_RECORD_FILES}
    summary = build_phase0_summary(
        analyzed_gt,
        detections,
        image_ids=[str(row["image_id"]) for row in images],
        n_boot=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        min_oracle_gain=args.min_oracle_gain,
    )
    summary["analysis_parameters"] = analysis_parameters
    write_phase0_outputs(summary, output_dir, environment)
    manifest["phase0_summary_sha256"] = _sha256(output_dir / "phase0_summary.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir
