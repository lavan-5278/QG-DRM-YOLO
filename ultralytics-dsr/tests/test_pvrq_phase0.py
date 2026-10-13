import numpy as np
import pytest
import json
import argparse
from pathlib import Path

import cv2
from ultralytics.data.utils import get_hash

from pvrq_diagnosis.phase0_metrics import (
    assign_detection_quality,
    bootstrap_mean_diff,
    compute_competition,
    fixed_match_ranking_summary,
    gate_g0,
    match_instances,
)
from pvrq_diagnosis.phase0_metrics import calibration_summary, compute_group_matched_scores
from pvrq_diagnosis.phase0_metrics import build_phase0_summary
from pvrq_diagnosis import phase0_runner
from pvrq_diagnosis.phase0_runner import (
    _validated_samples,
    collect_phase0_records,
    derive_analysis_records,
    run_phase0,
    write_instance_records,
)
from pvrq_diagnosis.run_phase0 import build_parser


def test_competition_combines_overlap_and_nearby_centres():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [40, 0, 50, 10]], dtype=np.float32)
    values = compute_competition(boxes, radius=2.0, beta=1.0, n_ref=4)
    assert values.shape == (3,)
    assert values[0] > values[2]
    assert values[1] > values[2]
    assert np.all((values >= 0.0) & (values < 1.0))


def test_matching_is_class_aware_confidence_ordered_and_one_to_one():
    gt_boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
    gt_cls = np.array([0, 0], dtype=np.int64)
    pred_boxes = np.array([[0, 0, 10, 10], [0, 0, 9, 9], [20, 0, 30, 10], [20, 0, 30, 10]], dtype=np.float32)
    pred_cls = np.array([0, 0, 1, 0], dtype=np.int64)
    pred_conf = np.array([0.90, 0.80, 0.99, 0.70], dtype=np.float32)
    rows = match_instances(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)
    assert [row["matched_pred_index"] for row in rows] == [0, 3]
    assert [row["matched_iou"] for row in rows] == [1.0, 1.0]
    assert rows[0]["matched_conf"] == pytest.approx(0.90)


def test_detection_quality_keeps_every_prediction_and_assigns_unmatched_fp_zero():
    gt_boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    gt_cls = np.array([0], dtype=np.int64)
    pred_boxes = np.array(
        [[0, 0, 10, 10], [0, 0, 9, 9], [20, 20, 30, 30], [0, 0, 10, 10]], dtype=np.float32
    )
    pred_cls = np.array([0, 0, 0, 1], dtype=np.int64)
    pred_conf = np.array([0.90, 0.80, 0.70, 0.99], dtype=np.float32)

    gt_matches, detections = assign_detection_quality(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)

    assert len(gt_matches) == 1
    assert len(detections) == len(pred_boxes)
    assert detections[0]["matched_gt_index"] == 0
    assert detections[0]["quality_iou"] == pytest.approx(1.0)
    assert [row["quality_iou"] for row in detections[1:]] == [0.0, 0.0, 0.0]
    assert all(row["pred_index"] == index for index, row in enumerate(detections))


def test_competition_stays_strictly_below_one_when_dense_and_overlapping():
    boxes = np.tile(np.array([[0, 0, 10, 10]], dtype=np.float32), (5, 1))
    values = compute_competition(boxes, radius=2.0, beta=1_000.0, n_ref=1)
    assert np.all(values < 1.0)


def test_bootstrap_and_gate_proceed_for_localization_deficit_and_oracle_reranking_gain():
    focus = np.full(16, 0.35, dtype=np.float64)
    reference = np.full(16, 0.70, dtype=np.float64)
    diff = bootstrap_mean_diff(focus, reference, n_boot=2000, seed=3407)
    assert diff["ci95_high"] < 0.0
    decision = gate_g0(
        iou_diff=diff,
        oracle_gain={"mean": 0.20, "ci95_low": 0.12, "ci95_high": 0.28},
        vertical_area_spearman=-0.45,
    )
    assert decision["decision"] == "PROCEED_PVRQ"
    assert decision["lpfield"] is True


def test_gate_selects_pvrf_only_when_localization_deficit_exists_without_reranking_headroom():
    decision = gate_g0(
        iou_diff={"mean": -0.2, "ci95_low": -0.3, "ci95_high": -0.1},
        oracle_gain={"mean": 0.0, "ci95_low": -0.01, "ci95_high": 0.01},
        vertical_area_spearman=0.4,
    )
    assert decision == {
        "decision": "PROCEED_PVRF_ONLY",
        "lpfield": True,
        "localization_deficit": True,
        "quality_headroom": False,
    }


def test_fixed_match_ranking_uses_all_detections_and_oracle_improves_bad_confidence_order():
    detections = [
        {"image_id": "a", "pred_index": 0, "conf": 0.99, "quality_iou": 0.0},
        {"image_id": "a", "pred_index": 1, "conf": 0.70, "quality_iou": 0.90},
        {"image_id": "b", "pred_index": 0, "conf": 0.95, "quality_iou": 0.0},
        {"image_id": "b", "pred_index": 1, "conf": 0.60, "quality_iou": 0.80},
    ]
    ranking = fixed_match_ranking_summary(detections, gt_count=2)
    assert ranking["n_predictions"] == 4
    assert ranking["n_unmatched_predictions"] == 2
    assert ranking["oracle_ap50_95"] > ranking["confidence_ap50_95"]
    assert ranking["oracle_gain_ap50_95"] > 0.0


def test_matched_scores_and_calibration_are_bounded_and_penalize_overconfidence():
    records = [
        {"image_id": "a", "gt_index": 0, "matched_iou": 1.0, "matched_conf": 0.95},
        {"image_id": "b", "gt_index": 0, "matched_iou": 0.20, "matched_conf": 0.95},
    ]
    scores = compute_group_matched_scores(records)
    cal = calibration_summary(records, n_bins=5)
    assert 0.0 <= scores["matched_score50"] <= 1.0
    assert 0.0 <= scores["matched_score50_95"] <= 1.0
    assert not any(key.startswith("AP") for key in scores)
    assert cal["mean_conf_minus_iou"] > 0.0
    assert cal["brier"] >= 0.0


def test_matched_score50_does_not_round_an_iou_just_below_the_threshold_upward():
    records = [{"matched_iou": 0.4999997, "matched_conf": 0.9}]
    assert compute_group_matched_scores(records)["matched_score50"] == 0.0


def test_record_writer_preserves_required_instance_fields(tmp_path: Path):
    item = {
        "image_id": "img.jpg", "gt_index": 0, "gt_cls": 1, "x_center_norm": 0.4,
        "y_center_norm": 0.6, "area_norm": 0.01, "log_area_norm": -4.60517,
        "competition": 0.3, "matched_pred_index": 2, "matched_iou": 0.7, "matched_conf": 0.8,
    }
    path = tmp_path / "instances.jsonl"
    write_instance_records([item], path)
    loaded = json.loads(path.read_text(encoding="utf-8").strip())
    assert loaded == item


class _FakeTensor:
    """Minimal tensor facade for the runner's result conversion path."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[10, 5, 30, 15], [30, 10, 39, 19]])
        self.cls = _FakeTensor([1, 1])
        self.conf = _FakeTensor([0.9, 0.8])

    def __len__(self):
        return 2


class _FakeModel:
    def predict(self, **kwargs):
        return [type("Result", (), {"boxes": _FakeBoxes()})()]


def test_validated_samples_use_ultralytics_duplicate_and_invalid_label_semantics(tmp_path: Path):
    root = tmp_path / "dataset"
    image_dir, label_dir = root / "images" / "val", root / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name in ("valid.jpg", "invalid.jpg"):
        assert cv2.imwrite(str(image_dir / name), np.zeros((20, 40, 3), dtype=np.uint8))
    duplicate = "1 0.5 0.5 0.5 0.5\n"
    (label_dir / "valid.txt").write_text(duplicate + duplicate, encoding="utf-8")
    (label_dir / "invalid.txt").write_text("3 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    samples, provenance = _validated_samples(
        {"val": str(image_dir), "names": {0: "zero", 1: "one"}, "channels": 3}, imgsz=640
    )

    assert len(samples) == 1
    assert Path(samples[0]["im_file"]).name == "valid.jpg"
    assert len(samples[0]["cls"]) == 1
    assert provenance["valid_image_count"] == 1
    assert provenance["valid_gt_count"] == 1
    # Provenance covers the complete source population, including the image
    # excluded by validation, exactly as the official cache hash does.
    expected_source_hash = get_hash(
        [str(label_dir / "invalid.txt"), str(label_dir / "valid.txt"),
         str(image_dir / "invalid.jpg"), str(image_dir / "valid.jpg")]
    )
    assert provenance["dataset_source_hash"] == expected_source_hash


def test_collect_raw_records_preserves_all_post_nms_predictions(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "labeled.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 40, 3), dtype=np.uint8))
    sample = {
        "im_file": str(image_path),
        "shape": (20, 40),
        "cls": np.array([[1]], dtype=np.float32),
        "bboxes": np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
    }
    monkeypatch.setattr(phase0_runner, "_validated_samples", lambda *_: ([sample], {"dataset_source_hash": "x"}))
    monkeypatch.setattr(phase0_runner, "check_det_dataset", lambda _: {"val": str(tmp_path), "names": {0: "a", 1: "b"}})
    monkeypatch.setattr(phase0_runner, "YOLO", lambda _: _FakeModel())

    images, ground_truth, predictions, manifest = collect_phase0_records(
        tmp_path / "best.pt", tmp_path / "data.yaml", 640, "cpu", 0.001, 0.7, 300
    )

    assert len(images) == 1
    assert len(ground_truth) == 1
    assert len(predictions) == 2
    assert [row["pred_index"] for row in predictions] == [0, 1]
    assert all("quality_iou" not in row for row in predictions)  # immutable raw inference, not derived analysis
    record = ground_truth[0]
    assert record["image_id"] == "labeled.jpg"
    assert record["gt_cls"] == 1
    assert record["area_norm"] == pytest.approx(0.25)
    assert "matched_iou" not in record
    assert manifest["image_count"] == 1
    assert manifest["gt_count"] == 1
    assert manifest["prediction_count"] == 2

    analyzed_gt, detections = derive_analysis_records(ground_truth, predictions)
    assert analyzed_gt[0]["matched_iou"] == pytest.approx(1.0)
    assert len(detections) == 2
    assert detections[1]["quality_iou"] == 0.0


def test_manifest_hash_and_reuse_mismatch_rejection(tmp_path: Path, monkeypatch):
    gt_record = {
        "image_id": "image.jpg", "gt_index": 0, "gt_cls": 0, "x_center_norm": 0.5,
        "y_center_norm": 0.5, "area_norm": 0.1, "log_area_norm": -2.3, "competition": 0.0,
        "xyxy": [0.0, 0.0, 10.0, 10.0],
    }
    pred_record = {"image_id": "image.jpg", "pred_index": 0, "pred_cls": 0, "conf": 0.8,
                   "xyxy": [0.0, 0.0, 10.0, 10.0]}
    (tmp_path / "best.pt").write_bytes(b"weights")
    (tmp_path / "data.yaml").write_text("names: [a]\n", encoding="utf-8")
    monkeypatch.setattr(
        phase0_runner,
        "collect_phase0_records",
        lambda *args: ([{"image_id": "image.jpg", "shape": [10, 10]}], [gt_record], [pred_record],
                       {"weights": str(Path(args[0]).resolve()), "data_yaml": str(Path(args[1]).resolve()),
                        "image_count": 1, "gt_count": 1, "prediction_count": 1, "imgsz": args[2],
                        "device": args[3], "conf": args[4], "iou": args[5], "max_det": args[6],
                        "utc_timestamp": "test", "dataset": {"dataset_source_hash": "x"}}),
    )
    args = argparse.Namespace(
        weights=tmp_path / "best.pt", data=tmp_path / "data.yaml", output=tmp_path / "output",
        imgsz=640, device="cpu", conf=0.001, iou=0.7, max_det=300, reuse_records=False,
        bootstrap_samples=40, bootstrap_seed=3407, min_oracle_gain=0.0,
    )
    run_phase0(args)
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_records_sha256"]["predictions.jsonl"] == phase0_runner._sha256(args.output / "predictions.jsonl")
    assert manifest["weights_sha256"] == phase0_runner._sha256(args.weights)
    assert manifest["data_yaml_sha256"] == phase0_runner._sha256(args.data)
    assert manifest["analysis_parameters"]["bootstrap_seed"] == 3407
    assert "phase0_metrics.py" in manifest["analysis_source_sha256"]

    original_environment = {"actual_device": "cuda:0", "cuda_available": True, "cpu_fallback": False,
                            "cpu_fallback_reason": None, "torch_version": "original"}
    manifest["execution_environment"] = original_environment
    (args.output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args.reuse_records = True
    assert run_phase0(args) == args.output.resolve()
    preserved = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    assert preserved["execution_environment"] == original_environment

    # A legacy manifest has no defensible environment provenance; do not infer
    # it from the runtime used merely to regenerate report files.
    del preserved["execution_environment"]
    (args.output / "manifest.json").write_text(json.dumps(preserved), encoding="utf-8")
    assert run_phase0(args) == args.output.resolve()
    unknown = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    assert unknown["execution_environment"]["actual_device"] == "unknown"
    assert unknown["execution_environment"]["cuda_available"] is None

    args.imgsz = 320
    with pytest.raises(ValueError, match="Refusing to reuse records"):
        run_phase0(args)


def test_phase0_cli_defaults_are_the_baseline_protocol():
    args = build_parser().parse_args(["--weights", "best.pt", "--data", "data.yaml", "--output", "output"])
    assert (args.imgsz, args.device, args.conf, args.iou, args.max_det, args.reuse_records) == (640, "0", 0.001, 0.7, 300, False)
    assert (args.bootstrap_samples, args.bootstrap_seed, args.min_oracle_gain) == (2000, 3407, 0.0)


def test_phase0_summary_has_predeclared_sections_and_decision():
    records = []
    for i in range(24):
        small_competitive = i < 12
        records.append({
            "image_id": f"{i // 2}", "gt_index": i, "y_center_norm": i / 24,
            "area_norm": 0.002 if small_competitive else 0.04,
            "log_area_norm": -6.2 if small_competitive else -3.2,
            "competition": 0.8 if small_competitive else 0.1,
            "matched_iou": 0.30 if small_competitive else 0.80,
            "matched_conf": 0.75 if small_competitive else 0.80,
        })
    detections = []
    for i, row in enumerate(records):
        detections.append({"image_id": row["image_id"], "pred_index": i, "conf": 0.99 - i * 0.01,
                           "quality_iou": row["matched_iou"]})
    summary = build_phase0_summary(records, detections, n_boot=200, seed=3407)
    assert set(summary) == {"population", "vertical_area", "competition", "quality", "correlations", "g0"}
    assert summary["g0"]["decision"] == "PROCEED_PVRQ"
    assert "AP50" not in summary["competition"]["bins"][0]
    assert summary["quality"]["calibration"]["n_predictions"] == len(detections)


def test_joint_image_bootstrap_preserves_focus_reference_image_dependence():
    # Every image has one focus and one reference row. Within either image the
    # difference is exactly +0.10, while independent focus/reference sampling
    # would spuriously create a wide distribution from cross-image pairings.
    records = []
    for image_id, focus_iou, reference_iou in (("a", 0.2, 0.1), ("b", 0.8, 0.7)):
        records.extend([
            {"image_id": image_id, "gt_index": 0, "y_center_norm": 0.2, "area_norm": 0.001,
             "log_area_norm": -6.9, "competition": 0.9, "matched_iou": focus_iou, "matched_conf": focus_iou + 0.1},
            {"image_id": image_id, "gt_index": 1, "y_center_norm": 0.8, "area_norm": 0.1,
             "log_area_norm": -2.3, "competition": 0.1, "matched_iou": reference_iou, "matched_conf": reference_iou},
        ])
    detections = [
        {"image_id": row["image_id"], "pred_index": index, "conf": row["matched_conf"],
         "quality_iou": row["matched_iou"]}
        for index, row in enumerate(records)
    ]
    summary = build_phase0_summary(records, detections, image_ids=["a", "b"], n_boot=200, seed=3407)
    iou_diff = summary["competition"]["focus_vs_reference_iou"]
    assert iou_diff == pytest.approx({"mean": 0.1, "ci95_low": 0.1, "ci95_high": 0.1})


def test_image_cluster_bootstrap_detects_oracle_reranking_headroom_with_fps():
    records, detections = [], []
    for image_id in ("a", "b", "c", "d"):
        records.extend([
            {"image_id": image_id, "gt_index": 0, "y_center_norm": 0.2, "area_norm": 0.001,
             "log_area_norm": -6.9, "competition": 0.9, "matched_iou": 0.4, "matched_conf": 0.6},
            {"image_id": image_id, "gt_index": 1, "y_center_norm": 0.8, "area_norm": 0.1,
             "log_area_norm": -2.3, "competition": 0.1, "matched_iou": 0.9, "matched_conf": 0.5},
        ])
        detections.extend([
            {"image_id": image_id, "pred_index": 0, "conf": 0.99, "quality_iou": 0.0},
            {"image_id": image_id, "pred_index": 1, "conf": 0.60, "quality_iou": 0.4},
            {"image_id": image_id, "pred_index": 2, "conf": 0.50, "quality_iou": 0.9},
        ])
    summary = build_phase0_summary(records, detections, image_ids=["a", "b", "c", "d"], n_boot=100, seed=9)
    gain = summary["quality"]["oracle_gain_bootstrap"]
    assert gain["ci95_low"] > 0.0
    assert summary["quality"]["calibration"]["n_predictions"] == 12


def test_unmatched_gt_is_localization_evidence_but_not_a_calibration_row():
    gt = [
        {"image_id": "a", "gt_index": 0, "y_center_norm": 0.2, "area_norm": 0.001,
         "log_area_norm": -6.9, "competition": 0.9, "matched_iou": 0.0, "matched_conf": 0.0,
         "matched_pred_index": None},
        {"image_id": "a", "gt_index": 1, "y_center_norm": 0.8, "area_norm": 0.1,
         "log_area_norm": -2.3, "competition": 0.1, "matched_iou": 0.9, "matched_conf": 0.8,
         "matched_pred_index": 0},
    ]
    detections = [{"image_id": "a", "pred_index": 0, "conf": 0.8, "quality_iou": 0.9}]
    summary = build_phase0_summary(gt, detections, image_ids=["a"], n_boot=20, seed=1)
    assert summary["population"]["gt_count"] == 2
    assert summary["quality"]["calibration"]["n_predictions"] == 1
