"""DSRDetectionValidator: strata / candidate recall / oracle (v1.1 7.2, 19.x, 20.3)."""

import json
import shutil
import tempfile
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.dsr.val import DSRDetectionValidator

ROOT = Path(__file__).resolve().parents[1]
DSR_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"
SMOKE_YAML = ROOT / "tests/data/smoke.yaml"


def _build_model():
    model = YOLO(str(DSR_YAML)).model
    dsr_cfg = model.yaml["dsr"]
    dsr_cfg["candidate_conf"] = 0.0001  # random-init base head -> low confidences
    dsr_cfg["pre_topk"] = 100
    dsr_cfg["relation_topk"] = 40
    head = model.model[-1]
    head._configure_dsr(dsr_cfg)
    for cv3 in head.cv3:
        cv3[-1].bias.data.zero_()
    return model


def test_validator_produces_scenario_report():
    work = Path(tempfile.mkdtemp(prefix="dsr_val_", dir=str(ROOT / "tests")))
    try:
        args = get_cfg(
            overrides={
                "model": str(DSR_YAML),
                "data": str(SMOKE_YAML),
                "imgsz": 128,
                "batch": 2,
                "device": "cpu",
                "workers": 0,
                "project": str(work / "runs"),
                "name": "val",
            }
        )
        validator = DSRDetectionValidator(args=args)
        model = _build_model()
        stats = validator(model=model)
        assert stats is not None
        report_path = Path(validator.save_dir) / "dsr_scenario.json"
        assert report_path.exists(), "dsr_scenario.json must be written by DSRDetectionValidator"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        per_image_path = Path(validator.save_dir) / "per_image_metrics.json"
        assert per_image_path.exists(), "per_image_metrics.json must be written for paired bootstrap"
        per_image = json.loads(per_image_path.read_text(encoding="utf-8"))
        assert len(per_image["images"]) > 0

        # candidate recall protocol (7.2)
        assert "candidate_recall" in report
        assert set(report["candidate_recall"].keys()) == {"0.01", "0.03", "0.05", "0.1"}
        for v in report["candidate_recall"].values():
            assert "recall_proposal@0.5" in v and "n_gt" in v

        # strata (19.1-19.4)
        assert set(report["strata"].keys()) == {"all", "occlusion", "density", "size"}
        assert set(report["strata"]["occlusion"].keys()) == {"无遮挡", "轻微遮挡", "中等遮挡", "严重遮挡"}
        all_metrics = report["strata"]["all"]
        assert all_metrics["n_gt"] > 0
        assert "AP50-95" in all_metrics and "Recall@0.5" in all_metrics
        assert "missed" in all_metrics and "misclassified" in all_metrics

        # relation effectiveness + base-vs-refined error attribution (19.5)
        assert "relation_stats" in report
        assert "base_vs_refined" in report
        bv = report["base_vs_refined"]
        assert bv["n_gt"] > 0 and "n_fix" in bv and "n_harm" in bv and "net_gain" in bv
        assert "recall_relation" in report

        # oracle (19.5)
        assert "oracle" in report
        assert report["oracle"]["mAP50-95"] is not None or report["oracle"]["mAP50-95"] == 0
        assert torch.isfinite(torch.tensor(float(report["oracle"]["mAP50-95"] or 0.0)))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_gt_ioa_helper():
    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0], [50.0, 50.0, 60.0, 60.0]],
        dtype=torch.float32,
    )
    ioa = DSRDetectionValidator._gt_ioa(boxes)
    # box0 overlaps box1 by 25/100, box1 overlaps box0 by 25/100, box2 isolated
    assert ioa[0].item() == 0.25
    assert ioa[1].item() == 0.25
    assert ioa[2].item() == 0.0
