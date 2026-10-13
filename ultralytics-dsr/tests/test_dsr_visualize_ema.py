"""P2: relation visualization (19.5) and EMA-candidate source (14.1)."""

import shutil
import tempfile
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.models.yolo.dsr.visualize import visualize_relation_stats

ROOT = Path(__file__).resolve().parents[1]
DSR_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"


def _model():
    model = YOLO(str(DSR_YAML))
    cfg = model.model.yaml["dsr"]
    cfg["candidate_conf"] = 0.0001
    cfg["pre_topk"] = 100
    cfg["relation_topk"] = 40
    head = model.model.model[-1]
    head._configure_dsr(cfg)
    for cv3 in head.cv3:
        cv3[-1].bias.data.zero_()
    return model


def test_visualize_relation_stats():
    work = Path(tempfile.mkdtemp(prefix="dsr_viz_", dir=str(ROOT / "tests")))
    try:
        rows = visualize_relation_stats(
            _model(), str(ROOT / "tests/data/smoke/images"), save_dir=work, imgsz=128, max_images=2, device="cpu"
        )
        assert len(rows) >= 1
        assert (work / "relation_stats.json").exists()
        pngs = list(work.glob("relation_*.png"))
        assert len(pngs) == len(rows)
        assert set(rows[0]["scale_weights"].keys()) == {"n"}  # v2: KNN-only
        assert "gate_mean" in rows[0] and "uncertainty_mean" in rows[0]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_ema_candidates_from_features():
    """v1.1 14.1: candidates can be generated from the frozen EMA head."""
    model = _model()
    head = model.model.model[-1]
    head.candidate_source = "ema"
    head.ema_update()  # creates + warms the EMA head
    model.model.eval()
    with torch.no_grad():
        model.model(torch.randn(1, 3, 128, 128))
    neck = head.neck_feats
    sel_cur = head.candidates_from_features(neck, (128, 128), source="current")
    sel_ema = head.candidates_from_features(neck, (128, 128), source="ema")
    assert sel_cur["mask"].shape == sel_ema["mask"].shape
    assert torch.isfinite(sel_ema["boxes_xyxy"]).all()
    assert sel_ema["idx"].shape == sel_cur["idx"].shape
