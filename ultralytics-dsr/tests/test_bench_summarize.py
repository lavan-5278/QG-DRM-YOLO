"""P1-7/8: efficiency profiling (20.2) and multi-seed summary + paired bootstrap (20.3/20.4)."""

import csv
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from scripts import summarize_runs
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DSR_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"


def _model():
    m = YOLO(str(DSR_YAML)).model
    head = m.model[-1]
    head.relation_topk = 20
    head.pre_topk = 60
    head.candidate_conf = 0.0001  # random-init base head -> low confidences
    for cv3 in head.cv3:
        cv3[-1].bias.data.zero_()
    return m


def test_profile_dsr_cpu():
    from ultralytics.models.yolo.dsr.bench import profile_dsr

    report = profile_dsr(
        _model(), imgsz=64, device="cpu", warmup=1, repeats=2, fixed_k=20, include_nms=True
    )
    run = report["runs"][0]
    assert run["precision"] == "fp32"
    assert run["total_mean_ms"] > 0 and run["fps"] > 0
    comp = run["components"]
    for key in ("base_detect", "candidate", "mdrr", "refine_head", "writeback"):
        assert key in comp, f"missing component timing: {key}"
    assert report["params"] > 0
    assert report["gflops_fp32"] > 0
    assert "nms_mean_ms" in run


def _write_fake_run(run_dir: Path, ap_values: list[float], seed: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["epoch", "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/recall(B)", "metrics/precision(B)"],
        )
        w.writeheader()
        w.writerow({"epoch": 1, "metrics/mAP50-95(B)": 0.5, "metrics/mAP50(B)": 0.7,
                    "metrics/recall(B)": 0.6, "metrics/precision(B)": 0.65})
    images = [
        {"im_file": f"img{i}.jpg", "n_gt": 3, "ap50_95": v, "recall05": min(1.0, v + 0.1)}
        for i, v in enumerate(ap_values)
    ]
    (run_dir / "per_image_metrics.json").write_text(
        json.dumps({"images": images}), encoding="utf-8"
    )


def test_summarize_and_compare():
    work = Path(tempfile.mkdtemp(prefix="dsr_sum_", dir=str(ROOT / "tests")))
    try:
        rng = np.random.default_rng(7)
        a_vals = rng.uniform(0.3, 0.5, 12).tolist()
        b_vals = rng.uniform(0.2, 0.4, 12).tolist()
        for exp, vals in (("expA", a_vals), ("expB", b_vals)):
            for seed in ("seed0", "seed42"):
                _write_fake_run(work / exp / seed, vals, seed)

        summary = summarize_runs.summarize(work)
        assert set(summary.keys()) == {"expA", "expB"}
        assert summary["expA"]["n_seeds"] == 2
        assert summary["expA"]["metrics/mAP50-95(B)_mean"] == 0.5
        assert summary["expA"]["metrics/mAP50-95(B)_std"] == 0.0

        result = summarize_runs.compare(work, "expA", "expB", iters=300, seed=0)
        assert result["n_seeds"] == 2
        m = result["metrics"]["ap50_95"]
        assert m["n_images"] == 24  # 12 images x 2 seeds (image-seed units)
        assert m["mean_diff_A_minus_B"] > 0
        assert m["ci95"][0] < m["ci95"][1]
        assert m["significant"] is True  # A consistently higher than B
        assert "recall05" in result["metrics"]
    finally:
        shutil.rmtree(work, ignore_errors=True)
