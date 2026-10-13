"""Experiment matrix runner tests (v1.1 Phase 5)."""

import argparse
from pathlib import Path

import pytest

from scripts import run_experiments as re

ROOT = Path(__file__).resolve().parents[1]
SMOKE_MATRIX = ROOT / "tests/data/matrix_smoke.yaml"
DEFAULT_MATRIX = ROOT / "ultralytics/cfg/experiments/matrix.yaml"


def test_matrix_yamls_load():
    """Default and smoke matrices must parse and expand into valid plans."""
    for matrix in (DEFAULT_MATRIX, SMOKE_MATRIX):
        args = argparse.Namespace(only=None, seeds=None, device=None)
        plans, g = re.build_plans(matrix, args)
        assert plans, matrix
        assert g["runs_root"]
        for p in plans:
            assert re.validate_plan(p) is None, p


def test_baseline_weights_blocking():
    """Plans that load a missing baseline are blocked, not hard errors."""
    args = argparse.Namespace(only=["smoke_dsr"], seeds=[0], device=None)
    plans, _ = re.build_plans(SMOKE_MATRIX, args)
    # smoke matrix has no load_weights -> never blocked
    assert all(re.plan_block_reason(p) is None for p in plans)
    fake = dict(plans[0])
    fake["load_weights"] = str(ROOT / "runs/definitely_missing.pt")
    assert re.plan_block_reason(fake) is not None


def test_baseline_weights_seed_template():
    """'baseline' load_weights resolves the {seed} template per plan."""
    args = argparse.Namespace(only=["smoke_dsr"], seeds=[0, 42], device=None)
    plans, g = re.build_plans(SMOKE_MATRIX, args)
    g["baseline_weights"] = "runs/matrix/baseline/seed{seed}/weights/best.pt"
    plans = [re.build_plan({"name": "x", "cfg": p["base_cfg"], "model": p["base_model"],
                            "trainer": "dsr", "load_weights": "baseline"}, seed, g)
             for p, seed in ((plans[0], 0), (plans[0], 42))]
    assert "seed0" in Path(plans[0]["load_weights"]).parts
    assert "seed42" in Path(plans[1]["load_weights"]).parts


def test_smoke_matrix_plan_contents():
    """DSR entries carry dsr overrides; baseline entry does not load weights."""
    args = argparse.Namespace(only=None, seeds=[0], device="cpu")
    plans, _ = re.build_plans(SMOKE_MATRIX, args)
    by_name = {p["exp"]: p for p in plans}
    assert set(by_name) == {"smoke_baseline", "smoke_dsr"}
    dsr = by_name["smoke_dsr"]
    assert dsr["trainer"] == "dsr"
    assert dsr["dsr"]["candidate_conf"] == 0.0001
    assert dsr["model_yaml"].endswith(".yaml")
    assert dsr["load_weights"] is None
    assert by_name["smoke_baseline"]["trainer"] is None


def test_only_filter_and_unknown():
    """--only filters by name and rejects unknown names."""
    args = argparse.Namespace(only=["smoke_dsr"], seeds=None, device=None)
    plans, _ = re.build_plans(SMOKE_MATRIX, args)
    assert [p["exp"] for p in plans] == ["smoke_dsr"]
    args2 = argparse.Namespace(only=["nope"], seeds=None, device=None)
    with pytest.raises(ValueError, match="unknown experiments"):
        re.build_plans(SMOKE_MATRIX, args2)


@pytest.mark.slow
def test_runner_smoke_matrix_end_to_end():
    """Full matrix run on the tiny dataset: train + val, manifest written."""
    rc = re.main(["--matrix", str(SMOKE_MATRIX), "--force"])
    assert rc == 0
    runs_root = ROOT / "runs/matrix_smoke"
    manifest = runs_root / re.MANIFEST_NAME
    assert manifest.exists()
    data = __import__("json").loads(manifest.read_text(encoding="utf-8"))
    assert len(data["plans"]) == 2
    for row in data["plans"]:
        assert row["train_done"] and row["val_done"] and not row["error"], row
        assert (Path(row["run_dir"]) / "per_image_metrics.json").exists()
