"""Contract tests for the SCB-Dataset3-U paired PNR generalization queue."""

from experiments.pnr_u_generalization_queue import DATA_YAML, Job, _training_source, build_jobs


def test_build_jobs_contains_only_paired_six_class_baseline_and_pnr_seeds():
    """A missing or duplicated seed would invalidate paired three-seed comparison."""
    jobs = build_jobs()

    assert {(job.variant, job.seed) for job in jobs} == {
        (variant, seed) for variant in ("baseline", "pnr_dfl") for seed in (42, 43, 44)
    }
    assert len(jobs) == 6


def test_training_source_uses_u_yaml_and_fixed_matched_protocol():
    """Changing dataset or protocol would make the reported U delta non-comparable."""
    source = _training_source(
        Job("pnr_dfl", 44, "ultralytics/cfg/models/11/yolo11n-pnr-dfl.yaml", "runs/pnr_u_generalization_3seed/pnr_dfl/seed44")
    )

    assert DATA_YAML == "ultralytics/cfg/datasets/SCB-Dataset3-U.yaml"
    assert 'data="ultralytics/cfg/datasets/SCB-Dataset3-U.yaml"' in source
    assert "epochs=300, patience=100" in source
    assert "imgsz=640, batch=16, device=0, workers=8" in source
    assert "seed=44, deterministic=True" in source
