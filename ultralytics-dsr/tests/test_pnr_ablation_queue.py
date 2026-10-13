"""Contract tests for the SCB-Dataset3-S PNR mechanism ablation queue."""

from experiments.pnr_ablation_queue import DATA_YAML, Job, _training_source, build_jobs


def test_build_jobs_contains_only_four_missing_mechanisms_for_three_seeds():
    """Baseline and full PNR already have three seeds, so rerunning either wastes controlled compute."""
    jobs = build_jobs()

    assert {(job.variant, job.seed) for job in jobs} == {
        (variant, seed)
        for variant in ("raw_uniform", "norm_uniform", "norm_size", "norm_vertical")
        for seed in (42, 43, 44)
    }
    assert len(jobs) == 12


def test_training_source_injects_only_the_declared_raw_uniform_mechanism():
    """If this mapping is omitted, every ablation silently becomes the full PNR configuration."""
    source = _training_source(
        Job("raw_uniform", 43, False, "uniform", "runs/pnr_mechanism_ablation_3seed/raw_uniform/seed43")
    )

    assert DATA_YAML == "ultralytics/cfg/datasets/SCB-Dataset3-S.yaml"
    assert "'normalize': False" in source
    assert "'weight_mode': 'uniform'" in source
    assert "seed=43, deterministic=True" in source
    assert "epochs=300, patience=100" in source
