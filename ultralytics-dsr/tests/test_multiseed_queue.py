"""Contracts for the local three-seed validation queue."""

from experiments.multiseed_queue import build_jobs


def test_three_seed_queue_contains_exactly_the_missing_paired_validation_runs():
    """Omitting baseline seeds or duplicating seed 42 would make module gains statistically incomparable."""
    jobs = build_jobs()

    assert len(jobs) == 12
    assert {(job.variant, job.seed) for job in jobs} == {
        ("baseline", 43),
        ("baseline", 44),
        ("bcsq", 43),
        ("bcsq", 44),
        ("sled", 43),
        ("sled", 44),
        ("pnr_dfl", 43),
        ("pnr_dfl", 44),
        ("sera", 43),
        ("sera", 44),
        ("hbpc", 43),
        ("hbpc", 44),
    }
    assert {job.yaml for job in jobs if job.variant == "baseline"} == {
        "ultralytics/cfg/models/11/yolo11n-baseline-matched.yaml"
    }
    assert {job.yaml for job in jobs if job.variant == "hbpc"} == {"ultralytics/cfg/models/11/yolo11n-hbpc.yaml"}
    assert len({job.run_dir for job in jobs}) == len(jobs)
