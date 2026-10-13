"""Queue contract for the two scratch YOLOv8n SATA internal variants."""

from experiments.train_hrw_yolov8n_sata_variants_queue import initial_state, reconcile_state


def test_sata_variant_queue_appends_reliability_then_dual_role_jobs_to_an_empty_state():
    """Reversing the order would prevent the reliability-first comparison requested for SATA refinement."""
    state = initial_state()
    state["jobs"] = []

    reconciled, changed = reconcile_state(state)

    assert changed is True
    assert reconciled["protocol"] == {
        "dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8,
        "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD",
    }
    assert [(job["id"], job["status"]) for job in reconciled["jobs"]] == [
        ("yolov8n_rgsata", "pending"), ("yolov8n_drsata", "pending")
    ]
