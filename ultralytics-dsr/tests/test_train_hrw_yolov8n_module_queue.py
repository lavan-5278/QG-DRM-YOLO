"""Queue-state migration tests for appending standalone YOLOv8n + SATA."""

from copy import deepcopy

from experiments.train_hrw_yolov8n_module_queue import initial_state, reconcile_state
from experiments.train_hrw_yolov8n_qg_clsc_queue import reconcile_state as reconcile_qg_clsc_state


def test_reconcile_state_preserves_completed_module_records_after_sata_is_transferred():
    """Retiring a transferred pending job must never discard completed benchmark evidence."""
    state = initial_state()
    for job in state["jobs"]:
        job["status"] = "completed"
        job["metrics"] = {"map5095": "0.5000"}
        job["summary"] = {"parameters": 1}
    original_jobs = deepcopy(state["jobs"])

    reconciled, changed = reconcile_state(state)

    assert changed is False
    assert reconciled["jobs"] == original_jobs


def test_reconcile_state_removes_only_the_transferred_pending_sata_record():
    """Leaving SATA pending in both queues would train the same experiment twice."""
    state = initial_state()
    state["jobs"].append(
        {"id": "yolov8n_sata", "name": "YOLOv8n + SATA（Backbone 社会上下文）", "yaml": "sata.yaml", "status": "pending"}
    )

    reconciled, changed = reconcile_state(state)

    assert changed is True
    assert [job["id"] for job in reconciled["jobs"]] == [
        "yolov8n_pnr", "yolov8n_sled", "yolov8n_bcsq", "yolov8n_afpn", "yolov8n_fadwc"
    ]


def test_qg_clsc_reconciliation_appends_sata_after_existing_two_jobs():
    """Appending SATA ahead of QG-AFPN or CLSC would violate the requested serial experiment order."""
    state = {
        "protocol": {"dataset": "5k_HRW"},
        "jobs": [
            {"id": "yolov8n_qg_afpn", "name": "YOLOv8n + QG-AFPN", "yaml": "qg.yaml", "status": "completed"},
            {"id": "yolov8n_clsc", "name": "YOLOv8n + CLSC Head", "yaml": "clsc.yaml", "status": "completed"},
        ],
        "running": None,
    }

    reconciled, changed = reconcile_qg_clsc_state(state)

    assert changed is True
    assert [job["id"] for job in reconciled["jobs"]] == ["yolov8n_qg_afpn", "yolov8n_clsc", "yolov8n_sata"]
    assert reconciled["jobs"][-1]["status"] == "pending"
