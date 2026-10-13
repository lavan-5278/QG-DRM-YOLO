"""Queue contract for the three LFM-inspired scratch YOLOv8n variants."""

import importlib.util
from pathlib import Path


def load_queue_module():
    script = Path(__file__).resolve().parents[1] / "experiments/train_hrw_yolov8n_lfm_backbone_queue.py"
    spec = importlib.util.spec_from_file_location("lfm_backbone_queue", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lfm_backbone_queue_initializes_ordered_scratch_jobs(tmp_path, monkeypatch):
    """LGC must establish the local-state baseline before LGRB and hierarchical LHGB run."""
    queue = load_queue_module()
    monkeypatch.setattr(queue, "STATE_PATH", tmp_path / "queue_state.json")

    state = queue.load_state()

    assert [job["id"] for job in state["jobs"]] == ["yolov8n_lgc", "yolov8n_lgrb", "yolov8n_lhgb"]
    assert all(job["status"] == "pending" for job in state["jobs"])
    assert state["protocol"] == {
        "dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8,
        "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD",
    }


def test_lfm_backbone_queue_reconciliation_rejects_order_changes():
    """A stale or manually reordered state must not silently compare different experiment definitions."""
    queue = load_queue_module()
    state = queue.initial_state()
    state["jobs"][0]["id"] = "wrong_order"

    try:
        queue.reconcile_state(state)
    except RuntimeError as error:
        assert "Queue definition changed" in str(error)
    else:
        raise AssertionError("Expected queue reconciliation to reject an altered first job")
