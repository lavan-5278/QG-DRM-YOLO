"""Queue contract for C2f-LGC partial/full replacement ablations."""

import importlib.util
from pathlib import Path


def load_queue_module():
    script = Path(__file__).resolve().parents[1] / "experiments/train_hrw_yolov8n_c2f_lgc_queue.py"
    spec = importlib.util.spec_from_file_location("c2f_lgc_queue", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_c2f_lgc_queue_orders_partial_before_full_replacement(tmp_path, monkeypatch):
    """The more conservative C2f-LGC comparison must finish before the high-risk full replacement."""
    queue = load_queue_module()
    monkeypatch.setattr(queue, "STATE_PATH", tmp_path / "queue_state.json")

    state = queue.load_state()

    assert [job["id"] for job in state["jobs"]] == ["yolov8n_c2f_lgc", "yolov8n_c2f_to_lgc"]
    assert state["protocol"]["pretrained"] is False
    assert state["protocol"]["epochs"] == 300
    assert state["protocol"]["batch"] == 8


def test_c2f_lgc_queue_waits_for_the_lhgb_predecessor(tmp_path, monkeypatch):
    """Starting a second queue before LHGB finishes would overlap GPU experiments and invalidate the protocol."""
    queue = load_queue_module()
    predecessor = tmp_path / "lfm_state.json"
    predecessor.write_text('{"jobs":[{"id":"yolov8n_lhgb","status":"running"}]}', encoding="utf-8")
    monkeypatch.setattr(queue, "LFM_STATE_PATH", predecessor)

    assert queue.predecessor_complete() is False
