# SATA Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the SATA social-anchor backbone adapter to YOLOv8n and append its scratch experiment to the completed module queue.

**Architecture:** SATA is a C4/C5-only, sparse learned-anchor relation block. It selects at most 32 feature anchors, performs coordinate-biased six-neighbor attention, and returns a zero-initialized residual feature update without changing detector tensor contracts.

**Tech Stack:** Python 3, PyTorch, Ultralytics YOLO, YAML, pytest.

## Global Constraints

- Dataset 5k_HRW; seed 42; no pretrained weights; batch 8; SGD; lr0 0.01; 300 epochs; 640 by 640.
- SATA is standalone: no AFPN, DSR, PNR, SLED, BCSQ, or FADWC stacking.
- Preserve existing completed queue records; append SATA only.
- Follow test-first red-green-refactor.

---

### Task 1: Prove the standalone SATA feature contract

**Files:**
- Create: `tests/test_sata_module.py`
- Modify: `ultralytics/nn/modules/block.py`

**Interfaces:**
- Produces: `SATA(c1: int, relation_dim: int = 64, max_anchors: int = 32, neighbor_k: int = 6)` returning a tensor with the same B, C, H, W shape.

- [ ] Write tests that call the not-yet-exported `SATA` on a literal `(2, 32, 10, 12)` tensor, assert a finite same-shape identity at initialization, then set `gamma` to one and assert a finite nonidentity output.
- [ ] Run `pytest tests/test_sata_module.py -v`; expect import failure because SATA does not exist.
- [ ] Implement minimal learned scoring, top-K gathering, coordinate-biased KNN attention, scatter-back, depth-wise diffusion and zero-scaled residual in `block.py`.
- [ ] Export SATA through `ultralytics/nn/modules/__init__.py` and register its single-input channel rule in `ultralytics/nn/tasks.py`.
- [ ] Run `pytest tests/test_sata_module.py -v`; expect pass.

### Task 2: Prove model-parser integration

**Files:**
- Modify: `tests/test_sata_module.py`
- Create: `ultralytics/cfg/models/v8/yolov8n-sata.yaml`

**Interfaces:**
- Consumes: `SATA` registered in the model parser.
- Produces: YOLOv8n configuration with SATA after C4 C2f and C5 SPPF.

- [ ] Add a test that constructs the SATA YAML and executes a no-grad 640 by 640 forward pass, asserting finite predictions with 7 channels for the three-class detector.
- [ ] Run the test; expect YAML file-not-found failure.
- [ ] Add the remapped YOLOv8n YAML where backbone layers 7 and 13 are SATA, and preserve the original P3/P4/P5 Detect connections.
- [ ] Run `pytest tests/test_sata_module.py -v`; expect pass.

### Task 3: Append the experiment without corrupting completed state

**Files:**
- Create: `tests/test_train_hrw_yolov8n_module_queue.py`
- Modify: `experiments/train_hrw_yolov8n_module_queue.py`

**Interfaces:**
- Produces: `reconcile_state(state: dict) -> tuple[dict, bool]`, preserving known jobs and appending the SATA job when missing.

- [ ] Add a test with five literal completed job records and assert reconciliation preserves their identifiers/statuses and appends `yolov8n_sata` as pending.
- [ ] Run `pytest tests/test_train_hrw_yolov8n_module_queue.py -v`; expect import/attribute failure because reconciliation does not exist.
- [ ] Add the SATA job tuple, reconciliation helper, report title update, and reconciliation call before report refresh; reject reordered or unknown jobs.
- [ ] Run the queue-state test and `python -m json.tool runs/hrw_yolov8n_modules_seed42_scratch/queue_state.json`; expect pass and parseable state.

### Task 4: Launch and verify queue operation

**Files:**
- Modify: `runs/hrw_yolov8n_modules_seed42_scratch/queue_state.json` only through the queue runner's atomic writer.

- [ ] Run the two SATA test files together.
- [ ] Start `experiments/train_hrw_yolov8n_module_queue.py --resume` as a hidden background process with the established environment Python and log files.
- [ ] Confirm queue state shows only SATA as running/pending, then confirm a results CSV appears.
- [ ] Update the monitoring automation to one-hour cadence and report progress, errors, and final benchmark/report row.
