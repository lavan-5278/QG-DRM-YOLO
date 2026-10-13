# SATA Reliability and Dual-Role Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and serially train RG-SATA and DR-SATA as independent YOLOv8n backbone experiments.

**Architecture:** RG-SATA gates a sparse relation graph with local reliability and semantic-geometric neighbor selection. DR-SATA splits sparse anchors into subject and context roles and performs only context-to-subject message passing. Both preserve the current SATA feature contract and begin as identity maps.

**Tech Stack:** Python 3.11, PyTorch, Ultralytics YOLO, YAML, pytest.

## Global Constraints

- Dataset 5k_HRW; seed 42; scratch/no pretrained weights; batch 8; SGD; lr0 0.01; 300 epochs; 640 by 640.
- Both variants run independently; do not stack with SATA, AFPN, QG-AFPN, DSR, PNR, SLED, BCSQ, FADWC, or CLSC.
- Preserve all completed QG-AFPN, CLSC, SATA, and prior module records.
- Write all new rows to the standard Chinese report with the fixed 50-warmup plus 200-forward benchmark protocol.

---

### Task 1: Define the two shape-preserving adapters by test

**Files:**
- Modify: `tests/test_sata_module.py`
- Modify: `ultralytics/nn/modules/block.py`

**Interfaces:**
- Produces: `RGSATA(c1, relation_dim=64, max_anchors=32, neighbor_k=6)`.
- Produces: `DRSATA(c1, relation_dim=64, subject_anchors=16, context_anchors=16, neighbor_k=6)`.

- [ ] Add tests that instantiate each adapter with literal C4-shaped inputs, assert finite output with identical B,C,H,W shape and exact initialization identity.
- [ ] Add tests that set `gamma=1` and assert a finite nonidentity update, naming the missing reliability/role scatter-back regression each protects.
- [ ] Run the SATA test file and verify import failure for RGSATA and DRSATA.
- [ ] Implement the minimal modules, export them, and add the single-input parser rules.
- [ ] Run the SATA tests and verify all adapter contracts pass.

### Task 2: Build both detector graphs before training

**Files:**
- Modify: `tests/test_sata_module.py`
- Create: `ultralytics/cfg/models/v8/yolov8n-rgsata.yaml`
- Create: `ultralytics/cfg/models/v8/yolov8n-drsata.yaml`

**Interfaces:**
- Consumes: parser-registered RGSATA and DRSATA modules.
- Produces: two three-class YOLOv8n C4/C5-backbone configurations with unmodified P3/P4/P5 Detect routing.

- [ ] Add a parametrized 640-pixel forward test for both YAML paths and assert finite 7-channel prediction tensors.
- [ ] Run the test and verify both YAML paths fail before files exist.
- [ ] Add remapped YOLOv8n YAMLs with C4/C5 modules and original neck routing.
- [ ] Run the full SATA test file and verify all graph tests pass.

### Task 3: Add a safe two-job serial experiment queue

**Files:**
- Create: `tests/test_train_hrw_yolov8n_sata_variants_queue.py`
- Create: `experiments/train_hrw_yolov8n_sata_variants_queue.py`

**Interfaces:**
- Produces: `initial_state()` and `reconcile_state(state) -> tuple[dict, bool]` for ordered `yolov8n_rgsata`, `yolov8n_drsata` jobs.

- [ ] Add a state test that starts from an empty job list and asserts reconciliation appends RG-SATA then DR-SATA, both pending, under the fixed protocol.
- [ ] Run the new test and verify import failure because the launcher does not exist.
- [ ] Implement atomic state writes, training kwargs, best-row extraction, checkpoint benchmark, report replacement marker, and serial training loop following the existing QG/CLSC launcher.
- [ ] Run the queue test and launcher `--dry-run`; verify both YAMLs build and JSON state parses.

### Task 4: Launch and monitor both variants

**Files:**
- Runtime output: `runs/hrw_yolov8n_sata_variants_seed42_scratch/`.

- [ ] Start the launcher hidden in the project root, with stdout/stderr under its `logs` directory.
- [ ] Confirm RG-SATA becomes running and a results CSV appears before leaving the queue unattended.
- [ ] Monitor hourly without changing the active process; after each completion verify report row and benchmark fields.
- [ ] When both jobs complete, compare P/R/mAP50/mAP50-95 against YOLOv8n and SATA, state the single-seed/single-split limitation, and remove the heartbeat.
