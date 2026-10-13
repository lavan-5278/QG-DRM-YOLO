# PNR-DFL + SLED-Head Implementation Plan

> **For agentic workers:** execute the checked tasks in order; each task has a test gate before the next task. Do not use Git.

**Goal:** Build a single YAML-selectable PNR-DFL + SLED-Head detector and launch a reproducible three-seed SCB-Dataset3-S stacking validation.

**Architecture:** The stack uses `SLEDDetect` unchanged as the terminal detection head and adds the existing top-level `pnr` loss configuration to the same model YAML. `DetectionModel.init_criterion()` already selects `PNRDetectionLoss` whenever the YAML has `pnr`, so no second head, loss dispatcher, tensor format, decoder, or NMS path is introduced. A serial local queue runs seeds 42, 43, and 44 under the strict matched protocol and records atomic JSON state plus per-job logs.

**Tech Stack:** Python, PyTorch, Ultralytics YOLO11n, pytest, Windows `nvidia-smi`.

## Global Constraints

- Dataset: `ultralytics/cfg/datasets/SCB-Dataset3-S.yaml`; no test-set or split changes.
- Protocol: yolo11n.pt initialization, 640 image size, batch 16, device 0, 300 epochs maximum, patience 100, seeds 42/43/44, deterministic=True, all other optimizer and augmentation settings identical to the matched baseline.
- The new combination is `SLEDDetect` + `PNRDetectionLoss`; PNR configuration is `{alpha: 1.0, lambda_pnr: 0.25, scale_floor: 1.0, normalize: true, weight_mode: joint}`.
- Output root: `runs/pnr_sled_stack_3seed/pnr_sled/seed{42,43,44}`. Never overwrite incomplete output.
- No Git commands or Git artifacts.

---

### Task 1: Define and verify the compositional YAML contract

**Files:**
- Create: `tests/test_pnr_sled_stack.py`
- Create: `ultralytics/cfg/models/11/yolo11n-pnr-sled.yaml`

**Interfaces:**
- Consumes: existing `SLEDDetect`, `PNRDetectionLoss`, and YAML parser registrations.
- Produces: `yolo11n-pnr-sled.yaml`, whose final layer is `SLEDDetect` and whose model criterion is `PNRDetectionLoss`.

- [ ] Write a failing test that asserts the new YAML exists, builds a three-class detector, emits standard `(B, 7, N)` inference predictions and 67-channel raw feature levels, constructs `SLEDDetect`, and selects `PNRDetectionLoss`.
- [ ] Run `pytest tests/test_pnr_sled_stack.py::test_pnr_sled_yaml_composes_sled_head_with_pnr_loss -v`; expect failure because the YAML does not exist.
- [ ] Create the YAML by copying the body/head of `yolo11n-sled.yaml` exactly and adding only the full-joint `pnr` mapping at top level.
- [ ] Re-run the test; expect pass.

### Task 2: Define and verify the three-seed queue contract

**Files:**
- Modify: `tests/test_pnr_sled_stack.py`
- Create: `experiments/pnr_sled_stack_queue.py`

**Interfaces:**
- Consumes: `Job(variant, seed, yaml, run_dir)` and the compositional YAML.
- Produces: `build_jobs() -> list[Job]` containing exactly the three stacked seeds and `_training_source(job) -> str` containing the strict protocol.

- [ ] Write failing tests that require only seeds 42/43/44, the combined YAML, the dedicated output root, PNR full-joint declaration in state, and exact 300-epoch deterministic protocol text.
- [ ] Run those tests; expect import failure because the queue module does not exist.
- [ ] Implement the serial/resumable queue: wait for optional PID, wait for GPU memory below 4000 MiB, write atomic queue state, reject incomplete output, and stop on the first failed job.
- [ ] Re-run the queue contract tests; expect pass.

### Task 3: Verify registration, shape, and training execution

**Files:**
- Modify: `tests/test_pnr_sled_stack.py` only if a regression is revealed.

**Interfaces:**
- Consumes: YAML model and queue training source.
- Produces: proof that SLED affects only classification features while PNR remains an additional regression loss.

- [ ] Run the PNR/SLED/stack test suite on CPU.
- [ ] Run a CUDA model build, pretrained-weight loading, one forward/backward loss pass, and a one-epoch `fraction` smoke training using an isolated `runs/pnr_sled_stack_smoke/seed42` directory.
- [ ] Inspect smoke logs/results for NaN, Inf, OOM, and model-output mismatch; stop rather than launch the three seeds if any appears.

### Task 4: Start and monitor strict three-seed training

**Files:**
- Create at runtime: `runs/pnr_sled_stack_3seed/queue_state.json`, `runs/pnr_sled_stack_3seed/logs/*.out`, and each seed run directory.

**Interfaces:**
- Consumes: `python experiments/pnr_sled_stack_queue.py`.
- Produces: seed 42/43/44 `results.csv` and `weights/best.pt` for paired analysis against existing Baseline, PNR, and SLED runs.

- [ ] Preflight GPU availability and confirm all three target run directories do not exist.
- [ ] Start the queue detached, recording its PID in the launcher files.
- [ ] Create a 15-minute status automation that reads queue state, current CSV/logs, GPU status, and immediately flags NaN/Inf/OOM/failure.
- [ ] At completion, compare each seed to the matching Baseline, PNR, and SLED best mAP50-95 values. Promote complementarity only if the stacked mean is at least +0.00200 above Baseline and at least +0.00200 above each individual constituent mean; otherwise report the measured result without a complementarity claim.
