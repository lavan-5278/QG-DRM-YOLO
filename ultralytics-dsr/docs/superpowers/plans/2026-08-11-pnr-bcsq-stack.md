# PNR-DFL + BCSQ-Head Implementation Plan

> **For agentic workers:** execute tasks in order with test gates. Do not use Git.

**Goal:** Build and validate the PNR-DFL + BCSQ-Head stack with one strict SCB-Dataset3-S run for each of seeds 42, 43, and 44.

**Architecture:** The terminal `BCSQDetect` continues to emit standard seven-channel inference predictions plus per-class side-quality logits during training. A new `PNRBCSQDetectionLoss` subclasses BCSQ loss semantics, retains the BCSQ side-quality classification supervision, and adds the existing full-joint PNR residual to the DFL loss item. The parser selects this hybrid only when a BCSQ head and a top-level `pnr` mapping coexist.

**Tech Stack:** Python, PyTorch, Ultralytics YOLO11n, pytest, CUDA 11.8, Windows `nvidia-smi`.

## Global Constraints

- Dataset and split: `ultralytics/cfg/datasets/SCB-Dataset3-S.yaml`; no split modifications.
- PNR config: `{alpha: 1.0, lambda_pnr: 0.25, scale_floor: 1.0, normalize: true, weight_mode: joint}`.
- BCSQ config remains exactly `{eta: 0.5, tau: 0.2, lambda_quality: 0.5, background_alpha: 0.75, background_gamma: 2.0}`.
- Protocol: yolo11n.pt initialization, `imgsz=640`, `batch=16`, `device=0`, `workers=8`, 300 epoch maximum, patience 100, deterministic seeds 42/43/44, and otherwise matched optimizer/augmentation settings.
- Output root: `runs/pnr_bcsq_stack_3seed/pnr_bcsq/seed{42,43,44}`. Never overwrite incomplete output.
- No Git commands or artifacts.

---

### Task 1: Define hybrid loss selection and model contract

**Files:**
- Create: `tests/test_pnr_bcsq_stack.py`
- Modify: `ultralytics/utils/loss.py`
- Modify: `ultralytics/nn/tasks.py`
- Create: `ultralytics/cfg/models/11/yolo11n-pnr-bcsq.yaml`

**Interfaces:**
- Produces `PNRBCSQDetectionLoss(model)`, selected by `DetectionModel.init_criterion()` when `isinstance(model.model[-1], BCSQDetect)` and `model.yaml['pnr']` exists.
- Produces an inference contract `(B, 7, N)` and raw training channels `64 + nc + 4*nc`.

- [x] Write a failing test requiring the new YAML, BCSQ terminal head, `PNRBCSQDetectionLoss`, standard inference tensor shape, and a finite backward gradient for `cvq`.
- [x] Run the test; expected failure observed because the YAML and hybrid criterion did not exist.
- [x] Implement the hybrid loss by preserving `BCSQDetectionLoss.__call__` parsing, assignment, BCE, and quality loss; add `lambda_pnr * _pnr_residual(...)` only to `loss[2]` when foreground anchors exist.
- [x] Add the loss import and selection branch before ordinary BCSQ selection; create the YAML with both unchanged mappings and BCSQ terminal head.
- [x] Re-run the test; passed.

### Task 2: Define strict three-seed queue contract

**Files:**
- Modify: `tests/test_pnr_bcsq_stack.py`
- Create: `experiments/pnr_bcsq_stack_queue.py`

**Interfaces:**
- Produces `build_jobs() -> list[Job]` with exactly seeds 42, 43, 44, and `_training_source(job) -> str` matching the fixed protocol.

- [x] Write failing queue tests for the combined YAML, exact output root, all three seeds, and 300-epoch deterministic arguments.
- [x] Run tests; expected missing queue-module failure observed.
- [x] Implement a serial atomic-state queue with GPU memory gate, incomplete-output refusal, per-job logs, and halt-on-first-failure behavior.
- [x] Re-run queue tests; passed.

### Task 3: CUDA smoke and full queue launch

**Files:**
- Create at runtime: `runs/pnr_bcsq_stack_smoke/seed42/*`, `runs/pnr_bcsq_stack_3seed/*`.

- [x] Run all BCSQ, PNR, and stack unit tests plus compilation (18 passed, 1 CUDA-only test skipped in CPU test environment).
- [x] Run a CUDA one-epoch fraction smoke under `lfm-yolo-mamba-stage14` and inspect metrics/logs for NaN, Inf, OOM, or output mismatch.
- [x] Preflight output paths and GPU, launch the three-seed queue detached, and create a 15-minute heartbeat monitor.
- [ ] At completion, compare per-seed and mean Precision, Recall, mAP50, and mAP50-95 against matching Baseline, PNR, and BCSQ. Promote only if stacked mean mAP50-95 is at least +0.00200 versus each comparator; otherwise reject the stack.
