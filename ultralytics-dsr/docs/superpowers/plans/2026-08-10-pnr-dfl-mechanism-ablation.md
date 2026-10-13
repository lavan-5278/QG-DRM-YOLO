# PNR-DFL 3-S Mechanism Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate whether PNR-DFL gains on SCB-Dataset3-S require relative-scale normalization and the joint perspective weight.

**Architecture:** Extend only the training-time PNR loss configuration with `normalize` and `weight_mode`; the YOLO head, assignment, decoding, standard DFL and all training hyperparameters stay unchanged. A sequential queue runs only the four missing ablation variants for seeds 42/43/44, while existing matched Baseline and Full-PNR three-seed runs provide the two endpoints.

**Tech Stack:** Python 3.11, PyTorch/CUDA, Ultralytics YOLO11, pytest, RTX 4080 SUPER.

## Global Constraints

- No Git operations or worktrees.
- Data: `SCB-Dataset3-S` official train/val split; 300 maximum epochs, patience 100, imgsz 640, batch 16, device 0, seed deterministic and established optimiser/augmentation protocol.
- Existing endpoint results are reused: Baseline seeds 42/43/44 and Full PNR seeds 42/43/44.
- New variants: `raw_uniform` (`normalize=false`, `weight_mode=uniform`), `norm_uniform` (`true`, `uniform`), `norm_size` (`true`, `size`), and `norm_vertical` (`true`, `vertical`), all with alpha=1 and lambda=0.25.
- Full PNR is `normalize=true`, `weight_mode=joint`; its weight is `1 + alpha*(1-y)*(1-r)`.
- Paired comparisons use each seed’s selected best mAP50-95 row and retain all metrics from that same row.

---

### Task 1: Make PNR mechanisms configurable and testable

**Files:**
- Modify: `ultralytics/utils/loss.py:413-500`
- Modify: `tests/test_pnr_dfl.py`

**Interfaces:**
- `PNRDetectionLoss` reads `pnr.normalize: bool` (default `true`) and `pnr.weight_mode: uniform|size|vertical|joint` (default `joint`).
- `_perspective_weights(...) -> Tensor` implements each configured geometry weighting.
- `_pnr_residual(...) -> Tensor` applies feature-grid scale division only when `normalize` is true.

- [ ] **Step 1:** Add failing numerical tests proving uniform weights are one, size/vertical weights isolate their own geometry factor, and raw residual exceeds normalized residual for the same small-target error.
- [ ] **Step 2:** Run `python -m pytest --basetemp .pytest-pnr-abl tests/test_pnr_dfl.py -v`; verify those mechanism tests fail under the current full-only implementation.
- [ ] **Step 3:** Add default-preserving configuration parsing and the minimal two branch points in loss computation.
- [ ] **Step 4:** Re-run the test suite and compile `loss.py`.

### Task 2: Queue the missing 3-S ablation cells

**Files:**
- Create: `experiments/pnr_ablation_queue.py`
- Create: `tests/test_pnr_ablation_queue.py`

**Interfaces:**
- `build_jobs() -> list[Job]` returns four mechanisms × seeds 42/43/44, 12 jobs.
- `_training_source(job: Job) -> str` sets the exact PNR mapping on the training model before its trainer is constructed.
- `run_queue(...) -> int` persists `runs/pnr_mechanism_ablation_3seed/queue_state.json` atomically and stops on first bad cell.

- [ ] **Step 1:** Write failing queue tests asserting exact 12 job identities and their injected mechanism configs.
- [ ] **Step 2:** Run the queue test and confirm it fails because the module does not exist.
- [ ] **Step 3:** Implement a resumable sequential CUDA queue with per-job stdout/stderr and output completeness checks.
- [ ] **Step 4:** Re-run queue tests, compile, and smoke-construct an ablation model with the modified PNR mapping.

### Task 3: Launch and analyse

**Files:**
- Create at runtime: `runs/pnr_mechanism_ablation_3seed/`
- Create after completion: `refine-logs/PNR-DFL_3S机制消融报告.md`

- [ ] **Step 1:** Confirm GPU 0 availability and launch the 12 missing jobs with CUDA Python 3.11.15 / torch 2.3.0+cu118.
- [ ] **Step 2:** Monitor at 15-minute intervals for progress, NaN/Inf, OOM, abnormal exit, or queue state failure.
- [ ] **Step 3:** Analyse all six rows (Baseline, four ablations, Full PNR) across three seeds; report P/R, mAP50, mAP50-95, train/val losses, paired deltas, and the mechanism verdict.
