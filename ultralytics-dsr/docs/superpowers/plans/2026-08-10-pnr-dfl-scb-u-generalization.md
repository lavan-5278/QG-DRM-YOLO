# PNR-DFL SCB-Dataset3-U Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a reproducible, paired three-seed comparison of Baseline and PNR-DFL on the official six-class SCB-Dataset3-U split.

**Architecture:** A dedicated local sequential queue declares the six fixed jobs, writes atomic queue state, and launches the same YOLO11n protocol already used for SCB-Dataset3-S. The only model difference is the PNR top-level loss configuration; Ultralytics obtains `nc=6` from the dataset YAML for both variants.

**Tech Stack:** Python 3, Ultralytics YOLO11, PyTorch/CUDA, pytest, NVIDIA RTX 4080 SUPER.

## Global Constraints

- No Git operations or worktrees.
- Dataset: `ultralytics/cfg/datasets/SCB-Dataset3-U.yaml`; official train/val split; six classes.
- Variants: baseline matched YAML and `yolo11n-pnr-dfl.yaml`; seeds 42, 43, 44.
- Fixed protocol: 300 maximum epochs, patience 100, imgsz 640, batch 16, device 0, workers 8, pretrained checkpoint loading, deterministic seed, and the established optimiser/augmentation settings.
- Compare paired best `metrics/mAP50-95(B)` deltas only within the same seed; do not compare U absolute mAP with 3-S absolute mAP.

---

### Task 1: Define and test the U generalization queue

**Files:**
- Create: `experiments/pnr_u_generalization_queue.py`
- Create: `tests/test_pnr_u_generalization_queue.py`

**Interfaces:**
- Produces `build_jobs() -> list[Job]` with exactly baseline/PNR × seeds 42/43/44.
- Produces `_training_source(job: Job) -> str` containing the fixed six-class protocol.
- Produces `run_queue(...) -> int`, atomically persisting `runs/pnr_u_generalization_3seed/queue_state.json`.

- [ ] **Step 1: Write failing tests** for the six job identities and U-specific generated training source.
- [ ] **Step 2: Run the test** with `python -m pytest --basetemp .pytest-pnr-u tests/test_pnr_u_generalization_queue.py -v`; verify expected import failure.
- [ ] **Step 3: Implement minimal queue** with complete-check validation (`results.csv` plus `weights/best.pt`), stdout/stderr files, process wait, and resumable status.
- [ ] **Step 4: Re-run test** and compile the queue.

### Task 2: Launch and verify the six paired jobs

**Files:**
- Create at runtime: `runs/pnr_u_generalization_3seed/queue_state.json`
- Create at runtime: `runs/pnr_u_generalization_3seed/logs/*.out`, `*.err`
- Create at runtime: `runs/pnr_u_generalization_3seed/{baseline,pnr_dfl}/seed{42,43,44}`

- [ ] **Step 1:** Check GPU 0 memory and ensure no training job owns it.
- [ ] **Step 2:** Launch queue in a hidden background PowerShell process with `POLARS_SKIP_CPU_CHECK=1`.
- [ ] **Step 3:** Verify queue process, state JSON, and first job log.
- [ ] **Step 4:** Monitor every 15 minutes; flag NaN/Inf, OOM, log errors, failed jobs, or incomplete output immediately.

### Task 3: Analyse the paired six-class result

**Files:**
- Create: `refine-logs/PNR-DFL_SCB-Dataset3-U_泛化报告.md`

- [ ] **Step 1:** Read best-mAP50-95 rows for every completed run, retaining P/R, mAP50, mAP50-95 and loss values from the same row.
- [ ] **Step 2:** Compute each PNR-minus-baseline paired delta and mean ± sample standard deviation.
- [ ] **Step 3:** Record the six-class scope, official split, fixed protocol, all raw metrics, verdict and the three-seed limitation.
