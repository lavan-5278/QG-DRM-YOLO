# PNR-DFL + AFPN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a standard three-scale AFPN Neck combined with the existing full-joint PNR-DFL criterion on SCB-Dataset3-S using three matched 300-epoch seeds.

**Architecture:** AFPN receives the existing P3/P4/P5 Neck features once, aligns their channel and spatial resolutions, and emits three refined features through asymptotic fusion. Three native `Index` nodes select the refined P3/P4/P5 tensors for the unchanged standard `Detect` head. PNRDetectionLoss remains selected solely through the existing top-level `pnr` mapping, so assignment, DFL decoding, inference layout, NMS, and prediction channels stay unchanged.

**Tech Stack:** PyTorch, Ultralytics YOLO11, CUDA Torch environment `lfm-yolo-mamba-stage14`, pytest, PowerShell.

## Global Constraints

- Do not use Git.
- Data split is exactly `ultralytics/cfg/datasets/SCB-Dataset3-S.yaml`.
- Training protocol is exactly YOLO11n pretrained initialization, 300 epochs, images 640, batch 16, GPU 0, workers 8, deterministic seeds 42/43/44.
- PNR config is `{alpha: 1.0, lambda_pnr: 0.25, scale_floor: 1.0, normalize: true, weight_mode: joint}`.
- AFPN is standard three-scale P3/P4/P5; no P2 branch is added.
- Promote only when mean mAP50-95 is at least +0.00200 versus Baseline and PNR-DFL.

---

### Task 1: AFPN Neck module and parser registration

**Files:**
- Modify: `ultralytics/nn/modules/block.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Test: `tests/test_pnr_afpn_stack.py`

**Interfaces:**
- Consumes: `list[Tensor]` of P3/P4/P5 features with channels supplied by `parse_model`.
- Produces: `AFPN(ch: tuple[int, int, int]) -> list[Tensor]`, preserving each source feature's batch size, channel count, height, and width.

- [ ] **Step 1: Write a failing shape/gradient test**

```python
afpn = AFPN((16, 32, 64))
features = [torch.randn(2, 16, 16, 16, requires_grad=True), torch.randn(2, 32, 8, 8, requires_grad=True), torch.randn(2, 64, 4, 4, requires_grad=True)]
outputs = afpn(features)
assert [x.shape for x in outputs] == [x.shape for x in features]
sum(x.mean() for x in outputs).backward()
assert all(x.grad is not None for x in features)
```

- [ ] **Step 2: Run the new test and verify it fails because `AFPN` is unavailable.**

Run: `python -m pytest --basetemp .pytest-pnr-afpn tests/test_pnr_afpn_stack.py::test_afpn_preserves_three_scale_shapes_and_gradients -v`

- [ ] **Step 3: Implement the minimal AFPN and parser branch.**

`AFPN` creates per-level 1×1 projections for every target level, resizes all projected inputs to the target spatial size with nearest-neighbour interpolation, then applies two residual `Conv` fusion stages in low-to-high asymptotic order. `parse_model` passes the three source channel counts into `AFPN`; regular `Index` nodes select the list outputs with their declared output channels.

- [ ] **Step 4: Run the Task 1 test and verify it passes.**

### Task 2: PNR-AFPN YAML and model contract

**Files:**
- Create: `ultralytics/cfg/models/11/yolo11n-pnr-afpn.yaml`
- Modify: `tests/test_pnr_afpn_stack.py`

**Interfaces:**
- Consumes: `AFPN`, `Index`, unchanged `Detect`, and existing `PNRDetectionLoss` selection.
- Produces: a model whose inference output is `(B, 7, N)`, whose raw levels have 67 channels, and whose criterion is `PNRDetectionLoss`.

- [ ] **Step 1: Write a failing YAML contract test.**

```python
model = YOLO(STACK_YAML).model.eval()
criterion_model = DetectionModel(STACK_YAML, verbose=False)
prediction, raw = model(torch.zeros(1, 3, 64, 64))
assert isinstance(model.model[23], AFPN)
assert prediction.shape[1] == 7
assert all(level.shape[1] == 67 for level in raw)
assert isinstance(criterion_model.init_criterion(), PNRDetectionLoss)
```

- [ ] **Step 2: Run the contract test and verify it fails because the YAML is missing.**

- [ ] **Step 3: Create the YAML.**

Keep the matched YOLO11n backbone and original Neck through layer 22. Add `[[16,19,22], 1, AFPN, []]`, then three `Index` nodes `[64,0]`, `[128,1]`, `[256,2]` at base scale, and connect those three nodes to `Detect`. Include only the standard full-joint top-level `pnr` mapping.

- [ ] **Step 4: Run the Task 1–2 tests and compile touched files.**

### Task 3: Reproducible three-seed training and final report

**Files:**
- Create: `experiments/pnr_afpn_stack_queue.py`
- Create: `experiments/report_pnr_afpn_stack.py`
- Modify: `tests/test_pnr_afpn_stack.py`

**Interfaces:**
- Consumes: PNR-AFPN YAML and existing Baseline/PNR matched `results.csv` files.
- Produces: serial seed42/43/44 runs under `runs/pnr_afpn_stack_3seed/pnr_afpn/seed{seed}` and `stack_comparison.md` with per-seed and paired all-metric summaries.

- [ ] **Step 1: Write failing queue/report tests.**

Assert exactly three `pnr_afpn` jobs, strict train source arguments, and final reporting of paired Precision/Recall/mAP50/mAP50-95 mean plus sample standard deviation versus Baseline and PNR-DFL.

- [ ] **Step 2: Run the tests and verify they fail because queue/report modules are missing.**

- [ ] **Step 3: Implement the queue and report.**

Use atomic UTF-8 queue state, block incomplete pre-existing output directories, gate launch until GPU memory is at most 4000 MiB, and halt on the first failed job. Select each run's highest mAP50-95 epoch for comparison. Mark `PROMOTE_STACK` only if the PNR-AFPN mean is at least +0.00200 over both endpoints.

- [ ] **Step 4: Run all AFPN/PNR tests, compile production scripts, then run a one-epoch CUDA smoke training.**

- [ ] **Step 5: Launch the serial three-seed queue and activate a 15-minute monitor.**

### Self-Review

- PNR remains an unchanged loss-only extension; AFPN is a three-scale Neck-only change.
- Every new runtime interface has a direct shape, criterion, queue, or report contract test.
- No P2 branch, changed training hyperparameter, overwritten run directory, or Git operation is allowed.
