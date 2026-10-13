# PVR-Qv2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and validate PVR-Qv2: perspective/scale-routed P2–P4 fusion plus all-anchor IoU-quality ranking in YOLO11n, then determine whether it improves SCB-Dataset3-S under the existing multi-seed protocol.

**Architecture:** `PVRF` refines only P3 using a scene-conditioned latent perspective field from P5 and three P2/P3/P4 feature routes. `PVQDetect` keeps the YOLO Detect output contract but learns one quality logit per anchor with all-anchor Quality Focal Loss: assigned foreground targets are detached predicted IoU and every background target is zero. Inference fuses class probability with predicted quality only after warm-up. There is no competition feature, competition target, hard-region loss, proposal graph, or image-level composition signal.

**Tech Stack:** CUDA Python `D:/Users/Admin/anaconda3/envs/lfm-yolo-mamba-stage14/python.exe`; Python 3.11; torch 2.3.0+cu118; vendored Ultralytics 8.3.247; pytest; SCB-Dataset3-S.

## Global Constraints

- Use only the verified Phase 0 evidence in `runs/pvrq_phase0/baseline_seed42_verified/`; do not use SCB-Dataset3-U as validation evidence.
- All training uses CUDA Python above, `imgsz=640`, batch 16, the existing SCB-Dataset3-S configuration and `yolo11n.pt` initialization.
- Core model has exactly LPF/PVRF and PVQDetect. Instance competition is not a core input, target, loss, claim or configuration key.
- `pvrq.enabled=false` or quality fusion gamma 0 must preserve the official Detect output numerically before optimization.
- PVQ quality targets must cover every anchor: foreground target is detached IoU, background target is zero.
- Compare only identical seed/weights/protocol runs. A claim of benefit requires three paired seeds and image-bootstrap CI excluding zero.
- No Git commands or commits.

---

### Task 1: PVRF/LPF module and YAML parser registration

**Files:**
- Create: `ultralytics/nn/modules/pvrq.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Create: `tests/test_pvrq_module.py`

**Interfaces:**
- `LatentPerspectiveField(c_p5: int, c_ctx: int, c_field: int) -> (field[B,c_field,H3,W3], log_scale[B,1,H3,W3])`.
- `PVRF(ch: tuple[int,int,int,int], cfg: dict).forward([p2,p3,p4,p5]) -> p3_refined` and exposes `last_aux={"log_scale": Tensor}` only while training.

- [ ] Write tests that instantiate `PVRF((32,64,128,256), cfg)`, verify P3 shape for `[2,32,160,160]`, `[2,64,80,80]`, `[2,128,40,40]`, `[2,256,20,20]`, finite gradients, and exact `P3` output at `fusion_alpha_init=0`.
- [ ] Implement depthwise stride-2 P2 projection, P3 projection, bilinear-upsampled P4 projection, LPF from P5 plus normalized coordinate Fourier features, three-way softmax gating, and zero-initialized residual alpha.
- [ ] Add `PVRF`/`LatentPerspectiveField` imports and exports. In `parse_model`, add PVRF special handling that resolves `pvrq`, injects four input channel counts, and reports `c2=ch[f[1]]`.
- [ ] Run `D:/Users/Admin/anaconda3/envs/lfm-yolo-mamba-stage14/python.exe -m pytest tests/test_pvrq_module.py -v`.

### Task 2: PVQDetect, all-anchor QFL criterion, and trainer integration

**Files:**
- Modify: `ultralytics/nn/modules/pvrq.py`
- Create: `ultralytics/models/yolo/pvrq/__init__.py`
- Create: `ultralytics/models/yolo/pvrq/model.py`
- Create: `ultralytics/models/yolo/pvrq/train.py`
- Modify: `ultralytics/nn/tasks.py`
- Create: `tests/test_pvrq_loss.py`

**Interfaces:**
- `PVQDetect(Detect)` outputs the normal detection tensor and stores `quality_logits: list[Tensor]` while training.
- `PVRQDetectionLoss(v8DetectionLoss).__call__(preds, batch, pvrf_aux) -> (loss, seven_loss_items)`.
- `PVRQDetectionModel.init_criterion()` returns PVRQDetectionLoss; `PVRQTrainer` names `box, cls, dfl, perspective, quality` losses.

- [ ] Write tests proving quality-off PVQDetect equals Detect, QFL includes a background logit with target zero and nonzero gradient, positive target is bounded detached IoU, empty GT batch is finite, and quality fusion changes only inference scores.
- [ ] Implement per-scale DWConv→1x1 quality logits. In loss, reuse the official assigner/decode; construct a `[B,A]` target initialized zero and fill `fg_mask` positions with detached IoU between assigned prediction and target box. Apply `abs(target-sigmoid(logit))**2 * BCEWithLogits` to all anchors, normalized by `max(target_scores.sum(),1)`. Sample LPF scale at positive P3 anchors and add SmoothL1 scale loss.
- [ ] Implement warm-up state from `head.train_epoch`: gamma 0 for epochs 0–4 and linearly to configured gamma by epoch 15. Preserve official output when disabled.
- [ ] Add parser registration for `PVQDetect`, PVRQ model/trainer package, and run CUDA Python tests for module/loss plus existing `tests/test_comp.py` and `tests/test_dsr_registration.py`.

### Task 3: Model YAML, equivalence checks, and CUDA smoke training

**Files:**
- Create: `ultralytics/cfg/models/11/yolo11n-pvrq.yaml`
- Create: `runs/pvrq/validate_pvrq.py`
- Create: `tests/test_pvrq_registration.py`

- [ ] Create YAML by copying official YOLO11n layers 0–22 byte-for-byte, append `[[1,16,19,22],1,PVRF,[pvrq]]` as layer 23 and `[[23,19,22],1,PVQDetect,[nc,pvrq]]` as layer 24; defaults `field_channels=16`, `lambda_pers=0.10`, `lambda_quality=0.25`, `quality_gamma=0.5`.
- [ ] Test construction, CPU/CUDA forward, disabled equivalence against official model after loading matching `yolo11n.pt`, loss finite for empty/nonempty batches, and model export-compatible output dimensions.
- [ ] Implement `validate_pvrq.py` to run baseline/PVRQ validation with the verified DSR validator, report parameters/GFLOPs/FPS, AP50-95/AP75/area bins and score-IoU rank correlation.
- [ ] Run 10-epoch CUDA smoke baseline and full PVR-Qv2 at seed 42. Stop and fix only if either loss is non-finite, baseline output diverges under disabled mode, or full model is more than 0.010 absolute mAP below baseline.

### Task 4: Causal ablation and confirmatory training

**Files:**
- Create: `runs/pvrq/run_ablation.py`
- Create: `runs/pvrq/ablation_manifest.yaml`
- Modify: `runs/pvrq/validate_pvrq.py`

- [ ] Define immutable variants: `baseline`, `p2_residual`, `pvrf`, `pvq`, `pvrqv2`; every variant uses yolo11n.pt, same dataset, 640/batch16/patience100, seeds 0/42/3407 and a separate run directory.
- [ ] Run the 300-epoch/patience protocol at seed 42 for all variants; use verified Phase 0 metrics to select one strongest partial ablation without tuning hyperparameters.
- [ ] Train baseline, selected partial ablation and PVR-Qv2 with all three seeds. Produce paired bootstrap CIs of mAP50-95, AP75, small-object AP and score/IoU ranking metrics.
- [ ] Declare effective only when PVR-Qv2 beats its paired baseline on all three seeds, image-bootstrap CI for mAP50-95 difference excludes zero, and AP75/small-object AP do not decrease. Otherwise write a negative conclusion and do not claim SOTA.
