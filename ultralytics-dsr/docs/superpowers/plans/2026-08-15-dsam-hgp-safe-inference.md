# DSAM/HGP 安全社交推理试点 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 YOLOv8n + DSAM/HGP 的社交推理改为“仅 NMS 后高质量候选 + 保守同类分数融合”，消除低置信背景候选被三类 Softmax 人为抬升的问题，并在暂停的 checkpoint 上先通过离线安全门槛再决定是否重训。

**Architecture:** Detect 头仍输出原始 `[B, 4 + nc, A]` 张量。社交分支使用 `non_max_suppression(..., return_idxs=True)` 精确取得每张图的原始 anchor 索引，只对经 NMS、置信度达标且数量受限的候选提取 RoI。融合仅读取原预测类别对应的社交 logit，保留原框和全类别向量；分数只允许保守上调，保证类别不被重分配。任何配置、候选、形状或数值异常均回退该图的原始预测。

**Tech Stack:** Python 3.11；PyTorch；Ultralytics 本地源码；`dsa_det_ssr` 外部包；pytest；5k_HRW 验证集；RTX 4080 SUPER。

## Global Constraints

- 仅修改 DSAM/HGP 的共享推理路径和其 YAML；不得恢复已暂停训练，亦不得启动 SSR/SSL 或完整 DSA 队列。
- 训练配置保持既定协议：5k_HRW、seed=42、scratch、batch=8、SGD、lr0=0.01、300 epoch、640×640。仅在离线验收通过后才允许安排新的 DSAM/HGP 训练试点。
- `candidate_confidence=0.25`、`max_candidates=64`、`score_alpha=0.15`、`min_candidates=2` 是本试点的固定默认值；缺省配置必须安全退化为关闭社交推理。
- 使用 `ultralytics.utils.nms.non_max_suppression(..., return_idxs=True)` 取得候选原始索引；禁止用 IoU/浮点分数近似反查索引。
- 社交推理不得改写边框、不得将整段类别向量清零、不得使用社交分类 `argmax` 改类、不得对未经过 NMS 的候选生成或抬升分数。
- 为保证原预测类别不会在后处理前被改写，融合采用单调保守形式：`p' = max(p, sigmoid(logit(clamp(p)) + alpha * tanh(s_c)))`。`tanh` 限制未校准社交 logit 的影响，`max` 禁止下调原预测类别分数；仅写回该类别位置。
- 若 `enabled=false`、候选少于 `min_candidates`、NMS 映射为空、RoI/社交输出形状不匹配、或出现 NaN/Inf，整张图返回未修改的原始预测。
- 当前外层工作区没有 Git 元数据。每个任务的“提交”步骤必须先运行 `git rev-parse --show-toplevel`；若结果为“not a git repository”，记录该事实并不得伪造提交。

## File Structure

| 路径 | 责任 |
|---|---|
| `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\dsa_det_ssr\engine\dsa_model.py` | 解析安全配置、NMS 后候选选择、同类保守融合、逐图回退。 |
| `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py` | 面向 `DSADetModel` 的配置、候选、融合和输出契约回归测试。 |
| `ultralytics\cfg\models\v8\yolov8n-dsam-hgp.yaml` | DSAM/HGP 试点启用的 `dsa.inference_refinement` 配置。 |
| `experiments\validate_dsam_hgp_safe_inference.py` | 对暂停 checkpoint 顺序运行 raw 与 safe 两种验证，写出可比较 JSON。 |
| `tests\test_safe_inference_validation.py` | 验收阈值和 JSON 指标差值计算的 CPU 单元测试。 |
| `runs\hrw_yolov8n_dsa_seed42_scratch\safe_inference_validation.json` | 离线验收产生的只读结果工件；不用于修改队列状态。 |

---

## Task 1: 为安全配置建立可测试的解析与向后兼容接口

**Files:**
- Modify: `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\dsa_det_ssr\engine\dsa_model.py`
- Modify: `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py`

**Interfaces:**

```python
DSADetModel.inference_refinement: dict[str, bool | float | int]
DSADetModel._inference_refinement_config() -> dict[str, bool | float | int]
```

The normalized dictionary must contain exactly `enabled`, `candidate_confidence`, `max_candidates`, `score_alpha`, and `min_candidates`; defaults are disabled with the fixed numerical values in Global Constraints.

- [ ] **Step 1: Add failing tests for defaults, YAML propagation, and invalid values.**

  Add three tests adjacent to `test_dsa_yaml_options_configure_social_modules`:
  - a DSA config without `inference_refinement` yields `enabled is False` and the fixed safe defaults;
  - an explicit config propagates all five fields to a model;
  - invalid bounds (`candidate_confidence=-0.01`, `score_alpha=1.01`, `max_candidates=0`, `min_candidates=1`) raise `ValueError` with the offending field name.

- [ ] **Step 2: Run the focused tests and confirm RED.**

  Run:

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -k "inference_refinement_config" -q
  ```

  Expected: test collection succeeds and the new tests fail because the normalized configuration interface does not yet exist.

- [ ] **Step 3: Implement normalized config parsing.**

  In `DSADetModel.__init__`, read `dsa_cfg.get("inference_refinement", {})`, normalize into `self.inference_refinement`, and validate exact bounds. Implement `_inference_refinement_config()` with `getattr(self, "inference_refinement", None)` fallback so pre-refactor pickled checkpoints safely behave as disabled. Do not access the new attribute directly elsewhere without this fallback.

- [ ] **Step 4: Run focused and existing integration tests (GREEN).**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -k "inference_refinement_config or dsa_yaml_options or inference_keeps" -q
  ```

- [ ] **Step 5: Check repository state before any commit.**

  ```powershell
  git rev-parse --show-toplevel
  ```

  If Git is unavailable in this outer workspace, record `not a git repository` in the task log and continue without a commit.

---

## Task 2: 实现 NMS 后的精确候选选择与逐图回退

**Files:**
- Modify: `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\dsa_det_ssr\engine\dsa_model.py`
- Modify: `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py`

**Interfaces:**

```python
DSADetModel._select_social_candidate_indices(
    prediction: torch.Tensor,  # [1, 4 + nc, A]
    nc: int,
) -> torch.Tensor  # [K] torch.long, original anchor indices, K <= max_candidates
```

The method calls `non_max_suppression(prediction, conf_thres=candidate_confidence, max_det=max_candidates, nc=nc, return_idxs=True)`. It returns only the first image’s kept indices, ordered by NMS confidence, or an empty long tensor on an unsupported/invalid result.

- [ ] **Step 1: Add failing tensor-level candidate tests.**

  Build a synthetic `[1, 7, 5]` three-class prediction tensor with:
  - two overlapping class-0 boxes at 0.90 and 0.80 (only the 0.90 anchor may survive NMS);
  - one isolated 0.70 box (must survive);
  - one 0.24 box (must not be selected);
  - one class-1 0.60 box overlapping the class-0 0.90 box (must survive class-aware NMS).

  Assert that all returned indices are original-anchor indices, all selected raw confidences are `>= 0.25`, no duplicate remains, and `max_candidates=2` truncates in descending confidence order. Add a test that an empty/invalid NMS result produces an empty `torch.long` tensor rather than an exception.

- [ ] **Step 2: Run candidate tests and confirm RED.**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -k "social_candidate" -q
  ```

  Expected: failures because candidate selection is still the old raw `conf_thr=0.05` mask.

- [ ] **Step 3: Implement the NMS-index selection helper.**

  Import `non_max_suppression` from `ultralytics.utils.nms`. Keep prediction channel-first for NMS. Request `return_idxs=True` and use the returned `keepi[0]` directly; do not reconstruct indices from coordinates. Validate type, dimensionality, dtype, range, uniqueness, and the confidence threshold on the source tensor. Apply the configurable maximum only after validation. Any exception returns an empty long tensor on the source device.

- [ ] **Step 4: Replace raw proposal masking in `_social_inference`.**

  For each image, make `pred.unsqueeze(0).transpose(1, 2)` into `[1, 4+nc, A]`, obtain exact selected indices, and append the unmodified `pred` if `enabled` is false or the count is below `min_candidates`. Construct RoI boxes, raw scores, and raw classes only from those indices. Retain the original `pred` for all nonselected candidates.

- [ ] **Step 5: Run candidate and output-contract tests (GREEN).**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -k "social_candidate or inference_keeps" -q
  ```

- [ ] **Step 6: Check repository state before any commit.**

  ```powershell
  git rev-parse --show-toplevel
  ```

  Only commit if this command identifies an actual repository; otherwise record the skipped commit.

---

## Task 3: 实现保守的同类残差融合，不改类、不改框

**Files:**
- Modify: `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\dsa_det_ssr\engine\dsa_model.py`
- Modify: `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py`

**Interfaces:**

```python
DSADetModel._fuse_social_class_scores(
    class_scores: torch.Tensor,  # [K, nc]
    original_classes: torch.Tensor,  # [K] torch.long
    social_logits: torch.Tensor,  # [K, nc]
) -> torch.Tensor  # [K, nc], finite, same dtype/device as class_scores
```

For row `i`, only `class_scores[i, original_classes[i]]` may change. The returned score is `max(p, sigmoid(logit(clamp(p, eps, 1-eps)) + score_alpha * tanh(s_c)))`, clamped to `[eps, 1-eps]`; this preserves the original argmax class.

- [ ] **Step 1: Add failing fusion tests.**

  Cover:
  - `score_alpha=0` is elementwise identical to input scores;
  - only each row’s original-class element can change and the original argmax is retained;
  - a positive same-class social logit can only raise, never lower, the detector score;
  - nonfinite score/logit or incompatible `[K, nc]` shape signals a failure value that makes `_social_inference` return the complete original image prediction;
  - outputs retain input dtype/device and are finite.

- [ ] **Step 2: Run fusion tests and confirm RED.**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -k "social_fusion" -q
  ```

  Expected: failures because the implementation presently takes `softmax(...).max`, zeroes every class, and rewrites the argmax class.

- [ ] **Step 3: Implement fusion and replace the destructive writeback.**

  Implement `_fuse_social_class_scores` without Python loops over classes. In `_social_inference`, call it only after verifying `cls_out.shape == (K, nc)`, `torch.isfinite(cls_out).all()`, and valid selected indices. Do not read or write `refined_boxes`; keep `pred[:, :4]` untouched. Write the fused `[K, nc]` rows into `new_pred[selected_indices, 4:]`, leaving all other rows untouched. If helper validation fails, append `pred`, not a partially modified clone.

- [ ] **Step 4: Add a full inference fallback regression test.**

  Monkeypatch `_run_social` to return a NaN class tensor for one image and verify `_social_inference` returns that image’s input prediction exactly. Add a complementary monkeypatch returning finite logits and assert only selected class entries can differ; all boxes, nonselected rows, and non-target class entries remain equal.

- [ ] **Step 5: Run all social-inference tests (GREEN).**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -k "social_candidate or social_fusion or inference_keeps" -q
  ```

- [ ] **Step 6: Check repository state before any commit.**

  ```powershell
  git rev-parse --show-toplevel
  ```

  Commit only in a real repository; otherwise retain the test command/output in the implementation record.

---

## Task 4: 启用 DSAM/HGP 试点 YAML，并验证 AMP/输出契约

**Files:**
- Modify: `ultralytics\cfg\models\v8\yolov8n-dsam-hgp.yaml`
- Modify: `tests\test_yolov8n_dsa_amp.py`
- Modify: `D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py`

**Interfaces:**

```yaml
dsa:
  inference_refinement:
    enabled: true
    candidate_confidence: 0.25
    max_candidates: 64
    score_alpha: 0.15
    min_candidates: 2
```

- [ ] **Step 1: Add failing YAML and AMP assertions.**

  Extend the local CUDA test to load the DSAM/HGP YAML and assert that the resulting model uses the five expected values, returns finite channel-first predictions under half-precision input, and preserves the existing FP32 social-island contract. Extend the external integration test to assert the DSAM/HGP YAML’s social inference config is enabled and parsed.

- [ ] **Step 2: Run the tests and confirm RED.**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest tests\test_yolov8n_dsa_amp.py "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -k "amp or inference_refinement" -q
  ```

  Expected: the YAML assertion fails until the new block is added.

- [ ] **Step 3: Add only the approved YAML block.**

  Add the `dsa.inference_refinement` block exactly as above. Do not alter training fields, module enables, `lambda_cls`, `lambda_box`, Mosaic/MixUp, or box-mixing settings in this task.

- [ ] **Step 4: Run targeted tests and the complete relevant regression set (GREEN).**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest tests\test_yolov8n_dsa_amp.py "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -q
  ```

- [ ] **Step 5: Check repository state before any commit.**

  ```powershell
  git rev-parse --show-toplevel
  ```

  Do not create a commit when this workspace remains outside a Git repository.

---

## Task 5: 用暂停 checkpoint 做 raw-vs-safe 离线验收，不启动训练

**Files:**
- Create: `experiments\validate_dsam_hgp_safe_inference.py`
- Create: `tests\test_safe_inference_validation.py`
- Create: `runs\hrw_yolov8n_dsa_seed42_scratch\safe_inference_validation.json`

**Interfaces:**

```text
python experiments\validate_dsam_hgp_safe_inference.py \
  --weights runs\hrw_yolov8n_dsa_seed42_scratch\native_runs\yolov8n_dsam_hgp\weights\best.pt \
  --data D:\YH_Project\DSA-Det\5k_HRW_yolo_Dataset\data.yaml \
  --output runs\hrw_yolov8n_dsa_seed42_scratch\safe_inference_validation.json
```

The JSON must contain `raw_detector`, `safe_social`, `delta`, `pass`, checkpoint path, data path, image size, seed, candidate configuration, and timestamp. Each metric object contains `precision`, `recall`, `map50`, and `map50_95`.

- [ ] **Step 1: Write failing tests for validation-result schema and gate calculation.**

  In `tests\test_safe_inference_validation.py`, import `evaluate_gate(raw_metrics, safe_metrics) -> dict` from the new script. Add pytest coverage that passes exactly at `safe.map50_95 == raw.map50_95`, fails below it, and preserves all four metric deltas. Keep tests CPU-only by using fixture dictionaries rather than loading checkpoints.

- [ ] **Step 2: Run the schema tests and confirm RED.**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest tests -k "safe_inference_validation" -q
  ```

  Expected: failure because the helper/script is absent.

- [ ] **Step 3: Implement a two-pass, no-training validator.**

  Load the existing `best.pt` once per pass. For `raw_detector`, set `model.model._dsa_initialized = False` exactly as the existing diagnostic path does. For `safe_social`, inject the normalized approved `inference_refinement` config onto the checkpoint model and keep `_dsa_initialized=True`. Call `model.val` with `split="val"`, `imgsz=640`, `batch=8`, `workers=0`, and no training API. Serialize metrics from `results.box` and compute `safe - raw` deltas. The gate is true only when all metrics are finite and `safe_social.map50_95 >= raw_detector.map50_95`; log an explicit no-retrain reason on failure.

- [ ] **Step 4: Run schema tests (GREEN), then execute the offline validation.**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest tests -k "safe_inference_validation" -q
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" experiments\validate_dsam_hgp_safe_inference.py --weights runs\hrw_yolov8n_dsa_seed42_scratch\native_runs\yolov8n_dsam_hgp\weights\best.pt --data "D:\YH_Project\DSA-Det\5k_HRW_yolo_Dataset\data.yaml" --output runs\hrw_yolov8n_dsa_seed42_scratch\safe_inference_validation.json
  ```

- [ ] **Step 5: Inspect and enforce the acceptance gate.**

  Inspect JSON fields and confirm raw reference is close to the previously measured `mAP50-95=0.5051` (allow ordinary validation nondeterminism). If `pass=false`, leave all training jobs paused and report `不启动重训`. If `pass=true`, prepare a separate user-visible request for authorization before launching any new scratch trial; do not launch it within this plan.

- [ ] **Step 6: Check repository state before any commit.**

  ```powershell
  git rev-parse --show-toplevel
  ```

  As above, only commit if a repository actually exists.

---

## Task 6: Final verification and handoff record

**Files:**
- Verify: `docs\superpowers\specs\2026-08-15-dsam-hgp-safe-inference-design.md`
- Verify: `runs\hrw_yolov8n_dsa_seed42_scratch\safe_inference_validation.json`
- Verify: `runs\hrw_yolov8n_dsa_seed42_scratch\queue_state.json`

- [ ] **Step 1: Run the full relevant test suite once more.**

  ```powershell
  & "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe" -m pytest tests\test_yolov8n_dsa_amp.py "D:\YH_Project\DSA-Det\dsa_det_ssr_pkg\tests\test_yolo_integration.py" -q
  ```

- [ ] **Step 2: Confirm queue non-interference.**

  Parse `runs\hrw_yolov8n_dsa_seed42_scratch\queue_state.json` with `ConvertFrom-Json` and assert `running` remains `null`; DSAM/HGP remains paused and SSR/SSL plus complete DSA remain pending. Do not edit this file in the verification task.

- [ ] **Step 3: Report the decision with exact evidence.**

  State the raw and safe Precision/Recall/mAP50/mAP50-95, all four deltas, configuration values, pass/fail state, and whether retraining remains prohibited. Reiterate that this is one checkpoint on one 5k_HRW split with seed 42; it is an inference-safety gate, not a generalization claim.
