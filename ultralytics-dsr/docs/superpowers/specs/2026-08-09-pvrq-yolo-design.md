# PVR-Q YOLO：透视—实例竞争路由的定位质量检测器

**状态：** 待用户审阅，尚未实现  
**主任务：** SCB-Dataset3-S 无泄漏划分上的 YOLO11n 目标检测  
**主指标：** 三 seed 的 mAP50-95；泛化只在主结论之后报告。

> **2026-08-09 证据修订（已替代原始竞争假设）：** 修复版 Phase 0 使用与 Ultralytics 一致的 1,670 张有效验证图、11,771 GT 和全部 95,772 条 NMS 后预测，已通过独立复审。小尺度 focus 的 matched-IoU 显著低于其余实例 `0.070756`（图像簇 95% CI `[-0.094364, -0.048537]`），且 oracle IoU 重排的 fixed-match AP50-95 上限增益为 `+0.150139`（95% CI `[+0.144494, +0.156081]`）。相反，competition 与定位误差的 Spearman 为 `-0.03949`，无支持证据。因此后续实现名称为 **PVR-Qv2**：保留 LPF/PVRF 与质量排序，删除竞争度辅助头、竞争损失和 `L_hloc`；竞争只允许作为后续可选消融，不能作为贡献或核心叙事。

## 0. PVR-Qv2 的冻结实现边界

PVR-Qv2 的模块图与原方案一致，但 PVRF 的门控输入固定为 `[P3, LPF.field]`，并仅在 `P2'/P3'/P4'` 三条特征源间路由。`local_competition_feature`、`competition_logits` 和所有由 GT 框重叠构造的训练目标均不实现。

`PVQDetect` 仍为 `Detect` 子类，但质量分支必须对**所有** anchor 输出一个无类别质量 logit。令正 anchor 的动态目标为当前预测框对其分配 GT 的 detached IoU，负 anchor 的目标为零。采用 Quality Focal Loss：

`L_q = sum_a |stopgrad(q_a)-sigmoid(z_a)|^beta * BCEWithLogits(z_a, stopgrad(q_a)) / max(sum_a q_a, 1)`，其中 `beta=2.0`。

这一定义保留 84,545 条未匹配预测的零质量监督，避免旧版“仅正样本 IoU 回归”无法学会抑制误检的缺陷。推理时用 `p_final = p_cls * sigmoid(z_q)^gamma` 重写分类分数；`gamma=0` warm-up 5 epoch，随后在 10 epoch 内线性升至 `0.5`。质量分支关闭时，输出必须与官方 `Detect` 数值等价。

总损失替换为：

`L = L_yolo + lambda_pers * L_pers + lambda_quality * L_q`，默认 `lambda_pers=0.10`、`lambda_quality=0.25`。原 `lambda_competition`、`lambda_hard_loc` 和 `competition_radius` 从配置中删除。

新的架构消融最小集合为：官方 YOLO11n、Naive P2-to-P3 residual、LPF/PVRF、PVQDetect、PVR-Qv2 完整模型。任何 competition 变体只能在完整模型取得稳定正收益后以额外实验出现。

## 1. 问题、证据与边界

### 1.1 已有证据

在相同训练协议下，YOLO11n 三 seed 基线为 `0.4890 ± 0.0094` mAP50-95。CompositionDetect 为 `0.4837 ± 0.0064`，关系精炼、选择性组成混合和 GICR 均未产生稳定正收益。因此本方案**禁止**将图像级行为组成、预测类别关系或 KNN 关系用于分类分数修正。

现有 `DSRDetectionValidator` 的基线诊断给出另一条更可操作的证据链：

- 召回率约 `0.974`，mAP50 为 `0.624`，mAP50-95 为 `0.445`；主要瓶颈是定位精度及其置信度排序，而非候选完全缺失。
- 面积分层的小目标 AP50-95 为 `0.1142`；严重遮挡分层为 `0.0623`。小尺度和遮挡竞争区域是主要失效区。
- SCB-Dataset3-S 是固定相机课堂场景。目标尺度、图像纵向位置与局部拥挤/遮挡存在共同变化，但不得把绝对位置当作类别先验。

### 1.2 研究命题

> 固定视角课堂中的检测误差主要来自透视尺度和实例竞争共同造成的定位质量下降。统一、位置无差别的 FPN 融合与分类置信度不能有效处理这一误差，因此应以场景条件化的细节路由和显式定位质量校准替代预测级关系先验。

### 1.3 非目标

- 不宣称 P2、坐标编码、IoU-aware score 或质量分支本身原创。
- 不使用 SCB-Dataset3-U 作有效性证据；其训练/验证布局重复率为 95.2%。
- 不在主实验中改变输入尺寸、骨干规模、预训练权重或训练预算；这些属于独立对照。
- 不声称自动从标注框得到像素级“真实可见度”。训练期构造的量统一称为**实例竞争度**（instance competition），不是 visibility ground truth。

## 2. 总体架构

模型由一个特征模块和一个检测头组成，保持官方 YOLO11n 的 P3/P4/P5 检测尺度以及标准推理输出格式。

```text
P2/4 (backbone layer 1) ─┐
P3/8 (head layer 16) ───┼── PVRF ── P3'/8 ─┐
P4/16 (head layer 19) ──┼───────────────────┼── PVQDetect ── boxes, classes
P5/32 (head layer 22) ──┘                   └── quality q (score calibration)
                                          LPF (inside PVRF)
```

`PVRF` 只替换送入检测头的 P3；P4、P5 的原始路径不变。这样能把风险限制在小目标路径，避免对清晰大目标进行无必要重写。`PVQDetect` 是 `Detect` 的子类，训练期保存质量 logits，推理期将其用于校准分类置信度，但不增加输出字段，故能复用 Ultralytics validator、NMS 和导出接口。

## 3. 模块接口

### 3.1 LPF：Latent Perspective Field

**目的：** 从当前图像的全局场景特征和相对坐标，输出 P3 分辨率的连续尺度场，而非硬编码“顶部/底部”的规则。

```python
class LatentPerspectiveField(nn.Module):
    def __init__(self, c_p5: int, c_ctx: int, c_field: int = 16) -> None: ...

    def forward(self, p5: Tensor, out_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
        # returns:
        #   field: [B, c_field, H3, W3]  scene-conditioned perspective features
        #   log_scale: [B, 1, H3, W3]    expected log normalized object side length
```

Implementation contract:

1. `p5 -> GAP -> MLP` gives per-image context `z`.
2. A fixed normalized `(x, y)` grid is encoded with two Fourier frequencies, concatenated with broadcast `z`, and processed by a 1x1/3x3 depthwise-separable predictor.
3. `log_scale` has no class input and no class output. It is supervised only at assigned positive locations with the matched GT normalized side length `log(sqrt(w*h) / sqrt(H*W))`.
4. The coordinate grid is relative, while `z` makes the field scene-conditioned. A cross-version test and an ablation without LPF are mandatory to rule out pure camera-position memorization.

### 3.2 PVRF：Perspective–Visibility Routed Feature Refiner

**目的：** 在 P3 的每个位置按透视需求和局部竞争线索选择 P2 的细节、P3 的当前表示或 P4 的语义上下文。这里的 `Visibility` 是特征上的可见性/竞争估计，不是标注的可见度。

```python
class PVRF(nn.Module):
    def __init__(
        self,
        ch: tuple[int, int, int, int],  # (C_P2, C_P3, C_P4, C_P5)
        cfg: dict,
    ) -> None: ...

    def forward(self, xs: list[Tensor]) -> Tensor:
        # xs = [P2/4, P3/8, P4/16, P5/32]
        # return P3_refined: [B, C_P3, H3, W3]
        # side effect for the training criterion only:
        # self.last_aux = {"log_scale": ..., "competition_logits": ...}
```

Internal computation:

1. Project P2/P3/P4 to `C_P3`; project P2 with a stride-2 depthwise 3x3 convolution and P4 with bilinear upsampling. P5 only supplies LPF, avoiding a costly full-resolution P5 fusion.
2. From `[P3, LPF.field, local_competition_feature]`, predict three per-location softmax gates `g2, g3, g4`.
3. Fuse as `F = P3 + alpha * Conv(g2*P2' + g3*P3' + g4*P4')`. `alpha` is initialized at zero, so a newly initialized model is functionally the official YOLO P3 path.
4. `local_competition_feature` is a single learned map from P3, not a predicted class relation or a proposal graph. During training it receives a weak auxiliary target described in section 5.2.

Default configuration:

```yaml
pvrq:
  enabled: true
  field_channels: 16
  gate_hidden_ratio: 0.50
  fusion_alpha_init: 0.0
  lpfield_warmup_epochs: 5
  competition_radius: 2.0
  competition_beta: 1.0
  lambda_perspective: 0.10
  lambda_competition: 0.15
  lambda_quality: 0.25
  lambda_hard_loc: 0.25
  quality_gamma: 0.5
```

### 3.3 PVQDetect：Perspective–Visibility Quality Detect

**目的：** 令最终检出分数同时反映类别存在概率和预测框的定位质量，减少“分类高分、IoU 不高”的错误排序。

```python
class PVQDetect(Detect):
    def __init__(self, nc: int = 80, ch: tuple = (), pvrq_cfg: dict | None = None) -> None: ...

    def forward(self, x: list[Tensor]) -> list[Tensor] | tuple[Tensor, list[Tensor]]: ...
    # side effects during training:
    # self.quality_logits: list[Tensor]  # each [B, 1, H_l, W_l]
```

`PVRF.forward()` stores `last_aux` only while training. `PVRQDetectionModel.loss()` obtains it from layer 23 (`model.model[-2]`) and passes it explicitly to `PVRQDetectionLoss`; `PVQDetect` is not responsible for transporting PVRF auxiliary maps. This keeps the feature-node and head-node interfaces valid under the existing sequential model executor.

Per scale, the quality branch is `DWConv(3x3) -> Conv(1x1) -> q_logit`. It is intentionally class agnostic. Standard `cv2` (DFL regression) and `cv3` (classification) are retained unchanged.

For inference, let `p_c = sigmoid(class_logit)` and `p_q = sigmoid(q_logit)`. Before `Detect._inference`, replace classification logits with:

`logit(clamp(p_c * p_q^gamma, eps, 1-eps))`, where `gamma = quality_gamma`.

This preserves the normal `Detect` tensor contract `[x1, y1, x2, y2, class scores]`, hence no validator or NMS modification is required. At initialization, set the final quality bias so `p_q` is approximately one and keep `gamma=0` for the first five warm-up epochs; this protects exact baseline behavior at the start of training.

## 4. YAML placement and parser changes

Starting from `ultralytics/cfg/models/11/yolo11.yaml`, create `ultralytics/cfg/models/11/yolo11n-pvrq.yaml`. Backbone and layers 0–22 remain byte-for-byte identical to official YOLO11n. Append:

```yaml
# PVR-Q: P2 from backbone index 1; official P3/P4/P5 from 16/19/22
- [[1, 16, 19, 22], 1, PVRF, [pvrq]]       # 23, returns refined P3/8
- [[23, 19, 22], 1, PVQDetect, [nc, pvrq]] # 24, Detect(P3', P4, P5)
```

`parse_model()` in `ultralytics/nn/tasks.py` needs two explicit special cases, mirroring existing `DSRDetect` and `CompositionDetect` behavior:

```python
elif m is PVRF:
    resolve "pvrq" to d.get("pvrq", {})
    args.insert(0, tuple(ch[x] for x in f))
    c2 = ch[f[1]]  # PVRF output has P3 channels
elif m is PVQDetect:
    resolve "pvrq" to d.get("pvrq", {})
    args.insert(1, [ch[x] for x in f])
    m.legacy = legacy
```

Registration locations:

| Component | File | Required export/registration |
|---|---|---|
| `LatentPerspectiveField`, `PVRF`, `PVQDetect` | `ultralytics/nn/modules/pvrq.py` | Add to `__all__` |
| module imports | `ultralytics/nn/modules/__init__.py` | Import and expose all three types |
| YAML parser | `ultralytics/nn/tasks.py` | Import through module wildcard; add the two branches above |
| model and criterion | `ultralytics/models/yolo/pvrq/model.py` | `PVRQDetectionModel`, `PVRQDetectionLoss` |
| trainer | `ultralytics/models/yolo/pvrq/train.py` | `PVRQTrainer`, five named losses |
| package export | `ultralytics/models/yolo/pvrq/__init__.py` | Export model/trainer |

No changes are required in the generic Ultralytics validator if `PVQDetect` preserves its detection output contract.

## 5. Losses

### 5.1 Base detection loss

Retain `v8DetectionLoss` unmodified as `L_yolo = L_box + L_cls + L_dfl`. The new criterion subclasses it, as `CompositionLoss` and `DSRDetectionLoss` already do in this repository. The task-aligned assigner stays unchanged in the first implementation.

### 5.2 Training-only instance competition target

For each ground-truth box `b_i`, construct a weak competition scalar from other GT boxes in the same image:

`o_i = 1 - exp[-( max_j area(b_i ∩ b_j)/area(b_i) + beta * N_i(r)/max(N_ref, 1) )]`.

`N_i(r)` counts centres of other boxes within `r * sqrt(area(b_i))`; `r=competition_radius` and `N_ref=4` initially. This represents overlap plus nearby-instance pressure. It requires no annotation change and is computed inside the criterion after target preprocessing.

At each assigned foreground anchor, sample the PVRF competition map `o_hat`. Use `BCEWithLogits(o_hat, stopgrad(o_i))`; denote it `L_comp`. The map is only an internal routing cue and is not exported as a visibility label.

### 5.3 Perspective-field supervision

At each assigned foreground anchor, let the target be

`s_i = log( sqrt(area(b_i)) / sqrt(H*W) )`.

Sample `s_hat` from `LPF.log_scale` and use robust loss:

`L_pers = mean_fg SmoothL1(s_hat, stopgrad(s_i))`.

This makes the spatial field observable and falsifiable. It encodes object scale, not behavior class.

### 5.4 Localization quality loss

For every foreground anchor, calculate `u_i = IoU(pred_box_i, target_box_i)` using the decoded current prediction. The quality branch learns the detached target:

`L_qual = mean_fg BCEWithLogits(q_logit_i, stopgrad(u_i))`.

### 5.5 Hard-region localization term

Define smallness `a_i = area(b_i)/(H*W)` and:

`w_i = sigmoid((log(t_small)-log(a_i))/T) * o_i`.

`t_small` is the training-set lower-tertile area quantile, frozen before training; `T=0.5`. The additional localization emphasis is

`L_hloc = sum_fg w_i * (1 - CIoU(pred_box_i, target_box_i)) / (sum_fg w_i + eps)`.

It is active only after warm-up and is deliberately additive rather than replacing DFL/CIoU. This avoids destabilizing the proven base loss.

### 5.6 Total objective and schedule

`L = L_yolo + lambda_pers*L_pers + lambda_comp*L_comp + lambda_quality*L_qual + lambda_hard_loc*L_hloc`.

For epochs `0..4`, set all four auxiliary coefficients and `quality_gamma` to zero; LPF/PVRF residual alpha remains zero. Linearly ramp to the configured coefficients over epochs `5..15`. Log seven scalar components:

`box_loss, cls_loss, dfl_loss, perspective_loss, competition_loss, quality_loss, hard_loc_loss`.

All seven components must be exposed; compact loss logging is not acceptable because it would hide an auxiliary-loss failure.

## 6. Stepwise experimental matrix

### Phase 0 — Go/no-go diagnosis (no new module)

Use the trained baseline predictions and annotations to produce:

1. 2D heat map: normalized vertical coordinate × log area → AP50-95 and mean matched IoU.
2. Competition bins (`o_i`) → AP50-95, AP75, matched IoU and classification error.
3. Reliability plot: confidence decile → matched IoU, overall and for small/competitive instances.
4. Spearman correlations for vertical coordinate–area, competition–IoU error, and confidence–IoU.

**Gate G0:** continue only if: (a) the small/competitive bins have worse IoU than the remainder with image-bootstrap 95% CI excluding zero, and (b) confidence–IoU calibration has visibly positive residual error in those bins. If no vertical effect exists, retain competition/quality but drop LPF and rename the study CQ-YOLO.

### Phase 1 — Architectural smoke tests

All use `yolo11n.pt`, SCB-Dataset3-S, 640, batch 16, the existing augmentation/hyperparameter protocol, seed 0, and 60 epochs only to reject broken variants.

| ID | Variant | Question answered | Continue criterion |
|---|---|---|---|
| S0 | Official YOLO11n | reference | — |
| S1 | Naive P2-to-P3 detail residual, no LPF/gates | Is extra detail alone useful? | drop from S0 is < 0.010 absolute mAP |
| S2 | PVRF, no auxiliary losses, inference quality off | Does routing train stably? | no worse than S1 by 0.003 absolute mAP |
| S3 | PVQDetect only, quality calibration on | Is quality prediction calibrated? | lower quality Brier error than S0 proxy |
| S4 | Full PVR-Q | interaction sanity | selected by mAP50-95 and AP75 |

Only S1/S2/S3/S4 that pass their numerical checks move forward. No claim is made from 60-epoch results.

### Phase 2 — Single-seed causal ablation

Train to the same 300-epoch/patience protocol as the existing complete experiment, seed 42:

| ID | Variant | Isolates |
|---|---|---|
| A0 | Official baseline | reproducibility anchor |
| A1 | Naive P2 detail residual | feature-resolution benefit |
| A2 | PVRF without LPF | learned feature routing alone |
| A3 | PVRF without competition target | perspective-only routing |
| A4 | PVRF without `L_hloc` | hard-region reweighting |
| A5 | PVQDetect only | score-calibration benefit |
| A6 | Full PVR-Q | complete mechanism |

Promotion rule: A6 must exceed both A0 and the strongest partial variant by at least `0.5` mAP50-95 on this diagnostic run, while AP75, small-object AP and competitive-bin AP do not decline. This is a prioritization threshold, not a final significance claim.

### Phase 3 — Confirmatory comparison

Train A0, strongest non-full ablation, and A6 with seeds `0, 42, 3407`, exactly matching the baseline schedule (same weights, images, batch, patience, evaluator and saved predictions). Report:

- mean ± sample standard deviation and paired seed differences for mAP50-95, mAP50, AP75, precision and recall;
- image-level paired bootstrap 95% CI for A6 − A0;
- AP by three fixed training-set area bins, three fixed competition bins and the existing occlusion bins;
- confidence–IoU calibration: Brier score and ECE over matched detections;
- parameters, GFLOPs, TensorRT/torch FPS on RTX 4080 SUPER, batch 1, 640 input.

**Claim gate G3:** describe A6 as a SCB-Dataset3-S SOTA candidate only if all three seeds improve over their paired A0 run, the paired bootstrap CI excludes zero, and the gain is not obtained by a material recall reduction. Otherwise report it as an ablation finding, not a SOTA result.

### Phase 4 — Generalization and anti-shortcut validation

Freeze architecture and hyperparameters selected on SCB-Dataset3-S. Train/evaluate on each clean available SCB version using its documented split; do not tune on those test sets. Include:

1. PVRF full model vs PVRF-without-LPF; a large degradation only on SCB-Dataset3-S signals camera-specific shortcut risk.
2. Vertical coordinate perturbation test at validation feature level, if it preserves image geometry; otherwise omit it rather than create unrealistic images.
3. Per-version small/competitive-bin gains, not only aggregate mAP.

SCB-Dataset3-U may appear only in a leakage audit appendix, not as generalization evidence.

## 7. Testing and acceptance criteria

Before every training stage:

- YAML parser unit test constructs `YOLO("yolo11n-pvrq.yaml")`, verifies PVRF is layer 23 and PVQDetect layer 24, and runs one CPU forward/backward pass.
- With `pvrq.enabled=false`, YAML must route to normal `Detect` or PVQDetect must be numerically equal to `Detect`; test matching outputs before any optimizer step.
- Verify `PVRF` output is `[B, C_P3, H3, W3]` for non-square inputs and batch sizes 1 and 2.
- Verify no target-derived competition tensor reaches inference; inference must accept image tensors alone.
- Verify loss components are finite with an empty-GT batch, a one-box image, and dense images.
- Verify exported ONNX/TorchScript uses the same final scores within tolerance; if quality calibration blocks export, keep it as a score post-processing node expressible in standard ONNX ops.

The full design is rejected or reduced if G0 fails, if the quality branch worsens calibration, if A6 does not outperform its partial ablations, or if the 3-seed comparison is not consistently positive. The desired outcome is a robust and falsifiable method, not a forced positive result.

## 8. Novelty wording and related-work positioning

The paper should state:

> Prior classroom detectors introduce generic attention, P2 heads, losses, or category/context modules. PVR-Q instead diagnoses a high-recall, low-localization-quality regime and jointly models scene-conditioned perspective scale and local instance competition to route multiscale features and calibrate localization quality.

Avoid claiming that perspective priors, dynamic multiscale fusion, or IoU-aware ranking are individually novel. The novelty claim is the specific **perspective-field × instance-competition** formulation, its training targets, and its evidence-linked application to fixed-view dense classroom detection. A separate, formal novelty search is required before submission.

## 9. Implementation order after spec approval

1. Implement the Phase-0 diagnostic script and decide G0.
2. Add PVRF/LPF registry and YAML parse support with shape/equivalence tests.
3. Add PVQDetect with a no-op/quality-off equivalence mode.
4. Add criterion and target construction, then smoke tests.
5. Run Phase 1 through Phase 4 without changing protocol mid-comparison.
