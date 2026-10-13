# VBR: Visible-Body Region Aligner

## Goal

Test one original, classroom-specific module that improves YOLO11n mAP50-95 on the leak-free SCB-Dataset3-S split by refining the P3 detection feature around the visible upper-body and behavior region.

## Research story

SCB labels denote the visible upper body and the action-bearing region, rather than complete human silhouettes. In classroom images, a distant student is often only a small head-shoulder-hand pattern; the lower body is hidden by a desk and lateral boundaries can be truncated by neighboring students. Standard FPN fusion treats all surrounding pixels and all body regions uniformly. PVRF tested generic multi-scale fusion and did not improve mAP50-95. VBR therefore uses only local, high-resolution P2 evidence and only at the P3 detection scale, with an upper-body behavior template that concentrates samples on head/shoulder, arm, and desk-interaction locations. The templates are deformable through learned offsets, so the method does not assume a fixed pose.

The claimed contribution is a label-definition-aligned local sampler for visible upper-body behavior detection, not generic attention, relation modeling, whole-image perspective routing, or super-resolution.

## Scope

- One inserted module: `VBR`.
- Standard Ultralytics `Detect` head and standard `v8DetectionLoss` remain unchanged.
- No new labels, no pseudo occlusion labels, no relation/composition branch, no quality head, and no custom trainer.
- Evaluation primary metric: validation mAP50-95; mAP50 is secondary.

## Module interface

```python
class VBR(nn.Module):
    def __init__(self, ch: tuple[int, int], cfg: dict[str, Any]) -> None: ...
    def forward(self, xs: list[Tensor]) -> Tensor: ...
```

- Inputs: `[p2, p3]`, where P2 has stride 4 and shape `(B, C2, 2H, 2W)`, and P3 has stride 8 and shape `(B, C3, H, W)`.
- Output: one refined P3 tensor of exact shape `(B, C3, H, W)`.
- YAML parser contract: `VBR` receives `tuple(ch[x] for x in f)` followed by the top-level `vbr` config dictionary. Parser output channel count is `ch[f[1]]`.

## Internal computation

1. Project P2 to `C3` channels with a 1x1 convolution followed by batch normalization and SiLU.
2. From P3, predict `K=5` two-dimensional offsets and `K` softmax sampling weights per P3 location.
3. Sample projected P2 with bilinear `grid_sample` at five upper-body template locations plus bounded learned offsets. The default template locations in P2-pixel units are `(-1.0,-1.5)`, `(0.0,-2.0)`, `(1.0,-1.5)`, `(-0.8,0.0)`, `(0.0,1.0)`: left shoulder/arm, head/raised hand, right shoulder/arm, arm/torso, and desk-interaction region.
4. Aggregate sampled features using the learned softmax weights, concatenate them with P3, and project back to `C3` channels.
5. Return `p3 + tanh(alpha) * residual`, with scalar `alpha` initialized to `0.0`. This preserves exact P3 identity at initialization and makes the module falsifiable: any change must be learned during training.

`offset_radius=2.0` P2 pixels is the default bound. Sampling grid normalization must use the P2 spatial dimensions and `align_corners=False`.

## YOLO YAML placement

Use the unchanged YOLO11n backbone and head through layer 22. Add:

```yaml
vbr:
  enabled: true
  samples: 5
  offset_radius: 2.0
  alpha_init: 0.0

head:
  # Existing YOLO11n layers 11 through 22
  - [[1, 16], 1, VBR, [vbr]]
  - [[23, 19, 22], 1, Detect, [nc]]
```

P2 is layer 1 and P3 is layer 16 in the YOLO11n graph. Layer 23 replaces only the P3 input consumed by `Detect`; P4 and P5 remain layers 19 and 22. A matched baseline YAML must use `[[16, 19, 22], 1, Detect, [nc]]` and otherwise have the same scales and all training arguments.

## Test acceptance criteria

1. VBR preserves P3 shape for two legal input sizes.
2. With `alpha=0`, output equals P3 bit-for-bit in evaluation mode.
3. With nonzero alpha, output is finite, differs from P3, and gradients reach P2, P3, offsets, weights, and fusion projection.
4. `YOLO(vbr_yaml)` builds; its penultimate module is `VBR`; a CUDA 640x640 forward pass succeeds.
5. Standard detection regression tests continue to pass.

## Experiment protocol

- Dataset: `ultralytics/cfg/datasets/SCB-Dataset3-S.yaml`, existing leak-free split.
- Initialization: `YOLO(vbr_yaml).load('yolo11n.pt')`.
- Training: 300 requested epochs, `patience=100`, `imgsz=640`, `batch=16`, `device=0`, `workers=8`, `seed=42`, `deterministic=True`.
- Baseline: the already completed matched baseline uses the same protocol and has best mAP50-95 `0.47010`.
- Candidate success gate: VBR best mAP50-95 must exceed `0.47010` by at least `0.0020` in this screening seed before any multi-seed confirmation.
- Candidate failure gate: VBR at or below `0.47010` is rejected for this dataset/protocol; no multi-seed run is authorized.

## Non-goals

This screening experiment does not claim SOTA, statistical significance, or cross-version generalization. A passing one-seed gate only authorizes a later three-seed confirmation and evaluation on the other dataset versions.
