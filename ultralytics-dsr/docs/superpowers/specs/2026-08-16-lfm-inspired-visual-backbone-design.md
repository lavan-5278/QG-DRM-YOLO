# LFM-Inspired Visual Backbone Blocks

## Goal

Evaluate three independent, lightweight YOLOv8n backbone modifications that translate the useful LFM principles to 2-D detection features: gated short convolutions, content-dependent state updates, and a small amount of sparse relation attention. The primary metric is validation mAP50-95 on 5k_HRW.

All variants preserve the YOLOv8n neck and Detect head. They are not stacked with SATA, RG-SATA, DR-SATA, AFPN, DSR, or other experimental modules.

## Shared interfaces and safety properties

Each block consumes and returns a feature tensor `x` of shape `[B, C, H, W]`. It preserves the number of channels and spatial resolution, so it can follow a C2f layer without changing parser routing.

Each residual update has a learnable scalar `gamma` initialized to zero. Therefore the initial block is exactly an identity mapping and does not destabilize from-scratch optimization. Every attention score uses normalized query/key vectors, a learned non-negative spatial penalty, and numerically safe clamps before logarithms.

The blocks are only placed after the C4 and C5 C2f layers, except that LHGB also uses a local-only C3 block. No sparse attention is used on C3.

## Variant 1: LGC — Liquid Gated Convolution

LGC isolates the short-convolution and dynamic-gating idea.

1. Project `x` with parallel depthwise 3x3 and depthwise 5x5 convolutions.
2. Compute a per-channel, per-location gate from the concatenated local responses.
3. Form the state `s = gate * conv3(x) + (1 - gate) * conv5(x)`.
4. Project `s` with a pointwise convolution and add `gamma * project(s)` to `x`.

LGC is used at C4 and C5. Its ablation answers whether local liquid-style state evolution is useful without any explicit relation operator.

## Variant 2: LGRB — Liquid Gated Relation Block

LGRB contains LGC as its local-state front end, then adds sparse relation reasoning.

1. Produce the LGC state `s`.
2. Score all positions with a saliency head and select at most 32 source anchors per image.
3. For each anchor, select at most six neighbours using cosine similarity minus a learned non-negative squared-coordinate penalty.
4. Compute attention only from each source anchor to its selected neighbours. Neighbour reliability is added as a log-prior to attention logits.
5. Gate the aggregated relation message by the source reliability, scatter it to the source positions, diffuse it through a 3x3 depthwise convolution, and fuse it with the local state through a pointwise projection.
6. Add the fused update to `x` through the zero-initialized residual scalar.

LGRB is used at C4 and C5. It tests whether a small relation budget can improve occlusion and crowded-scene evidence without the expense of dense attention.

## Variant 3: LHGB — Liquid Hierarchical Gated Backbone

LHGB is a backbone-level composition, not a separate neck.

1. Insert an LGC block after C3. Its output remains a normal feature map for the next backbone stage.
2. Insert an LHGB block after C4 with two inputs: the current C4 feature map and the retained C3 feature map. LHGB globally pools and projects C3 internally to obtain a compact stage state, then broadcasts it into its gating head before sigmoid gating.
3. Insert an LHGB block after C5 with the current C5 map and the retained C4 output as its two inputs, using the same context-pooling mechanism.

Thus information flows only bottom-up within the backbone: local detail at C3 influences the state selection/gating at C4, and C4 influences C5. Each LHGB still returns one ordinary feature tensor, so it does not change the features exported to the neck other than through the normal residual outputs. It does not route C5 backward into C4.

## Model configurations and experiment order

Create separate configurations:

- `yolov8n-lgc.yaml`
- `yolov8n-lgrb.yaml`
- `yolov8n-lhgb.yaml`

Run them serially in the order LGC, LGRB, LHGB. Each job is independent and uses the exact fixed protocol: 5k_HRW, scratch initialization with no pretrained weights, seed 42, 300 epochs, 640x640 input, batch size 8, SGD, and `lr0=0.01`.

## Validation and reporting

Before training, tests must verify identity initialization, non-identity finite output after enabling the residual, shape preservation, model-YAML parsing, and finite YOLO forward output at 640x640. The queue state must be atomically writable and parseable. After each successful training job, record best validation metrics, actual parameter count, `best.pt` size, FLOPs at 640, and RTX 4080 SUPER FP32 batch-1 latency using 50 warmups plus 200 forwards.

Append all results to `runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md`. Compare each variant with YOLOv8n scratch and the existing SATA, RG-SATA, and DR-SATA runs, while explicitly noting the single-seed, single-split limitation.
