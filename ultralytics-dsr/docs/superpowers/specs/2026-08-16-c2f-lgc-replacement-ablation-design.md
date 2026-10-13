# C2f-LGC Replacement Ablation Design

## Motivation

The completed YOLOv8n + LGC experiment reaches mAP50-95 0.52833, exceeding the YOLOv8n scratch reference (0.51690) by 0.01143 under the fixed 5k_HRW seed-42 protocol. This ablation determines whether the gain depends on preserving C2f and adding LGC after it, or whether liquid-gated local processing can replace C2f compute.

## Scope

Train two independent YOLOv8n variants after the already-running LHGB job completes. They are not stacked with post-C2f LGC, LGRB, LHGB, SATA, or any neck/head module.

Both use 5k_HRW, scratch initialization with no pretrained weights, seed 42, 300 epochs, batch size 8, 640x640 input, SGD, and lr0=0.01.

## Variant A: C2f-LGC partial internal replacement

`C2fLGC` preserves the original C2f outer contract: input 1x1 split projection, sequential transformed branch, feature concatenation, and output 1x1 projection. Its internal `m` list contains a mix of standard `Bottleneck` and `LGCBottleneck` blocks. `LGCBottleneck` keeps a 1x1 channel-mixing projection before and after one LGC residual unit, so it is a proper branch transformation rather than a post-hoc layer.

At the scaled YOLOv8n C4 C2f, there are two internal blocks: first standard Bottleneck, second LGCBottleneck. At C5, scale depth produces a single internal block, so its unique block is LGCBottleneck. This replaces one of two C4 units and the sole C5 unit; it is the closest reproducible per-stage approximation to “about half” under the existing YOLOv8n-n depth multiplier.

## Variant B: full C2f-to-LGC replacement

Remove the C4 and C5 C2f modules entirely. After each preceding stride-2 convolution, place an `LGC` module that preserves the incoming channel count and resolution. The C3 C2f and all Neck/Detect layers remain exactly standard YOLOv8n. This is a deliberately lower-capacity, lower-compute contrast to quantify how much C2f feature reuse contributes.

## Initialization and parser contracts

Every LGCBottleneck contains an LGC residual scale initialized to zero. Therefore the liquid subpath initially leaves its surrounding 1x1 projections as the active transform, while its parameters can learn progressively. C2fLGC has exactly the C2f input/output channel contract and receives `[c1, c2, n, shortcut, g, e]` from the normal parser repeat path. LGC keeps the already-tested shape-preserving parser contract.

## Validation and reporting

Tests must verify C2fLGC identity/finite behavior for the nested LGC unit, block-count composition at C4/C5, model parsing and finite 640 predictions, queue job order, and atomic parseable state. Each completed run must record best metrics, actual parameters, best.pt size, FLOPs at 640, and RTX 4080 SUPER FP32 batch-1 latency (50 warmups plus 200 forwards) in the standard Chinese report. Compare against YOLOv8n scratch, post-C2f LGC, LGRB, SATA, RG-SATA, and DR-SATA; state the single-split/single-seed limitation.
