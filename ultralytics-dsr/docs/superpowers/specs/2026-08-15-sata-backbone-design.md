# SATA Backbone Design

## Goal

Evaluate whether sparse social context inside the YOLOv8n backbone improves 5k_HRW detection under the established scratch protocol: seed 42, batch 8, SGD, lr0 0.01, 300 epochs and 640 by 640 inputs.

## Scope

SATA (Social Anchor Token Adapter) is a shape-preserving backbone adapter.  It is evaluated alone on YOLOv8n, not stacked with AFPN, DSR, PNR, or another experimental module.  The run is appended after the five completed YOLOv8n module experiments and uses the same queue, report, benchmark, and checkpoint conventions.

## Alternatives considered

1. Global all-position relation attention: strongest nominal context coverage, but quadratic spatial cost and high risk of background noise on 5k_HRW.
2. Detection-head candidate graph: directly models boxes, but confounds the test with NMS/candidate-selection behavior and is not a backbone intervention.
3. Sparse anchor-token relation adapter: select a small learned set of high-response C4/C5 anchors, exchange only local geometric context, and write it back with a neutral residual. This is selected because it isolates backbone social context at small cost.

## Architecture

Two SATA blocks are inserted after the C4 C2f stage and after the C5 SPPF stage. C3 is intentionally untouched so high-resolution small-object texture is preserved.

For input feature map X with shape B by C by H by W:

1. A learned one-channel score map selects K = min(32, H times W) anchor locations per image.
2. The selected C-dimensional features are projected to query, key, and value tokens of dimension 64.
3. Normalized anchor coordinates provide a learned relative-coordinate bias. Each anchor attends only to its nearest min(6, K) anchors, including itself.
4. The fused messages are projected back to C channels, scattered to their anchor positions, locally diffused with depth-wise convolution, and added to X through a zero-initialized scalar residual.

The zero-initialized residual guarantees the module starts as an exact identity mapping, so scratch training is not destabilized by an abrupt feature-distribution change. The output keeps exactly the original B, C, H, W contract and works with the unchanged YOLOv8 Detect head.

## Reliability and errors

The module rejects non-4D tensors or wrong channel counts. K is clamped to the number of spatial positions and neighbor count is clamped to K, so 1 by 1 feature maps remain finite. All attention masking retains self-links; no row is fully masked.

## Verification

Tests must be written before implementation and demonstrate: (1) C4/C5-shaped inputs retain shape and are finite, (2) zero residual is an identity, (3) enabling the residual produces an actual context update, and (4) the SATA YAML constructs a complete YOLO detector with finite 640-pixel outputs. Queue-state reconciliation must preserve the five completed jobs while appending a pending SATA job.

## Success criterion

The experiment is successful operationally if it trains to completion, its best checkpoint is benchmarked, and metrics, parameter count, checkpoint size, FLOPs, and inference time are written to the standard Chinese baseline report. Detection quality is interpreted against YOLOv8n baseline and AFPN only as a single-seed, single-split screening result.
