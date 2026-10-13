# HBPC-Head Design

## Goal

Add a single, registrable YOLO11 detection-head module for SCB-Dataset3-S: **HBPC-Head** (Head–Behaviour-Part Complementarity Head). It calibrates the classification feature at each detection scale with complementary local cues from the head/shoulder side and the hand–book–desk interaction side of a visible upper-body instance. Box regression remains on the unmodified feature.

## Dataset rationale

The three SCB-Dataset3-S classes (hand raising, reading, writing) are defined by a visible upper body plus a small action-bearing region. A local hand or book cue can be ambiguous without head/shoulder pose context; conversely, head/torso posture alone cannot reliably distinguish reading from writing. Shared YOLO features give both classification and localisation the same representation. HBPC makes the class branch ask whether two directional, local pieces of evidence agree, while retaining the unmodified feature for localisation.

This differs from SLED-Head: SLED reconstructs a feature gate from global vertical bands. HBPC uses no image-coordinate bands. At every feature-map location it samples a one-cell upward neighbourhood (head/shoulder evidence) and a one-cell downward neighbourhood (hand/book/desk evidence), so it remains translation equivariant and can apply to people anywhere in the image.

## Module contract

`HBPC(c: int, hbpc_cfg: dict | None)` accepts one feature map `x` of shape `[B, C, H, W]` and returns `[B, C, H, W]` with the same dtype/device.

1. Form `head` by vertically shifting `x` down by `shift` cells; output row `y` receives source row `y-shift`, which is the local upper cue.
2. Form `behaviour` by vertically shifting `x` up by `shift` cells; output row `y` receives source row `y+shift`, which is the local lower interaction cue.
3. Project both paths from `C` to `hidden=max(8, round(C*reduction))` with separate 1x1 convolutions.
4. Combine them as `head + behaviour + head * behaviour`, apply a depth-wise 3x3 mixer and SiLU, then project to `C` channels.
5. Return `x * (1 + gain * scale * (sigmoid(delta) - 0.5))`, where `gain` is a scalar parameter initialized to zero.

The zero-initialised gain makes HBPC an exact identity at initialization, protecting fair pretrained transfer. Setting a nonzero gain must provide gradient to both directional projections.

Default YAML configuration is `shift: 1`, `reduction: 0.25`, `scale: 1.0`.

## Detect integration

`HBPCDetect(Detect)` owns one `HBPC` per P3/P4/P5 input. For each scale, it sends the original feature to `cv2[i]` (DFL/box regression) and the calibrated feature to `cv3[i]` (class logits), then concatenates the normal YOLO output. It preserves normal training raw output and inference/NMS contracts, the existing `cv2`/`cv3` parameter names, and does not support end-to-end Detect heads.

`yolo11n-hbpc.yaml` is an exact copy of `yolo11n-baseline-matched.yaml`, except for an `hbpc` configuration block and replacement of the final `Detect` with `HBPCDetect`; its three sources remain `[16, 19, 22]`.

## Validation and decision rule

Before training, tests must prove exact identity and shape preservation, bidirectional gradient flow after enabling `gain`, and construction/inference contract from the YAML. A one-epoch smoke run must complete without NaN, OOM, or model-parser errors.

The screening run uses the strict matched protocol: SCB-Dataset3-S, YOLO11n initial weights, 640 px, batch 16, seed 42, deterministic mode, device 0, all augmentation/optimizer settings equal to the matched baseline, up to 300 epochs and patience 100. Compare the best CSV `metrics/mAP50-95(B)` to baseline `0.47010`. Promote only if delta is at least `+0.00200`; otherwise reject. The first result is a single-seed screen, not a final statistical claim.
