# SLED-Head Design

## Goal

Improve action-category discrimination for partially visible upper bodies without changing YOLO detection-box regression, output tensors, post-processing, or the matched training protocol.

## Motivation

SCB-Dataset3-S labels three upper-body behaviours: hand raising, reading and writing.  The useful evidence is vertically structured: head/raised hand, torso/book, and desk/arm regions have different relevance.  A standard shared feature map asks the classification and localisation branches to use the same representation.  This can let behaviour context perturb box localisation, particularly for truncated persons.

## Module

`SLED` means **Spatial Layout Evidence Decoupling**.  `SLEDDetect` subclasses the normal YOLO `Detect` head.  At each P3/P4/P5 level it sends the original feature map unchanged to the box-regression branch and sends a layout-gated copy only to the class branch:

1. `SLEDLayoutGate` adaptively pools a feature map to three vertical bins and one horizontal bin.
2. A lightweight 1x1 reduction, 3x1 vertical mixing layer, and 1x1 projection predict a channel-wise vertical gate.
3. The gate is bilinearly restored to the input feature size and applies the residual transform `x * (1 + scale * (sigmoid(gate) - 0.5))`.
4. The final projection is zero-initialised, so the untrained gate is exactly the identity; the pretrained YOLO classification path is therefore preserved before optimisation.

This decouples behaviour-layout evidence from bounding-box evidence, but does not add labels, inference heads, custom decoding, or a new loss.

## Interfaces and registration

- Module: `ultralytics.nn.modules.sled.SLEDLayoutGate(c, bands=3, reduction=0.25, scale=1.0)` and `SLEDDetect(nc, ch, sled_cfg)`.
- Training return: one raw prediction tensor per level, each with the unmodified YOLO channel count `4 * reg_max + nc`.
- Inference return: standard `Detect` prediction `[B, 4 + nc, anchors]`, so NMS is unchanged.
- Parser: `parse_model` passes the top-level YAML mapping `sled` into `SLEDDetect` with feature channels from P3/P4/P5.
- Loss: use the ordinary `v8DetectionLoss`, because the prediction layout is unchanged.

## Configuration and evaluation

`yolo11n-sled.yaml` will duplicate the strict matched baseline body and replace only its final head with `SLEDDetect`.  Configuration is `bands: 3`, `reduction: 0.25`, `scale: 1.0`.  It will load `yolo11n.pt` and use the identical SCB-Dataset3-S protocol: 640px, batch 16, seed 42, device 0, 300 maximum epochs, patience 100, all augmentations and optimiser values matched to BCSQ/baseline.

The module is promoted only when its best validation mAP50-95 is at least 0.47210, i.e. +0.00200 over the strict matched baseline 0.47010.  A one-seed result is screening evidence only.

## Tests

Tests must first fail before the module is implemented.  They verify identity initialisation, tensor shape preservation, standard inference contract, gradients to the new classification-only gate, YAML parser registration/configuration, and one CUDA AMP smoke training epoch.  Full training starts only after these tests pass.
