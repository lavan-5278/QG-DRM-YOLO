# PNR-DFL Design

## Goal

Improve localisation of small, high-image upper bodies through a training-only Perspective-Normalized Residual Distribution Focal Loss (PNR-DFL), while retaining the exact YOLO detection head, output tensors, decoding and NMS.

## Motivation and method

The Phase-0 diagnosis found a positive relationship between vertical image coordinate and log-area (Spearman 0.58247).  Upper, farther examples are smaller and localisation errors consume a larger fraction of their visible body extent.  Standard DFL applies its distributional target loss uniformly after matching.  PNR-DFL retains it and adds a relative DFL expectation residual to the normal DFL component.

For each matched anchor and LTRB side, let `p` be the `reg_max`-bin softmax distribution, `mu=sum_k k*p_k`, and `d*` the ordinary DFL target distance.  With target box width/height in the current feature-grid coordinates `(w,h)`, scale `s=max(sqrt(w*h), 1)`.  Let `y` be the target centre normalised by image height and `r=sqrt((w_px/W)*(h_px/H))` clamped to `[0,1]`.  The residual is:

`L_pnr = sum_fg w * SmoothL1((mu-d*)/s) / (4 * sum_fg w)`,

where `w=1+alpha*(1-y)*(1-r)`.  The final loss is `L_yolo + lambda_pnr*L_pnr`.  This keeps the DFL bin supervision but makes relative distance calibration more influential for far/small upper bodies.  `alpha=1.0`, `lambda_pnr=0.25`, and `scale_floor=1.0` are the initial fixed settings.

## Integration

`PNRDetectionLoss(v8DetectionLoss)` overrides only the normal detection-loss call.  It preserves the Task-Aligned Assigner, BCE class term, CIoU term, standard DFL, and three logged loss columns.  PNR loss is added to the logged DFL value after the normal `hyp.dfl` gain.  The top-level YAML mapping `pnr` causes `DetectionModel.init_criterion()` to select PNRDetectionLoss; a normal `Detect` head remains unchanged.

## Evaluation

`yolo11n-pnr-dfl.yaml` is body-identical to `yolo11n-baseline-matched.yaml`, with only the top-level `pnr` mapping added.  Training loads `yolo11n.pt` and uses SCB-Dataset3-S at 640px, batch 16, seed 42, device 0, 300 maximum epochs, patience 100, and the same optimiser/augmentation protocol as baseline and SLED.  Promote only at best mAP50-95 >= 0.47210.  One seed is screening evidence only.

## Tests

Tests first fail before the loss is implemented.  They use hand-derived predictions/targets to verify: (1) a small high target has a larger PNR weight than a large low target; (2) matching expected distributions gives zero PNR residual; (3) wrong distributions yield finite nonzero gradients; (4) YAML selection chooses PNRDetectionLoss but standard Detect output shapes are unchanged; (5) CUDA AMP smoke training completes.
