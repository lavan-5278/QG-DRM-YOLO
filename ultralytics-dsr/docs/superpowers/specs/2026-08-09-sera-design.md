# SERA Design

## Goal

Improve upper-body behaviour recognition by injecting only semantically compatible P4/P5 context into P3 hand/arm evidence, without changing YOLO regression, detection outputs or NMS.

## Module

SERA means **Semantic Evidence Relation Aggregator**.  It consumes post-neck P3, P4 and P5 features.  P4/P5 are projected to P3 channels and bilinearly upsampled.  A lightweight spatial gate predicts where their sum agrees with P3; the result is added only as a residual to P3:

`P3_out = P3 + gamma * sigmoid(gate([P3, context])) * context`.

`gamma` is a learnable scalar initialized to zero, so SERA initially returns P3 exactly.  Consequently transferred `yolo11n.pt` behaviour is unchanged before optimisation.  P4/P5 pass directly into `Detect`, preserving box/class head interfaces.

## Integration and evaluation

The module has interface `SERA(ch=(c3,c4,c5), sera_cfg)` and returns a P3 tensor `[B,c3,H3,W3]`.  YAML inserts `[[16,19,22],1,SERA,[sera]]`; Detect consumes `[SERA_output,19,22]`.  The body otherwise matches `yolo11n-baseline-matched.yaml`.  No custom loss is used.

The test protocol verifies identity initialization, P3 shape, P4/P5 gradient route after nonzero gain, parser registration, standard 67 raw channels and 7 inference channels, then CUDA AMP smoke training.  Formal screening uses the identical 300-maximum-epoch protocol, seed 42 and baseline promotion threshold mAP50-95 >= 0.47210.  A single seed remains screening evidence only.
