# SATA Reliability and Dual-Role Design

## Goal

Evaluate two internal SATA revisions, independently and under the existing YOLOv8n scratch protocol, to determine whether precision-aware social context or directed role-aware social context improves on SATA's 0.52162 mAP50-95 result.

## Evidence motivating the revisions

The completed SATA run improved Recall from 0.6492 to 0.6798 and mAP50-95 from 0.5169 to 0.5216 against YOLOv8n, but reduced Precision from 0.6794 to 0.6586 and mAP50 from 0.7039 to 0.7022. The current adapter selects anchors with a single learned score and limits relations by Euclidean nearest neighbors. It has no explicit mechanism to reject isolated high-response background anchors or distinguish a focal subject from its context.

## Common experimental protocol

Both variants use 5k_HRW, seed 42, scratch/no pretrained weights, batch 8, SGD, lr0 0.01, 300 epochs, and 640 by 640 images. They are independent YOLOv8n Backbone experiments, placed after C4 and C5 only, with unchanged Detect head, loss, data split, augmentation, and benchmark protocol. They run serially in the order RG-SATA then DR-SATA and write a separate marker section into the existing Chinese standard-results report.

## Variant A: RG-SATA (Reliability-Gated SATA)

RG-SATA retains K=32 anchors and up to six neighbors. Every spatial location obtains a saliency logit and a local-reliability logit. The product of their sigmoid probabilities selects anchors and gates their outgoing messages. Neighbor eligibility is determined by a combined semantic-and-geometric affinity, rather than Euclidean distance alone: normalized query/key similarity minus a learned nonnegative spatial-distance penalty. Attention receives the existing learned relative-coordinate bias plus the log reliability of the candidate neighbor. The returned message is therefore suppressed if either endpoint is unreliable.

This directly tests the Precision hypothesis: high saliency without local feature support should not propagate social context.

## Variant B: DR-SATA (Dual-Role SATA)

DR-SATA learns distinct subject and context score maps. It selects up to 16 subject anchors and 16 context anchors independently. Each subject attends only to its best six context anchors according to semantic-and-geometric affinity; messages flow context to subject, not symmetrically. The subject confidence gates the final update and the context confidence enters the attention score. The scattered subject updates are lightly depth-wise diffused before a zero-initialized residual addition.

This tests whether a directed focal-subject/context decomposition is more useful than undifferentiated social-anchor interaction. It has no object, pose, relation, or extra annotation input.

## Reliability and tensor contract

Both modules require B,C,H,W input with their configured channel count and return exactly B,C,H,W. Anchor counts and neighbor counts clamp to available spatial positions; each attention row has at least one valid candidate; invalid shapes fail with a clear ValueError. A zero-initialized scalar residual makes each module exactly identity at initialization.

## Success criteria

Operational success requires both checkpoints, summaries, and report rows with best metrics, parameter count, best.pt size, FLOPs, and fixed-protocol inference time. Comparative success is screened primarily on mAP50-95 while reporting the Precision/Recall trade-off. No result is treated as statistically conclusive because this is a single data split and seed.
