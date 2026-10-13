# PVR-Q Phase 0 diagnostic report (repaired)

## G0 rule

The gate uses two image-cluster bootstrap quantities: (1) small/competitive GT focus-minus-reference matched IoU, and (2) the gain from perfectly reranking **all post-NMS detections** by their class-aware one-to-one assigned IoU quality. Unmatched predictions have quality 0; unmatched GTs are not calibration samples.

- A localization deficit exists when the IoU-difference CI upper bound is below 0.
- Quality-head headroom exists when the oracle-gain CI lower bound exceeds the declared minimum.
- With both: `PROCEED_PVRQ`; with localization deficit only: `PROCEED_PVRF_ONLY`; without a localization deficit: `STOP`.
- LPF eligibility is reported separately using `|Spearman(y, log-area)| >= 0.20`.

## Result

- Decision: **STOP**
- Localization deficit: **False**
- Quality-head headroom: **False**
- Latent perspective field eligible: **False**
- Focus-minus-reference IoU: 0.000000 (95% CI [0.000000, 0.000000])
- Oracle fixed-match AP50-95 ranking gain: 0.000000 (95% CI [0.000000, 0.000000])
- Spearman(y-center, log-area): 0.000000
- Population: 1 valid images, 1 GT, 1 post-NMS predictions
- Inference device: `unknown`; CUDA available: `None`
- CPU fallback: `None` — Not recorded in the manifest.

The oracle metric freezes confidence-ordered matches and is a diagnostic ranking proxy, not detector AP or a claim of achievable mAP gain. G0 authorizes implementation experiments; only controlled training can establish an actual improvement.
