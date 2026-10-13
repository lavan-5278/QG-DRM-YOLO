# LFM-Inspired Visual Backbone Blocks: Implementation Plan

> The `writing-plans` skill file was unavailable in the current plugin cache, so this plan follows the approved design and the repository's existing SATA implementation/testing conventions.

## Scope

Implement and independently train LGC, LGRB, and LHGB on YOLOv8n. Preserve the neck and Detect head, use no pretrained weights, and run the three jobs serially under the fixed 5k_HRW seed-42 protocol.

## 1. Write failing module and integration tests

Files:

- Modify `tests/test_sata_module.py` or add `tests/test_lfm_backbone_blocks.py`.
- Add `tests/test_train_hrw_yolov8n_lfm_blocks_queue.py`.

Tests to add before implementation:

1. `LGC`, `LGRB`, and `LHGB` are importable from `ultralytics.nn.modules`.
2. Each block returns exactly its input at initialization because `gamma=0`.
3. With `gamma=1`, each block returns a finite non-identical tensor with unchanged `[B,C,H,W]` shape.
4. LGRB handles feature maps with fewer cells than the configured anchor count.
5. All three YAML models construct and produce finite 640x640 predictions with the expected Detect output channels.
6. Queue initialization/reconciliation generates LGC, LGRB, LHGB in that exact order and retains the fixed scratch protocol.

Expected first test result: import and YAML failures, because the blocks/configurations/queue do not exist yet.

## 2. Implement reusable local liquid state primitives

File: `ultralytics/nn/modules/block.py`.

1. Add `LiquidGatedConv` as an internal utility: parallel depthwise 3x3 and 5x5 paths, pointwise gate, pointwise output projection, zero-initialized residual scalar.
2. Add `LGC` wrapper with input validation and a shape-preserving forward contract.
3. Add shared sparse-relation helper logic for anchor coordinates, semantic-geometric neighbour selection, safe attention, scatter-back, and diffusion. Keep the relation budget explicit and bounded.

## 3. Implement LGRB and LHGB

File: `ultralytics/nn/modules/block.py`.

1. Build `LGRB` from the local liquid state plus 32-anchor / 6-neighbour sparse relation updates, with source/target reliability gates and zero-initialized residual output.
2. Implement `LHGB` as a two-input C4/C5 stage-aware block: it accepts the current stage feature and the preceding retained feature map, pools/projects the preceding map internally, and broadcasts that context into its local gate and relation reliability head. It returns only the current-stage feature tensor.
3. The implementation must keep C3→C4→C5 state propagation entirely within parser-routed backbone inputs without adding a Neck edge or hidden cross-batch mutable state.

## 4. Register parser-visible modules and define model YAMLs

Files:

- `ultralytics/nn/modules/block.py`
- `ultralytics/nn/modules/__init__.py`
- `ultralytics/nn/tasks.py`
- `ultralytics/cfg/models/v8/yolov8n-lgc.yaml`
- `ultralytics/cfg/models/v8/yolov8n-lgrb.yaml`
- `ultralytics/cfg/models/v8/yolov8n-lhgb.yaml`

1. Export LGC, LGRB, and LHGB and add them to the shape-preserving parser branch.
2. Use LGC at C4/C5 for LGC.
3. Use LGRB at C4/C5 for LGRB.
4. For LHGB, use LGC at C3 and two-input stage-conditioned LGRB-equivalent blocks at C4/C5. Add a dedicated parser branch that derives current and context channel counts from the two YAML inputs. The YAML must preserve the standard YOLOv8n P3/P4/P5 routing into the unmodified neck.

## 5. Implement serial training and reporting queue

File: `experiments/train_hrw_yolov8n_lfm_backbone_queue.py`.

1. Follow the existing SATA queue's atomic JSON, reconciliation, benchmark, and report-update patterns.
2. Set output root to `runs/hrw_yolov8n_lfm_backbone_seed42_scratch`.
3. Train LGC → LGRB → LHGB using 5k_HRW, seed 42, scratch/no pretrained weights, batch 8, SGD, lr0=0.01, 300 epochs, 640x640.
4. Benchmark every best weight on RTX 4080 SUPER FP32 batch 1 at 640 after 50 warmups and 200 forwards.
5. Replace a marker-bounded “LFM 启发式 Backbone 实验” section in `runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md`, including metrics, parameter count, best-weight size, FLOPs, latency, and comparisons against baseline/SATA/RG-SATA/DR-SATA.

## 6. Verify before launch

Run, in order:

```powershell
D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m pytest tests\test_lfm_backbone_blocks.py tests\test_train_hrw_yolov8n_lfm_blocks_queue.py -v
D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe experiments\train_hrw_yolov8n_lfm_backbone_queue.py --dry-run
D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m json.tool runs\hrw_yolov8n_lfm_backbone_seed42_scratch\queue_state.json
```

Then start the queue in a hidden background process and create/replace the hourly monitoring automation. Do not silently restart a failed job. On completion, validate `results.csv`, best weights, generated summaries, report rows, and the queue state before reporting a conclusion.
