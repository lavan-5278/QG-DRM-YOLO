# C2f-LGC Replacement Ablation: Implementation Plan

## 1. Add red tests

- Extend `tests/test_lfm_backbone_blocks.py` to import `C2fLGC`, check mixed internal block composition, zero-residual behavior of its LGCBottleneck, finite non-identity update after enabling its LGC residual, and 640 predictions for both ablation YAMLs.
- Add `tests/test_train_hrw_yolov8n_c2f_lgc_queue.py` to assert ordered jobs and fixed scratch protocol.
- Run the tests before implementation; expect import/config/queue failures.

## 2. Implement C2fLGC

- In `ultralytics/nn/modules/block.py`, add `LGCBottleneck` with `Conv -> LGC -> Conv` and optional shortcut.
- Add `C2fLGC` by retaining C2f's split/concat/output structure and deterministically using LGC for the final `ceil(n/2)` internal units. This yields one standard plus one LGC unit at C4 (`n=2`) and one LGC unit at C5 (`n=1`).
- Export both public modules in `block.py` and `ultralytics/nn/modules/__init__.py`.

## 3. Register parser and YAMLs

- Add `C2fLGC` to the C2f parser/repeat-module families in `ultralytics/nn/tasks.py` so it receives C2f-equivalent arguments and repeat count.
- Create `yolov8n-c2f-lgc.yaml`: replace C4/C5 C2f with C2fLGC; do not insert post-C2f LGC.
- Create `yolov8n-c2f-to-lgc.yaml`: replace C4/C5 C2f with standalone LGC; preserve C3, Neck, and Detect.

## 4. Build serial queue/report integration

- Add `experiments/train_hrw_yolov8n_c2f_lgc_queue.py` using atomic state, dry-run graph validation, training, summaries, benchmarks, and marker-bounded report replacement patterns established by the LFM queue.
- Queue `C2f-LGC` first then `C4/C5 C2f→LGC` second; wait for `runs/hrw_yolov8n_lfm_backbone_seed42_scratch/queue_state.json` to show all three prior jobs completed before it uses the GPU.
- Append metrics and efficiency results to the existing Chinese standard report.

## 5. Verify and run

```powershell
D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m pytest tests\test_lfm_backbone_blocks.py tests\test_train_hrw_yolov8n_c2f_lgc_queue.py -v
D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe experiments\train_hrw_yolov8n_c2f_lgc_queue.py --dry-run
D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m json.tool runs\hrw_yolov8n_c2f_lgc_seed42_scratch\queue_state.json
```

Start the waiting queue as a hidden process. It must not alter, stop, or overlap the active LHGB process. Update the hourly heartbeat monitor to include the new queue, then complete final report audit after both ablations finish.
