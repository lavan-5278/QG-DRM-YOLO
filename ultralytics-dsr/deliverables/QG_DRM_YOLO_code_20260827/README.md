# QG-DRM-YOLO 源码包

本包对应当前已完成实验中 **mAP50-95 最高的单模型**：

`YOLOv8s + C4 RGDRC2fV3 + QG-AFPN + P3/P4/P5 MLKCM + P3DetailRestore`

模型保持 YOLOv8s 的 P2/P3、C5、PAN-FPN 前段和 Detect 头不变，仅在以下位置改造：

- Backbone C4：`RGDRC2fV3`
- 三尺度跨尺度融合：`QGAFPN`
- Detect 前 P3/P4/P5：`MLKCM`
- P3 的 MLKCM 输出后：`P3DetailRestore`

## 目录

- `ultralytics/`：包含当前项目注册代码的完整 Python 源码快照。
- `ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail.yaml`：最终模型配置。
- `train_qg_drm_yolo.py`：独立训练入口，固定复现实验协议。
- `evidence/args.yaml`：原实验实际参数。
- `evidence/results.csv`：完整 300 epoch 记录。
- `evidence/native_summary.json`：参数量、FLOPs、推理时间及学习到的残差系数。
- `docs/方法章节_按ScientificReports结构_QG_DRM_YOLO.md`：方法论章节草稿。

## 关键源文件

- `ultralytics/nn/modules/dr_c2f_v3.py`
- `ultralytics/nn/modules/dr_c2f_v3_gate.py`
- `ultralytics/nn/modules/pfom_mlkcm.py`
- `ultralytics/nn/modules/p3_detail_restore.py`
- `ultralytics/nn/modules/block.py`：包含 `QGAFPN`。
- `ultralytics/nn/modules/__init__.py`：模块导出注册。
- `ultralytics/nn/tasks.py`：YAML 解析与通道推导注册。

## 安装

建议在新的虚拟环境中执行：

```powershell
pip install -e .
```

若需要严格复用本机环境，可参考 `requirements-lock.txt`。该锁文件包含原环境的完整依赖快照，跨机器安装时可能需要按 CUDA/PyTorch 版本调整。

## 数据集配置

默认数据配置位于：

`ultralytics/cfg/datasets/5k-HRW.yaml`

该文件中的 `path` 是原实验机器上的绝对路径。迁移到其他机器后，请修改 `path`，或训练时通过 `--data` 指定新的数据 YAML。

## 结构检查

```powershell
python train_qg_drm_yolo.py --dry-run
```

脚本会确认 C4、C5、QG-AFPN、三个 MLKCM、P3DetailRestore 和 Detect 的位置，防止误加载其他组合配置。

## 同配置训练

```powershell
python train_qg_drm_yolo.py --data ultralytics/cfg/datasets/5k-HRW.yaml
```

训练协议固定为：5k_HRW、seed=42、scratch、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。可使用 `--project`、`--name`、`--device` 和 `--workers` 调整输出路径与运行设备；这些参数不会改变模型结构。

## 已完成结果

- Precision：0.71728
- Recall：0.69138
- mAP50：0.74480
- mAP50-95：0.57765
- 参数量：19,142,575
- FLOPs（640）：42.2296 G
- 推理时间：11.078 ms/图

推理时间协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向平均，不含数据读取、预处理和 NMS。

上述结果仅来自单一数据划分和 seed=42，应通过多 seed 实验确认统计稳定性。

## 权重说明

本压缩包是源码包，不包含 `best.pt`。原权重位于原项目的：

`runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch/native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail/weights/best.pt`

