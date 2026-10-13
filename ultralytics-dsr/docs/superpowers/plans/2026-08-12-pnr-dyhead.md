# PNR-DFL + DyHead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个保持三尺度 Detect 接口不变的轻量 DyHead，并在严格三 seed 协议下验证其与 PNR-DFL 的叠加效益。

**Architecture:** DyHead 在 P3/P4/P5 做动态尺度融合、深度空间门控与通道任务门控，返回三个原始尺度形状；通过 Index 将输出接回普通 Detect，PNRDetectionLoss 不变。

**Tech Stack:** PyTorch、Ultralytics YOLO、pytest、Windows CUDA、nvidia-smi。

## Global Constraints

- 不使用 Git；不覆盖已存在的训练端点。
- SCB-Dataset3-S、YOLO11n、300 epoch、seed42/43/44，训练参数与 PNR-DFL 完全匹配。
- 正式队列只在 GPU 使用量不高于 4000 MiB 时启动。
- 推广阈值：组合平均 mAP50-95 同时相对 Baseline、PNR-DFL 不低于 +0.00200。

---

### Task 1: DyHead 接口、形状与模型解析

**Files:**
- Modify: `ultralytics/nn/modules/block.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Test: `tests/test_pnr_dyhead_stack.py`

- [ ] 写入 `DyHead((16,32,64))` 对 P3/P4/P5 保持形状、输入有梯度的失败测试。
- [ ] 运行测试，预期 `DyHead` 导入失败。
- [ ] 实现动态尺度、空间和任务门控，注册导出和 `parse_model` 多输入分支。
- [ ] 运行单元测试，预期通过。

### Task 2: YAML 与 PNR 损失契约

**Files:**
- Create: `ultralytics/cfg/models/11/yolo11n-pnr-dyhead.yaml`
- Modify: `tests/test_pnr_dyhead_stack.py`

- [ ] 写入 YAML 不存在的失败测试，断言三层 DyHead 输出经 Index 接入 Detect 且选择 `PNRDetectionLoss`。
- [ ] 从 PNR-DFL YAML 拷贝网络，在 Detect 前增加 DyHead 与三个 Index。
- [ ] 运行 YAML/模块测试，预期通过。

### Task 3: 队列与配对报告

**Files:**
- Create: `experiments/pnr_dyhead_stack_queue.py`
- Create: `experiments/report_pnr_dyhead_stack.py`
- Modify: `tests/test_pnr_dyhead_stack.py`

- [ ] 写入三个新 seed 作业、严格训练参数、最佳 epoch 与配对样本标准差的失败测试。
- [ ] 实现原子状态、GPU 守门、完整输出检查、同 seed Baseline/PNR 报告。
- [ ] 运行 DyHead + PNR 全部测试，预期通过。

### Task 4: CUDA 冒烟与正式三 seed

**Files:**
- Output: `runs/pnr_dyhead_stack_smoke/`
- Output: `runs/pnr_dyhead_stack_3seed/`

- [ ] 运行 1 epoch 极小数据 CUDA 冒烟，检查 results.csv/best.pt/无 NaN、OOM、确定性警告。
- [ ] 启动 300 epoch × seed42/43/44 串行队列。
- [ ] 每 15 分钟报告状态；完成后生成配对报告和严格推广结论。
