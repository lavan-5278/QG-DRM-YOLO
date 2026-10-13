# PNR-DFL + FADWC-Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将保持尺度不变的频率自适应动态小波卷积头注册到 YOLO11n，并完成严格三 seed PNR-DFL 叠加验证。

**Architecture:** 三个独立 FADWC 层分别细化 P3/P4/P5；每层使用固定 Haar 频率滤波、细节/上下文双候选和空间门控，残差输出保持 `B,C,H,W`。YAML 仍将三个尺度送入原始 Detect，PNR-DFL 仍由已有损失选择路径处理。

**Tech Stack:** PyTorch、Ultralytics YOLO、pytest、Windows CUDA、nvidia-smi。

## Global Constraints

- 不使用 Git，不覆盖既有训练目录。
- 训练协议固定为 SCB-Dataset3-S、YOLO11n、300 epoch、seed 42/43/44。
- 只在 GPU 空闲（使用量不高于 4000 MiB）时启动正式队列。
- 最终以每个 seed 的最佳 mAP50-95 做配对比较，推广阈值为相对 Baseline 与 PNR-DFL 均不低于 +0.00200。

---

### Task 1: FADWC 单模块及解析注册

**Files:**
- Modify: `ultralytics/nn/modules/block.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Test: `tests/test_pnr_fadwc_stack.py`

**Interfaces:**
- Produces: `FADWC(c1: int)`，输入输出均为 `torch.Tensor[B,C,H,W]`。

- [ ] 写入断言 `FADWC(16)` 对 `(2,16,15,17)` 保持形状且输入/参数可反向传播的失败测试。
- [ ] 运行该测试，预期导入 `FADWC` 失败。
- [ ] 用固定 Haar 滤波、动态门控、深度可分细节/扩张上下文卷积和零初始化残差实现最小模块；在导出表和模型解析器注册。
- [ ] 运行单元测试，预期通过。

### Task 2: PNR-FADWC YAML 组合契约

**Files:**
- Create: `ultralytics/cfg/models/11/yolo11n-pnr-fadwc.yaml`
- Modify: `tests/test_pnr_fadwc_stack.py`

**Interfaces:**
- Consumes: `FADWC(c1)`。
- Produces: P3/P4/P5 经 FADWC 后进入普通 `Detect` 的完整模型 YAML。

- [ ] 写入 YAML 构建、raw Detect 每尺度 67 channel、预测第二维 7、`PNRDetectionLoss` 选择的失败测试。
- [ ] 运行测试，预期 YAML 文件不存在。
- [ ] 从 PNR-DFL 网络复制骨干与 FPN，在 Detect 前增加三个 FADWC，并连接 `[23,24,25]` 到 Detect。
- [ ] 运行 YAML 契约和 Task 1 测试，预期通过。

### Task 3: 三 seed 队列和配对报告

**Files:**
- Create: `experiments/pnr_fadwc_stack_queue.py`
- Create: `experiments/report_pnr_fadwc_stack.py`
- Modify: `tests/test_pnr_fadwc_stack.py`

**Interfaces:**
- Produces: `runs/pnr_fadwc_stack_3seed/queue_state.json` 和 `stack_comparison.md`。

- [ ] 写入三个新作业（42/43/44）和严格训练参数的失败测试，以及报告最佳 epoch/样本标准差测试。
- [ ] 运行测试，预期导入队列和报告模块失败。
- [ ] 以 AFPN 队列的原子状态、GPU 守门、完整输出检查模式实现 FADWC 队列与报告；报告读取同 seed Baseline/PNR 端点。
- [ ] 运行所有 FADWC 与已有 PNR 测试，预期通过。

### Task 4: CUDA 冒烟与正式验证

**Files:**
- Output: `runs/pnr_fadwc_stack_smoke/`
- Output: `runs/pnr_fadwc_stack_3seed/`

- [ ] 在极小数据分数下运行 1 epoch CUDA 冒烟，检查 `results.csv`、`best.pt`、无 NaN/OOM。
- [ ] 以三 seed 队列启动严格 300 epoch 训练，记录 PID 与状态。
- [ ] 每 15 分钟读取队列、当前 CSV、错误日志和 GPU 状态。
- [ ] 完成后生成配对报告并按阈值报告 `PROMOTE_STACK` 或 `REJECT_STACK`。
