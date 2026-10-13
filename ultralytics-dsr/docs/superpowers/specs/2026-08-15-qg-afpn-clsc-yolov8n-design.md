# QG-AFPN 与 CLSC Head：YOLOv8n 独立实验设计

## 目标

在 5k_HRW 上以 YOLOv8n 为基础，分别验证两个独立模块：

1. **QG-AFPN**：定位于 Neck 末端，以当前 AFPN 为基线，增加局部尺度一致性质量调制。
2. **CLSC Head**：定位于 Detect 输入/内部，为分类与定位分支提供不同的残差适配特征。

两项实验不互相叠加，均与现有 YOLOv8n 和 YOLOv8n+AFPN 单模块结果比较。

## 固定实验协议

- 数据集：5k_HRW。
- 模型规模：YOLOv8n。
- 初始化：从零训练；不使用预训练权重。
- 随机种子：42；batch=8；SGD；lr0=0.01；300 epoch；输入 640×640。
- 执行：单 GPU 串行；使用 RTX 4080 SUPER。
- 记录：最佳 mAP50-95 对应 epoch 的 P、R、mAP50、mAP50-95 和训练/验证损失；实际参数量、best.pt 大小、640 FLOPs、50 次预热后 200 次 FP32 batch=1 前向平均延迟。

## QG-AFPN

### 接口与位置

`QGAFPN(ch: tuple[int, int, int]) -> list[Tensor]` 接收 Neck 的 P3/P4/P5，返回通道数和空间尺寸不变的 P3'/P4'/P5'，供原始 `Detect` 直接消费。

### 数据流

对每个目标尺度，先保留 AFPN 的顺序式 P3→P4→P5 投影、尺寸对齐、门控与 3×3 融合。每一条来源尺度注入路径增加：

- 来源质量图 `q_s = sigmoid(Q_s(F_s))`；
- 尺度一致性 `a = exp(-mean_c(|LN(F_t)-LN(F_s)|))`；
- 可学习调制 `m = 1 + tanh(alpha) * (q_s * a - 1)`；
- 有效门控 `g_eff = g_afpn * m`。

`alpha` 零初始化，因此初始 `m=1`，前向严格等价于 AFPN；原 AFPN 作为稳定回退路径。质量图只由检测损失端到端学习，不引入额外监督或额外损失。

## CLSC Head

### 接口与位置

`CLSC(ch: tuple[int, int, int])` 对 P3/P4/P5 分别返回 `(F_cls, F_loc)`；`CLSCDetect` 继承原 `Detect` 的输出、解码和损失契约，但将分类卷积 `cv3` 输入 `F_cls`，将回归卷积 `cv2` 输入 `F_loc`。

### 适配器

每个尺度均有两条无通道扩张的残差适配支路：

- 分类：`F_cls = F + alpha_cls * PW(DWConv5×5(F))`，提高局部上下文感受野；
- 定位：`F_loc = F + alpha_loc * PW(DWConv3×3(F))`，保留局部边界细节。

`alpha_cls` 和 `alpha_loc` 均零初始化，故训练初始行为等价于普通 YOLOv8 Detect。CLSC 不修改标注、损失函数、DFL、框解码或 NMS。

## 注册、配置与验证

- 在 `ultralytics.nn.modules` 和模型解析器注册 `QGAFPN`、`CLSC`、`CLSCDetect`。
- 新增 `yolov8n-qg-afpn.yaml` 和 `yolov8n-clsc.yaml`。
- 每个模型先进行：YAML 解析、CPU 构建、640×640 前向形状/有限值检查、训练脚本语法检查。
- 新队列必须采用原队列的原子 JSON 状态写入模式；完成项不会重跑；失败项保留错误文本。

## 报告与成功判据

结果追加到 `runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md` 中单独的 “QG-AFPN / CLSC 独立补充实验” 区段，绝不覆盖既有模块表。

单 seed Pilot 的预注册正信号：相对 YOLOv8n+AFPN，QG-AFPN 的 mAP50-95 至少 +0.003；相对 YOLOv8n，CLSC 的 mAP50-95 至少 +0.003。达不到阈值的结果保留并报告为无足够正信号，而不进行模块叠加。

## 限制

单 seed、单数据划分只能提供可复现实验信号，不能作为统计显著性或原创性结论。QG-AFPN 和 CLSC 均为待验证的原创候选，须在有正信号后进行系统文献新颖性检索和多 seed 验证。
