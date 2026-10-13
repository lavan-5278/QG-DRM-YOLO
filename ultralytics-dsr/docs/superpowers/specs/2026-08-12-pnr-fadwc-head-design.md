# PNR-DFL + FADWC-Head 设计说明

## 目标

在不改变 YOLO11n 的 P3/P4/P5 检测接口和 PNR-DFL 损失的前提下，验证频率自适应动态小波卷积头（FADWC-Head）是否能提升 SCB-Dataset3-S 上可见上半身行为区域的定位。

## 模块定义

每个检测尺度独立使用一个 `FADWC(c1)`。输入为 `x∈R^(B×C×H×W)`，输出保持相同形状。

1. 以固定 Haar 低通核和三个高通核进行逐通道卷积，得到 `LL, LH, HL, HH`；采用 stride=1、确定性的零填充，因此不改变 H/W。
2. 高频能量 `E=(|LH|+|HL|+|HH|)/3` 经 1×1 门控映射得到 `g∈[0,1]`。
3. 分别提取细节候选 `D=DWConv3×3(E)` 与上下文候选 `C=DWConv3×3,dilation=2(LL)`，并经 1×1 投影到 C 通道。
4. 残差输出 `y=x+γ·Proj(g·D+(1-g)·C)`，其中 γ 从 0 初始化，保证初始行为接近原始 Detect 输入。

这使高频边缘/手臂器具纹理强的区域偏向细节分支，平滑遮挡或主体区域偏向上下文分支；并且没有跨尺度融合，因而与 PNR-DFL 的标签空间加权机制互补。

## 接口与注册

- 实现：`ultralytics.nn.modules.block.FADWC(c1: int)`，`forward(x) -> x.shape`。
- 注册：`ultralytics.nn.modules.__init__` 与 `ultralytics.nn.tasks.parse_model`。
- YAML：在原 PNR-DFL 的 Detect 前，对 P3/P4/P5 分别插入三个 FADWC 层，再送入普通 `Detect`；损失仍由 YAML 的 `pnr` 配置选择 `PNRDetectionLoss`。

## 验证协议

- 先做 CPU 单元测试：输出形状、反向梯度、YAML 的 Detect 原始输出以及 PNR 损失选择。
- 再做 CUDA 极小数据冒烟训练，排除 NaN/OOM/解析错误。
- 正式训练：SCB-Dataset3-S，YOLO11n，300 epochs，seed 42/43/44，所有其余训练参数与已完成 Baseline/PNR-DFL 完全相同。
- 终点：每个 seed 取最大验证 mAP50-95；与同 seed Baseline 和 PNR-DFL 配对比较 P/R/mAP50/mAP50-95，报告均值与样本标准差。
- 仅当组合 mAP50-95 均值同时相对 Baseline 与 PNR-DFL 不低于 +0.00200 时，结论为 `PROMOTE_STACK`，否则 `REJECT_STACK`。

## 边界

FADWC 是一次单划分、三随机种子的叠加筛选，不替代跨数据集或跨划分泛化结论。
