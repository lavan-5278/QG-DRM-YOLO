# PNR-DFL + DyHead 设计说明

## 目标

测试 PNR-DFL 的目标级定位重加权能否与 DyHead 的特征级动态注意力互补，以改善 SCB-Dataset3-S 中不同尺度、遮挡和上半身行为区域的检测。

## 最小部署实现

`DyHead(ch)` 接收 P3/P4/P5 三个尺度并返回同样三组形状：

1. **尺度注意力**：每个目标尺度将相邻尺度经 1×1 对齐、上/下采样后相加；源尺度全局池化得到 softmax 权重，抑制语义不匹配的来源。
2. **空间注意力**：每个融合特征使用深度 3×3 动态空间门，作为轻量且 CUDA 确定性的可变形采样替代，不引入 torchvision 可变形卷积的版本依赖。
3. **任务注意力**：SE 风格通道门控，按当前行为类别的共享检测表征重新标定通道。
4. 每尺度以零初始化残差系数回注输入，确保初始模型接近原 P3/P4/P5 路径；输出不改变通道数、空间尺寸或普通 `Detect` 的接口。

## 组合与验证

- YAML 在 P3/P4/P5 后插入一个多输入 DyHead，并用三个 `Index` 取回 P3/P4/P5 后输入原始 `Detect`。
- PNR 配置保持 `alpha=1.0, lambda_pnr=0.25, normalize=true, weight_mode=joint`，损失仍是 `PNRDetectionLoss`。
- 训练固定 SCB-Dataset3-S、YOLO11n、300 epoch、seed 42/43/44、同一严格协议。
- 每 seed 取最佳 mAP50-95，给出 Precision/Recall/mAP50/mAP50-95 对同 seed Baseline 和 PNR-DFL 的配对差、均值和样本标准差。
- 仅当 mAP50-95 平均值相对 Baseline 与 PNR-DFL 均至少 +0.00200 时保留组合。

## 边界

本模块实现的是与 DyHead 三类注意力对应的轻量可部署变体，不宣称逐行复现原始 DCNv2 DyHead；结论限于单官方划分、三随机种子。
