# 第 3 章 方法论

> 本章采用“总体框架—关键模块—检测头与损失函数”的组织方式。为避免与既有方法混淆，本文将最终网络暂记为 **QG-DRM-YOLO**（Quality-Guided Multi-scale Detail Relay YOLO）。该名称仅用于本文叙述；模型由 YOLOv8s、RGDRC2fV3、QG-AFPN、MLKCM 和 P3DetailRestore 组成。

## 3.1 总体框架

5k_HRW 场景中的检测对象具有尺度变化大、局部纹理易受背景干扰以及相邻类别外观相似等特点。若只依赖浅层局部纹理，模型难以建立足够的语义区分；若过早、过强地引入高层上下文，则可能损失小目标边界。因此，本文以 YOLOv8s 为基础，构建由“深层选择性语义中继、质量约束跨尺度融合、全尺度上下文建模和 P3 细节补偿”组成的 QG-DRM-YOLO。

给定输入图像 $\mathbf{I}\in\mathbb{R}^{H\times W\times3}$，YOLOv8s Backbone 与原始 PAN-FPN 首先生成步长为 $8$、$16$ 和 $32$ 的多尺度特征 $\{\mathbf{F}_3,\mathbf{F}_4,\mathbf{F}_5\}$。在 Backbone 中，本文仅用可靠性门控差分中继模块替换 C4 阶段的 C2f，而保留 P2/P3 的局部特征提取路径和 C5 的高层语义提取路径。随后，QG-AFPN 对三个尺度进行逐级跨尺度融合；三个 MLKCM 对融合特征进行多尺度上下文编码；最后，仅在 P3 分支中加入局部细节恢复模块。处理后的 P3、P4 和 P5 特征直接送入未改动的 YOLOv8 Detect 头。

该设计有两项原则。第一，C4 是局部结构向高层语义过渡的阶段，适合实施受控的语义中继，而不应扰动 P2/P3 的原始纹理表征。第二，跨尺度融合后的检测特征同时需要上下文与边界信息；MLKCM 负责前者，P3DetailRestore 仅在最细尺度上补偿后者。网络的预测头、正负样本分配策略和总体损失保持不变，因此性能变化可以主要归因于特征表达和融合路径的调整。

## 3.2 C4 阶段的可靠性门控差分中继

### 3.2.1 设计动机

直接以复杂模块替换全部 Backbone C2f 容易改变浅层和高层特征的统计特性，并增加从零训练的优化难度。本文仅在 C4 阶段使用可靠性门控差分中继 C2f（RGDRC2fV3）。该模块将原始 C2f 的稳定局部表示作为基底，并通过一条轻量差分中继分支提供补充语义；空间可靠性门控再决定补充信息的注入位置和强度。

### 3.2.2 差分中继分支

对输入特征 $\mathbf{X}$，原始 C2f 分支和中继分支分别输出

$$
\mathbf{B}=\operatorname{C2f}(\mathbf{X}),\qquad
\mathbf{R}=\operatorname{DRC2fV3}(\mathbf{X}),
$$

其中 $\mathbf{B}$ 表示基底特征。中继分支首先将 $1\times1$ 投影结果分解为锚定状态 $\mathbf{X}_b$ 和初始中继状态 $\mathbf{u}_0$。第 $i$ 个 Bottleneck 对前一中继状态计算残差差分

$$
\Delta_i=\mathcal{B}_i(\mathbf{u}_{i-1})-\mathbf{u}_{i-1}.
$$

使用可学习的差分系数 $\alpha_i$ 与锚定系数 $\beta_i$ 更新状态：

$$
\mathbf{u}_i=\alpha_i\Delta_i+\beta_i\mathbf{u}_0,\qquad
\mathbf{s}_i=\mathbf{s}_{i-1}+\mathbf{u}_i,
$$

其中 $\mathbf{s}_0=\mathbf{u}_0$。经由两次中继更新后，模块将锚定状态、累计状态和最终状态拼接并映射为 $\mathbf{R}$：

$$
\mathbf{R}=\operatorname{Conv}\left([\mathbf{X}_b,\mathbf{s}_n,\mathbf{u}_n]\right).
$$

该递推使后续中继响应显式依赖前一阶段的增量，同时保留初始状态的锚定信息，避免语义传播完全脱离原始局部表征。

### 3.2.3 可靠性门控融合

并非每个空间位置都需要中继信息。本文根据基底特征预测单通道可靠性图：

$$
\mathbf{G}_r=\sigma\big(\phi_r(\mathbf{B})\big),
$$

其中 $\phi_r(\cdot)$ 为 $1\times1$ 卷积，$\sigma(\cdot)$ 为 Sigmoid。最终输出定义为

$$
\mathbf{Y}=\mathbf{B}+\lambda_r\,\mathbf{G}_r\odot(\mathbf{R}-\mathbf{B}),
$$

其中 $\lambda_r$ 是零初始化的可学习残差尺度，$\odot$ 表示逐元素乘法。训练开始时 $\lambda_r=0$，模块严格退化为原始 C2f 输出 $\mathbf{B}$；随着训练进行，网络再按数据需要学习中继残差的方向和幅度。这种渐进接入方式能够降低 Backbone 替换对从零训练稳定性的影响。

## 3.3 质量引导的渐进跨尺度特征融合

### 3.3.1 基础渐进融合

不同尺度的语义特征并不总是具有相同的可信度。例如，高层特征能提供类别上下文，但也可能将背景或遮挡信息传递至细粒度目标区域。为此，本文在 AFPN 的逐级融合框架上加入质量调制机制，构建 QG-AFPN。

对于目标尺度 $t\in\{3,4,5\}$，先由 P3 特征构造初始表征：

$$
\mathbf{Z}_{t}^{(0)}=\Pi_{t,3}(\mathbf{F}_3),
$$

其中 $\Pi_{t,s}$ 为 $1\times1$ 通道投影。来自尺度 $s\in\{4,5\}$ 的源特征经投影和最近邻尺度对齐后记为

$$
\mathbf{S}_{t,s}=\mathcal{A}_{t}\big(\Pi_{t,s}(\mathbf{F}_{s})\big).
$$

基础空间门控由当前目标特征与源特征共同预测：

$$
\mathbf{G}_{t,s}=\sigma\left(\phi_{t,s}\left([\mathbf{Z}_{t}^{(s-1)},\mathbf{S}_{t,s}]\right)\right).
$$

### 3.3.2 源质量与跨尺度一致性调制

为区分“是否需要注入”和“待注入信息是否可靠”两类问题，QG-AFPN 额外计算源质量图与跨尺度一致性图。源质量图写为

$$
\mathbf{Q}_{t,s}=\sigma\left(\psi_{t,s}(\mathbf{S}_{t,s})\right),
$$

一致性由归一化后目标特征和源特征的通道平均绝对差刻画：

$$
\mathbf{A}_{t,s}=\exp\left[-\frac{1}{C_t}\sum_{c=1}^{C_t}
\left|\operatorname{LN}(\mathbf{Z}_{t}^{(s-1)})-
\operatorname{LN}(\mathbf{S}_{t,s})\right|\right].
$$

据此构造质量调制系数

$$
\mathbf{M}_{t,s}=1+\tanh(\eta_{t,s})
\left(\mathbf{Q}_{t,s}\odot\mathbf{A}_{t,s}-1\right),
$$

并执行逐级融合：

$$
\mathbf{Z}_{t}^{(s)}=\mathcal{F}_{t,s}\left(
\mathbf{Z}_{t}^{(s-1)}+
\mathbf{G}_{t,s}\odot\mathbf{M}_{t,s}\odot\mathbf{S}_{t,s}
\right),
$$

其中 $\mathcal{F}_{t,s}$ 为 $3\times3$ 融合卷积。$\eta_{t,s}$ 零初始化，因此训练初期 $\mathbf{M}_{t,s}=1$，新增质量调制不会改变基础 AFPN 的融合行为。基础空间门控负责选择注入区域，质量和一致性则抑制低可靠或与目标尺度不一致的源信息。

## 3.4 多尺度大核上下文与 P3 细节恢复

### 3.4.1 多尺度大核卷积模块

跨尺度融合后，检测特征仍需同时建模局部边界、相邻区域和较大范围上下文。本文在 QG-AFPN 的 P3、P4 和 P5 输出后分别配置一个多尺度大核卷积模块（MLKCM）。以尺度 $k$ 的融合特征 $\widetilde{\mathbf{F}}_k$ 为例，首先进行通道投影和归一化：

$$
\mathbf{U}_k=\operatorname{LN}\left(\Pi_k(\widetilde{\mathbf{F}}_k)\right).
$$

经 $1\times1$ 卷积后，$\mathbf{U}_k$ 沿通道均衡划分为三组 $\{\mathbf{U}_{k}^{(j)}\}_{j=1}^{3}$。第 $j$ 个分支采用核大小 $k_j$ 和膨胀率 $d_j$ 的深度卷积，得到

$$
\mathbf{D}_{k}^{(j)}=\mathbf{U}_{k}^{(j)}\odot
\operatorname{PW}_{j}\left(
\operatorname{DW}_{d_j}\left(
\operatorname{DW}_{k_j}(\mathbf{U}_{k}^{(j)})
\right)\right).
$$

三条分支的 $(k_j,d_j)$ 分别取 $(3,1)$、$(5,2)$ 和 $(7,3)$。拼接后的特征通过逐点门控融合：

$$
\mathbf{V}_k=\operatorname{PC}\left(
\left[\mathbf{D}_{k}^{(1)},\mathbf{D}_{k}^{(2)},\mathbf{D}_{k}^{(3)}\right]
\odot\operatorname{PWG}(\mathbf{U}_k)
\right).
$$

随后，多轴块分别汇聚垂直方向最大池化响应和水平方向平均池化响应，并生成方向感知门控：

$$
\mathbf{H}_k=\sigma\left(
\varphi\left([operatorname{Pool}_{h}^{\max}(\mathbf{V}_k),
\operatorname{Pool}_{w}^{\operatorname{avg}}(\mathbf{V}_k)]\right)
\right),
$$

$$
\widehat{\mathbf{F}}_k=\mathbf{V}_k+\mathbf{H}_k\odot\mathbf{V}_k.
$$

该模块在三个检测尺度上以不同感受野抽取局部—全局信息，并通过多轴池化调整空间方向上的响应。

### 3.4.2 P3 局部细节恢复

较大感受野和跨尺度融合有助于稳定语义，却可能弱化 P3 的边界纹理。鉴于 P3 最接近小目标和细粒度结构，本文只在其 MLKCM 输出后加入 P3DetailRestore。对于 $\widehat{\mathbf{F}}_3$，局部细节分支和空间门控分别为

$$
\mathbf{D}_3=\phi_p\left(operatorname{DWConv}_{3\times3}(\widehat{\mathbf{F}}_3)\right),
\qquad
\mathbf{G}_p=\sigma\left(\phi_g(\widehat{\mathbf{F}}_3)\right).
$$

最终 P3 特征为

$$
\mathbf{P}_3=\widehat{\mathbf{F}}_3+
\lambda_p\mathbf{G}_p\odot\mathbf{D}_3,
$$

其中 $\lambda_p$ 同样采用零初始化。因而，模块在训练初期为恒等映射；训练完成后仅在网络判定为可靠的空间位置补偿局部响应。P4 和 P5 直接采用各自的 MLKCM 输出，以避免在低分辨率层过度强调局部算子。

## 3.5 检测头与损失函数

本文保留 YOLOv8 的 anchor-free 解耦 Detect 头。检测头接收 $\{\mathbf{P}_3,\widehat{\mathbf{F}}_4,\widehat{\mathbf{F}}_5\}$，并分别完成分类预测和边界框回归。训练过程继续采用任务对齐的正样本分配策略，联合优化 CIoU 边界框损失、二元交叉熵分类损失和分布焦点损失：

$$
\mathcal{L}=\lambda_{box}\mathcal{L}_{CIoU}+
\lambda_{cls}\mathcal{L}_{BCE}+
\lambda_{dfl}\mathcal{L}_{DFL}.
$$

本实验中，$\lambda_{box}=7.5$、$\lambda_{cls}=0.5$、$\lambda_{dfl}=1.5$。保留原始检测头与损失配置的目的在于控制变量：本文的结构改动集中于 Backbone 的 C4 阶段和检测特征融合路径，而不将检测头更新带来的变化混入模块有效性判断。

## 3.6 实现配置与可复现性

最终结构仅使用以下改动：C4 的 RGDRC2fV3、P3/P4/P5 的 QG-AFPN 与 MLKCM，以及 P3DetailRestore。Backbone 的 P2/P3 和 C5 C2f、原 PAN-FPN 前段及 Detect 头保持 YOLOv8s 默认实现。模型配置文件为 [yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail.yaml](D:/YH_Project/DSA-Det/DSA-Det/ultralytics-dsr/ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail.yaml)。

为与基线保持一致，所有训练均采用 5k_HRW 数据集、固定随机种子 42、从零初始化、输入分辨率 $640\times640$、batch size 为 8、SGD 优化器、初始学习率 0.01 和最多 300 个 epoch。结果章节应另外报告 Precision、Recall、mAP50、mAP50--95、参数量、计算量和推理时间；这些性能数据不作为本章方法有效性的先验依据。由于当前结论来自单一随机种子和单一数据划分，后续仍需通过多随机种子或交叉划分验证其稳定性。

## 3.7 本章小结

本文提出 QG-DRM-YOLO，以最小化且可控的结构改动增强 YOLOv8s 的多尺度检测能力：RGDRC2fV3 在 C4 阶段按空间可靠性选择性引入差分中继；QG-AFPN 用源质量和跨尺度一致性约束高层上下文的注入；MLKCM 在三尺度检测特征上建立多感受野上下文；P3DetailRestore 通过零初始化残差方式补偿细粒度边界信息。四类改动均采用残差或调制形式，旨在保持原始特征路径的稳定性，并使增强分支由数据驱动地逐步发挥作用。

---

**写作说明。** 本章仅借鉴参考文献的方法章节组织方式，即先给出整体网络，再分别阐释模块机理，最后说明检测头、损失与实现设置；不复用其文字、模型命名或实验结论。正式投稿时，应为 YOLOv8、AFPN/FPN、CIoU、DFL、任务对齐分配以及所采用的 MLKCM/PFOM 原始论文补充经核验的正式参考文献。
