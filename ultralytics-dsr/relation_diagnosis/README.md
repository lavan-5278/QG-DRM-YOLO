# SCB-Dataset 关系可研究性诊断

依据《SCB-Dataset关系可研究性诊断实验方案》实现的 **模型无关（Model-free）数据关系可行性分析**：
先验证"数据里是否存在可利用的关系信号"，再决定是否值得继续 DSR 类关系建模。

## 运行

```powershell
python relation_diagnosis/run_diagnosis.py
```

可选参数：

- `--datasets scb3 scb3s scb3u`：只跑指定数据集（默认全部）；
- `--perm-b 1000`：KNN 置换检验次数（默认 1000）；
- `--boot-b 1000`：图像级 bootstrap 次数（默认 1000）；
- `--configs / --outputs`：配置与输出目录。

## 目录结构

```text
relation_diagnosis/
├── configs/            # scb3 / scb3s / scb3u 数据集配置
├── src/
│   ├── parse_yolo_labels.py    # 标签解析 -> 统一实例表
│   ├── descriptive_stats.py    # Exp0 审计 + Exp1 熵 + Exp2 共现 + 泄漏检查
│   ├── relation_stats.py       # Exp3 KNN+置换 / Exp4 距离曲线 / Exp5 转移矩阵
│   │                           # Exp6 同配系数 / Exp10 互信息 / Exp12 同配-异配
│   ├── predictive.py           # Exp7 Neighbor-only / Exp8 Oracle / Exp11 类别级 RPG
│   ├── report.py               # 表格 / 图表 / Bootstrap CI / 最终决策
│   └── __init__.py
├── run_diagnosis.py            # 编排入口
└── outputs/
    ├── <dataset>/tables/       # 01-13 结果表
    ├── <dataset>/figures/      # fig1-fig9
    ├── <dataset>_diagnosis.md  # 各数据集诊断报告（含判定）
    ├── tables/13b_dataset_comparison.csv   # 三数据集总表
    └── figures/fig10_dataset_comparison.png
```

## 核心口径

- **关系有效比例**：每图 ≥2 目标的比例；
- **ΔR_K**：KNN 同类率减去图内标签置换基线（K=1/3/5/8，B=1000，p 值为单侧置换 p）；
- **距离效应**：分位数距离分箱（0-20% … 80-100%）的同类概率净增益，要求近箱明显高于远箱；
- **Neighbor-only 预测**：只用邻居 GT 标签与几何预测中心行为（Logistic Regression + HistGradientBoosting，按 image 划分，Macro-F1 为主指标）；
- **GT Social Oracle**：已知其他所有学生的 GT 框与类别时的预测上限；
- **Bootstrap**：全部按 image 重抽样（B=1000），报告 ΔR / RPG / OG 的 95% CI。

## 最终判定

| 证据 | 条件 |
|---|---|
| A 空间关联 | ΔR3 ≥ 0.05 且 p < 0.01 |
| B 距离效应 | 近箱净增益明显高于远箱且趋势递减 |
| C Neighbor-only | RPG_local ≥ 0.05 |
| D Oracle | OG ≥ 0.08 |

- 满足 ≥3 项：**CONTINUE**（继续 DSR）；
- 满足 1-2 项：**NARROW**（选择性关系建模）；
- ΔR3<0.02 且 RPG<0.03 且 OG<0.03 且置换不显著：**STOP**。

## 已知注意事项

1. **SCB-Dataset3-U 存在严重 train/val 泄漏**：95.2%（199/209）的 val 图与 train 有完全相同的框布局（同一视频的近重复帧）。树模型仅用几何特征即达 0.93 Macro-F1（LR 仅 0.31），属于记忆化。诊断工具会自动检测并在该情况下改用 LR 证据做最终判定；U 版的结论需按视频/场景分组重新划分后才可采信。
2. 3 类版本（SCB-Dataset3 / S）无布局重复，划分干净，结论可信。
3. KNN/置换/距离曲线等描述性统计不受 train/val 泄漏影响（对全部标签直接分析，置换基线已控制类别频数）。
