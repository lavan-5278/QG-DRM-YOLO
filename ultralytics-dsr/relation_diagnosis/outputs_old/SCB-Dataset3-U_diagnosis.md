# SCB-Dataset3-U 关系可研究性诊断报告

> 依据《SCB-Dataset关系可研究性诊断实验方案》自动生成（模型无关，仅依赖 GT 标签）。

## 0. 数据质量检查（泄漏风险）

- train/val 框布局完全重复比例：0.952（199/209 张 val 图）

- 几何特征 Tree vs LR Macro-F1 差距：0.933 vs 0.306 → **存在记忆化风险，最终判定采用 LR 证据**

## 1. 基础审计

| Split | Images | Instances | Avg/img | Median | Max | Relation-valid | Imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 797 | 23608 | 29.62 | 30.0 | 41 | 1.000 | 518.78 |
| val | 209 | 6045 | 28.92 | 29.0 | 40 | 1.000 | 598.00 |

- 类别实例：{0: 18, 1: 9338, 2: 3524, 3: 8363, 4: 1143, 5: 1222}

## 2. 群体结构（Exp1）

- Mean Dominance = 0.478，Median = 0.467，Mean H_norm = 0.671，低熵比例 (H<0.4) = 0.006

## 3. KNN 同类率与置换检验（Exp3/9）

| K | R_obs | Perm Mean | ΔR | Z | p |
|---|---:|---:|---:|---:|---:|
| 1 | 0.3786 | 0.3317 | 0.0469 | 15.36 | 0.0010 |
| 3 | 0.3584 | 0.3317 | 0.0267 | 15.25 | 0.0010 |
| 5 | 0.3495 | 0.3318 | 0.0177 | 13.08 | 0.0010 |
| 8 | 0.3451 | 0.3318 | 0.0134 | 13.45 | 0.0010 |

## 4. 距离效应（Exp4）

| Bin | n_pairs | P_same_obs | Null | Net |
|---|---:|---:|---:|---:|
| 0 | 139208 | 0.3589 | 0.3302 | 0.0287 |
| 1 | 139212 | 0.3427 | 0.3307 | 0.0120 |
| 2 | 139206 | 0.3136 | 0.3290 | -0.0154 |
| 3 | 139214 | 0.3121 | 0.3305 | -0.0184 |
| 4 | 139212 | 0.3247 | 0.3316 | -0.0069 |
| 0 | 34800 | 0.3586 | 0.3305 | 0.0281 |
| 1 | 34800 | 0.3410 | 0.3311 | 0.0099 |
| 2 | 34800 | 0.3141 | 0.3293 | -0.0152 |
| 3 | 34800 | 0.3096 | 0.3302 | -0.0206 |
| 4 | 34800 | 0.3295 | 0.3316 | -0.0021 |

## 5. 同配/异配（Exp5/12）

| Class | T_cc | Lift(diag) | H_c | E_c | Type |
|---|---:|---:|---:|---:|---:|
| hand-raising | 0.000 | nan | -0.001 | 0.097 | heterophily |
| reading | 0.221 | 0.560 | -0.174 | 0.145 | heterophily |
| writing | 0.202 | 1.338 | 0.051 | 0.096 | heterophily |
| using-phone | 0.233 | 0.661 | -0.120 | 0.105 | heterophily |
| head-down | 0.078 | 1.623 | 0.030 | 0.062 | heterophily |
| lying-on-desk | 0.103 | 1.973 | 0.051 | 0.071 | heterophily |

## 6. 预测性分析（Exp7/8/11）

| Model | LR Macro-F1 | Tree Macro-F1 |
|---|---:|---:|
| B0_majority | 0.0945 | 0.0945 |
| B1_prior_random | 0.1633 | 0.1633 |
| B2_geometry | 0.3058 | 0.9328 |
| B3_global | 0.1878 | 0.6940 |
| B4_neighbor | 0.2612 | 0.7889 |
| B5_geom_neighbor | 0.3878 | 0.9529 |
| B6_geom_neighbor_global | 0.4162 | 0.9521 |
| Oracle_full | 0.3952 | 0.8980 |

## 7. 核心证据与判定

- Evidence A（空间关联 ΔR3≥0.05 且 p<0.01）：**False**（ΔR3=0.0267, p=0.0010）

- Evidence B（距离效应）：**True**（最近箱净增益 0.0286，最远箱 -0.0059）

- Evidence C（RPG_local≥0.05）：**True**（RPG_local=0.0820）

- Evidence D（Oracle Gain≥0.08）：**True**（OG=0.0894）

- RPG_global=-0.1180，RPG_all=0.1104


**最终判定：继续 DSR（CONTINUE；判定依据：lr (tree geometry-only gap suggests memorization)）**


---

附表：Neighbor-only 详细指标见 `tables/08_neighbor_prediction.csv`，类别级 RPG 见 `tables/12_classwise_rpg.csv`，Bootstrap CI 见 `tables/13_final_decision.csv`。