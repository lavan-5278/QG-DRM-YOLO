# ultralytics-dsr

DSR-Det（Differential Social Relation Detection）的 **Ultralytics 原生集成骨架（Phase 0/1）**，
依据《基于 Ultralytics 原生扩展的 DSR-Det 单帧课堂行为检测技术方案》v1.1 搭建。

> 📖 所有脚本、库函数与测试命令的完整用法见 [SCRIPTS.md](SCRIPTS.md)。

## DSR-Det v2（当前实现）

按《DSR-Det_v2_优化方案》重构为 **选择性差分社会关系精炼（Selective Differential
Social Refinement）**：

- **只改分类、不改 bbox**：YOLO 负责定位，DSR 仅做行为分类纠错（bbox 残差已删除）；
- **不确定性选择**：`u_i = 1 - (p_top1 - p_top2)`，仅 `u > uncertainty_threshold`
  的模糊候选进入关系分支，高置信候选保持 YOLO 结果；
- **仅 KNN 关系图**：删除 Global/Region 关系（避免多数类偏置），保留尺度感知 KNN；
- **可靠邻居加权**：`alpha_ij = softmax(s_ij + log(1 - u_j + eps))`，优先参考高置信邻居；
- **零初始化小残差**：`F_i' = F_i + rho * R_i`，`rho=0` 起步，训练中自动学习；
- **保真损失**：`L_preserve = KL(stop-grad(P_YOLO) || P_DSR)` 约束高置信样本不被破坏；
- **删除 GICR**；损失为 `L = L_YOLO + λr·L_relation + λp·L_preserve`（5 组件）；
- 新增 **Fix / Harm / NetGain** 归因指标（`dsr_scenario.json → base_vs_refined`）。

## 核心原则（v1.1 入口约束）

- 所有训练、验证、预测只通过官方入口 `model.train()` / `model.val()` / `model.predict()` 完成；
- 项目内不存在独立的 `train.py` / `val.py` / `evaluate_dsr.py`；
- 困难场景分层、候选召回、oracle 上界等扩展统计由 `DSRDetectionValidator` 在 `model.val()` 内输出（Phase 5 填充）；
- 官方文件最小补丁仅两处：`ultralytics/nn/modules/__init__.py`（导出）与
  `ultralytics/nn/tasks.py`（parse_model 注册），diff 保存在 `patches/`。

## 环境（Phase 0 基线锁定）

本项目自带固定版本 vendored 包（`ultralytics/`，8.3.247，直接拷贝自已验证环境的
site-packages）。推荐使用已验证环境
（Python 3.12.12、CUDA 12.6、torch 2.9.1+cu126），依赖版本见
[requirements-lock.txt](requirements-lock.txt)。

```powershell
# 在项目根目录运行，确保 import ultralytics 解析到本项目的 vendored 树
cd I:\DSA-Det\ultralytics-dsr
H:\anaconda3\envs\oulad-dp-cu12\python.exe -m pytest
```

## Phase 1 验收对应的骨架能力

| v1.1 要求 | 位置 |
|---|---|
| `DSRDetect` 注册（parse_model + 导出） | `ultralytics/nn/modules/dsr.py`、`tasks.py`、`modules/__init__.py` |
| `yolo11n-dsr.yaml` / `yolo11s-dsr.yaml` | `ultralytics/cfg/models/11/` |
| `dsr.enabled=false` 输出与官方 YOLO 完全一致 | `tests/test_dsr_registration.py::test_enabled_false_output_identical_to_official` |
| 零初始化时精炼输出等于基础预测 | `tests/test_dsr_registration.py::test_enabled_zero_init_output_equals_base` |
| 官方权重加载不报错 | `tests/test_dsr_registration.py::test_load_official_yolo11_weights` |
| `model.info()` / FLOPs / train/val/predict 冒烟 | `tests/test_dsr_train_step.py`（train 冒烟为 `slow` 标记） |
| 候选匹配 v1.1（重复正样本按忽略） | `tests/test_dsr_loss.py::test_duplicate_positive_is_ignored_not_background` |
| 训练/推理候选一致性（同一 `DSRDetect.forward` 代码路径） | `ultralytics/nn/modules/dsr.py` |
| 关系算子 R0–R7 配置切换（`dsr.relation_type`） | `tests/test_dsr_relations.py` |
| MDRR 内部消融 M1–M5（`dsr.relation_variant`） | `tests/test_dsr_relations.py::test_signed_variants_run` |
| EMA 候选源（`dsr.candidate_source: ema`，Trainer 自动更新） | `tests/test_dsr_train_step.py::test_candidate_source_ema_step` |
| 候选集稳定性探针（`candidate_stability.json`） | `ultralytics/models/yolo/dsr/train.py` |
| Stage B1/B2 两阶段调度（基础检测器冻结，B2 降 LR） | `ultralytics/models/yolo/dsr/train.py::_on_train_epoch_start_stage` |
| 门控消融 `gate_mode`（fixed/scalar/conf/feature/uncertainty，18.5） | `tests/test_dsr_gate_uncertainty.py` |
| 不确定度 margin/entropy + u 分布审计（7.1/11.1，`uncertainty_audit.json`） | `tests/test_dsr_gate_uncertainty.py` |
| 困难场景分层 / 候选召回 / oracle（`dsr.validator: true`，`dsr_scenario.json`） | `ultralytics/models/yolo/dsr/val.py`、`tests/test_dsr_validator.py` |
| 效率测量（20.2：分项延迟 / FLOPs / 峰值显存 / FP32-FP16） | `ultralytics/models/yolo/dsr/bench.py`、`tests/test_bench_summarize.py` |
| 多种子汇总 + 图像级配对 bootstrap（20.3/20.4） | `scripts/summarize_runs.py`、`tests/test_bench_summarize.py` |
| 关系可视化（19.5：尺度权重/门控热图，`relation_stats.json`） | `ultralytics/models/yolo/dsr/visualize.py`、`tests/test_dsr_visualize_ema.py` |
| GICR 验收（off==Phase3、稀有类无 NaN、同类对比例） | `tests/test_dsr_gate_uncertainty.py` |
| base-vs-refined 错误归因（19.5） | `ultralytics/models/yolo/dsr/val.py`、`tests/test_dsr_validator.py` |

## 目录结构

```text
ultralytics-dsr/
├── ultralytics/                      # vendored 8.3.247 + 最小补丁 + DSR 扩展
│   ├── nn/modules/dsr.py             # DSRDetect / MDRR / RelationRefinementHead
│   ├── nn/modules/__init__.py        # [PATCH] 导出 DSRDetect/MDRR
│   ├── nn/tasks.py                   # [PATCH] parse_model 注册 DSRDetect
│   ├── models/yolo/dsr/              # DSRDetectionModel / Loss / Trainer / Validator / Predictor
│   └── cfg/
│       ├── models/11/yolo11{n,s}-dsr.yaml
│       ├── datasets/SCB-Dataset3-{S,U}.yaml
│       └── experiments/              # baseline / continue / dsr_full / ablations
├── patches/                          # 官方文件 diff（静态文本，无需 git）
├── tests/                            # 单元测试 + 冒烟（slow 标记）
├── scripts/                          # 实验矩阵运行器 + runs 产物汇总（训练/评估仍走官方入口）
├── pyproject.toml
├── requirements-lock.txt
└── README.md
```

## 使用方式

```python
from ultralytics import YOLO
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer

# Phase 1 冒烟：构建 + 推理
model = YOLO("ultralytics/cfg/models/11/yolo11n-dsr.yaml")
model.load("../../DSA-Det/yolo11n.pt")  # 官方权重，仅基础层生效
results = model.predict("tests/data/smoke/images", imgsz=640)

# Stage A 基线（Phase 0，官方训练入口）
model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
model.train(data="ultralytics/cfg/datasets/SCB-Dataset3-S.yaml",
            cfg="ultralytics/cfg/experiments/scb_yolo11n_baseline.yaml")

# Stage B：DSR 精炼训练（B1+B2，v1.1 14.1）
model = YOLO("ultralytics/cfg/models/11/yolo11n-dsr.yaml")
model.load("runs/scb/yolo11n_baseline/weights/best.pt")
model.train(trainer=DSRDetectionTrainer,
            cfg="ultralytics/cfg/experiments/scb_dsr_full.yaml")
```

## 测试

```powershell
H:\anaconda3\envs\oulad-dp-cu12\python.exe -m pytest            # 快速单元测试（默认排除 slow）
H:\anaconda3\envs\oulad-dp-cu12\python.exe -m pytest -m slow    # 含 model.train()/model.val() 冒烟
```

## 效率测量（v1.1 20.2）

```python
from ultralytics import YOLO
from ultralytics.models.yolo.dsr.bench import profile_dsr, peak_memory_train

model = YOLO("ultralytics/cfg/models/11/yolo11n-dsr.yaml")
report = profile_dsr(model.model, imgsz=640, device="0", warmup=10, repeats=50,
                     fixed_k=100, compare_half=True)   # FP32 + FP16 对比
peak_memory_train(model.model, imgsz=640, batch=8, device="0")
```

报告包含：参数量、固定 `relation_topk` 下的 GFLOPs（thop 实测，动态算子口径已注明）、
分项延迟（base detect / candidate / RoI Align / graphs / aggregate / gate /
refine head / write-back / NMS）、单图总延迟与 FPS、推理/训练峰值显存。

## 统计检验（v1.1 20.3 / 20.4）

三种子（0 / 42 / 3407）通过同一入口运行，目录约定 `runs/<exp>/seed{0,42,3407}/`
（或直接放 `runs/<exp>/`）：

```powershell
python scripts/summarize_runs.py runs                    # 每种子的 μ±σ 汇总
python scripts/summarize_runs.py runs --compare yolo11n_dsr yolo11n_continue
```

`--compare` 使用 `DSRDetectionValidator` 写出的 `per_image_metrics.json` 做
图像级配对 bootstrap 95% CI（默认 2000 次），报告均值差、CI 与显著性（CI 是否含 0）。

## 阶段状态

- **Phase 0**：基线锁定（环境/依赖/vendored 包已固定；SCB 数据路径已配置；正式基线训练待跑）；
- **Phase 1**：注册骨架完成（DSRDetect 注册、YAML、输出契约、权重加载、冒烟测试）；
- **Phase 2**：个体精炼基线（R0）与关系算子 R0–R7、M1–M5 内部消融已可通过配置切换；
- **Phase 3**：MDRR 核心关系模块（三尺度图 + 差分聚合 + 不确定性门控消融）、EMA 候选源、
  候选稳定性探针（`candidate_stability.json`）与 Stage B1/B2 两阶段调度已接入 Trainer；
- **Phase 4**：GICR 完整消融（含 warm-up、关闭=Phase3 验收、稀有类别无 NaN、
  有效同类对比例 `gicr_pair_stats.json`）与 margin/entropy 不确定度已实现；
- **Phase 5（评估主体）**：`DSRDetectionValidator` 已实现遮挡/密度/尺寸分层、
  逐层 AP/Recall/漏检/误分类与图像级 bootstrap CI、候选召回 conf 扫描、oracle 上界；
  关系有效性（尺度权重/门控统计）与 base-vs-refined 错误归因已并入报告。
- **Phase 5**：困难场景分层 / 候选召回 / oracle 上界由 `DSRDetectionValidator` 填充；
- **Phase 6**：论文与开源整理。
