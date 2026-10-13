# ultralytics-dsr 脚本使用说明

本文档介绍本项目所有脚本、库函数与测试命令的用途和用法。

> 项目核心原则(v1.1):训练、验证、预测只通过官方入口 `model.train()` / `model.val()` / `model.predict()` 完成,项目内不存在独立的 `train.py` / `val.py` / `evaluate_dsr.py`。`scripts/` 下的脚本只负责**读取和分析训练产物**,不承担训练与评估。

---

## 0. 环境准备

推荐使用已验证的 conda 环境(本机为 `lfm-yolo-mamba-stage14`):

```powershell
$env:PY = "D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe"
& $env:PY -c "import ultralytics; print(ultralytics.__version__)"   # 8.3.247
```

所有命令都在**项目根目录**下运行:

```powershell
cd D:\YH_Project\DSA-Det\DSA-Det\ultralytics-dsr
```

`conftest.py` 会保证 `import ultralytics` 解析到本项目的 vendored 源码树(固定 8.3.247),而不是 site-packages 里的副本。

### 脚本/工具总览

| 工具 | 类型 | 作用 |
|---|---|---|
| `scripts/run_experiments.py` | CLI | 实验矩阵运行器:批量编排对比/消融实验(训练 + 统一验证) |
| `scripts/dataset_stats.py` | CLI | 数据集统计(类别实例数、每图目标分布、面积比、无效标注) |
| `scripts/summarize_runs.py` | CLI | 多 seed 训练结果汇总(μ±σ)与两组实验的配对 bootstrap 显著性检验 |
| `scripts/scenario_analysis.py` | CLI | 打印 `dsr_scenario.json` 报告(分层/候选召回/oracle) |
| `ultralytics/models/yolo/dsr/bench.py` | 库函数 | DSR 效率测量:分项延迟、FLOPs、FP32/FP16、峰值显存 |
| `ultralytics/models/yolo/dsr/visualize.py` | 库函数 | 关系可视化:尺度融合权重、门控曲线 |
| `tests/`(pytest) | 测试 | 单元测试 + 冒烟测试(默认排除 slow) |

---

## 1. 标准训练 / 验证 / 预测流程

多数分析脚本读取的是训练/验证产生的产物,因此先说明主流程。

### 1.1 Stage A:官方 YOLO11n 基线

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
model.train(cfg="ultralytics/cfg/experiments/scb_yolo11n_baseline.yaml")
```

### 1.2 Stage B:DSR 精炼训练(B1 + B2)

```python
from ultralytics import YOLO
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer

model = YOLO("ultralytics/cfg/models/11/yolo11n-dsr.yaml")
model.load("runs/scb/yolo11n_baseline/weights/best.pt")   # 官方/基线权重,仅基础层
model.train(trainer=DSRDetectionTrainer,
            cfg="ultralytics/cfg/experiments/scb_dsr_full.yaml")
```

### 1.3 继续训练对照组(与 DSR 同协议、无关系分支)

```python
model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")
model.load("runs/scb/yolo11n_baseline/weights/best.pt")
model.train(cfg="ultralytics/cfg/experiments/scb_yolo11n_continue.yaml")
```

### 1.4 多 seed 约定

论文实验建议 3 个 seed(0 / 42 / 3407),目录约定:

```text
runs/<exp>/seed0/
runs/<exp>/seed42/
runs/<exp>/seed3407/
```

运行方式(以 DSR 为例):

```python
model.train(trainer=DSRDetectionTrainer,
            cfg="ultralytics/cfg/experiments/scb_dsr_full.yaml",
            seed=0, project="runs/scb/yolo11n_dsr", name="seed0")
```

不按 seed 分目录时,直接放在 `runs/<exp>/` 下也可以,汇总脚本会按 `default` 处理。

### 1.5 验证

训练过程中的验证:若希望输出 DSR 场景报告(`dsr_scenario.json`、`per_image_metrics.json`),需在模型 yaml 中设 `dsr.validator: true`。

独立验证(推荐所有实验统一使用,保证 `--compare` 可用):

```python
from ultralytics import YOLO
from ultralytics.models.yolo.dsr.val import DSRDetectionValidator

m = YOLO("runs/scb/yolo11n_dsr/seed0/weights/best.pt")
m.val(data="ultralytics/cfg/datasets/SCB-Dataset3-S.yaml",
      validator=DSRDetectionValidator, imgsz=640, batch=16, device=0)
```

> `DSRDetectionValidator` 对**没有 DSR 头的普通模型(纯 YOLO、RT-DETR 等)同样可用**,会降级为通用统计并照常写出 `per_image_metrics.json`。对比实验要统一用它评测,配对 bootstrap 才能计算。

### 1.6 预测

```python
model = YOLO("runs/scb/yolo11n_dsr/seed0/weights/best.pt")
results = model.predict("path/to/images", imgsz=640, device=0)
```

---

## 2. `scripts/dataset_stats.py` —— 数据集统计

读取 YOLO 格式数据集 yaml 及其标注,输出:每个 split 的图片数、实例数、每类实例数、每图目标分布(mean/median/max/p90)、空图数、无效标注数、归一化面积比分布(p25/median/p75/mean)。

### 用法

```powershell
python scripts/dataset_stats.py ultralytics/cfg/datasets/SCB-Dataset3-S.yaml
python scripts/dataset_stats.py ultralytics/cfg/datasets/SCB-Dataset3-S.yaml --save-json runs/dataset_stats/SCB-Dataset3-S.json
```

### 参数

| 参数 | 说明 |
|---|---|
| `data_yaml`(位置参数) | 数据集 yaml 路径 |
| `--save-json PATH` | 可选,把统计结果保存为 JSON |

### 输出示例

```text
dataset: ultralytics\cfg\datasets\SCB-Dataset3-S.yaml  (3 classes: hand_raising, reading, writing)
  [train] images=5193 instances=34522 per_image(mean/median/max/p90)=6.65/4.0/59/17.0 empty=0 invalid=2
      per-class: {'hand_raising': 10536, 'reading': 17539, 'writing': 6447}
  [val] images=1671 instances=12846 per_image(mean/median/max/p90)=7.69/4.0/52/21.0 empty=0 invalid=2
```

---

## 3. `scripts/summarize_runs.py` —— 多 seed 汇总与对比

读取 `results.csv`(训练最后一行)与 `per_image_metrics.json`(由 `DSRDetectionValidator` 写出),完成两类任务。

### 3.1 多 seed 汇总(μ ± σ)

```powershell
python scripts/summarize_runs.py runs
python scripts/summarize_runs.py runs/scb/yolo11n_dsr
```

输出每组的 seed 数、`metrics/mAP50-95(B)` 的均值与标准差、是否有 per-image 指标。

### 3.2 两组配对 bootstrap 显著性检验

```powershell
python scripts/summarize_runs.py runs --compare yolo11n_dsr yolo11n_continue
python scripts/summarize_runs.py runs --compare yolo11n_dsr yolo11n_continue --iters 2000 --seed 0
```

对两组实验按图像做**配对**差异(实验 A 减实验 B),bootstrap 95% CI;若 CI 不含 0 则判为显著。输出 `ap50_95` 与 `recall05` 两个指标的均值差、CI、显著性和参与图像数。

### 参数

| 参数 | 说明 |
|---|---|
| `runs_dir`(位置参数,默认 `runs`) | 扫描目录,按 `runs/<exp>/seedX/` 或 `runs/<exp>/` 识别实验 |
| `--compare EXP_A EXP_B` | 指定两个实验名做配对 bootstrap |
| `--iters N` | bootstrap 次数,默认 2000 |
| `--seed N` | 随机种子,默认 0 |

### 前提

`--compare` 要求两个实验目录下都有 `per_image_metrics.json`(由 `DSRDetectionValidator` 写出)。缺失时会在对应指标处报 `per_image_metrics.json missing`。对照组必须用 1.5 节的方式评测。

---

## 4. `scripts/run_experiments.py` —— 实验矩阵运行器

批量编排模型对比与消融实验:读取矩阵 YAML,为每个"实验 × seed"生成独立的
模型 yaml(`dsr:` 覆盖)与训练 cfg,通过官方 `model.train()` / `model.val()`
执行,目录布局与 `summarize_runs.py` 直接兼容。

### 用法

```powershell
# 预览计划(不实际运行)
python scripts/run_experiments.py --matrix ultralytics/cfg/experiments/matrix.yaml --dry-run

# 运行整个矩阵(默认 3 seed × 全部实验;DSR/消融需要先有 Stage A 基线权重)
python scripts/run_experiments.py

# 只跑部分实验 / 单个 seed / 指定设备
python scripts/run_experiments.py --only dsr_full continue ablate_r0_none --seeds 0
python scripts/run_experiments.py --only dsr_full --device 1

# 多 GPU 并发(每个实验独立子进程)
python scripts/run_experiments.py --parallel 2

# 强制重跑已完成实验
python scripts/run_experiments.py --force
```

### 参数

| 参数 | 说明 |
|---|---|
| `--matrix PATH` | 矩阵 YAML,默认 `ultralytics/cfg/experiments/matrix.yaml` |
| `--dry-run` | 只打印计划(标注 RUN / SKIP / BLOCK) |
| `--only NAME...` | 只跑指定实验(支持逗号分隔) |
| `--seeds N...` | 覆盖种子列表 |
| `--device X` | 覆盖设备(如 `0` 或 `cpu`) |
| `--parallel N` | 并发 worker 数(多 GPU 时用,默认 1) |
| `--force` | 重跑已完成/被阻断的实验 |
| `--no-val` | 跳过验证阶段(调试用) |
| `--run-plan PATH` | 内部 worker 模式,一般不要直接使用 |

### 矩阵文件与目录约定

- 默认矩阵 [ultralytics/cfg/experiments/matrix.yaml](ultralytics/cfg/experiments/matrix.yaml):
  模型对比三组(`baseline` / `continue` / `dsr_full`)+ 全套消融(关系算子 R0-R7、
  MDRR 变体 M1-M5、v2 选择机制(uncertainty 阈值/可靠邻居/保真损失)、门控模式、
  不确定性类型、EMA 候选、候选参数)。
- 目录约定:`runs/matrix/<experiment>/seed{0,42,3407}/`,与 `summarize_runs.py` 兼容:

```powershell
python scripts/summarize_runs.py runs/matrix
python scripts/summarize_runs.py runs/matrix --compare dsr_full continue
```

- 每个实验条目字段:`name`(唯一名)、`cfg`(实验配置模板)、`model`(基础模型 yaml)、
  `trainer`(`dsr` 用 DSRDetectionTrainer,省略用官方)、`load_weights`(`baseline` =
  `global.baseline_weights`,或显式路径)、`dsr`(配置覆盖,即消融开关)、
  `train`(可选训练参数覆盖,如 `epochs`)。

### 运行器行为

- DSR 实验自动附加 `dsr.validator: true`,训练过程与最终验证都会写场景报告;
- 验证统一使用 `DSRDetectionValidator`,保证 `--compare` 有 per-image 数据;
- Stage A 基线权重缺失的实验会标记 **BLOCK** 并跳过(先跑 `baseline` 再跑 DSR/消融);
- 已完成实验自动跳过(可断点续跑),单个实验失败不中断矩阵;
- `runs/matrix/manifest.json` 记录所有实验的状态(含错误堆栈)。

### 快速验证(小数据 1 epoch)

```powershell
python scripts/run_experiments.py --matrix tests/data/matrix_smoke.yaml
```

---

## 5. `scripts/scenario_analysis.py` —— 打印场景报告

递归扫描目录下所有 `dsr_scenario.json`,打印:候选召回扫描、各分层(遮挡/密度/尺寸)的 AP50-95 与 Recall、oracle 上界。

### 用法

```powershell
python scripts/scenario_analysis.py runs
python scripts/scenario_analysis.py runs/scb/yolo11n_dsr/seed0
```

该脚本**只打印** `DSRDetectionValidator` 已生成的结果,不含任何评估逻辑。

---

## 6. `bench.py` —— DSR 效率测量(库函数)

位置:`ultralytics/models/yolo/dsr/bench.py`。提供两个函数,均返回报告字典并打印日志。

### 5.1 `profile_dsr` —— 推理分项延迟 / FLOPs / 显存

```python
from ultralytics import YOLO
from ultralytics.models.yolo.dsr.bench import profile_dsr

model = YOLO("ultralytics/cfg/models/11/yolo11n-dsr.yaml")
report = profile_dsr(model.model, imgsz=640, device="0",
                     warmup=10, repeats=50, fixed_k=100,
                     compare_half=True)          # FP32 + FP16 对比
```

参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | — | 传入 `YOLO(...).model`(即 `DetectionModel`),内部会 unwrap |
| `imgsz` | 640 | 输入边长 |
| `batch` | 1 | 批量大小(计时会折算到单图) |
| `device` | auto | `"0"` 或 `"cpu"` |
| `warmup` / `repeats` | 10 / 50 | 预热与重复次数 |
| `half` | False | 是否跑 FP16 |
| `fixed_k` | 100 | 固定候选 top-K(即 `relation_topk`) |
| `include_nms` | True | 是否附加 NMS 计时 |
| `compare_half` | False | 为 True 时同时输出 FP32 与 FP16 两组结果 |

报告包含:设备、参数量、FP32 GFLOPs、峰值推理显存,以及每个组件的 mean/median 毫秒数(`base_detect` / `candidate` / `mdrr` / `refine_head` / `writeback` / `nms`)和 FPS。

> 注意:`gflops_fp32` 由 thop 统计,**MDRR 的动态算子(RoIAlign、K×K×C 配对展开等)会被按静态算子计入**,数值明显高于官方 YOLO11n(约 96.8 vs 6.5 GFLOPs)。写论文引用时需按报告中的 `gflops_note` 说明口径。

### 5.2 `peak_memory_train` —— 训练峰值显存

```python
from ultralytics.models.yolo.dsr.bench import peak_memory_train

peak_memory_train(model.model, imgsz=640, batch=8, device="0")
```

在合成 batch 上做一次前向 + 反向,报告 `peak_training_memory_mb`。

---

## 7. `visualize.py` —— 关系可视化(库函数)

位置:`ultralytics/models/yolo/dsr/visualize.py`。通过官方 `model.predict()` 逐张推理,保存每张图的尺度融合权重与门控曲线。

```python
from ultralytics import YOLO
from ultralytics.models.yolo.dsr.visualize import visualize_relation_stats

model = YOLO("runs/scb/yolo11n_dsr/seed0/weights/best.pt")
rows = visualize_relation_stats(
    model,
    "tests/data/smoke/images",          # 图片目录或单张图片
    save_dir="runs/relation_viz",
    imgsz=640,
    max_images=None,
    device=0,
)
```

产物(保存到 `save_dir`):

- `relation_stats.json`:每张图的候选数、尺度权重(`g`/`r`/`n`)、门控均值、不确定性均值;
- `relation_<i>.png`:尺度权重柱状图 + 排序后的门控曲线。

返回值 `rows` 与 JSON 内容一致(每张图一个 dict)。

---

## 8. 测试

### 快速单元测试(默认排除 slow)

```powershell
python -m pytest
```

### 只跑 slow 冒烟测试(真实 `model.train()` / `model.val()` / `resume`)

```powershell
python -m pytest -m slow
```

### 指定文件 / 用例

```powershell
python -m pytest tests/test_dsr_registration.py -q
python -m pytest tests/test_dsr_loss.py::test_duplicate_positive_is_ignored_not_background -q
```

说明:

- slow 冒烟包括 `test_trainer_smoke_one_epoch`(在小数据集上跑 1 epoch 训练 + 验证)与 `test_official_resume_link`(子进程训练后中断再 `resume=True` 续训)。后者会启动 `tests/resume_worker.py`,该文件是**测试基础设施**,不是给用户直接使用的脚本。
- `tests/test_run_experiments.py` 覆盖矩阵运行器:快速用例校验计划生成/过滤/阻断逻辑,`test_runner_smoke_matrix_end_to_end`(slow)在小数据上跑完整矩阵。
- `tests/data/smoke.yaml` 未写 `path`,ultralytics 自动以 yaml 所在目录为数据集根,因此测试可在任意机器运行。
- 当前机器测试状态:快速测试 64 通过、0 失败(3 个 slow 未跑)。

---

## 9. 训练/验证产物文件一览

以下文件由官方入口自动生成,是上述脚本的数据来源:

| 文件 | 生成者 | 内容 |
|---|---|---|
| `results.csv` | Trainer | 逐 epoch 训练/验证指标(含 5 项损失:box/cls/dfl/relation/preserve) |
| `weights/best.pt`、`weights/last.pt` | Trainer | 最优/最新权重 |
| `dsr_scenario.json` | `DSRDetectionValidator` | 分层 AP/Recall、候选召回、oracle、关系统计、base-vs-refined 归因 |
| `per_image_metrics.json` | `DSRDetectionValidator` | 每图 AP50-95 / Recall,供配对 bootstrap |
| `uncertainty_audit.json` | Trainer(epoch 结束) | u 分布审计(u_mean/u_p50/u_p90/ratio_u_gt_0.9/conf_mean) |
| `candidate_stability.json` | Trainer(epoch 结束) | 候选集稳定性(相对参考 epoch 的平均 max-IoU) |

---

## 10. 常见问题与注意事项

1. **数据集路径**:`ultralytics/cfg/datasets/SCB-Dataset3-{S,U}.yaml` 与 `SCB-Dataset3.yaml` 的 `path` 指向 `D:/YH_Project/DSA-Det/DSA-Det/data/...`。换机器时需同步修改。
2. **门控消融的坑**:做 `gate_mode` 消融时必须设 `use_uncertainty_gate: false`,否则会被强制覆盖为 `uncertainty`(yaml 注释里已标明)。
3. **对比实验的评测一致性**:DSR 与对照组必须用同一个 `DSRDetectionValidator` 评测,`--compare` 的配对 bootstrap 才有意义。
4. **GFLOPs 口径**:`model.info()` / `profile_dsr` 的 FLOPs 包含 MDRR 动态算子,不能直接与官方 YOLO 数值对比,引用时需注明。
5. **旧机器缓存**:`tests/data/smoke/labels.cache` 是旧机器生成的,ultralytics 首次加载发现 hash 不匹配会自动重建,无需手动删除。
6. **种子可复现**:正式实验请固定 `seed`(0/42/3407),并关闭不确定的增强(DSR 阶段已默认 `mosaic: 0`、`mixup: 0`)。
7. **polars 兼容性(本机)**:conda 环境的 polars≥1.x 在部分 CPU 上会误报
   `RuntimeError: unknown feature flag: 'sse3'`,导致任何 `model.train()` 在收尾
   保存权重时报错。矩阵运行器已自动设置 `POLARS_SKIP_CPU_CHECK=1`;
   若绕过运行器直接跑训练,请先手动设置该环境变量:

```powershell
$env:POLARS_SKIP_CPU_CHECK = "1"
```
