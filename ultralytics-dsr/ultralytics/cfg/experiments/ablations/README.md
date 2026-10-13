# 消融实验配置（Phase 5）

所有消融都通过“同一 `model.train()` 入口 + 切换配置”完成，不编写独立训练脚本。

- 关系算子 R0–R7（No Relation / Dot / Concat / Product / Abs Diff / EdgeConv /
  Signed / Transformer Encoder）：通过 `dsr.relation_type` 配置切换
  （`none | dot | concat | product | abs_diff | edgeconv | signed_diff | transformer`）；
- 模块消融 A0–A7：通过 `dsr.use_global / use_region / use_neighbor /
  use_uncertainty_gate / neighbor_reliability / lambda_preserve / uncertainty_threshold` 切换；
- MDRR 内部机制消融 M1–M5：M1（完整 Signed）为默认；M2–M5 通过
  `dsr.relation_variant` 切换：
  `no_geo`（去掉几何编码）、`no_gate`（去掉通道门控）、`flip`（方向翻转）、
  `shared`（边权与消息共用同一投影空间）；
- 门控消融：通过 `dsr.uncertainty_type`（max_prob / margin / entropy）与
  `dsr.gate_mode`（fixed / scalar / conf / feature / uncertainty，v1.1 18.5）切换；
  `dsr.use_uncertainty_gate: false` 时使用 `gate_mode`，为 true 时强制 uncertainty；
- 候选参数消融：`relation_topk / candidate_conf / global_topk / neighbor_k /
  relation_dim / roi_size`。
- 候选稳定性对照：`dsr.candidate_source: current | ema`（EMA 冻结副本生成候选，
  由 Trainer 在 `optimizer_step` 后更新）；候选集漂移曲线由
  `stability_ref_epoch / stability_every / stability_probe_batches` 控制，
  结果写入 `runs/<exp>/candidate_stability.json`。
- 训练阶段调度（v1.1 14.1）：前 `stage_b1_epochs` 冻结
  `stage_b1_freeze`，之后沿用 `stage_b2_freeze` 并将 LR 乘以
  `stage_b2_lr_scale`。总训练轮数由训练配置中的 `epochs` 明确控制。
- 不确定性分布审计（v1.1 7.1）：`track_uncertainty: true` 时每个 epoch 输出
  `runs/<exp>/uncertainty_audit.json`（u_mean / u_p50 / u_p90 / ratio_u_gt_0.9 /
  conf_mean），用于校验 `uncertainty_type` 输入有效性。
- 困难场景评估（v1.1 7.2 / 19.x / 20.3）：`dsr.validator: true` 时 Trainer 使用
  `DSRDetectionValidator`，在 `model.val()` 内输出 `dsr_scenario.json`：
  - 遮挡（IoA）/ 密度 / 尺寸分层 + 每层 AP50、AP50-95、Recall、漏检、误分类与
    图像级 bootstrap 95% CI（`scenario_bootstrap`，独立 `model.val()` 时启用）；
  - 候选召回协议：`scenario_conf_sweep`（默认 0.01/0.03/0.05/0.10）逐档
    `Recall_proposal@0.5`，验收门 ≥95%；
  - oracle 上界：`oracle: true` 时以 GT 框为候选跑 MDRR+精炼头并与预测候选 mAP 对比。
- 统计检验（v1.1 20.3/20.4）：每种子跑 3 次（seed 0/42/3407，目录
  `runs/<exp>/seed{0,42,3407}/`），`scripts/summarize_runs.py` 输出 μ±σ；
  `--compare` 用 `per_image_metrics.json` 做图像级配对 bootstrap CI。
- 效率测量（v1.1 20.2）：`profile_dsr()`（分项延迟/FLOPs/峰值显存/FP32-FP16）与
  `peak_memory_train()` 在标准模型前向内采集，不涉及独立训练/评估脚本。
- 关系可视化（v1.1 19.5）：`visualize_relation_stats(model, 图片目录)` 通过
  `model.predict()` 逐图保存尺度融合权重/门控曲线（`relation_*.png`）与
  `relation_stats.json`；Validator 报告同时输出关系统计与 base-vs-refined
  错误归因（fixed / introduced / kept / added / removed）。
- v2 选择机制消融：`uncertainty_threshold`（0.1/0.5 等）、`neighbor_reliability`、
  `lambda_preserve`（保真损失开关）；GICR 与 bbox 精炼已在 v2 中移除。

每个消融项生成一个实验 YAML（由 `scripts/run_experiments.py` 读取
`ultralytics/cfg/experiments/matrix.yaml` 批量生成与运行），
结果统一写入 `runs/`，由 `scripts/summarize_runs.py` 汇总。
