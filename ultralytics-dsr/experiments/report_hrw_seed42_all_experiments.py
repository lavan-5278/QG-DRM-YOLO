"""Create a Chinese Markdown report for the 5k_HRW seed-42 experiment suite."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path("runs/hrw_seed42_all_experiments")
DATASET_ROOT = Path(r"D:/YH_Project/DSA-Det/5k_HRW_yolo_Dataset")
METRIC_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map5095": "metrics/mAP50-95(B)",
    "train_box": "train/box_loss",
    "train_cls": "train/cls_loss",
    "train_dfl": "train/dfl_loss",
    "val_box": "val/box_loss",
    "val_cls": "val/cls_loss",
    "val_dfl": "val/dfl_loss",
}


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(METRIC_COLUMNS[key], "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.5f}" if signed else f"{value:.5f}"


def _best_result(csv_path: Path) -> dict[str, Any] | None:
    if not csv_path.is_file():
        return None
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if _number(row, "map5095") is not None]
    if not rows:
        return None
    best = max(rows, key=lambda row: _number(row, "map5095") or float("-inf"))
    return {
        "best_epoch": int(float(best.get("epoch", "0"))),
        "last_epoch": int(float(rows[-1].get("epoch", "0"))),
        **{key: _number(best, key) for key in METRIC_COLUMNS},
    }


def _model_artifacts(run_dir: Path) -> tuple[str, str]:
    """Return the actual best-checkpoint parameter count and on-disk size."""
    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.is_file():
        return "—", "—"

    size_mib = weights_path.stat().st_size / (1024 * 1024)
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            model = checkpoint.get("ema")
            if model is None:
                model = checkpoint.get("model")
        else:
            model = checkpoint
        parameters = sum(parameter.numel() for parameter in model.parameters())
        parameter_text = f"{parameters / 1_000_000:.3f} M"
    except Exception:
        parameter_text = "读取失败"
    return parameter_text, f"{size_mib:.2f} MiB"


def _label_counts() -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for label_path in (DATASET_ROOT / "labels").rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and fields[0].isdigit() and int(fields[0]) in counts:
                counts[int(fields[0])] += 1
    return counts


def _image_count(split: str) -> int:
    return sum(1 for path in (DATASET_ROOT / "images" / split).iterdir() if path.is_file())


def _load_state(output_root: Path) -> dict[str, Any]:
    state_path = output_root / "queue_state.json"
    if not state_path.is_file():
        return {"jobs": {}}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _phase0_lines(output_root: Path) -> list[str]:
    summary_path = output_root / "pvrq_phase0" / "phase0_summary.json"
    if not summary_path.is_file():
        return ["PVR-Q Phase-0 尚未完成；最终报告更新后将补充诊断结果。"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    g0 = summary.get("g0", {})
    focus = summary.get("competition", {}).get("focus_vs_reference_iou", {})
    oracle = summary.get("quality", {}).get("oracle_gain_bootstrap", {})
    correlation = summary.get("correlations", {}).get("y_center_vs_log_area_spearman")
    return [
        f"- G0 决策：**{g0.get('decision', '—')}**；局部定位缺口：{g0.get('localization_deficit', '—')}；质量排序余量：{g0.get('quality_headroom', '—')}。",
        f"- 聚焦区域 IoU 差：{_fmt(focus.get('mean'))}（95% CI [{_fmt(focus.get('ci95_low'))}, {_fmt(focus.get('ci95_high'))}]）。",
        f"- Oracle AP50-95 排序代理增益：{_fmt(oracle.get('mean'))}（95% CI [{_fmt(oracle.get('ci95_low'))}, {_fmt(oracle.get('ci95_high'))}]）。",
        f"- y-center 与 log-area 的 Spearman：{_fmt(correlation)}。",
        "- 这是诊断代理指标，不是检测 mAP，不能被表述为可直接实现的 mAP 增益。",
    ]


def _conclusion_lines(results: dict[str, dict[str, Any] | None]) -> list[str]:
    baseline = results.get("baseline")
    pnr = results.get("pnr_dfl")
    completed = {name: result for name, result in results.items() if result is not None}
    lines: list[str] = []
    if not baseline:
        return ["Baseline 尚未完成，暂不能做有效对照。"]
    if pnr:
        delta = (pnr["map5095"] or 0.0) - (baseline["map5095"] or 0.0)
        verdict = "候选有效" if delta >= 0.002 else "未达到预设 +0.00200 筛选阈值"
        lines.append(f"- PNR-DFL 相对 Baseline 的 mAP50-95 为 {_fmt(delta, signed=True)}：{verdict}（仅 seed42）。")
    if completed:
        best_name, best = max(completed.items(), key=lambda item: item[1]["map5095"] or float("-inf"))
        lines.append(f"- 已完成模型中 mAP50-95 最高的是 **{best_name}**：{_fmt(best['map5095'])}（最佳 epoch {best['best_epoch']}）。")
    for stack in ("pnr_sled", "pnr_bcsq", "pnr_afpn", "pnr_fadwc", "pnr_dyhead"):
        result = results.get(stack)
        if result and pnr:
            delta_pnr = (result["map5095"] or 0.0) - (pnr["map5095"] or 0.0)
            verdict = "候选叠加有效" if delta_pnr >= 0.002 else "不支持叠加"
            lines.append(f"- {stack} 相对 PNR-DFL 的 mAP50-95 为 {_fmt(delta_pnr, signed=True)}：{verdict}（仅 seed42）。")
    lines.append("- 本套实验只有一个随机种子和一个固定划分，只能用于筛选；不能据此作稳定性、显著性或论文最终结论。")
    return lines


def build_report(output_root: Path) -> str:
    """Return the complete Markdown report; safe to call while the queue is still running."""
    state = _load_state(output_root)
    jobs: dict[str, dict[str, Any]] = state.get("jobs", {})
    rows: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    for name, job in jobs.items():
        if job.get("kind") == "phase0":
            continue
        run_dir = WORKSPACE / job["run_dir"]
        is_skipped = str(job.get("status", "")).startswith("skipped")
        rows.append((name, job, None if is_skipped else _best_result(run_dir / "results.csv")))
    results = {name: result for name, _, result in rows}
    baseline = results.get("baseline")
    pnr = results.get("pnr_dfl")
    counts = _label_counts() if DATASET_ROOT.is_dir() else {0: 0, 1: 0, 2: 0}
    phase0_status = jobs.get("pvrq_phase0", {}).get("status", "pending")

    lines = [
        "# 5k_HRW 数据集：YOLO11n 扩展实验结果（seed42）",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验边界与可比性",
        "",
        "- 数据集：`D:/YH_Project/DSA-Det/5k_HRW_yolo_Dataset`；训练/验证划分固定，所有训练型实验均使用 seed=42。",
        "- 标准训练协议：YOLO11n，640×640，batch=16，300 epochs，patience=100，预训练 `yolo11n.pt`，其余超参数与此前 SCB-Dataset3-S 严格协议一致。",
        "- DSR-Det 保留其专用 DSR trainer 与阶段性冻结策略；Composition 保留其专用 auxiliary trainer。二者的优化路径不同，只可作本数据集上的探索性比较。",
        "- 标签源仅提供 0/1/2 数字类别，本文暂以 `class_0/class_1/class_2` 标记；类别语义需由数据集提供方补充。",
        "- 每行取验证集 mAP50-95 最高的 epoch；损失也取同一 epoch。所有差值均为“方法 − 对照”。参数量取该作业 `best.pt` 中 EMA/模型的实际总参数量；权重大小为 `best.pt` 的磁盘大小。",
        "",
        "## 数据集审计",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 训练图像 | {_image_count('train') if DATASET_ROOT.is_dir() else '—'} |",
        f"| 验证图像 | {_image_count('val') if DATASET_ROOT.is_dir() else '—'} |",
        f"| class_0 标注框 | {counts[0]} |",
        f"| class_1 标注框 | {counts[1]} |",
        f"| class_2 标注框 | {counts[2]} |",
        "",
        "## 训练结果总表",
        "",
        "| 类别 | 方法 | 状态 | 参数量 | best.pt 大小 | 最佳 Epoch | P | R | mAP50 | mAP50-95 | 相对 Baseline | 相对 PNR | train box/cls/dfl | val box/cls/dfl |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, job, result in rows:
        run_dir = WORKSPACE / job["run_dir"]
        if result is None:
            lines.append(f"| {job.get('category', '—')} | {job.get('display_name', name)} | {job.get('status', 'pending')} | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        parameter_text, size_text = _model_artifacts(run_dir)
        delta_base = None if baseline is None else (result["map5095"] or 0.0) - (baseline["map5095"] or 0.0)
        delta_pnr = None if pnr is None else (result["map5095"] or 0.0) - (pnr["map5095"] or 0.0)
        train_losses = "/".join(_fmt(result[key]) for key in ("train_box", "train_cls", "train_dfl"))
        val_losses = "/".join(_fmt(result[key]) for key in ("val_box", "val_cls", "val_dfl"))
        lines.append(
            f"| {job.get('category', '—')} | {job.get('display_name', name)} | {job.get('status', 'completed')} | {parameter_text} | {size_text} | {result['best_epoch']} | "
            f"{_fmt(result['precision'])} | {_fmt(result['recall'])} | {_fmt(result['map50'])} | {_fmt(result['map5095'])} | "
            f"{_fmt(delta_base, signed=True)} | {_fmt(delta_pnr, signed=True)} | {train_losses} | {val_losses} |"
        )

    lines += ["", "## PNR-DFL 机制消融", "", "| 变体 | 与完整 PNR 的区别 | mAP50-95 | 相对 Baseline | 相对完整 PNR |", "|---|---|---:|---:|---:|"]
    variants = {
        "raw_uniform": "不做尺度归一化；uniform 权重",
        "norm_uniform": "尺度归一化；uniform 权重",
        "norm_size": "尺度归一化；size 权重",
        "norm_vertical": "尺度归一化；vertical 权重",
        "pnr_dfl": "尺度归一化；joint 权重（完整 PNR）",
    }
    for name, description in variants.items():
        result = results.get(name)
        map95 = None if result is None else result["map5095"]
        delta_base = None if baseline is None or map95 is None else map95 - (baseline["map5095"] or 0.0)
        delta_pnr = None if pnr is None or map95 is None else map95 - (pnr["map5095"] or 0.0)
        lines.append(f"| {name} | {description} | {_fmt(map95)} | {_fmt(delta_base, signed=True)} | {_fmt(delta_pnr, signed=True)} |")

    lines += ["", "## PNR-DFL 叠加验证", "", "| 组合 | mAP50-95 | 相对 Baseline | 相对 PNR-DFL | 单 seed 筛选结论 |", "|---|---:|---:|---:|---|"]
    for name in ("pnr_sled", "pnr_bcsq", "pnr_afpn", "pnr_fadwc", "pnr_dyhead"):
        job = jobs.get(name, {})
        result = results.get(name)
        map95 = None if result is None else result["map5095"]
        delta_base = None if baseline is None or map95 is None else map95 - (baseline["map5095"] or 0.0)
        delta_pnr = None if pnr is None or map95 is None else map95 - (pnr["map5095"] or 0.0)
        verdict = "已按用户要求跳过" if str(job.get("status", "")).startswith("skipped") else ("待完成" if map95 is None else ("候选有效" if delta_base is not None and delta_pnr is not None and delta_base >= 0.002 and delta_pnr >= 0.002 else "不支持叠加"))
        lines.append(f"| {job.get('display_name', name)} | {_fmt(map95)} | {_fmt(delta_base, signed=True)} | {_fmt(delta_pnr, signed=True)} | {verdict} |")

    lines += ["", "## PVR-Q Phase-0 诊断", "", f"状态：`{phase0_status}`", "", *_phase0_lines(output_root), "", "## 自动结论", "", *_conclusion_lines(results), ""]
    return "\n".join(lines)


def write_report(output_root: Path) -> Path:
    """Write and return the stable final/interim Markdown report path."""
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "5k_HRW_seed42_全部实验结果.md"
    report_path.write_text(build_report(output_root), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(write_report(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
