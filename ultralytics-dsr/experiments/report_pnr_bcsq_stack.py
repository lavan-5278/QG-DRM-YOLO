"""Paired all-metric report for the PNR-DFL + BCSQ-Head three-seed stack."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}
RUNS = {
    "Baseline": {
        42: Path("runs/pvrf_matched_baseline_300/seed42/results.csv"),
        43: Path("runs/multiseed_3seed/baseline/seed43/results.csv"),
        44: Path("runs/multiseed_3seed/baseline/seed44/results.csv"),
    },
    "PNR-DFL": {
        42: Path("runs/pnr_dfl_full_300/seed42/results.csv"),
        43: Path("runs/multiseed_3seed/pnr_dfl/seed43/results.csv"),
        44: Path("runs/multiseed_3seed/pnr_dfl/seed44/results.csv"),
    },
    "BCSQ-Head": {
        42: Path("runs/bcsq_full_300/seed42/results.csv"),
        43: Path("runs/multiseed_3seed/bcsq/seed43/results.csv"),
        44: Path("runs/multiseed_3seed/bcsq/seed44/results.csv"),
    },
    "PNR-DFL + BCSQ-Head": {
        42: Path("runs/pnr_bcsq_stack_3seed/pnr_bcsq/seed42/results.csv"),
        43: Path("runs/pnr_bcsq_stack_3seed/pnr_bcsq/seed43/results.csv"),
        44: Path("runs/pnr_bcsq_stack_3seed/pnr_bcsq/seed44/results.csv"),
    },
}


def best_metrics(csv_path: Path) -> dict[str, float | int]:
    """Return all primary metrics at the row with the run's highest mAP50-95."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No metric rows in {csv_path}")
    best = max(rows, key=lambda row: float(row[METRICS["mAP50-95"]]))
    return {"epoch": int(float(best["epoch"])), **{name: float(best[column]) for name, column in METRICS.items()}}


def paired_summary(
    candidate: dict[int, dict[str, float | int]], reference: dict[int, dict[str, float | int]], metric: str = "mAP50-95"
) -> dict[str, list[float] | float]:
    """Return seed-paired candidate-minus-reference deltas, mean, and sample standard deviation."""
    seeds = tuple(seed for seed in SEEDS if seed in candidate and seed in reference)
    deltas = [float(candidate[seed][metric]) - float(reference[seed][metric]) for seed in seeds]
    mean = sum(deltas) / len(deltas)
    std = math.sqrt(sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)) if len(deltas) > 1 else 0.0
    return {"deltas": deltas, "mean": mean, "std": std}


def collect() -> dict[str, dict[int, dict[str, float | int]]]:
    """Read every mandatory completed endpoint, failing rather than reporting an incomplete comparison."""
    result: dict[str, dict[int, dict[str, float | int]]] = {}
    for name, seed_paths in RUNS.items():
        result[name] = {}
        for seed, relative in seed_paths.items():
            csv_path = WORKSPACE / relative
            if not csv_path.is_file():
                raise FileNotFoundError(f"Required completed result is missing: {csv_path}")
            result[name][seed] = best_metrics(csv_path)
    return result


def render_markdown(results: dict[str, dict[int, dict[str, float | int]]]) -> str:
    """Render a concise, all-metric paired comparison suitable for the experiment record."""
    candidate = results["PNR-DFL + BCSQ-Head"]
    lines = [
        "# PNR-DFL + BCSQ-Head 三 Seed 叠加验证",
        "",
        "每个 seed 取其验证集 mAP50-95 最高的 epoch；差值均为组合模型减去同 seed 对照。",
        "",
        "| 模型 | Seed | 最佳 Epoch | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, by_seed in results.items():
        for seed in SEEDS:
            row = by_seed[seed]
            lines.append(
                f"| {name} | {seed} | {row['epoch']} | {row['precision']:.5f} | {row['recall']:.5f} | "
                f"{row['mAP50']:.5f} | {row['mAP50-95']:.5f} |"
            )

    lines += ["", "## 组合相对各端点的配对差（组合 − 对照）", ""]
    for reference_name in ("Baseline", "PNR-DFL", "BCSQ-Head"):
        lines += [f"### 相对 {reference_name}", "", "| 指标 | seed42 | seed43 | seed44 | 均值 ± 标准差 |", "|---|---:|---:|---:|---:|"]
        for metric in METRICS:
            summary = paired_summary(candidate, results[reference_name], metric)
            values = summary["deltas"]
            lines.append(
                f"| {metric} | {values[0]:+.5f} | {values[1]:+.5f} | {values[2]:+.5f} | "
                f"{summary['mean']:+.5f} ± {summary['std']:.5f} |"
            )

    m95 = {name: paired_summary(candidate, results[name]) for name in ("Baseline", "PNR-DFL", "BCSQ-Head")}
    promote = all(float(summary["mean"]) >= 0.002 for summary in m95.values())
    conclusion = "PROMOTE_STACK" if promote else "REJECT_STACK"
    lines += [
        "",
        f"**结论：{conclusion}。** 规则为组合平均 mAP50-95 必须同时比 Baseline、PNR-DFL、BCSQ-Head 高至少 +0.00200。",
        "该结论来自 SCB-Dataset3-S 的单一官方划分与三个随机种子，不能替代跨数据集/跨划分泛化结论。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write JSON and Markdown only once all four endpoints are available."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/pnr_bcsq_stack_3seed")
    args = parser.parse_args()
    results = collect()
    output = WORKSPACE / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "stack_comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "stack_comparison.md").write_text(render_markdown(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
