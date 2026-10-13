"""Plot the recorded 5k_HRW convergence curves for YOLOv8n and QG-DRM-YOLO."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs/hrw_standard_baselines_seed42_scratch/visualizations"
REPORT = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
RUNS = {
    "YOLOv8n": ROOT / "runs/hrw_standard_baselines_seed42_scratch/native_runs/yolov8n/results.csv",
    "QG-DRM-YOLO": ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch/native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail/results.csv",
}
COLORS = {"YOLOv8n": "#276FBF", "QG-DRM-YOLO": "#D1495B"}
LOSS_PANELS = (
    ("train/box_loss", "Train box loss"),
    ("train/cls_loss", "Train cls loss"),
    ("train/dfl_loss", "Train DFL loss"),
    ("val/box_loss", "Validation box loss"),
    ("val/cls_loss", "Validation cls loss"),
    ("val/dfl_loss", "Validation DFL loss"),
)
METRIC_PANELS = (
    ("metrics/precision(B)", "Precision"),
    ("metrics/recall(B)", "Recall"),
    ("metrics/mAP50(B)", "mAP@0.5"),
    ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = rows[0].keys()
    return {field: np.asarray([float(row[field]) for row in rows], dtype=np.float64) for field in fields}


def setup_axis(axis: plt.Axes, title: str) -> None:
    axis.set_title(title, fontsize=11, weight="bold")
    axis.set_xlabel("Epoch")
    axis.grid(alpha=0.22, linewidth=0.6)
    axis.set_xlim(1, 300)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_panels(data: dict[str, dict[str, np.ndarray]], panels: tuple[tuple[str, str], ...], shape: tuple[int, int], output: Path, suptitle: str) -> None:
    figure, axes = plt.subplots(*shape, figsize=(13.6, 7.6), constrained_layout=True)
    for axis, (field, title) in zip(np.asarray(axes).ravel(), panels):
        for name, values in data.items():
            axis.plot(values["epoch"], values[field], label=name, color=COLORS[name], linewidth=1.7)
            last_epoch = int(values["epoch"][-1])
            axis.scatter(last_epoch, values[field][-1], color=COLORS[name], s=18, zorder=3)
        setup_axis(axis, title)
        axis.set_ylabel("Loss" if "loss" in field else "Value")
    handles, labels = np.asarray(axes).ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    figure.suptitle(suptitle, fontsize=14, weight="bold", y=1.03)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summary(data: dict[str, dict[str, np.ndarray]]) -> dict[str, object]:
    output: dict[str, object] = {"protocol_epochs": 300, "models": {}}
    for name, values in data.items():
        model: dict[str, object] = {"last_recorded_epoch": int(values["epoch"][-1])}
        for field, label in METRIC_PANELS:
            tail = values[field][-min(30, len(values[field])) :]
            model[label] = {
                "best": round(float(values[field].max()), 6),
                "last": round(float(values[field][-1]), 6),
                "tail30_mean": round(float(tail.mean()), 6),
                "tail30_std": round(float(tail.std(ddof=0)), 6),
            }
        output["models"][name] = model
    return output


def update_report(loss_path: Path, metric_path: Path, summary_path: Path) -> None:
    start = "<!-- 5K_HRW_TRAINING_CURVES_START -->"
    end = "<!-- 5K_HRW_TRAINING_CURVES_END -->"
    block = "\n".join((
        start,
        "## 5k_HRW 训练曲线：YOLOv8n 与 QG-DRM-YOLO",
        "",
        "两图均直接由各自 `results.csv` 绘制，横轴统一到 300 epoch；末端圆点表示当前记录的最后一个 epoch。损失越低越好，Precision、Recall、mAP 越高越好。两次记录分别截至 epoch 276 与 297，因此仅比较已记录的收敛过程，不将缺失尾段视为完成训练。",
        "",
        f"![YOLOv8n 与 QG-DRM-YOLO 损失曲线]({loss_path.as_posix()})",
        "",
        f"![YOLOv8n 与 QG-DRM-YOLO 指标曲线]({metric_path.as_posix()})",
        "",
        f"对应的末段稳定性和最优/末次指标见 [JSON]({summary_path.as_posix()})。",
        end,
    ))
    report = REPORT.read_text(encoding="utf-8")
    if start in report and end in report:
        before, after_start = report.split(start, 1)
        _, after = after_start.split(end, 1)
        report = before.rstrip() + "\n\n" + block + after
    else:
        report = report.rstrip() + "\n\n" + block + "\n"
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {name: read_csv(path) for name, path in RUNS.items()}
    loss_path = OUT_DIR / "5k_HRW_training_curves_yolov8n_vs_qg_drm_losses.png"
    metric_path = OUT_DIR / "5k_HRW_training_curves_yolov8n_vs_qg_drm_metrics.png"
    plot_panels(data, LOSS_PANELS, (2, 3), loss_path, "5k_HRW loss convergence (same scratch protocol)")
    plot_panels(data, METRIC_PANELS, (2, 2), metric_path, "5k_HRW validation metrics (same scratch protocol)")
    results = summary(data)
    summary_path = OUT_DIR / "5k_HRW_training_curves_yolov8n_vs_qg_drm_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    update_report(loss_path, metric_path, summary_path)
    print(json.dumps({"loss": str(loss_path), "metrics": str(metric_path), "summary": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
