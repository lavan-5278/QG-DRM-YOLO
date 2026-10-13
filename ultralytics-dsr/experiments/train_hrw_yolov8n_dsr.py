"""Train YOLOv8n + DSR on 5k_HRW with the fixed scratch-baseline protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer
from ultralytics.utils.torch_utils import get_flops


MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8n-dsr.yaml"
DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
DEFAULT_OUTPUT = ROOT / "runs/hrw_yolov8n_dsr_seed42_scratch/native_runs/yolov8n_dsr"
STANDARD_REPORT = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8N_DSR_FINAL_START -->"
REPORT_END = "<!-- YOLOV8N_DSR_FINAL_END -->"


def build_train_kwargs(run_dir: Path) -> dict:
    """Return the protocol shared with the completed 5k_HRW scratch baselines."""
    return {
        "data": str(DATA_YAML),
        "epochs": 300,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.01,
        "optimizer": "SGD",
        "seed": 42,
        "deterministic": True,
        "device": 0,
        "workers": 4,
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "pretrained": False,
        "val": True,
        "save": True,
        "plots": True,
        "verbose": True,
    }


def write_summary(run_dir: Path) -> dict:
    """Measure the completed best checkpoint using the baseline efficiency protocol."""
    best = run_dir / "weights/best.pt"
    checkpoint_model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().cuda().eval()
    params = sum(v.numel() for v in checkpoint_model.parameters())
    flops_g = float(get_flops(checkpoint_model, imgsz=640))
    sample = torch.zeros((1, 3, 640, 640), device="cuda")
    with torch.inference_mode():
        for _ in range(50):
            checkpoint_model(sample)
        torch.cuda.synchronize()
        timings = []
        for _ in range(200):
            start = time.perf_counter()
            checkpoint_model(sample)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    summary = {
        "model": "yolov8n_dsr",
        "framework": "Ultralytics + DSRDetectionTrainer",
        "status": "completed",
        "run_dir": str(run_dir),
        "best_weight": str(best),
        "parameters": int(params),
        "weight_bytes": best.stat().st_size,
        "flops_g": flops_g,
        "inference_ms_b1_640": statistics.mean(timings),
        "inference_protocol": "RTX 4080 SUPER, CUDA, FP32, batch=1, 640x640, 50 warmup + 200 iterations, forward only",
        "pretraining": "无：从头训练",
    }
    (run_dir / "native_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def read_best_metrics(run_dir: Path) -> dict[str, str]:
    """Read the highest mAP50-95 validation row from an Ultralytics results file."""
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    map95_key = next(key for key in rows[0] if "mAP50-95" in key)
    best = max(rows, key=lambda row: float(row[map95_key]))
    return {
        "epoch": best["epoch"],
        "precision": best["metrics/precision(B)"],
        "recall": best["metrics/recall(B)"],
        "map50": best["metrics/mAP50(B)"],
        "map5095": best[map95_key],
        "train_box": best["train/box_loss"],
        "train_cls": best["train/cls_loss"],
        "train_dfl": best["train/dfl_loss"],
        "val_box": best["val/box_loss"],
        "val_cls": best["val/cls_loss"],
        "val_dfl": best["val/dfl_loss"],
    }


def format_report_section(metrics: dict[str, str], summary: dict) -> str:
    """Format the final DSR result as an idempotent supplementary report section."""
    params = f"{int(summary['parameters']):,}"
    weight_mb = summary["weight_bytes"] / 1024 / 1024
    loss = f"train: {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f}; val: {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f}"
    return "\n".join((
        REPORT_START,
        "## YOLOv8n + DSR（补充实验）",
        "",
        "该项使用与标准基线相同的 5k_HRW、seed=42、batch=8、lr0=0.01、300 epoch、640×640、SGD、无预训练权重协议；DSR 的随机初始化冻结阶段已禁用，以保证从零训练公平性。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best 权重大小 | 训练/验证损失 | FLOPs（G，640×640） | 推理耗时（ms/图） |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        f"| YOLOv8n + DSR | completed | {int(metrics['epoch']) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {params} | {weight_mb:.2f} MB | {loss} | {float(summary['flops_g']):.3f} | {float(summary['inference_ms_b1_640']):.3f} |",
        "",
        "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640、50 次预热后 200 次前向均值；不含图像读取、预处理与 NMS。",
        REPORT_END,
    ))


def write_final_report(run_dir: Path, summary: dict) -> Path:
    """Insert or replace the completed DSR section in the user-requested standard-baseline report."""
    metrics = read_best_metrics(run_dir)
    section = format_report_section(metrics, summary)
    existing = STANDARD_REPORT.read_text(encoding="utf-8") if STANDARD_REPORT.exists() else "# 5k_HRW 标准基线结果\n"
    if REPORT_START in existing:
        before, remainder = existing.split(REPORT_START, 1)
        _, after = remainder.split(REPORT_END, 1)
        content = before.rstrip() + "\n\n" + section + after
    else:
        content = existing.rstrip() + "\n\n" + section + "\n"
    STANDARD_REPORT.write_text(content, encoding="utf-8")
    return STANDARD_REPORT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-only", action="store_true", help="Measure an existing best.pt and update the standard report.")
    args = parser.parse_args()
    if args.finalize_only:
        summary = write_summary(args.output)
        summary["standard_report"] = str(write_final_report(args.output, summary))
        (args.output / "native_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_YAML))
    result = model.train(trainer=DSRDetectionTrainer, **build_train_kwargs(args.output))
    summary = write_summary(Path(result.save_dir))
    summary["standard_report"] = str(write_final_report(Path(result.save_dir), summary))
    (Path(result.save_dir) / "native_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
