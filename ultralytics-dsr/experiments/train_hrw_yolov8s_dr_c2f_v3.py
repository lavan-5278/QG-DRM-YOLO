"""Train standalone scratch YOLOv8s + DR-C2f v3 Backbone ablation on 5k_HRW."""

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
from ultralytics.utils.torch_utils import get_flops

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8s-dr-c2f-v3.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_dr_c2f_v3_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_dr_c2f_v3"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_DR_C2F_V3_FINAL_START -->"
REPORT_END = "<!-- YOLOV8S_DR_C2F_V3_FINAL_END -->"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def initial_state() -> dict:
    return {"protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"}, "job": {"id": "yolov8s_dr_c2f_v3", "name": "YOLOv8s + DR-C2f v3（Backbone 全替换）", "yaml": str(MODEL_YAML), "status": "pending"}}


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def read_best_metrics(run_dir: Path) -> dict[str, str]:
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("results.csv has no validation rows")
    map95_key = next(key for key in rows[0] if "mAP50-95" in key)
    best = max(rows, key=lambda row: float(row[map95_key]))
    keys = {"epoch": "epoch", "precision": "metrics/precision(B)", "recall": "metrics/recall(B)", "map50": "metrics/mAP50(B)", "map5095": map95_key, "train_box": "train/box_loss", "train_cls": "train/cls_loss", "train_dfl": "train/dfl_loss", "val_box": "val/box_loss", "val_cls": "val/cls_loss", "val_dfl": "val/dfl_loss"}
    return {name: best.get(key, "") for name, key in keys.items()}


def benchmark_checkpoint(best: Path) -> dict:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().cuda().eval()
    sample = torch.zeros((1, 3, 640, 640), device="cuda")
    with torch.inference_mode():
        for _ in range(50): model(sample)
        torch.cuda.synchronize()
        timings = []
        for _ in range(200):
            start = time.perf_counter(); model(sample); torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {"parameters": sum(p.numel() for p in model.parameters()), "weight_bytes": best.stat().st_size, "flops_g": float(get_flops(model, imgsz=640)), "inference_ms_b1_640": statistics.mean(timings), "inference_protocol": "RTX 4080 SUPER, CUDA, FP32, batch=1, 640x640, 50 warmup + 200 forward passes"}


def learned_relay_coefficients(best: Path) -> dict[str, dict[str, list[float]]]:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().eval()
    result = {name: module.relay_coefficients() for name, module in model.named_modules() if callable(getattr(module, "relay_coefficients", None))}
    if not result: raise RuntimeError("No DRC2fV3 relay modules in best.pt")
    return result


def refresh_report(state: dict) -> None:
    job = state["job"]; metrics, summary = job.get("metrics"), job.get("summary")
    lines = [REPORT_START, "## YOLOv8s DR-C2f v3 Backbone 替换实验", "", "DRC2fV3 仅替换四个 Backbone C2f；Neck、Detect 与 YOLOv8s scratch 其余配置不变，未叠加其他模块，也不调用可选 training-time clamp。", "", "统一协议：5k_HRW、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "", "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    if job["status"] == "completed" and metrics and summary:
        lines.append(f"| {job['name']} | completed | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
        lines += ["", "学习到的中继系数：", "", "```json", json.dumps(job["relay_coefficients"], ensure_ascii=False, indent=2), "```"]
    else: lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines); previous = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in previous and REPORT_END in previous:
        prefix, tail = previous.split(REPORT_START, 1); _, suffix = tail.split(REPORT_END, 1); content = prefix.rstrip()+"\n\n"+section+suffix
    else: content = previous.rstrip()+"\n\n"+section+"\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    state = load_state(); refresh_report(state); model = YOLO(str(MODEL_YAML))
    if args.dry_run: model.info(); return
    job = state["job"]
    if job["status"] == "completed": return
    job.update({"status": "running", "run_dir": str(RUN_DIR), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}); atomic_json(STATE_PATH, state); refresh_report(state)
    try:
        result = model.train(data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42, deterministic=True, device=0, workers=4, project=str(RUN_DIR.parent), name=RUN_DIR.name, exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True)
        final_dir = Path(result.save_dir); best = final_dir / "weights/best.pt"; job["metrics"] = read_best_metrics(final_dir); job["summary"] = benchmark_checkpoint(best); job["relay_coefficients"] = learned_relay_coefficients(best)
        (final_dir / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"], "relay_coefficients": job["relay_coefficients"]}, ensure_ascii=False, indent=2), encoding="utf-8"); job["status"] = "completed"
    except Exception as exc:
        job.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}); raise
    finally:
        job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S"); atomic_json(STATE_PATH, state); refresh_report(state)


if __name__ == "__main__": main()
