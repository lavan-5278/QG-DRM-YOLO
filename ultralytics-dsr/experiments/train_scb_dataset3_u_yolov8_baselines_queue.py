"""Strictly serial native YOLOv8n -> YOLOv8s scratch baselines for SCB-Dataset3-U."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = OUTPUT_ROOT / "SCB-Dataset3-U_YOLOv8基线迁移筛查报告.md"
LOG_DIR = OUTPUT_ROOT / "logs"
REPORT_START = "<!-- SCB_DATASET3_U_YOLOV8_BASELINES_START -->"
REPORT_END = "<!-- SCB_DATASET3_U_YOLOV8_BASELINES_END -->"

JOBS = [
    {"scale": "n", "id": "yolov8n_scb_u_seed42", "name": "YOLOv8n（原生基线）"},
    {"scale": "s", "id": "yolov8s_scb_u_seed42", "name": "YOLOv8s（原生基线）"},
]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def initial_state() -> dict:
    return {
        "dataset": "SCB-Dataset3-U",
        "data_yaml": str(ROOT / "ultralytics/cfg/datasets/SCB-Dataset3-U.yaml"),
        "protocol": {"seed": 42, "scratch": True, "pretrained": False, "batch": 8, "optimizer": "SGD", "lr0": 0.01, "epochs": 300, "imgsz": 640},
        "validity_notice": "本数据集既有审计发现 train/val 框布局高重复；本队列仅报告官方划分下的基准/迁移筛查，不能据此宣称严格跨域泛化。",
        "status": "pending",
        "jobs": [{**job, "status": "pending"} for job in JOBS],
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def render_report(state: dict) -> None:
    lines = [
        REPORT_START,
        "# SCB-Dataset3-U YOLOv8 原生基线迁移筛查报告",
        "",
        "本报告与 5k_HRW 实验完全隔离。模型为原生 YOLOv8n、YOLOv8s，不插入 DSR、LGC、DR-C2f 或其他自定义模块。",
        "",
        "统一协议：SCB-Dataset3-U 官方 train/val 划分、6 类、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。顺序为 YOLOv8n 后 YOLOv8s。",
        "",
        "> 有效性限制：已有数据审计显示该数据集 train/val 的框布局存在约 95.2% 的高重复。因此结果仅作为官方划分下的基准/迁移筛查，不能作为严格跨域泛化或无泄漏泛化结论。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for job in state["jobs"]:
        metrics = job.get("metrics")
        summary = job.get("summary")
        if job["status"] == "completed" and metrics and summary:
            lines.append(
                f"| {job['name']} | completed | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |"
            )
        else:
            lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += [
        "",
        "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理和 NMS。",
        "单一随机种子、单一官方划分的结果不构成统计显著性结论。",
        REPORT_END,
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ingest_summary(job: dict) -> bool:
    summary_path = OUTPUT_ROOT / "native_runs" / job["id"] / "native_summary.json"
    if not summary_path.exists():
        return False
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        return False
    job.update({"status": "completed", "metrics": payload["metrics"], "summary": {key: payload[key] for key in ("parameters", "weight_bytes", "flops_g", "inference_ms_b1_640", "inference_protocol")}, "finished_at": payload.get("finished_at", time.strftime("%Y-%m-%dT%H:%M:%S"))})
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    state = load_state()
    state["status"] = "running"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for job in state["jobs"]:
        if job["status"] == "completed" or ingest_summary(job):
            atomic_json(STATE_PATH, state)
            render_report(state)
            continue
        if job["status"] == "failed" and not args.resume:
            raise RuntimeError(f"previous failed job requires explicit --resume: {job['id']}")
        job.update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_dir": str(OUTPUT_ROOT / "native_runs" / job["id"])})
        atomic_json(STATE_PATH, state)
        render_report(state)
        command = [sys.executable, str(ROOT / "experiments/train_scb_dataset3_u_yolov8_baseline.py"), "--scale", job["scale"]]
        with (LOG_DIR / f"{job['id']}.out").open("a", encoding="utf-8") as stdout, (LOG_DIR / f"{job['id']}.err").open("a", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if completed.returncode != 0 or not ingest_summary(job):
            job.update({"status": "failed", "returncode": completed.returncode, "error": f"worker exited with code {completed.returncode}; inspect {LOG_DIR / (job['id'] + '.err')}"})
            state["status"] = "failed"
            atomic_json(STATE_PATH, state)
            render_report(state)
            raise RuntimeError(job["error"])
        atomic_json(STATE_PATH, state)
        render_report(state)
    state["status"] = "completed"
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(STATE_PATH, state)
    render_report(state)


if __name__ == "__main__":
    main()
