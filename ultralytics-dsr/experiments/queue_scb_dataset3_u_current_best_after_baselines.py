"""Append the current-best YOLOv8s topology after SCB-U native baselines finish."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "runs/scb_dataset3_u_yolov8_baselines_seed42_scratch"
BASE_STATE = OUTPUT_ROOT / "queue_state.json"
EXTENSION_STATE = OUTPUT_ROOT / "current_best_queue_state.json"
LOG_DIR = OUTPUT_ROOT / "logs"
REPORT_PATH = OUTPUT_ROOT / "SCB-Dataset3-U_YOLOv8基线迁移筛查报告.md"

JOB = {
    "id": "yolov8s_current_best_topology_scb_u_seed42",
    "name": "YOLOv8s 当前最优结构（QG-AFPN + C4 RGDRC2fV3 + MLKCM + P3DetailRestore）",
    "architecture": "QG-AFPN + C4-only RGDRC2fV3 + MLKCM(P3/P4/P5) + P3DetailRestore",
    "status": "waiting_for_native_baselines",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_report(state: dict) -> None:
    rows = state["jobs"]
    lines = [
        "<!-- SCB_DATASET3_U_YOLOV8_BASELINES_START -->",
        "# SCB-Dataset3-U YOLOv8 基线与当前最优结构迁移筛查报告", "",
        "前两项是原生 YOLOv8n、YOLOv8s 基线；第三项是从 5k_HRW 单模型筛选得到的最优 YOLOv8s 拓扑，重新从零训练，未加载 5k_HRW 权重。", "",
        "统一协议：SCB-Dataset3-U 官方 train/val 划分、6 类、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640；严格串行。", "",
        "> 有效性限制：已有数据审计显示该数据集 train/val 的框布局存在约 95.2% 的高重复。因此结果仅作为官方划分下的基准/迁移筛查，不能作为严格跨域泛化或无泄漏泛化结论。", "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for job in rows:
        metrics, summary = job.get("metrics"), job.get("summary")
        if job["status"] == "completed" and metrics and summary:
            lines.append(f"| {job['name']} | completed | {int(float(metrics['epoch']))+1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
        else:
            lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "当前最优结构组成：QG-AFPN 位于 Neck；仅 Backbone C4 使用零初始化可靠性门控残差 DR-C2f v3；MLKCM 位于 P3/P4/P5 Detect 输入；P3DetailRestore 位于 P3 MLKCM 之后。Neck 其余层与 Detect 保持 YOLOv8s 原生结构。", "", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理和 NMS。", "单一随机种子、单一官方划分的结果不构成统计显著性结论。", "<!-- SCB_DATASET3_U_YOLOV8_BASELINES_END -->"]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    extension = {"job": JOB.copy(), "status": "waiting_for_native_baselines", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        if not BASE_STATE.exists():
            atomic_json(EXTENSION_STATE, extension)
            time.sleep(30)
            continue
        state = json.loads(BASE_STATE.read_text(encoding="utf-8"))
        if state.get("status") == "failed":
            extension.update({"status": "blocked_native_baseline_failed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            atomic_json(EXTENSION_STATE, extension)
            return
        if state.get("status") == "completed":
            break
        atomic_json(EXTENSION_STATE, extension)
        time.sleep(30)

    job = JOB.copy()
    job.update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "run_dir": str(OUTPUT_ROOT / "native_runs" / JOB["id"])})
    state["jobs"] = [item for item in state["jobs"] if item["id"] != job["id"]] + [job]
    state["status"] = "running_current_best_topology"
    atomic_json(BASE_STATE, state)
    update_report(state)
    extension.update({"status": "running", "job": job})
    atomic_json(EXTENSION_STATE, extension)

    command = [sys.executable, str(ROOT / "experiments/train_scb_dataset3_u_current_best.py")]
    with (LOG_DIR / f"{JOB['id']}.out").open("a", encoding="utf-8") as stdout, (LOG_DIR / f"{JOB['id']}.err").open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
    summary_path = OUTPUT_ROOT / "native_runs" / JOB["id"] / "native_summary.json"
    if completed.returncode != 0 or not summary_path.exists():
        job.update({"status": "failed", "returncode": completed.returncode, "error": f"worker exited with code {completed.returncode}"})
        state["status"] = "failed"
        extension["status"] = "failed"
    else:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        job.update({"status": "completed", "metrics": payload["metrics"], "summary": {key: payload[key] for key in ("parameters", "weight_bytes", "flops_g", "inference_ms_b1_640", "inference_protocol")}, "adapter_states": payload["adapter_states"]})
        state["status"] = "completed"
        extension["status"] = "completed"
    job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    extension["finished_at"] = job["finished_at"]
    atomic_json(BASE_STATE, state)
    atomic_json(EXTENSION_STATE, extension)
    update_report(state)


if __name__ == "__main__":
    main()
