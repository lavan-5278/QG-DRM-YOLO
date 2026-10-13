"""Strict serial queue for four complementary YOLOv8s improvement routes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_four_improvements_seed42_scratch"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
LOG_DIR = OUTPUT_ROOT / "logs"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_FOUR_IMPROVEMENT_ROUTES_START -->"
REPORT_END = "<!-- YOLOV8S_FOUR_IMPROVEMENT_ROUTES_END -->"
PAPER_TARGET = {"map50": 0.75145, "map5095": 0.56894}
JOBS = [
    {
        "id": "yolov8s_best_cbqfl",
        "name": "当前最优结构 + CB-QFL",
        "kind": "single_model_training",
        "command": [sys.executable, "experiments/train_hrw_yolov8s_four_improvements.py", "--job", "cbqfl"],
        "summary": OUTPUT_ROOT / "native_runs/yolov8s_best_cbqfl/native_summary.json",
    },
    {
        "id": "yolov8s_best_pfom_ensemble",
        "name": "当前最优 + PFOM+MLKCM 原始输出融合",
        "kind": "ensemble_evaluation",
        "command": [sys.executable, "experiments/evaluate_hrw_yolov8s_best_pfom_ensemble.py"],
        "summary": OUTPUT_ROOT / "native_runs/yolov8s_best_pfom_ensemble/ensemble_summary.json",
    },
    {
        "id": "yolov8s_best_dual_teacher_distill",
        "name": "当前最优结构 + 双教师蒸馏",
        "kind": "single_model_training",
        "command": [sys.executable, "experiments/train_hrw_yolov8s_four_improvements.py", "--job", "distill"],
        "summary": OUTPUT_ROOT / "native_runs/yolov8s_best_dual_teacher_distill/native_summary.json",
    },
    {
        "id": "yolov8s_best_p2_detect",
        "name": "当前最优结构 + P2 检测层",
        "kind": "single_model_training",
        "command": [sys.executable, "experiments/train_hrw_yolov8s_four_improvements.py", "--job", "p2"],
        "summary": OUTPUT_ROOT / "native_runs/yolov8s_best_p2_detect/native_summary.json",
    },
]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def initial_state() -> dict:
    return {
        "status": "pending",
        "protocol": {
            "dataset": "5k_HRW",
            "seed": 42,
            "student_initialization": "scratch/no pretrained weights",
            "batch": 8,
            "optimizer": "SGD",
            "lr0": 0.01,
            "epochs": 300,
            "imgsz": 640,
        },
        "target": {**PAPER_TARGET, "criterion": "both metrics must be strictly exceeded"},
        "order": [job["id"] for job in JOBS],
        "jobs": [
            {
                "id": job["id"],
                "name": job["name"],
                "kind": job["kind"],
                "status": "pending",
                "summary_path": str(job["summary"]),
            }
            for job in JOBS
        ],
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def refresh_report(state: dict) -> None:
    lines = [
        REPORT_START,
        "## YOLOv8s 四路线定向提升实验",
        "",
        "目标：在当前单模型最优 mAP50-95=0.5777 的基础上补足 mAP50，并严格检验是否同时超过论文结构复现的 mAP50=0.75145、mAP50-95=0.56894。",
        "",
        "训练项统一协议：5k_HRW、seed=42、学生模型从零训练且无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。第二项是两个既有模型的验证集融合评估，不是单模型训练，单独标注以避免误读。",
        "",
        "| 路线 | 类型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | 权重大小 | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for job in state["jobs"]:
        metrics, summary = job.get("metrics"), job.get("summary")
        if job["status"] == "completed" and metrics and summary:
            passed = float(metrics["map50"]) > PAPER_TARGET["map50"] and float(metrics["map5095"]) > PAPER_TARGET["map5095"]
            status = "completed / 双指标通过" if passed else "completed / 未双指标通过"
            epoch = "—" if job["kind"] == "ensemble_evaluation" else str(int(float(metrics["epoch"])) + 1)
            lines.append(
                f"| {job['name']} | {job['kind']} | {status} | {epoch} | "
                f"{float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | "
                f"{float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | "
                f"{int(summary['parameters']):,} | {float(summary['weight_bytes'])/1024/1024:.2f} MB | "
                f"{float(summary['flops_g']):.3f} | {float(summary['inference_ms_b1_640']):.3f} |"
            )
        else:
            lines.append(f"| {job['name']} | {job['kind']} | {job['status']} | — | — | — | — | — | — | — | — | — |")
    lines += [
        "",
        "融合项在固定权重网格上使用同一验证集选择权重，属于验证集调参结果；部署成本为两模型总和，不能作为单模型参数/FLOPs成绩。",
        "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含读取、预处理与 NMS。",
        "单一数据划分与单一 seed 结果仅用于可重复初筛，不构成统计显著性结论。",
        REPORT_END,
    ]
    section = "\n".join(lines)
    old = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in old and REPORT_END in old:
        prefix, tail = old.split(REPORT_START, 1)
        _, suffix = tail.split(REPORT_END, 1)
        content = prefix.rstrip() + "\n\n" + section + suffix
    else:
        content = old.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def ingest(job_state: dict, spec: dict) -> None:
    payload = json.loads(spec["summary"].read_text(encoding="utf-8"))
    job_state["metrics"] = payload["metrics"]
    job_state["summary"] = {
        key: payload[key]
        for key in ("parameters", "weight_bytes", "flops_g", "inference_ms_b1_640", "inference_protocol")
    }
    if "selected" in payload:
        job_state["selected"] = payload["selected"]
        job_state["trials"] = payload["trials"]
    job_state["dual_target_passed"] = (
        float(payload["metrics"]["map50"]) > PAPER_TARGET["map50"]
        and float(payload["metrics"]["map5095"]) > PAPER_TARGET["map5095"]
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state["status"] = "running"
    state.setdefault("started_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    atomic_json(STATE_PATH, state)
    refresh_report(state)
    try:
        for spec, job_state in zip(JOBS, state["jobs"]):
            if spec["summary"].exists():
                ingest(job_state, spec)
                job_state["status"] = "completed"
                atomic_json(STATE_PATH, state)
                refresh_report(state)
                continue
            job_state["status"] = "running"
            job_state["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            state["current_job"] = spec["id"]
            atomic_json(STATE_PATH, state)
            refresh_report(state)
            stdout_path = LOG_DIR / f"{spec['id']}.out"
            stderr_path = LOG_DIR / f"{spec['id']}.err"
            with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
                completed = subprocess.run(spec["command"], cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            if completed.returncode != 0 or not spec["summary"].exists():
                raise RuntimeError(
                    f"{spec['id']} exited with code {completed.returncode}; expected summary {spec['summary']}"
                )
            ingest(job_state, spec)
            job_state["status"] = "completed"
            job_state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            atomic_json(STATE_PATH, state)
            refresh_report(state)
        state["status"] = "completed"
        state["current_job"] = None
    except Exception as exc:
        state["status"] = "failed"
        if state.get("current_job"):
            current = next(job for job in state["jobs"] if job["id"] == state["current_job"])
            current["status"] = "failed"
            current["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_json(STATE_PATH, state)
        refresh_report(state)


if __name__ == "__main__":
    main()
