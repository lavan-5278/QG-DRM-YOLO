"""Serial scratch ablation for three YOLOv8s DSwin integration schemes."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from train_hrw_yolov8s_module_singletons_queue import DATA_YAML, atomic_json, benchmark, read_best

OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_dswin_3schemes_seed42_scratch"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
PREDECESSOR = ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_drc2fv3_combinations_seed42_scratch/queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_DSWIN_3SCHEMES_START -->"
REPORT_END = "<!-- YOLOV8S_DSWIN_3SCHEMES_END -->"

JOBS = (
    ("sppf_dswin", "YOLOv8s + SPPFDSwin（方案 A）"),
    ("c2f_post_dswin", "YOLOv8s + C2fPostDSwin（方案 B）"),
    ("c2f_dswin", "YOLOv8s + C2fDSwin（方案 C）"),
)


def backbone(scheme: str) -> list[list]:
    c4 = "C2fPostDSwin" if scheme == "c2f_post_dswin" else "C2fDSwin" if scheme == "c2f_dswin" else "C2f"
    c4_args = [512, True, 1, 0.5, 8, 5, 1.0] if c4 != "C2f" else [512, True]
    c5_args = [1024, True, 1, 0.5, 8, 5, 1.0] if c4 != "C2f" else [1024, True]
    sppf = "SPPFDSwin" if scheme == "sppf_dswin" else "SPPF"
    sppf_args = [1024, 5, 8, 5, 1.0] if sppf == "SPPFDSwin" else [1024, 5]
    return [
        [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]], [-1, 6, "C2f", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, c4, c4_args], [-1, 1, "Conv", [1024, 3, 2]], [-1, 3, c4, c5_args], [-1, 1, sppf, sppf_args],
    ]


def model_dict(scheme: str) -> dict:
    return {
        "nc": 3, "scale": "s",
        "scales": {"s": [0.33, 0.50, 1024]},
        "backbone": backbone(scheme),
        "head": [
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 6], 1, "Concat", [1]], [-1, 3, "C2f", [512]],
            [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],
            [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]], [-1, 3, "C2f", [512]],
            [-1, 1, "Conv", [512, 3, 2]], [[-1, 9], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],
            [[15, 18, 21], 1, "Detect", [3]],
        ],
    }


def write_configs() -> dict[str, Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    result = {}
    for job_id, _ in JOBS:
        path = CONFIG_ROOT / f"yolov8s-{job_id}.yaml"
        path.write_text(yaml.safe_dump(model_dict(job_id), allow_unicode=True, sort_keys=False), encoding="utf-8")
        result[job_id] = path
    return result


def initial_state(paths: dict[str, Path]) -> dict:
    return {
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"},
        "design": {"attention": "pure-PyTorch DSwin-inspired deformable sliding-window attention", "heads": 8, "kernel": 5, "offset_range_factor": 1.0, "schemes": [job_id for job_id, _ in JOBS]},
        "queue_status": "pending", "running": None,
        "jobs": [{"id": job_id, "name": name, "yaml": str(paths[job_id]), "status": "pending", "run_dir": str(OUTPUT_ROOT / "native_runs" / f"yolov8s_{job_id}")} for job_id, name in JOBS],
    }


def load_state(paths: dict[str, Path]) -> dict:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state(paths)
    if [job.get("id") for job in state.get("jobs", [])] != [job_id for job_id, _ in JOBS]:
        raise RuntimeError("Unexpected DSwin queue definition")
    for job in state["jobs"]:
        job["yaml"] = str(paths[job["id"]])
    return state


def row(job: dict) -> str:
    metrics, summary = job.get("metrics"), job.get("summary")
    if job.get("status") != "completed" or not metrics or not summary:
        return f"| {job['name']} | {job.get('status', 'pending')} | — | — | — | — | — | — | — | — | — | — | — |"
    train = "/".join(f"{float(metrics[key]):.4f}" for key in ("train_box", "train_cls", "train_dfl"))
    val = "/".join(f"{float(metrics[key]):.4f}" for key in ("val_box", "val_cls", "val_dfl"))
    return f"| {job['name']} | completed | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes'] / 1024 / 1024:.2f} MB | {train} | {val} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |"


def refresh_report(state: dict) -> None:
    lines = [
        REPORT_START, "## YOLOv8s DSwin Attention 三方案独立实验", "",
        "DSwin 为纯 PyTorch 的检测适配：在滑动局部邻域内计算注意力，并用内容驱动的偏移对 K/V 特征进行可形变采样；它不是 DSwinIR 的逐行复现。三项均独立实验、不与其他改进模块叠加。",
        "方案 A：`SPPF → SPPFDSwin`；方案 B：Backbone P4/P5 的 `C2f → C2fPostDSwin`；方案 C：Backbone P4/P5 的 `C2f → C2fDSwin`。",
        "统一协议：5k_HRW、seed=42、从零训练且无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
        *(row(job) for job in state["jobs"]), "",
        "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。",
        "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END,
    ]
    section = "\n".join(lines)
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        prefix, tail = existing.split(REPORT_START, 1)
        _, suffix = tail.split(REPORT_END, 1)
        content = prefix.rstrip() + "\n\n" + section + suffix
    else:
        content = existing.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")


def wait_for_predecessor(state: dict) -> None:
    state["queue_status"] = "waiting_for_predecessor"
    atomic_json(STATE_PATH, state)
    refresh_report(state)
    while True:
        if not PREDECESSOR.exists():
            time.sleep(60)
            continue
        predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
        status = predecessor.get("queue_status")
        if status == "completed":
            time.sleep(10)
            return
        if status == "failed":
            raise RuntimeError(f"Predecessor queue failed: {predecessor.get('error', '')}")
        time.sleep(60)


def validate(paths: dict[str, Path]) -> None:
    for path in paths.values():
        model = DetectionModel(str(path), nc=3, verbose=False)
        model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = write_configs()
    validate(paths)
    state = load_state(paths)
    atomic_json(STATE_PATH, state)
    refresh_report(state)
    if args.dry_run:
        return
    try:
        wait_for_predecessor(state)
        state["queue_status"] = "running"
        atomic_json(STATE_PATH, state)
        refresh_report(state)
        for job in state["jobs"]:
            if job["status"] != "pending":
                continue
            run_dir = Path(job["run_dir"])
            job.update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            state["running"] = job["id"]
            atomic_json(STATE_PATH, state)
            refresh_report(state)
            result = YOLO(job["yaml"]).train(data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42, deterministic=True, device=0, workers=4, project=str(run_dir.parent), name=run_dir.name, exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True)
            final = Path(result.save_dir)
            best = final / "weights/best.pt"
            job["metrics"] = read_best(final)
            job["summary"] = benchmark(best)
            (final / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"]}, ensure_ascii=False, indent=2), encoding="utf-8")
            job.update({"status": "completed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            state["running"] = None
            atomic_json(STATE_PATH, state)
            refresh_report(state)
        state["queue_status"] = "completed"
    except Exception as exc:
        state["queue_status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_json(STATE_PATH, state)
        refresh_report(state)


if __name__ == "__main__":
    main()
