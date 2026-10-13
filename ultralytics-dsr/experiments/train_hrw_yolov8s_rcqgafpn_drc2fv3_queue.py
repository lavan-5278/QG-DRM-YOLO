"""Serial scratch ablation for relay-calibrated competitive QG-AFPN on YOLOv8s + DR-C2f v3."""
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

OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_rcqgafpn_drc2fv3_seed42_scratch"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_RCQGAFPN_DRC2FV3_START -->"
REPORT_END = "<!-- YOLOV8S_RCQGAFPN_DRC2FV3_END -->"

JOBS = (
    ("relay_calibration", "YOLOv8s + DR-C2f v3 + Relay-Calibrated QG-AFPN", "RCAQGAFPN"),
    ("competitive_fusion", "YOLOv8s + DR-C2f v3 + Competitive QG-AFPN", "CQGAFPN"),
    ("rc_qgafpn", "YOLOv8s + DR-C2f v3 + RC-QG-AFPN", "RCQGAFPN"),
)


def model_dict(neck: str) -> dict:
    """Use DR-C2f v3 only for four Backbone C2f blocks; Neck/Detect contracts remain YOLOv8s-compatible."""
    backbone = [
        [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, "DRC2fV3", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]], [-1, 6, "DRC2fV3", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, "DRC2fV3", [512, True]], [-1, 1, "Conv", [1024, 3, 2]], [-1, 3, "DRC2fV3", [1024, True]],
        [-1, 1, "SPPF", [1024, 5]],
    ]
    head = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 6], 1, "Concat", [1]], [-1, 1, "C2f", [512]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 4], 1, "Concat", [1]], [-1, 1, "C2f", [256]],
        [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]], [-1, 1, "C2f", [512]],
        [-1, 1, "Conv", [512, 3, 2]], [[-1, 9], 1, "Concat", [1]], [-1, 1, "C2f", [1024]],
        [[15, 18, 21], 1, neck, []], [-1, 1, "Index", [128, 0]], [22, 1, "Index", [256, 1]],
        [22, 1, "Index", [512, 2]], [[23, 24, 25], 1, "Detect", [3]],
    ]
    return {"nc": 3, "scales": {"s": [0.33, 0.50, 1024]}, "backbone": backbone, "head": head}


def write_configs() -> dict[str, Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for job_id, _, neck in JOBS:
        path = CONFIG_ROOT / f"yolov8s-{job_id}.yaml"
        path.write_text(yaml.safe_dump(model_dict(neck), allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths[job_id] = path
    return paths


def initial_state(paths: dict[str, Path]) -> dict:
    return {
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"},
        "design": {
            "base": "YOLOv8s + four Backbone DRC2fV3 blocks + QG-AFPN feature pyramid",
            "order": ["relay calibration", "competitive fusion", "full RC-QG-AFPN"],
            "relay_calibration": "zero-initialized residual, bounded per-channel calibration before the Neck",
            "competitive_fusion": "zero-initialized two-source softmax competition with 2*softmax neutral initialization",
            "full": "relay calibration + competitive fusion + per-location channel cosine agreement",
        },
        "queue_status": "pending",
        "running": None,
        "jobs": [
            {"id": job_id, "name": name, "neck": neck, "yaml": str(paths[job_id]), "status": "pending", "run_dir": str(OUTPUT_ROOT / "native_runs" / f"yolov8s_{job_id}")}
            for job_id, name, neck in JOBS
        ],
    }


def load_state(paths: dict[str, Path]) -> dict:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state(paths)
    if [job.get("id") for job in state.get("jobs", [])] != [job_id for job_id, _, _ in JOBS]:
        raise RuntimeError("Unexpected RC-QG-AFPN queue definition")
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
        REPORT_START,
        "## YOLOv8s DR-C2f v3 + RC-QG-AFPN 定向优化实验",
        "",
        "动机：既有 QG-AFPN + DR-C2f v3 提升 mAP50-95 但使 Precision 降低。本节按顺序检验中继统计校准、跨尺度竞争注入及完整 RC-QG-AFPN。三项均为独立实验，不叠加 DSR、PFOM 或 MLKCM。",
        "统一协议：5k_HRW、seed=42、从零训练且无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。DR-C2f v3 仅替换四个 Backbone C2f；Neck 的普通 C2f、Detect 与数据增强保持不变。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
        *(row(job) for job in state["jobs"]),
        "",
        "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。",
        "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。",
        REPORT_END,
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


def validate(paths: dict[str, Path]) -> None:
    for path in paths.values():
        DetectionModel(str(path), nc=3, verbose=False).eval()


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
            result = YOLO(job["yaml"]).train(
                data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42,
                deterministic=True, device=0, workers=4, project=str(run_dir.parent), name=run_dir.name,
                exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True,
            )
            final = Path(result.save_dir)
            best = final / "weights/best.pt"
            job["metrics"] = read_best(final)
            job["summary"] = benchmark(best)
            (final / "native_summary.json").write_text(
                json.dumps({"model": job["id"], "status": "completed", **job["summary"]}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
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
