"""Serial YOLOv8s scratch ablation for PFOM and MLKCM paper-structure modules."""
from __future__ import annotations

import argparse
import json
import shutil
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

OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_pfom_mlkcm_seed42_scratch"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
PREDECESSOR = ROOT / "runs/hrw_yolov8s_dswin_3schemes_seed42_scratch/queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_PFOM_MLKCM_START -->"
REPORT_END = "<!-- YOLOV8S_PFOM_MLKCM_END -->"

JOBS = (
    ("pfom", "YOLOv8s + PFOM（论文结构复现）", True, False),
    ("mlkcm", "YOLOv8s + MLKCM（论文结构复现）", False, True),
    ("pfom_mlkcm", "YOLOv8s + PFOM + MLKCM（论文结构复现）", True, True),
)


def feature_module(use_pfo: bool, channels: int, shortcut: bool = True) -> tuple[str, list]:
    return ("PFOM", [channels]) if use_pfo else ("C2f", [channels, shortcut])


def model_dict(use_pfo: bool, use_mlkcm: bool) -> dict:
    p2_name, p2_args = feature_module(False, 128)
    p3_name, p3_args = feature_module(use_pfo, 256)
    p4_name, p4_args = feature_module(use_pfo, 512)
    p5_name, p5_args = feature_module(use_pfo, 1024)
    backbone = [
        [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, p2_name, p2_args],
        [-1, 1, "Conv", [256, 3, 2]], [-1, 6, p3_name, p3_args], [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, p4_name, p4_args], [-1, 1, "Conv", [1024, 3, 2]], [-1, 3, p5_name, p5_args], [-1, 1, "SPPF", [1024, 5]],
    ]
    head = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 6], 1, "Concat", [1]], [-1, 3, p4_name, p4_args],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 4], 1, "Concat", [1]], [-1, 3, p3_name, p3_args],
    ]
    if use_mlkcm:
        head += [
            [-1, 1, "MLKCM", [256]], [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]],
            [-1, 3, p4_name, p4_args], [-1, 1, "MLKCM", [512]], [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 9], 1, "Concat", [1]], [-1, 3, p5_name, p5_args], [-1, 1, "MLKCM", [1024]], [[16, 20, 24], 1, "Detect", [3]],
        ]
    else:
        head += [
            [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]], [-1, 3, p4_name, p4_args],
            [-1, 1, "Conv", [512, 3, 2]], [[-1, 9], 1, "Concat", [1]], [-1, 3, p5_name, p5_args], [[15, 18, 21], 1, "Detect", [3]],
        ]
    return {"nc": 3, "scale": "s", "scales": {"s": [0.33, 0.50, 1024]}, "backbone": backbone, "head": head}


def write_configs() -> dict[str, Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    configs = {}
    for job_id, _, use_pfo, use_mlkcm in JOBS:
        path = CONFIG_ROOT / f"yolov8s-{job_id}.yaml"
        path.write_text(yaml.safe_dump(model_dict(use_pfo, use_mlkcm), allow_unicode=True, sort_keys=False), encoding="utf-8")
        configs[job_id] = path
    return configs


def initial_state(paths: dict[str, Path]) -> dict:
    return {
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"},
        "implementation": {"source": "paper-structure reproduction package", "dcp_kernels": [3, 5, 7], "dcp_dilations_assumption": [1, 2, 3], "pfom_gate": "sigmoid-bounded GhostModule multiplicative gate for scratch-SGD stability", "note": "The supplied package is not official author code; undocumented details remain explicit reproduction assumptions."},
        "queue_status": "pending", "running": None,
        "jobs": [{"id": job_id, "name": name, "yaml": str(paths[job_id]), "status": "pending", "run_dir": str(OUTPUT_ROOT / "native_runs" / f"yolov8s_{job_id}")} for job_id, name, _, _ in JOBS],
    }


def load_state(paths: dict[str, Path]) -> dict:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state(paths)
    if [job.get("id") for job in state.get("jobs", [])] != [job_id for job_id, *_ in JOBS]:
        raise RuntimeError("Unexpected PFOM/MLKCM queue definition")
    for job in state["jobs"]:
        job["yaml"] = str(paths[job["id"]])
    return state


def report_row(job: dict) -> str:
    metrics, summary = job.get("metrics"), job.get("summary")
    if job.get("status") != "completed" or not metrics or not summary:
        return f"| {job['name']} | {job.get('status', 'pending')} | — | — | — | — | — | — | — | — | — | — | — |"
    train = "/".join(f"{float(metrics[key]):.4f}" for key in ("train_box", "train_cls", "train_dfl"))
    val = "/".join(f"{float(metrics[key]):.4f}" for key in ("val_box", "val_cls", "val_dfl"))
    return f"| {job['name']} | completed | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes'] / 1024 / 1024:.2f} MB | {train} | {val} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |"


def refresh_report(state: dict) -> None:
    lines = [
        REPORT_START, "## YOLOv8s PFOM / MLKCM 论文结构复现实验", "",
        "本节使用提供的论文结构复现包进行三项独立消融：PFOM 替换 Backbone 与 Neck 的 C2f 角色；MLKCM 放在 P3/P4/P5 Detect 输入之前；全模型同时采用两者。",
        "说明：该实现不是论文作者官方源代码。论文未公开 DCP 具体膨胀率等细节，本复现包固定使用 kernels=(3,5,7)、dilations=(1,2,3)，因此结果应解释为该明确实现的复现实验。",
        "稳定性说明：初次 PFOM 训练在 epoch 165 的训练 batch 出现 NaN。复训将 GhostModule 的乘性门控显式改为 sigmoid 有界门控；该实现选择也属于复现假设的一部分。",
        "统一协议：5k_HRW、seed=42、从零训练且无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
        *(report_row(job) for job in state["jobs"]), "",
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
        if predecessor.get("queue_status") == "completed":
            time.sleep(10)
            return
        if predecessor.get("queue_status") == "failed":
            raise RuntimeError(f"DSwin predecessor failed: {predecessor.get('error', '')}")
        time.sleep(60)


def validate(paths: dict[str, Path]) -> None:
    for path in paths.values():
        DetectionModel(str(path), nc=3, verbose=False).eval()


def restart_failed_queue(state: dict) -> dict:
    """Archive invalid attempt outputs and reset unfinished jobs for a scratch retry."""
    if state.get("queue_status") != "failed":
        raise RuntimeError("--restart-failed requires a failed PFOM/MLKCM queue")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for job in state["jobs"]:
        if job.get("status") == "completed":
            continue
        run_dir = Path(job["run_dir"])
        if run_dir.exists():
            archive = run_dir.with_name(f"{run_dir.name}_failed_nan_{stamp}")
            if archive.exists():
                raise RuntimeError(f"Refusing to overwrite existing archive: {archive}")
            shutil.move(str(run_dir), str(archive))
            job["archived_failed_run"] = str(archive)
        for key in ("metrics", "summary", "started_at", "finished_at"):
            job.pop(key, None)
        job["status"] = "pending"
    state.pop("error", None)
    state.pop("finished_at", None)
    state["queue_status"] = "pending"
    state["running"] = None
    state["restart_note"] = (
        "Previous PFOM attempt was archived after NaN/Inf. "
        "PFOM GhostModule gates now use bounded sigmoid gating for scratch-SGD stability."
    )
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart-failed", action="store_true", help="archive failed outputs and rerun unfinished jobs from scratch")
    args = parser.parse_args()
    paths = write_configs()
    validate(paths)
    state = load_state(paths)
    if args.restart_failed:
        state = restart_failed_queue(state)
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
