"""Run the three YOLOv8n + DSA ablations after the active DSR experiment.

All jobs use the established 5k_HRW scratch protocol.  The full DSA variant
alone disables Mosaic/MixUp and uses the social branch's progressive
GT-to-prediction RoI mixing schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(r"D:\YH_Project\DSA-Det\dsa_det_ssr_pkg")
for path in (ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dsa_det_ssr  # noqa: F401 - registers the dsa-det task with Ultralytics.
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8n_dsa_seed42_scratch"
DSR_FINALIZATION = ROOT / "runs/hrw_yolov8n_dsr_seed42_scratch/native_runs/yolov8n_dsr/native_summary.json"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = OUTPUT_ROOT / "5k_HRW_seed42_DSA补充实验结果.md"

JOBS = (
    ("yolov8n_dsam_hgp", "YOLOv8n + DSAM/HGP", ROOT / "ultralytics/cfg/models/v8/yolov8n-dsam-hgp.yaml", False),
    ("yolov8n_ssr_ssl", "YOLOv8n + SSR/SSL", ROOT / "ultralytics/cfg/models/v8/yolov8n-ssr-ssl.yaml", False),
    ("yolov8n_dsa_full", "完整 DSA", ROOT / "ultralytics/cfg/models/v8/yolov8n-dsa-full.yaml", True),
)


def atomic_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def initial_state() -> dict:
    return {
        "protocol": {
            "dataset": "5k_HRW",
            "seed": 42,
            "pretrained": False,
            "batch": 8,
            "lr0": 0.01,
            "epochs": 300,
            "imgsz": 640,
            "optimizer": "SGD",
        },
        "jobs": [{"id": item[0], "name": item[1], "status": "pending"} for item in JOBS],
        "running": None,
    }


def common_train_kwargs(run_dir: Path) -> dict:
    return {
        "data": str(DATA_YAML), "epochs": 300, "batch": 8, "imgsz": 640,
        "lr0": 0.01, "optimizer": "SGD", "seed": 42, "deterministic": True,
        "device": 0, "workers": 4, "project": str(run_dir.parent), "name": run_dir.name,
        "exist_ok": True, "pretrained": False, "val": True, "save": True,
        "plots": True, "verbose": True,
    }


def read_best_metrics(run_dir: Path) -> dict[str, str]:
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No metrics in {run_dir / 'results.csv'}")
    map95_key = next(key for key in rows[0] if "mAP50-95" in key)
    best = max(rows, key=lambda row: float(row[map95_key]))
    result = {"epoch": best["epoch"], "precision": best["metrics/precision(B)"], "recall": best["metrics/recall(B)"], "map50": best["metrics/mAP50(B)"], "map5095": best[map95_key]}
    for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"):
        result[key] = best.get(key, "")
    return result


def benchmark_checkpoint(best: Path) -> dict:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().cuda().eval()
    params = sum(parameter.numel() for parameter in model.parameters())
    flops = float(get_flops(model, imgsz=640))
    sample = torch.zeros((1, 3, 640, 640), device="cuda")
    with torch.inference_mode():
        for _ in range(50):
            model(sample)
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(200):
            start = time.perf_counter()
            model(sample)
            torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - start) * 1000)
    return {
        "parameters": params,
        "weight_bytes": best.stat().st_size,
        "flops_g": flops,
        "inference_ms_b1_640": statistics.mean(elapsed),
        "inference_protocol": "RTX 4080 SUPER, CUDA, FP32, batch=1, 640x640, 50 warmup + 200 forward passes",
    }


def refresh_report(state: dict) -> None:
    lines = [
        "# 5k_HRW seed42：YOLOv8n + DSA 补充实验",
        "",
        "统一设置：从零训练、无预训练权重、seed=42、batch=8、lr0=0.01、SGD、300 epoch、640×640。完整 DSA 额外关闭 Mosaic/MixUp，并采用 GT→预测框渐进混合的社会分支 RoI 训练。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for job in state["jobs"]:
        summary = job.get("summary")
        metrics = job.get("metrics")
        if job["status"] == "completed" and summary and metrics:
            lines.append(
                f"| {job['name']} | 完成 | {int(metrics['epoch']) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes'] / 1024 / 1024:.2f} MB | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |"
            )
        else:
            lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — |")
    lines += ["", "注：本轮仅使用单一数据划分与单个 seed，结果用于可重复的初筛比较，不能替代多 seed 统计检验。"]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_queue(wait_for_dsr: bool) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if wait_for_dsr and not DSR_FINALIZATION.exists():
        raise RuntimeError(f"DSR finalization has not completed: {DSR_FINALIZATION}")
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()
    refresh_report(state)
    for job, spec in zip(state["jobs"], JOBS):
        job_id, _display, yaml_path, disable_augmentation = spec
        assert job["id"] == job_id
        run_dir = OUTPUT_ROOT / "native_runs" / job_id
        if job["status"] == "completed":
            continue
        job.update({"status": "running", "run_dir": str(run_dir)})
        state["running"] = job_id
        atomic_json(STATE_PATH, state)
        try:
            kwargs = common_train_kwargs(run_dir)
            if disable_augmentation:
                kwargs.update({"mosaic": 0.0, "mixup": 0.0, "close_mosaic": 0})
            result = YOLO(str(yaml_path)).train(**kwargs)
            final_dir = Path(result.save_dir)
            best = final_dir / "weights" / "best.pt"
            job["metrics"] = read_best_metrics(final_dir)
            job["summary"] = benchmark_checkpoint(best)
            job["status"] = "completed"
        except Exception as exc:
            job.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            state["running"] = None
            atomic_json(STATE_PATH, state)
            refresh_report(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-dsr-finalization", action="store_true")
    args = parser.parse_args()
    run_queue(args.wait_for_dsr_finalization)


if __name__ == "__main__":
    main()
