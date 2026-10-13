"""Train four independent YOLOv8s single-module scratch ablations on 5k_HRW serially."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.models.yolo.dsr.model import DSRDetectionModel
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
DSR_SOURCE_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8n-dsr.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_LGC_DRSATA_QGAFPN_DSR_SINGLETONS_START -->"
REPORT_END = "<!-- YOLOV8S_LGC_DRSATA_QGAFPN_DSR_SINGLETONS_END -->"
JOBS = (("lgc", "YOLOv8s + LGC"), ("drsata", "YOLOv8s + DR-SATA"), ("qgafpn", "YOLOv8s + QG-AFPN"), ("dsr", "YOLOv8s + DSR"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def dsr_config() -> dict:
    config = yaml.safe_load(DSR_SOURCE_YAML.read_text(encoding="utf-8")).get("dsr")
    if not isinstance(config, dict):
        raise RuntimeError(f"Missing dsr configuration in {DSR_SOURCE_YAML}")
    return config


def model_dict(active: str) -> dict:
    """Build the YOLOv8s graph, preserving each existing YOLOv8n singleton's insertion contract."""
    if active not in {job_id for job_id, _ in JOBS}:
        raise ValueError(active)
    backbone = [
        [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]], [-1, 6, "C2f", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, "C2f", [512, True]],
    ]
    if active == "lgc": backbone.append([-1, 1, "LGC", []])
    if active == "drsata": backbone.append([-1, 1, "DRSATA", [64, 16, 16, 6]])
    c4 = len(backbone) - 1
    backbone += [[-1, 1, "Conv", [1024, 3, 2]], [-1, 3, "C2f", [1024, True]], [-1, 1, "SPPF", [1024, 5]]]
    if active == "lgc": backbone.append([-1, 1, "LGC", []])
    if active == "drsata": backbone.append([-1, 1, "DRSATA", [64, 16, 16, 6]])
    c5 = len(backbone) - 1
    head: list = []
    def add(from_, module, args):
        head.append([from_, 1, module, args]); return len(backbone) + len(head) - 1
    add(-1, "nn.Upsample", [None, 2, "nearest"]); add([-1, c4], "Concat", [1]); p4_td = add(-1, "C2f", [512])
    add(-1, "nn.Upsample", [None, 2, "nearest"]); add([-1, 4], "Concat", [1]); p3 = add(-1, "C2f", [256])
    add(-1, "Conv", [256, 3, 2]); add([-1, p4_td], "Concat", [1]); p4 = add(-1, "C2f", [512])
    add(-1, "Conv", [512, 3, 2]); add([-1, c5], "Concat", [1]); p5 = add(-1, "C2f", [1024])
    if active == "qgafpn":
        qg = add([p3, p4, p5], "QGAFPN", []); p3 = add(-1, "Index", [128, 0]); p4 = add(qg, "Index", [256, 1]); p5 = add(qg, "Index", [512, 2])
    add([p3, p4, p5], "DSRDetect" if active == "dsr" else "Detect", [3, "dsr"] if active == "dsr" else [3])
    model = {"nc": 3, "scales": {"s": [0.33, 0.50, 1024]}, "backbone": backbone, "head": head}
    if active == "dsr": model["dsr"] = dsr_config()
    return model


def write_configs() -> dict[str, Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True); paths = {}
    for job_id, _ in JOBS:
        path = CONFIG_ROOT / f"yolov8s-{job_id}.yaml"
        path.write_text(yaml.safe_dump(model_dict(job_id), allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths[job_id] = path
    return paths


def new_state(paths: dict[str, Path]) -> dict:
    return {"protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"}, "design": {"type": "four independent YOLOv8s singleton ablations", "order": [job_id for job_id, _ in JOBS]}, "queue_status": "pending", "running": None, "jobs": [{"id": job_id, "name": name, "yaml": str(paths[job_id]), "status": "pending", "run_dir": str(OUTPUT_ROOT / "native_runs" / f"yolov8s_{job_id}")} for job_id, name in JOBS]}


def load_state(paths: dict[str, Path]) -> dict:
    if not STATE_PATH.exists(): return new_state(paths)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if [job.get("id") for job in state.get("jobs", [])] != [job_id for job_id, _ in JOBS]: raise RuntimeError("Unexpected singleton queue definition")
    for job in state["jobs"]: job["yaml"] = str(paths[job["id"]])
    return state


def read_best(run_dir: Path) -> dict[str, str]:
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as file: rows = list(csv.DictReader(file))
    if not rows: raise RuntimeError("results.csv has no rows")
    map95 = next(key for key in rows[0] if "mAP50-95" in key); row = max(rows, key=lambda item: float(item[map95]))
    fields = {"epoch": "epoch", "precision": "metrics/precision(B)", "recall": "metrics/recall(B)", "map50": "metrics/mAP50(B)", "map5095": map95, "train_box": "train/box_loss", "train_cls": "train/cls_loss", "train_dfl": "train/dfl_loss", "val_box": "val/box_loss", "val_cls": "val/cls_loss", "val_dfl": "val/dfl_loss"}
    return {name: row.get(key, "") for name, key in fields.items()}


def benchmark(best: Path) -> dict:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().cuda().eval(); x = torch.zeros((1, 3, 640, 640), device="cuda")
    with torch.inference_mode():
        for _ in range(50): model(x)
        torch.cuda.synchronize(); timings = []
        for _ in range(200):
            start = time.perf_counter(); model(x); torch.cuda.synchronize(); timings.append((time.perf_counter()-start)*1000)
    return {"parameters": sum(parameter.numel() for parameter in model.parameters()), "weight_bytes": best.stat().st_size, "flops_g": float(get_flops(model, imgsz=640)), "inference_ms_b1_640": statistics.mean(timings), "inference_protocol": "RTX 4080 SUPER, CUDA, FP32, batch=1, 640x640, 50 warmup + 200 forward passes"}


def report(state: dict) -> None:
    lines = [REPORT_START, "## YOLOv8s LGC / DR-SATA / QG-AFPN / DSR 单模块实验", "", "四项为独立 YOLOv8s 实验：LGC 与 DR-SATA 分别插入 C4/C5，QG-AFPN 位于三尺度 Neck 融合处，DSR 替换 Detect；不叠加模块。", "", "统一协议：5k_HRW、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "", "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for job in state["jobs"]:
        metrics, summary = job.get("metrics"), job.get("summary")
        if job["status"] == "completed" and metrics and summary:
            lines.append(f"| {job['name']} | completed | {int(float(metrics['epoch']))+1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
        else: lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines); existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        prefix, tail = existing.split(REPORT_START, 1); _, suffix = tail.split(REPORT_END, 1); content = prefix.rstrip()+"\n\n"+section+suffix
    else: content = existing.rstrip()+"\n\n"+section+"\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(content, encoding="utf-8")


def validate(paths: dict[str, Path]) -> None:
    for job_id, _ in JOBS:
        (DSRDetectionModel if job_id == "dsr" else DetectionModel)(str(paths[job_id]), nc=3, verbose=False)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    paths = write_configs(); validate(paths); state = load_state(paths); atomic_json(STATE_PATH, state); report(state)
    if args.dry_run: return
    try:
        state["queue_status"] = "running"; atomic_json(STATE_PATH, state); report(state)
        for job in state["jobs"]:
            if job["status"] == "completed": continue
            job.update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}); state["running"] = job["id"]; atomic_json(STATE_PATH, state); report(state)
            run_dir = Path(job["run_dir"]); trainer = DSRDetectionTrainer if job["id"] == "dsr" else None
            result = YOLO(job["yaml"]).train(trainer=trainer, data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42, deterministic=True, device=0, workers=4, project=str(run_dir.parent), name=run_dir.name, exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True)
            final = Path(result.save_dir); best = final / "weights/best.pt"; job["metrics"] = read_best(final); job["summary"] = benchmark(best)
            (final / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"]}, ensure_ascii=False, indent=2), encoding="utf-8")
            job.update({"status": "completed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}); state["running"] = None; atomic_json(STATE_PATH, state); report(state)
        state["queue_status"] = "completed"
    except Exception as exc:
        state["queue_status"] = "failed"; state["error"] = f"{type(exc).__name__}: {exc}"; raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S"); atomic_json(STATE_PATH, state); report(state)


if __name__ == "__main__": main()
