"""Train the exact current-best module topology at YOLOv8n scale under the fixed 5k_HRW protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from train_hrw_yolov8s_dr_c2f_v3 import benchmark_checkpoint, read_best_metrics
from train_hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm import PAPER_TARGET, atomic_json

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8n-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8n_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8n_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8N_CURRENT_BEST_TOPOLOGY_START -->"
REPORT_END = "<!-- YOLOV8N_CURRENT_BEST_TOPOLOGY_END -->"


def initial_state() -> dict:
    return {
        "status": "pending",
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "optimizer": "SGD", "lr0": 0.01, "epochs": 300, "imgsz": 640},
        "architecture": "YOLOv8n scale; QG-AFPN + C4-only RGDRC2fV3 + MLKCM(P3/P4/P5) + P3DetailRestore; no additional modules",
        "comparison_target": {"PFOM+MLKCM paper reproduction": PAPER_TARGET},
        "job": {"id": RUN_DIR.name, "name": "YOLOv8n 当前最优结构迁移", "status": "pending", "run_dir": str(RUN_DIR), "yaml": str(MODEL_YAML)},
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def verify_structure(model: YOLO) -> None:
    layers = model.model.model
    expected = ["MLKCM", "MLKCM", "MLKCM", "P3DetailRestore"]
    if layers[6].__class__.__name__ != "RGDRC2fV3" or layers[8].__class__.__name__ != "C2f":
        raise RuntimeError("Expected C4-only RGDRC2fV3 while preserving native C5 C2f")
    if [layers[index].__class__.__name__ for index in (26, 27, 28, 29)] != expected:
        raise RuntimeError("Expected MLKCM(P3/P4/P5) followed by P3DetailRestore")
    if list(map(float, layers[-1].stride)) != [8, 16, 32]:
        raise RuntimeError(f"Expected P3/P4/P5 Detect strides [8, 16, 32], got {layers[-1].stride}")


def refresh_report(state: dict) -> None:
    job, metrics, summary = state["job"], state["job"].get("metrics"), state["job"].get("summary")
    lines = [
        REPORT_START,
        "## YOLOv8n 当前最优结构迁移实验",
        "",
        "结构保持 YOLOv8s 当前最优方案的模块拓扑：QG-AFPN、仅 Backbone C4 的 RGDRC2fV3、P3/P4/P5 MLKCM 和 P3DetailRestore；仅按 YOLOv8n 宽度缩放，未叠加其他模块。",
        "",
        "统一协议：5k_HRW、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    if job["status"] == "completed" and metrics and summary:
        passed = float(metrics["map50"]) > PAPER_TARGET["map50"] and float(metrics["map5095"]) > PAPER_TARGET["map5095"]
        status = "completed / 双指标超过目标" if passed else "completed / 未双指标超过目标"
        lines.append(f"| {job['name']} | {status} | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
    else:
        lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复初筛，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines)
    old = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in old and REPORT_END in old:
        prefix, tail = old.split(REPORT_START, 1)
        _, suffix = tail.split(REPORT_END, 1)
        updated = prefix.rstrip() + "\n\n" + section + suffix
    else:
        updated = old.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.write_text(updated, encoding="utf-8")


def finalize(state: dict) -> None:
    best = RUN_DIR / "weights/best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best}")
    job = state["job"]
    job["metrics"], job["summary"] = read_best_metrics(RUN_DIR), benchmark_checkpoint(best)
    job["dual_target_passed"] = float(job["metrics"]["map50"]) > PAPER_TARGET["map50"] and float(job["metrics"]["map5095"]) > PAPER_TARGET["map5095"]
    (RUN_DIR / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", "architecture": state["architecture"], "metrics": job["metrics"], **job["summary"], "dual_target_passed": job["dual_target_passed"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    job["status"] = state["status"] = "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    state, model = load_state(), YOLO(str(MODEL_YAML))
    verify_structure(model)
    refresh_report(state)
    if args.dry_run:
        model.info(verbose=False)
        return
    try:
        if args.finalize_only:
            finalize(state)
            return
        if state["job"]["status"] == "completed":
            return
        state["status"] = state["job"]["status"] = "running"
        state["job"]["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_json(STATE_PATH, state)
        refresh_report(state)
        model.train(data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42, deterministic=True, device=0, workers=4, project=str(RUN_DIR.parent), name=RUN_DIR.name, exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True)
        finalize(state)
    except Exception as exc:
        state["status"] = state["job"]["status"] = "failed"
        state["job"]["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_json(STATE_PATH, state)
        refresh_report(state)


if __name__ == "__main__":
    main()
