"""Train the first targeted challenger to the PFOM+MLKCM reproduction result."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics import YOLO
from train_hrw_yolov8s_dr_c2f_v3 import benchmark_checkpoint, read_best_metrics

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_MLKCM_START -->"
REPORT_END = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_MLKCM_END -->"
PAPER_TARGET = {"map50": 0.7514, "map5095": 0.5689}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def initial_state() -> dict:
    return {
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"},
        "comparison_target": {"model": "PFOM+MLKCM paper-structure reproduction", **PAPER_TARGET, "criterion": "both mAP50 and mAP50-95 must exceed target"},
        "design": {"neck": "QG-AFPN", "backbone_adapter": "C4-only zero-init reliability-gated DR-C2f v3", "detect_input": "MLKCM at P3/P4/P5 after neck construction", "preserved": "P2/P3/C5 native C2f; Detect unchanged", "excluded_modules": ["LGC", "DR-SATA", "DSR", "PFOM"]},
        "status": "pending",
        "job": {"id": "yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm", "name": "YOLOv8s + QG-AFPN + C4 RGDRC2fV3 + MLKCM", "yaml": str(MODEL_YAML), "run_dir": str(RUN_DIR), "status": "pending"},
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def adapter_state(best: Path) -> dict[str, dict]:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().eval()
    result = {name: module.adapter_state() for name, module in model.named_modules() if callable(getattr(module, "adapter_state", None))}
    if len(result) != 1:
        raise RuntimeError(f"Expected exactly one RGDRC2fV3, found {len(result)}")
    return result


def verify_structure_and_identity(model: YOLO) -> None:
    modules = model.model.model
    adapters = [(index, item) for index, item in enumerate(modules) if item.__class__.__name__ == "RGDRC2fV3"]
    if len(adapters) != 1 or adapters[0][0] != 6:
        raise RuntimeError(f"Expected one C4 RGDRC2fV3 at layer 6, got {[index for index, _ in adapters]}")
    if modules[8].__class__.__name__ != "C2f":
        raise RuntimeError("C5 must remain native C2f")
    if [modules[index].__class__.__name__ for index in (26, 27, 28)] != ["MLKCM"] * 3:
        raise RuntimeError("MLKCM must refine P3, P4 and P5 exactly once")
    adapter = adapters[0][1].eval()
    sample = torch.randn(1, adapter.base.cv1.conv.in_channels, 40, 40)
    with torch.inference_mode():
        expected, actual = adapter.base(sample), adapter(sample)
    if not torch.equal(actual, expected):
        raise RuntimeError(f"Zero-init identity failed, max error={(actual - expected).abs().max().item()}")


def refresh_report(state: dict) -> None:
    job, metrics, summary = state["job"], state["job"].get("metrics"), state["job"].get("summary")
    lines = [
        REPORT_START, "## YOLOv8s QG-AFPN + C4 门控 DR-C2f v3 + MLKCM 定向挑战实验", "",
        "目的：以 PFOM+MLKCM 论文结构复现实验（mAP50=0.7514、mAP50-95=0.5689）为双指标硬门槛。保留当前 mAP50-95 较强的 QG-AFPN 与 C4 零初始化可靠性门控 DR-C2f v3；MLKCM 只作用于 QG-AFPN 输出的三尺度 Detect 输入，针对误差诊断中的候选覆盖、漏检与定位不足。",
        "", "统一协议：5k_HRW、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    if job["status"] == "completed" and metrics and summary:
        passes = float(metrics["map50"]) > PAPER_TARGET["map50"] and float(metrics["map5095"]) > PAPER_TARGET["map5095"]
        lines.append(f"| {job['name']} | {'completed / 双指标超过目标' if passes else 'completed / 未双指标超过目标'} | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes'] / 1024 / 1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
        lines += ["", "学习到的 C4 适配器状态：", "", "```json", json.dumps(job["adapter_state"], ensure_ascii=False, indent=2), "```"]
    else:
        lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines)
    existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        prefix, tail = existing.split(REPORT_START, 1)
        _, suffix = tail.split(REPORT_END, 1)
        content = prefix.rstrip() + "\n\n" + section + suffix
    else:
        content = existing.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.write_text(content, encoding="utf-8")


def finalize(state: dict) -> None:
    job, best = state["job"], RUN_DIR / "weights/best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Cannot finalize without {best}")
    job["metrics"], job["summary"], job["adapter_state"] = read_best_metrics(RUN_DIR), benchmark_checkpoint(best), adapter_state(best)
    job["dual_target_passed"] = float(job["metrics"]["map50"]) > PAPER_TARGET["map50"] and float(job["metrics"]["map5095"]) > PAPER_TARGET["map5095"]
    (RUN_DIR / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"], "adapter_state": job["adapter_state"], "dual_target_passed": job["dual_target_passed"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    job["status"], state["status"] = "completed", "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    state, model = load_state(), YOLO(str(MODEL_YAML))
    verify_structure_and_identity(model)
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
