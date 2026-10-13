"""Train the P4-semantic-guided P3 refinement on the current best YOLOv8s design."""

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
from train_hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm import PAPER_TARGET, atomic_json

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3semantic.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3semantic_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3semantic"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_MLKCM_P3SEMANTIC_START -->"
REPORT_END = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_MLKCM_P3SEMANTIC_END -->"


def initial_state() -> dict:
    return {
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"},
        "comparison_target": {"model": "PFOM+MLKCM paper-structure reproduction", **PAPER_TARGET, "criterion": "both mAP50 and mAP50-95 must exceed target"},
        "design": {
            "base": "QG-AFPN + C4 RGDRC2fV3 + MLKCM(P3/P4/P5) + P3 local restoration",
            "replacement": "P3SemanticDetailRestore: P3 local detail conditioned on aligned P4 semantic context",
            "initialization": "zero residual scale and zero reliability-logit, exact P3 identity at initialization",
            "hypothesis": "P4 semantic context improves P3 class discrimination and mAP50 while retaining the proven localization margin",
            "preserved": "QG-AFPN, C4-only RGDRC2fV3, all three MLKCM blocks, Detect, data and training protocol",
        },
        "status": "pending",
        "job": {"id": "yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3semantic", "name": "YOLOv8s + QG-AFPN + C4 RGDRC2fV3 + MLKCM + P4-guided P3 restore", "yaml": str(MODEL_YAML), "run_dir": str(RUN_DIR), "status": "pending"},
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def adapter_states(best: Path) -> dict[str, dict]:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().eval()
    states = {name: module.adapter_state() for name, module in model.named_modules() if callable(getattr(module, "adapter_state", None))}
    classes = {module.__class__.__name__ for module in model.modules() if callable(getattr(module, "adapter_state", None))}
    expected = {"RGDRC2fV3", "P3SemanticDetailRestore"}
    if classes != expected or len(states) != 2:
        raise RuntimeError(f"Expected one RGDRC2fV3 and P3SemanticDetailRestore, got {classes} / {len(states)}")
    return states


def verify_structure_and_identity(model: YOLO) -> None:
    layers = model.model.model
    if layers[6].__class__.__name__ != "RGDRC2fV3" or layers[8].__class__.__name__ != "C2f":
        raise RuntimeError("C4 must be RGDRC2fV3 and C5 must remain native C2f")
    if [layers[index].__class__.__name__ for index in (26, 27, 28, 29)] != ["MLKCM", "MLKCM", "MLKCM", "P3SemanticDetailRestore"]:
        raise RuntimeError("Expected MLKCM at P3/P4/P5 followed by P4-guided P3 restore")
    c4 = layers[6]
    p3_restore = layers[29]
    c4.eval()
    p3_restore.eval()
    with torch.inference_mode():
        c4_input = torch.randn(1, c4.base.cv1.conv.in_channels, 40, 40)
        if not torch.equal(c4(c4_input), c4.base(c4_input)):
            raise RuntimeError("C4 zero-init identity failed")
        p3 = torch.randn(1, p3_restore.channels, 80, 80)
        p4 = torch.randn(1, p3_restore.context_channels, 40, 40)
        if not torch.equal(p3_restore(p3, p4), p3):
            raise RuntimeError("P3 semantic adapter zero-init identity failed")


def refresh_report(state: dict) -> None:
    job, metrics, summary = state["job"], state["job"].get("metrics"), state["job"].get("summary")
    lines = [
        REPORT_START,
        "## YOLOv8s QG-AFPN + C4 门控 DR-C2f v3 + MLKCM + P4 语义引导 P3 恢复",
        "",
        "动机：当前最优方案 mAP50-95=0.5777 已有目标余量，但 mAP50=0.7494 仍差 0.0020。将原 P3 局部恢复替换为 P4 语义引导的 P3 局部-语义残差，以提升 P3 类别判别；残差严格零初始化，其余结构与当前最优方案保持一致。",
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
        lines += ["", "学习到的残差适配器状态：", "", "```json", json.dumps(job["adapter_states"], ensure_ascii=False, indent=2), "```"]
    else:
        lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines)
    old = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in old and REPORT_END in old:
        prefix, tail = old.split(REPORT_START, 1)
        _, suffix = tail.split(REPORT_END, 1)
        new = prefix.rstrip() + "\n\n" + section + suffix
    else:
        new = old.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.write_text(new, encoding="utf-8")


def finalize(state: dict) -> None:
    job, best = state["job"], RUN_DIR / "weights/best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Cannot finalize without {best}")
    job["metrics"], job["summary"], job["adapter_states"] = read_best_metrics(RUN_DIR), benchmark_checkpoint(best), adapter_states(best)
    job["dual_target_passed"] = float(job["metrics"]["map50"]) > PAPER_TARGET["map50"] and float(job["metrics"]["map5095"]) > PAPER_TARGET["map5095"]
    (RUN_DIR / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"], "adapter_states": job["adapter_states"], "dual_target_passed": job["dual_target_passed"]}, ensure_ascii=False, indent=2), encoding="utf-8")
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
