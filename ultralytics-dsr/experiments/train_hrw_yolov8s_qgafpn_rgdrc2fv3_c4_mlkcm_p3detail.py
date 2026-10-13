"""Train all-scale MLKCM plus zero-init P3 detail restoration as the next challenger."""

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
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-mlkcm-p3detail.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_MLKCM_P3DETAIL_START -->"
REPORT_END = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_MLKCM_P3DETAIL_END -->"


def initial_state() -> dict:
    return {
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"},
        "comparison_target": {"model": "PFOM+MLKCM paper-structure reproduction", **PAPER_TARGET, "criterion": "both mAP50 and mAP50-95 must exceed target"},
        "design": {"neck": "QG-AFPN", "backbone_adapter": "C4-only zero-init reliability-gated DR-C2f v3", "detect_input": "MLKCM at QG P3/P4/P5 outputs", "addition": "post-MLKCM P3DetailRestore with zero-init residual scale", "rationale": "all-scale MLKCM has mAP50-95 margin; historical P3 detail restoration recovers mAP50", "excluded_modules": ["LGC", "DR-SATA", "DSR", "PFOM"]},
        "status": "pending",
        "job": {"id": "yolov8s_qgafpn_rgdrc2fv3_c4_mlkcm_p3detail", "name": "YOLOv8s + QG-AFPN + C4 RGDRC2fV3 + MLKCM + P3DetailRestore", "yaml": str(MODEL_YAML), "run_dir": str(RUN_DIR), "status": "pending"},
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def adapter_states(best: Path) -> dict[str, dict]:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().eval()
    result = {name: module.adapter_state() for name, module in model.named_modules() if callable(getattr(module, "adapter_state", None))}
    classes = {module.__class__.__name__ for module in model.modules() if callable(getattr(module, "adapter_state", None))}
    if classes != {"RGDRC2fV3", "P3DetailRestore"} or len(result) != 2:
        raise RuntimeError(f"Expected one RGDRC2fV3 and P3DetailRestore, got {classes} / {len(result)}")
    return result


def verify_structure_and_identity(model: YOLO) -> None:
    layers = model.model.model
    if layers[6].__class__.__name__ != "RGDRC2fV3" or layers[8].__class__.__name__ != "C2f":
        raise RuntimeError("C4 must be RGDRC2fV3 and C5 must remain native C2f")
    if [layers[index].__class__.__name__ for index in (26, 27, 28, 29)] != ["MLKCM", "MLKCM", "MLKCM", "P3DetailRestore"]:
        raise RuntimeError("Expected MLKCM at P3/P4/P5 followed by P3DetailRestore")
    checks = ((layers[6], torch.randn(1, layers[6].base.cv1.conv.in_channels, 40, 40), "c4"), (layers[29], torch.randn(1, 64, 80, 80), "p3"))
    for module, sample, kind in checks:
        module.eval()
        with torch.inference_mode():
            output = module(sample)
            expected = module.base(sample) if kind == "c4" else sample
        if not torch.equal(output, expected):
            raise RuntimeError(f"{kind} zero-init identity failed: {(output-expected).abs().max().item()}")


def refresh_report(state: dict) -> None:
    job, metrics, summary = state["job"], state["job"].get("metrics"), state["job"].get("summary")
    lines = [
        REPORT_START, "## YOLOv8s QG-AFPN + C4 门控 DR-C2f v3 + MLKCM + P3DetailRestore 定向挑战", "",
        "动机：全尺度 MLKCM 已获得 mAP50-95=0.5752 的目标余量，但 mAP50=0.7459。这里在其 P3 Detect 输入后加入零初始化、空间可靠性门控的局部细节残差，尝试恢复小目标候选覆盖；其余结构与全尺度 MLKCM 候选一致。",
        "", "统一协议：5k_HRW、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    if job["status"] == "completed" and metrics and summary:
        passed = float(metrics["map50"]) > PAPER_TARGET["map50"] and float(metrics["map5095"]) > PAPER_TARGET["map5095"]
        status = "completed / 双指标超过目标" if passed else "completed / 未双指标超过目标"
        lines.append(f"| {job['name']} | {status} | {int(float(metrics['epoch']))+1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
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
