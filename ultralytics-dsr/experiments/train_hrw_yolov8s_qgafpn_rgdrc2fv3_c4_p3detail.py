"""Train QG-AFPN + C4 RGDRC2fV3 + P3DetailRestore on 5k_HRW from scratch."""

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
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8s-qgafpn-rgdrc2fv3-c4-p3detail.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_qgafpn_rgdrc2fv3_c4_p3detail_seed42_scratch"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_qgafpn_rgdrc2fv3_c4_p3detail"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_P3DETAIL_START -->"
REPORT_END = "<!-- YOLOV8S_QGAFPN_RGDRC2FV3_C4_P3DETAIL_END -->"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def initial_state() -> dict:
    return {
        "protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"},
        "design": {
            "base": "YOLOv8s + QG-AFPN + C4 RGDRC2fV3",
            "addition": "P3DetailRestore after QG-AFPN P3 output only",
            "initialization": "Both residual scales are zero; both adapters start as exact identity mappings.",
            "preserved": "P2/P3/C5 Backbone C2f, P4/P5 detection features and Detect remain unchanged.",
            "excluded_modules": ["LGC", "DR-SATA", "DSR"],
        },
        "status": "pending",
        "job": {"id": "yolov8s_qgafpn_rgdrc2fv3_c4_p3detail", "name": "YOLOv8s + QG-AFPN + C4门控DR-C2f v3 + P3细节恢复", "yaml": str(MODEL_YAML), "run_dir": str(RUN_DIR), "status": "pending"},
    }


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()


def adapter_states(best: Path) -> dict[str, dict]:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().eval()
    result = {name: module.adapter_state() for name, module in model.named_modules() if callable(getattr(module, "adapter_state", None))}
    expected = {"RGDRC2fV3", "P3DetailRestore"}
    kinds = {module.__class__.__name__ for module in model.modules() if callable(getattr(module, "adapter_state", None))}
    if kinds != expected or len(result) != 2:
        raise RuntimeError(f"Expected one C4 RGDRC2fV3 and one P3DetailRestore, got {kinds} / {len(result)} states")
    return result


def verify_structure_and_identities(model: YOLO) -> None:
    layers = model.model.model
    if layers[6].__class__.__name__ != "RGDRC2fV3" or layers[8].__class__.__name__ != "C2f":
        raise RuntimeError("C4 must be RGDRC2fV3 and C5 must remain native C2f")
    if layers[26].__class__.__name__ != "P3DetailRestore":
        raise RuntimeError("P3DetailRestore must be inserted at layer 26 after QG-AFPN P3 Index")
    checks = [(layers[6], torch.randn(1, layers[6].base.cv1.conv.in_channels, 40, 40), "base"), (layers[26], torch.randn(1, 128, 80, 80), "identity")]
    for adapter, sample, kind in checks:
        adapter.eval()
        with torch.inference_mode():
            expected = adapter.base(sample) if kind == "base" else sample
            actual = adapter(sample)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Zero-init identity check failed for {adapter.__class__.__name__}: max error {(actual - expected).abs().max().item()}")


def refresh_report(state: dict) -> None:
    job = state["job"]; metrics, summary = job.get("metrics"), job.get("summary")
    lines = [
        REPORT_START,
        "## YOLOv8s QG-AFPN + C4 门控 DR-C2f v3 + P3 细节恢复实验",
        "",
        "设计：沿用 mAP50-95 最优的 C4 可靠性门控残差 DR-C2f v3，仅在 QG-AFPN 输出的 P3 检测特征上增加零初始化、空间可靠性门控的局部细节残差。两个适配器初始均为恒等映射，以保持 QG-AFPN 的候选覆盖；P2/P3/C5 Backbone、P4/P5 检测特征与 Detect 不变。",
        "",
        "统一协议：5k_HRW、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    if job["status"] == "completed" and metrics and summary:
        lines.append(f"| {job['name']} | completed | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
        lines += ["", "学习到的适配器状态：", "", "```json", json.dumps(job["adapter_states"], ensure_ascii=False, indent=2), "```"]
    else:
        lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines); existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        prefix, tail = existing.split(REPORT_START, 1); _, suffix = tail.split(REPORT_END, 1); content = prefix.rstrip() + "\n\n" + section + suffix
    else:
        content = existing.rstrip() + "\n\n" + section + "\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(content, encoding="utf-8")


def finalize(state: dict) -> None:
    job = state["job"]; best = RUN_DIR / "weights/best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Cannot finalize without {best}")
    job["metrics"] = read_best_metrics(RUN_DIR); job["summary"] = benchmark_checkpoint(best); job["adapter_states"] = adapter_states(best)
    (RUN_DIR / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"], "adapter_states": job["adapter_states"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    job["status"] = "completed"; state["status"] = "completed"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--finalize-only", action="store_true"); args = parser.parse_args()
    state = load_state(); model = YOLO(str(MODEL_YAML)); verify_structure_and_identities(model); refresh_report(state)
    if args.dry_run:
        model.info(verbose=False); return
    try:
        if args.finalize_only:
            finalize(state); return
        if state["job"]["status"] == "completed": return
        state["status"] = "running"; state["job"].update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}); atomic_json(STATE_PATH, state); refresh_report(state)
        model.train(data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42, deterministic=True, device=0, workers=4, project=str(RUN_DIR.parent), name=RUN_DIR.name, exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True)
        finalize(state)
    except Exception as exc:
        state["status"] = "failed"; state["job"].update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}); raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S"); atomic_json(STATE_PATH, state); refresh_report(state)


if __name__ == "__main__":
    main()
