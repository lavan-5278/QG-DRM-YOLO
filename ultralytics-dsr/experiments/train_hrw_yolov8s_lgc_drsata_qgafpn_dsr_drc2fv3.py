"""Wait for YOLOv8s singleton queue, then train the one full five-module scratch combination."""

from __future__ import annotations

import argparse
import json
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

from train_hrw_yolov8s_dr_c2f_v3 import benchmark_checkpoint, read_best_metrics

DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
DSR_SOURCE_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8n-dsr.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_drc2fv3_full_seed42_scratch"
CONFIG_PATH = OUTPUT_ROOT / "configs/yolov8s-lgc-drsata-qgafpn-dsr-drc2fv3.yaml"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
PREDECESSOR = ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/queue_state.json"
RUN_DIR = OUTPUT_ROOT / "native_runs/yolov8s_lgc_drsata_qgafpn_dsr_drc2fv3"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_LGC_DRSATA_QGAFPN_DSR_DRC2FV3_FULL_START -->"
REPORT_END = "<!-- YOLOV8S_LGC_DRSATA_QGAFPN_DSR_DRC2FV3_FULL_END -->"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def build_model_dict() -> dict:
    dsr = yaml.safe_load(DSR_SOURCE_YAML.read_text(encoding="utf-8")).get("dsr")
    if not isinstance(dsr, dict):
        raise RuntimeError("Missing validated DSR config")
    # DRC2fV3 replaces all four Backbone C2f; LGC -> DR-SATA is the fixed C4/C5 order.
    backbone = [
        [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, "DRC2fV3", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]], [-1, 6, "DRC2fV3", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, "DRC2fV3", [512, True]], [-1, 1, "LGC", []], [-1, 1, "DRSATA", [64, 16, 16, 6]],
        [-1, 1, "Conv", [1024, 3, 2]], [-1, 3, "DRC2fV3", [1024, True]], [-1, 1, "SPPF", [1024, 5]],
        [-1, 1, "LGC", []], [-1, 1, "DRSATA", [64, 16, 16, 6]],
    ]
    # C4 after DRSATA is index 8; C5 after DRSATA is index 13.
    head = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 8], 1, "Concat", [1]], [-1, 3, "C2f", [512]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]], [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],
        [-1, 1, "Conv", [256, 3, 2]], [[-1, 16], 1, "Concat", [1]], [-1, 3, "C2f", [512]],
        [-1, 1, "Conv", [512, 3, 2]], [[-1, 13], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],
        [[19, 22, 25], 1, "QGAFPN", []], [-1, 1, "Index", [128, 0]], [26, 1, "Index", [256, 1]],
        [26, 1, "Index", [512, 2]], [[27, 28, 29], 1, "DSRDetect", [3, "dsr"]],
    ]
    return {"nc": 3, "scales": {"s": [0.33, 0.50, 1024]}, "dsr": dsr, "backbone": backbone, "head": head}


def write_config() -> Path:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(build_model_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return CONFIG_PATH


def new_state() -> dict:
    return {"protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"}, "design": {"modules": ["LGC", "DR-SATA", "QG-AFPN", "DSR", "DR-C2f v3"], "dr_c2f_v3_scope": "four Backbone C2f", "c4_c5_order": "LGC -> DR-SATA", "qgafpn": "Neck", "dsr": "Detect"}, "status": "pending", "predecessor_state": str(PREDECESSOR), "job": {"id": "yolov8s_lgc_drsata_qgafpn_dsr_drc2fv3", "name": "YOLOv8s + LGC + DR-SATA + QG-AFPN + DSR + DR-C2f v3", "yaml": str(CONFIG_PATH), "run_dir": str(RUN_DIR), "status": "pending"}}


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else new_state()


def relay_coefficients(best: Path) -> dict[str, dict[str, list[float]]]:
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().eval()
    result = {name: module.relay_coefficients() for name, module in model.named_modules() if callable(getattr(module, "relay_coefficients", None))}
    if not result: raise RuntimeError("No DR-C2f v3 coefficients found in best.pt")
    return result


def refresh_report(state: dict) -> None:
    job = state["job"]; metrics, summary = job.get("metrics"), job.get("summary")
    lines = [REPORT_START, "## YOLOv8s 五模块完整组合：LGC / DR-SATA / QG-AFPN / DSR / DR-C2f v3", "", "单个完整组合实验：DR-C2f v3 替换四个 Backbone C2f；LGC → DR-SATA 固定插入 C4/C5；QG-AFPN 位于三尺度 Neck；DSR 替换 Detect。其余 Neck 与训练设置保持 YOLOv8s scratch 一致。", "", "统一协议：5k_HRW、seed=42、从零训练、无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "", "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    if job["status"] == "completed" and metrics and summary:
        lines.append(f"| {job['name']} | completed | {int(float(metrics['epoch']))+1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {float(metrics['train_box']):.4f}/{float(metrics['train_cls']):.4f}/{float(metrics['train_dfl']):.4f} | {float(metrics['val_box']):.4f}/{float(metrics['val_cls']):.4f}/{float(metrics['val_dfl']):.4f} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |")
        lines += ["", "学习到的 DR-C2f v3 中继系数：", "", "```json", json.dumps(job["relay_coefficients"], ensure_ascii=False, indent=2), "```"]
    else: lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines); existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        prefix, tail = existing.split(REPORT_START, 1); _, suffix = tail.split(REPORT_END, 1); content = prefix.rstrip()+"\n\n"+section+suffix
    else: content = existing.rstrip()+"\n\n"+section+"\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(content, encoding="utf-8")


def wait_for_singletons(state: dict) -> None:
    state["status"] = "waiting_for_singletons"; atomic_json(STATE_PATH, state); refresh_report(state)
    while True:
        if not PREDECESSOR.exists(): time.sleep(30); continue
        predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
        status = predecessor.get("queue_status")
        if status == "completed": time.sleep(10); return
        if status == "failed": raise RuntimeError(f"Singleton predecessor failed: {predecessor.get('error', '')}")
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    config = write_config(); DSRDetectionModel(str(config), nc=3, verbose=False)
    state = load_state(); state["job"]["yaml"] = str(config); atomic_json(STATE_PATH, state); refresh_report(state)
    if args.dry_run: return
    job = state["job"]
    if job["status"] == "completed": return
    try:
        wait_for_singletons(state); state["status"] = "running"; job.update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}); atomic_json(STATE_PATH, state); refresh_report(state)
        result = YOLO(str(config)).train(trainer=DSRDetectionTrainer, data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42, deterministic=True, device=0, workers=4, project=str(RUN_DIR.parent), name=RUN_DIR.name, exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True)
        final = Path(result.save_dir); best = final / "weights/best.pt"; job["metrics"] = read_best_metrics(final); job["summary"] = benchmark_checkpoint(best); job["relay_coefficients"] = relay_coefficients(best)
        (final / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"], "relay_coefficients": job["relay_coefficients"]}, ensure_ascii=False, indent=2), encoding="utf-8"); job["status"] = "completed"; state["status"] = "completed"
    except Exception as exc:
        state["status"] = "failed"; job.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}); raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S"); atomic_json(STATE_PATH, state); refresh_report(state)


if __name__ == "__main__": main()
