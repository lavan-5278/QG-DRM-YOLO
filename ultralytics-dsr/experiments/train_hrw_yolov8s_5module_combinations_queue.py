"""Run the 26 non-singleton subsets of LGC, DR-SATA, QG-AFPN, DSR and DR-C2f v3 on YOLOv8s."""

from __future__ import annotations

import argparse
import itertools
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
from ultralytics.nn.tasks import DetectionModel

from train_hrw_yolov8s_module_singletons_queue import DATA_YAML, atomic_json, benchmark, read_best

MODULES = ("lgc", "drsata", "qgafpn", "dsr", "drc2fv3")
LABELS = {"lgc": "LGC", "drsata": "DR-SATA", "qgafpn": "QG-AFPN", "dsr": "DSR", "drc2fv3": "DR-C2f v3"}
COMBINATIONS = tuple(combo for size in range(2, len(MODULES) + 1) for combo in itertools.combinations(MODULES, size))
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_drc2fv3_combinations_seed42_scratch"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
PREDECESSOR = ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/queue_state.json"
DSR_SOURCE = ROOT / "ultralytics/cfg/models/v8/yolov8n-dsr.yaml"
REPORT_PATH = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8S_5MODULE_COMBINATIONS_START -->"
REPORT_END = "<!-- YOLOV8S_5MODULE_COMBINATIONS_END -->"
SINGLETON_RUNS = {
    "lgc": ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/native_runs/yolov8s_lgc",
    "drsata": ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/native_runs/yolov8s_drsata",
    "qgafpn": ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/native_runs/yolov8s_qgafpn",
    "dsr": ROOT / "runs/hrw_yolov8s_lgc_drsata_qgafpn_dsr_singletons_seed42_scratch/native_runs/yolov8s_dsr",
    "drc2fv3": ROOT / "runs/hrw_yolov8s_dr_c2f_v3_seed42_scratch/native_runs/yolov8s_dr_c2f_v3",
}


def dsr_config() -> dict:
    config = yaml.safe_load(DSR_SOURCE.read_text(encoding="utf-8")).get("dsr")
    if not isinstance(config, dict): raise RuntimeError("Missing DSR config")
    return config


def job_id(combo: tuple[str, ...]) -> str:
    return "_".join(combo)


def model_dict(combo: tuple[str, ...]) -> dict:
    active = set(combo)
    c2f = "DRC2fV3" if "drc2fv3" in active else "C2f"
    backbone = [[-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, c2f, [128, True]], [-1, 1, "Conv", [256, 3, 2]], [-1, 6, c2f, [256, True]], [-1, 1, "Conv", [512, 3, 2]], [-1, 6, c2f, [512, True]]]
    if "lgc" in active: backbone.append([-1, 1, "LGC", []])
    if "drsata" in active: backbone.append([-1, 1, "DRSATA", [64, 16, 16, 6]])
    c4 = len(backbone) - 1
    backbone += [[-1, 1, "Conv", [1024, 3, 2]], [-1, 3, c2f, [1024, True]], [-1, 1, "SPPF", [1024, 5]]]
    if "lgc" in active: backbone.append([-1, 1, "LGC", []])
    if "drsata" in active: backbone.append([-1, 1, "DRSATA", [64, 16, 16, 6]])
    c5 = len(backbone) - 1
    head: list = []
    def add(src, module, args):
        head.append([src, 1, module, args]); return len(backbone) + len(head) - 1
    add(-1, "nn.Upsample", [None, 2, "nearest"]); add([-1, c4], "Concat", [1]); p4_td = add(-1, "C2f", [512])
    add(-1, "nn.Upsample", [None, 2, "nearest"]); add([-1, 4], "Concat", [1]); p3 = add(-1, "C2f", [256])
    add(-1, "Conv", [256, 3, 2]); add([-1, p4_td], "Concat", [1]); p4 = add(-1, "C2f", [512])
    add(-1, "Conv", [512, 3, 2]); add([-1, c5], "Concat", [1]); p5 = add(-1, "C2f", [1024])
    if "qgafpn" in active:
        qg = add([p3, p4, p5], "QGAFPN", []); p3 = add(-1, "Index", [128, 0]); p4 = add(qg, "Index", [256, 1]); p5 = add(qg, "Index", [512, 2])
    add([p3, p4, p5], "DSRDetect" if "dsr" in active else "Detect", [3, "dsr"] if "dsr" in active else [3])
    model = {"nc": 3, "scales": {"s": [0.33, 0.50, 1024]}, "backbone": backbone, "head": head}
    if "dsr" in active: model["dsr"] = dsr_config()
    return model


def write_configs() -> dict[str, Path]:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True); paths = {}
    for combo in COMBINATIONS:
        key = job_id(combo); path = CONFIG_ROOT / f"yolov8s-{key}.yaml"
        path.write_text(yaml.safe_dump(model_dict(combo), allow_unicode=True, sort_keys=False), encoding="utf-8"); paths[key] = path
    return paths


def initial_state(paths: dict[str, Path]) -> dict:
    return {"protocol": {"dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8, "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD"}, "design": {"universe": list(MODULES), "existing_singletons": 5, "new_non_singleton_jobs": len(COMBINATIONS), "c4_c5_joint_order": "LGC -> DR-SATA", "dr_c2f_v3_scope": "four Backbone C2f"}, "queue_status": "pending", "running": None, "jobs": [{"id": job_id(combo), "name": "YOLOv8s + " + " + ".join(LABELS[item] for item in combo), "modules": list(combo), "yaml": str(paths[job_id(combo)]), "status": "pending", "run_dir": str(OUTPUT_ROOT / "native_runs" / f"yolov8s_{job_id(combo)}")} for combo in COMBINATIONS]}


def load_state(paths: dict[str, Path]) -> dict:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state(paths)
    expected = [job_id(combo) for combo in COMBINATIONS]
    if [job.get("id") for job in state.get("jobs", [])] != expected: raise RuntimeError("Unexpected 5-module combination definition")
    for job in state["jobs"]: job["yaml"] = str(paths[job["id"]])
    return state


def singleton_records() -> list[dict]:
    records = []
    for key in MODULES:
        run = SINGLETON_RUNS[key]; summary = run / "native_summary.json"
        if (run / "results.csv").exists() and summary.exists():
            records.append({"name": "YOLOv8s + " + LABELS[key] + "（既有单模块）", "status": "completed_existing", "metrics": read_best(run), "summary": json.loads(summary.read_text(encoding="utf-8"))})
        else:
            # The DSR singleton can still be running while this successor queue is waiting.
            records.append({"name": "YOLOv8s + " + LABELS[key] + "（既有单模块）", "status": "waiting_for_singleton"})
    return records


def row(record: dict) -> str:
    metrics, summary = record.get("metrics"), record.get("summary")
    if not (metrics and summary) or record.get("status") not in {"completed", "completed_existing"}: return f"| {record['name']} | {record.get('status','pending')} | — | — | — | — | — | — | — | — | — | — | — |"
    train = "/".join(f"{float(metrics[key]):.4f}" for key in ("train_box", "train_cls", "train_dfl")); val = "/".join(f"{float(metrics[key]):.4f}" for key in ("val_box", "val_cls", "val_dfl"))
    status = "既有完成" if record["status"] == "completed_existing" else "completed"
    return f"| {record['name']} | {status} | {int(float(metrics['epoch']))+1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {summary['parameters']:,} | {summary['weight_bytes']/1024/1024:.2f} MB | {train} | {val} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |"


def refresh_report(state: dict) -> None:
    lines = [REPORT_START, "## YOLOv8s LGC / DR-SATA / QG-AFPN / DSR / DR-C2f v3 多模块组合消融（31 个唯一子集）", "", "五模块全集共 31 个唯一非空子集：5 个单模块采用既有完成证据。根据用户指令，所有含 DSR 的多模块组合均不再训练；不含 DSR 的 11 个多模块组合继续从零训练。DR-C2f v3 仅替换四个 Backbone C2f；LGC 与 DR-SATA 联合时固定顺序为 `LGC → DR-SATA`；QG-AFPN 位于 Neck。", "", "统一协议：5k_HRW、seed=42、从零训练且无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。", "", "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    lines.extend(row(record) for record in singleton_records() + state["jobs"])
    lines += ["", "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。", "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。", REPORT_END]
    section = "\n".join(lines); existing = REPORT_PATH.read_text(encoding="utf-8") if REPORT_PATH.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        prefix, tail = existing.split(REPORT_START, 1); _, suffix = tail.split(REPORT_END, 1); content = prefix.rstrip()+"\n\n"+section+suffix
    else: content = existing.rstrip()+"\n\n"+section+"\n"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True); REPORT_PATH.write_text(content, encoding="utf-8")


def validate(paths: dict[str, Path]) -> None:
    for combo in COMBINATIONS:
        key = job_id(combo); (DSRDetectionModel if "dsr" in combo else DetectionModel)(str(paths[key]), nc=3, verbose=False)


def wait_for_singletons(state: dict) -> None:
    state["queue_status"] = "waiting_for_singletons"; atomic_json(STATE_PATH, state); refresh_report(state)
    while True:
        if not PREDECESSOR.exists(): time.sleep(30); continue
        predecessor = json.loads(PREDECESSOR.read_text(encoding="utf-8")); status = predecessor.get("queue_status")
        if status == "completed": time.sleep(10); return
        if status == "failed": raise RuntimeError(f"Singleton predecessor failed: {predecessor.get('error','')}")
        time.sleep(30)


def exclude_dsr_combinations(state: dict) -> None:
    """Persist the user's decision to omit every multi-module combination containing DSR."""
    for job in state["jobs"]:
        if "dsr" not in job["modules"] or job["status"] in {"completed", "completed_existing"}:
            continue
        previous = job["status"]
        job["status"] = "stopped_by_user_dsr_policy" if previous == "running" else "skipped_by_user_dsr_policy"
        job["policy_updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if state.get("running") == job["id"]:
            state["running"] = None


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true"); args = parser.parse_args()
    paths = write_configs(); validate(paths); state = load_state(paths); exclude_dsr_combinations(state); atomic_json(STATE_PATH, state); refresh_report(state)
    if args.dry_run: return
    try:
        wait_for_singletons(state); state["queue_status"] = "running"; atomic_json(STATE_PATH, state); refresh_report(state)
        for job in state["jobs"]:
            if job["status"] != "pending": continue
            run_dir = Path(job["run_dir"]); job.update({"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}); state["running"] = job["id"]; atomic_json(STATE_PATH, state); refresh_report(state)
            trainer = DSRDetectionTrainer if "dsr" in job["modules"] else None
            result = YOLO(job["yaml"]).train(trainer=trainer, data=str(DATA_YAML), epochs=300, batch=8, imgsz=640, lr0=0.01, optimizer="SGD", seed=42, deterministic=True, device=0, workers=4, project=str(run_dir.parent), name=run_dir.name, exist_ok=True, pretrained=False, val=True, save=True, plots=True, verbose=True)
            final = Path(result.save_dir); best = final / "weights/best.pt"; job["metrics"] = read_best(final); job["summary"] = benchmark(best)
            (final / "native_summary.json").write_text(json.dumps({"model": job["id"], "status": "completed", **job["summary"]}, ensure_ascii=False, indent=2), encoding="utf-8")
            job.update({"status": "completed", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}); state["running"] = None; atomic_json(STATE_PATH, state); refresh_report(state)
        state["queue_status"] = "completed"
    except Exception as exc:
        state["queue_status"] = "failed"; state["error"] = f"{type(exc).__name__}: {exc}"; raise
    finally:
        state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S"); atomic_json(STATE_PATH, state); refresh_report(state)


if __name__ == "__main__": main()
