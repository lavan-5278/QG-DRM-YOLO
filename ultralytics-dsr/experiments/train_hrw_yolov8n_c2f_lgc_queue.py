"""Run C2f-LGC partial/full replacement ablations after the LFM backbone queue completes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


DATA_YAML = ROOT / "ultralytics/cfg/datasets/5k-HRW.yaml"
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8n_c2f_lgc_seed42_scratch"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
LFM_STATE_PATH = ROOT / "runs/hrw_yolov8n_lfm_backbone_seed42_scratch/queue_state.json"
STANDARD_REPORT = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8N_C2F_LGC_ABLATION_FINAL_START -->"
REPORT_END = "<!-- YOLOV8N_C2F_LGC_ABLATION_FINAL_END -->"
JOBS = (
    ("yolov8n_c2f_lgc", "YOLOv8n + C2f-LGC（C4/C5 内部混合替换）", ROOT / "ultralytics/cfg/models/v8/yolov8n-c2f-lgc.yaml"),
    ("yolov8n_c2f_to_lgc", "YOLOv8n + C4/C5 C2f→LGC（完整替换）", ROOT / "ultralytics/cfg/models/v8/yolov8n-c2f-to-lgc.yaml"),
)
REFERENCE_RUNS = (
    ("YOLOv8n scratch", ROOT / "runs/hrw_standard_baselines_seed42_scratch/native_runs/yolov8n"),
    ("后插 LGC", ROOT / "runs/hrw_yolov8n_lfm_backbone_seed42_scratch/native_runs/yolov8n_lgc"),
    ("LGRB", ROOT / "runs/hrw_yolov8n_lfm_backbone_seed42_scratch/native_runs/yolov8n_lgrb"),
    ("LHGB", ROOT / "runs/hrw_yolov8n_lfm_backbone_seed42_scratch/native_runs/yolov8n_lhgb"),
    ("原 SATA", ROOT / "runs/hrw_yolov8n_qg_clsc_seed42_scratch/native_runs/yolov8n_sata"),
    ("RG-SATA", ROOT / "runs/hrw_yolov8n_sata_variants_seed42_scratch/native_runs/yolov8n_rgsata"),
    ("DR-SATA", ROOT / "runs/hrw_yolov8n_sata_variants_seed42_scratch/native_runs/yolov8n_drsata"),
)


def atomic_json(path: Path, data: dict) -> None:
    """Persist state atomically so monitors never observe a half-written JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def initial_state() -> dict:
    """Return the fixed scratch protocol and conservative-to-aggressive ablation order."""
    return {
        "protocol": {
            "dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8,
            "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD",
        },
        "jobs": [{"id": job_id, "name": name, "yaml": str(yaml_path), "status": "pending"} for job_id, name, yaml_path in JOBS],
        "running": None,
    }


def reconcile_state(state: dict) -> tuple[dict, bool]:
    """Append only missing declared jobs and forbid reordered persisted experiments."""
    jobs = state.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("Queue state must contain a jobs list")
    if len(jobs) > len(JOBS):
        raise RuntimeError("Queue state contains more jobs than the current queue definition")
    for index, job in enumerate(jobs):
        if job.get("id") != JOBS[index][0]:
            raise RuntimeError(f"Queue definition changed at {job.get('id')}")
    changed = False
    for job_id, name, yaml_path in JOBS[len(jobs):]:
        jobs.append({"id": job_id, "name": name, "yaml": str(yaml_path), "status": "pending"})
        changed = True
    if state.get("running") not in {None, *(job_id for job_id, _name, _yaml in JOBS)}:
        raise RuntimeError(f"Queue state references unknown running job {state['running']}")
    return state, changed


def load_state() -> dict:
    """Load/reconcile local replacement-ablation state."""
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state()
    state, changed = reconcile_state(state)
    if changed:
        atomic_json(STATE_PATH, state)
    return state


def predecessor_complete() -> bool:
    """Return true only after all LGC/LGRB/LHGB predecessor jobs completed successfully."""
    if not LFM_STATE_PATH.exists():
        return False
    try:
        predecessor = json.loads(LFM_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    jobs = predecessor.get("jobs", [])
    required = {"yolov8n_lgc", "yolov8n_lgrb", "yolov8n_lhgb"}
    statuses = {job.get("id"): job.get("status") for job in jobs}
    return all(statuses.get(job_id) == "completed" for job_id in required)


def wait_for_predecessor() -> None:
    """Wait without modifying the active predecessor queue or sharing GPU training time."""
    while not predecessor_complete():
        print("Waiting for LGC/LGRB/LHGB predecessor queue to complete before starting replacement ablations.", flush=True)
        time.sleep(60)


def train_kwargs(run_dir: Path, resume: bool) -> dict:
    """Return the exact user-approved scratch protocol."""
    kwargs = {
        "data": str(DATA_YAML), "epochs": 300, "batch": 8, "imgsz": 640,
        "lr0": 0.01, "optimizer": "SGD", "seed": 42, "deterministic": True,
        "device": 0, "workers": 4, "project": str(run_dir.parent), "name": run_dir.name,
        "exist_ok": True, "pretrained": False, "val": True, "save": True, "plots": True, "verbose": True,
    }
    last = run_dir / "weights" / "last.pt"
    if resume and last.exists():
        kwargs["resume"] = str(last)
    return kwargs


def read_best_metrics(run_dir: Path) -> dict[str, str]:
    """Read the validation row selected by best mAP50-95."""
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No result rows found in {run_dir / 'results.csv'}")
    map95_key = next(key for key in rows[0] if "mAP50-95" in key)
    best = max(rows, key=lambda row: float(row[map95_key]))
    keys = {
        "epoch": "epoch", "precision": "metrics/precision(B)", "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)", "map5095": map95_key, "train_box": "train/box_loss",
        "train_cls": "train/cls_loss", "train_dfl": "train/dfl_loss", "val_box": "val/box_loss",
        "val_cls": "val/cls_loss", "val_dfl": "val/dfl_loss",
    }
    return {name: best.get(key, "") for name, key in keys.items()}


def benchmark_checkpoint(best: Path) -> dict:
    """Collect actual parameter/FLOP/latency figures on the fixed 4080 SUPER protocol."""
    model = torch.load(best, map_location="cpu", weights_only=False)["model"].float().cuda().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    sample = torch.zeros((1, 3, 640, 640), device="cuda")
    with torch.inference_mode():
        for _ in range(50):
            model(sample)
        torch.cuda.synchronize()
        timings = []
        for _ in range(200):
            start = time.perf_counter()
            model(sample)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {
        "parameters": parameters, "weight_bytes": best.stat().st_size, "flops_g": float(get_flops(model, imgsz=640)),
        "inference_ms_b1_640": statistics.mean(timings),
        "inference_protocol": "RTX 4080 SUPER, CUDA, FP32, batch=1, 640x640, 50 warmup + 200 forward passes",
    }


def reference_map95() -> dict[str, float]:
    """Read all available same-protocol comparison mAP50-95 values from CSV evidence."""
    return {
        name: float(read_best_metrics(run_dir)["map5095"])
        for name, run_dir in REFERENCE_RUNS
        if (run_dir / "results.csv").exists()
    }


def report_section(state: dict) -> str:
    """Render a marker-bounded Chinese ablation table with comparator evidence."""
    references = reference_map95()
    baseline = references.get("YOLOv8n scratch")
    lines = [
        REPORT_START,
        "## YOLOv8n C2f-LGC 替换消融（部分替换 / C4+C5 完整替换）",
        "",
        "统一协议：5k_HRW、seed=42、从零训练且不使用预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。两项均不与后插 LGC、关系模块、SATA 或 Neck/Head 改进叠加。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 相对 YOLOv8n mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for job in state["jobs"]:
        metrics, summary = job.get("metrics"), job.get("summary")
        if job["status"] == "completed" and metrics and summary:
            train_loss = "/".join(f"{float(metrics[key]):.4f}" for key in ("train_box", "train_cls", "train_dfl"))
            val_loss = "/".join(f"{float(metrics[key]):.4f}" for key in ("val_box", "val_cls", "val_dfl"))
            delta = "—" if baseline is None else f"{float(metrics['map5095']) - baseline:+.5f}"
            lines.append(
                f"| {job['name']} | completed | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {delta} | {summary['parameters']:,} | {summary['weight_bytes'] / 1024 / 1024:.2f} MB | {train_loss} | {val_loss} | {summary['flops_g']:.3f} | {summary['inference_ms_b1_640']:.3f} |"
            )
        else:
            lines.append(f"| {job['name']} | {job['status']} | — | — | — | — | — | — | — | — | — | — | — | — |")
    comparator = "；".join(f"{name}={value:.4f}" for name, value in references.items()) or "参考结果不可用"
    lines.extend((
        "",
        f"mAP50-95 参考：{comparator}。",
        "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。",
        "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。",
        REPORT_END,
    ))
    return "\n".join(lines)


def refresh_standard_report(state: dict) -> None:
    """Replace only this ablation section while preserving all earlier report content."""
    section = report_section(state)
    existing = STANDARD_REPORT.read_text(encoding="utf-8") if STANDARD_REPORT.exists() else "# 5k_HRW seed42 标准基线结果\n"
    if REPORT_START in existing and REPORT_END in existing:
        before, tail = existing.split(REPORT_START, 1)
        _, after = tail.split(REPORT_END, 1)
        content = before.rstrip() + "\n\n" + section + after
    else:
        content = existing.rstrip() + "\n\n" + section + "\n"
    STANDARD_REPORT.parent.mkdir(parents=True, exist_ok=True)
    STANDARD_REPORT.write_text(content, encoding="utf-8")


def validate_models() -> None:
    """Build both model graphs before either can consume training GPU time."""
    for _job_id, _name, yaml_path in JOBS:
        YOLO(str(yaml_path))


def run_queue(resume: bool, dry_run: bool, wait_for_lhgb: bool) -> None:
    """Optionally wait for LHGB, then run conservative and full-replacement jobs serially."""
    state = load_state()
    atomic_json(STATE_PATH, state)
    refresh_standard_report(state)
    if dry_run:
        validate_models()
        return
    if wait_for_lhgb:
        wait_for_predecessor()
    elif not predecessor_complete():
        raise RuntimeError("LFM predecessor queue is incomplete; pass --wait-for-lhgb to wait safely")
    for job, (job_id, _name, yaml_path) in zip(state["jobs"], JOBS):
        if job["id"] != job_id:
            raise RuntimeError(f"Queue definition changed at {job['id']}")
        if job["status"] == "completed":
            continue
        run_dir = OUTPUT_ROOT / "native_runs" / job_id
        job.update({"status": "running", "run_dir": str(run_dir), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        state["running"] = job_id
        atomic_json(STATE_PATH, state)
        refresh_standard_report(state)
        try:
            result = YOLO(str(yaml_path)).train(**train_kwargs(run_dir, resume=resume))
            final_dir = Path(result.save_dir)
            best = final_dir / "weights" / "best.pt"
            job["metrics"] = read_best_metrics(final_dir)
            job["summary"] = benchmark_checkpoint(best)
            (final_dir / "native_summary.json").write_text(
                json.dumps({"model": job_id, "status": "completed", **job["summary"]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            job["status"] = "completed"
            job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as exc:
            job.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        finally:
            state["running"] = None
            atomic_json(STATE_PATH, state)
            refresh_standard_report(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume a locally interrupted unfinished replacement job from last.pt.")
    parser.add_argument("--dry-run", action="store_true", help="Initialize state and validate both graphs without training.")
    parser.add_argument("--wait-for-lhgb", action="store_true", help="Wait until the LGC/LGRB/LHGB queue completes before training.")
    args = parser.parse_args()
    run_queue(resume=args.resume, dry_run=args.dry_run, wait_for_lhgb=args.wait_for_lhgb)


if __name__ == "__main__":
    main()
