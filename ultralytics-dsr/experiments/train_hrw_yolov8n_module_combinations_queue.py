"""Run the 11 unseen unique LGC/DR-SATA/QG-AFPN/DSR combinations on 5k_HRW.

The four singleton experiments were completed previously.  This queue contains the
remaining six pairs, four triples and one four-module model, so its report covers
the full 2**4 - 1 unique-subset ablation without treating module order as a new
experiment.  LGC and DR-SATA share C4/C5 insertion points; their joint architecture
therefore uses the single, declared order LGC -> DR-SATA at both stages.
"""

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
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8n_lgc_drsata_qgafpn_dsr_combinations_seed42_scratch"
CONFIG_ROOT = OUTPUT_ROOT / "configs"
STATE_PATH = OUTPUT_ROOT / "queue_state.json"
STANDARD_REPORT = ROOT / "runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md"
REPORT_START = "<!-- YOLOV8N_LGC_DRSATA_QGAFPN_DSR_COMBINATIONS_START -->"
REPORT_END = "<!-- YOLOV8N_LGC_DRSATA_QGAFPN_DSR_COMBINATIONS_END -->"

# The order is complexity-monotonic inside each cardinality, allowing useful results
# to arrive early while preserving every unique non-singleton subset exactly once.
JOBS = (
    ("lgc_drsata", "YOLOv8n + LGC + DR-SATA", ("lgc", "drsata")),
    ("lgc_qgafpn", "YOLOv8n + LGC + QG-AFPN", ("lgc", "qgafpn")),
    ("lgc_dsr", "YOLOv8n + LGC + DSR", ("lgc", "dsr")),
    ("drsata_qgafpn", "YOLOv8n + DR-SATA + QG-AFPN", ("drsata", "qgafpn")),
    ("drsata_dsr", "YOLOv8n + DR-SATA + DSR", ("drsata", "dsr")),
    ("qgafpn_dsr", "YOLOv8n + QG-AFPN + DSR", ("qgafpn", "dsr")),
    ("lgc_drsata_qgafpn", "YOLOv8n + LGC + DR-SATA + QG-AFPN", ("lgc", "drsata", "qgafpn")),
    ("lgc_drsata_dsr", "YOLOv8n + LGC + DR-SATA + DSR", ("lgc", "drsata", "dsr")),
    ("lgc_qgafpn_dsr", "YOLOv8n + LGC + QG-AFPN + DSR", ("lgc", "qgafpn", "dsr")),
    ("drsata_qgafpn_dsr", "YOLOv8n + DR-SATA + QG-AFPN + DSR", ("drsata", "qgafpn", "dsr")),
    ("lgc_drsata_qgafpn_dsr", "YOLOv8n + LGC + DR-SATA + QG-AFPN + DSR", ("lgc", "drsata", "qgafpn", "dsr")),
)

EXISTING_SINGLETONS = (
    ("lgc", "YOLOv8n + LGC（既有单模块）", ROOT / "runs/hrw_yolov8n_lfm_backbone_seed42_scratch/native_runs/yolov8n_lgc"),
    ("drsata", "YOLOv8n + DR-SATA（既有单模块）", ROOT / "runs/hrw_yolov8n_sata_variants_seed42_scratch/native_runs/yolov8n_drsata"),
    ("qgafpn", "YOLOv8n + QG-AFPN（既有单模块）", ROOT / "runs/hrw_yolov8n_qg_clsc_seed42_scratch/native_runs/yolov8n_qg_afpn"),
    ("dsr", "YOLOv8n + DSR（既有单模块）", ROOT / "runs/hrw_yolov8n_dsr_seed42_scratch/native_runs/yolov8n_dsr"),
)
BASELINE_RUN = ROOT / "runs/hrw_standard_baselines_seed42_scratch/native_runs/yolov8n"


def atomic_json(path: Path, data: dict) -> None:
    """Atomically persist queue state so heartbeat readers always receive valid JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _dsr_config() -> dict:
    """Load the validated DSR configuration shared by standalone and combined runs."""
    source = yaml.safe_load(DSR_SOURCE_YAML.read_text(encoding="utf-8"))
    config = source.get("dsr")
    if not isinstance(config, dict):
        raise RuntimeError(f"Missing dsr mapping in {DSR_SOURCE_YAML}")
    return config


def _append(layers: list, from_: int | list[int], module: str, args: list) -> int:
    """Append a one-repeat YAML layer and return its absolute layer index."""
    layers.append([from_, 1, module, args])
    return len(layers) - 1


def combo_model_dict(active_modules: tuple[str, ...]) -> dict:
    """Build one unique architecture with the declared C4/C5 and neck/head contracts."""
    active = set(active_modules)
    unexpected = active.difference({"lgc", "drsata", "qgafpn", "dsr"})
    if unexpected:
        raise ValueError(f"Unknown module(s): {sorted(unexpected)}")

    backbone: list = []
    _append(backbone, -1, "Conv", [64, 3, 2])
    _append(backbone, -1, "Conv", [128, 3, 2])
    _append(backbone, -1, "C2f", [128, True])
    _append(backbone, -1, "Conv", [256, 3, 2])
    _append(backbone, -1, "C2f", [256, True])
    _append(backbone, -1, "Conv", [512, 3, 2])
    _append(backbone, -1, "C2f", [512, True])
    if "lgc" in active:
        _append(backbone, -1, "LGC", [])
    if "drsata" in active:
        _append(backbone, -1, "DRSATA", [64, 16, 16, 6])
    c4 = len(backbone) - 1
    _append(backbone, -1, "Conv", [1024, 3, 2])
    _append(backbone, -1, "C2f", [1024, True])
    _append(backbone, -1, "SPPF", [1024, 5])
    if "lgc" in active:
        _append(backbone, -1, "LGC", [])
    if "drsata" in active:
        _append(backbone, -1, "DRSATA", [64, 16, 16, 6])
    c5 = len(backbone) - 1

    head: list = []

    def add(from_: int | list[int], module: str, args: list) -> int:
        head.append([from_, 1, module, args])
        return len(backbone) + len(head) - 1

    add(-1, "nn.Upsample", [None, 2, "nearest"])
    add([-1, c4], "Concat", [1])
    p4_top_down = add(-1, "C2f", [512])
    add(-1, "nn.Upsample", [None, 2, "nearest"])
    add([-1, 4], "Concat", [1])
    p3 = add(-1, "C2f", [256])
    add(-1, "Conv", [256, 3, 2])
    add([-1, p4_top_down], "Concat", [1])
    p4 = add(-1, "C2f", [512])
    add(-1, "Conv", [512, 3, 2])
    add([-1, c5], "Concat", [1])
    p5 = add(-1, "C2f", [1024])
    if "qgafpn" in active:
        qg = add([p3, p4, p5], "QGAFPN", [])
        p3 = add(-1, "Index", [64, 0])
        p4 = add(qg, "Index", [128, 1])
        p5 = add(qg, "Index", [256, 2])
    detect_module = "DSRDetect" if "dsr" in active else "Detect"
    detect_args = [3, "dsr"] if "dsr" in active else [3]
    add([p3, p4, p5], detect_module, detect_args)

    model = {"nc": 3, "scales": {"n": [0.33, 0.25, 1024]}, "backbone": backbone, "head": head}
    if "dsr" in active:
        model["dsr"] = _dsr_config()
    return model


def write_combo_yamls(config_dir: Path = CONFIG_ROOT) -> dict[str, Path]:
    """Materialize reproducible generated YAMLs and return their job-id paths."""
    config_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for job_id, _name, active in JOBS:
        path = config_dir / f"yolov8n-{job_id}.yaml"
        path.write_text(yaml.safe_dump(combo_model_dict(active), allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths[job_id] = path
    return paths


def job_uses_dsr(job_id: str) -> bool:
    """Return whether a declared job needs DSRDetectionModel and DSRDetectionLoss."""
    return "dsr" in job_id.split("_")


def trainer_for(job_id: str):
    """Choose the DSR trainer only for DSR-containing architectures."""
    return DSRDetectionTrainer if job_uses_dsr(job_id) else None


def initial_state(config_paths: dict[str, Path] | None = None) -> dict:
    """Return the immutable scratch protocol and the 11 pending combinations."""
    config_paths = config_paths or write_combo_yamls()
    return {
        "protocol": {
            "dataset": "5k_HRW", "seed": 42, "pretrained": False, "batch": 8,
            "lr0": 0.01, "epochs": 300, "imgsz": 640, "optimizer": "SGD",
        },
        "design": {
            "subsets": 15, "existing_singletons": 4, "new_training_jobs": 11,
            "c4_c5_joint_order": "LGC -> DR-SATA",
        },
        "jobs": [
            {"id": job_id, "name": name, "modules": list(active), "yaml": str(config_paths[job_id]), "status": "pending"}
            for job_id, name, active in JOBS
        ],
        "running": None,
    }


def reconcile_state(state: dict, config_paths: dict[str, Path]) -> tuple[dict, bool]:
    """Verify ordered job identity while allowing a safely interrupted queue to resume."""
    jobs = state.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != len(JOBS):
        raise RuntimeError("Queue state must contain exactly the declared 11 combination jobs")
    changed = False
    for job, (job_id, name, active) in zip(jobs, JOBS):
        if job.get("id") != job_id or job.get("modules") != list(active):
            raise RuntimeError(f"Queue definition changed at {job.get('id')}")
        if job.get("name") != name:
            raise RuntimeError(f"Queue display name changed at {job_id}")
        if job.get("yaml") != str(config_paths[job_id]):
            job["yaml"] = str(config_paths[job_id])
            changed = True
    known = {job_id for job_id, _name, _active in JOBS}
    if state.get("running") not in known | {None}:
        raise RuntimeError(f"Queue state references unknown running job {state['running']}")
    return state, changed


def load_state(config_paths: dict[str, Path]) -> dict:
    """Load an existing parseable state or create the fixed new-combination queue."""
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else initial_state(config_paths)
    state, changed = reconcile_state(state, config_paths)
    if changed:
        atomic_json(STATE_PATH, state)
    return state


def train_kwargs(run_dir: Path, resume: bool) -> dict:
    """Return the exact scratch protocol used for every completed 5k_HRW baseline."""
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
    """Read the validation row selected by highest mAP50-95 from native results.csv."""
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
    """Measure an EMA best checkpoint on the fixed RTX 4080 SUPER FP32 protocol."""
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


def singleton_records() -> list[dict]:
    """Load the four completed singleton evidence records reused in the 15-subset table."""
    records = []
    for job_id, name, run_dir in EXISTING_SINGLETONS:
        summary_path = run_dir / "native_summary.json"
        if not (run_dir / "results.csv").exists() or not summary_path.exists():
            raise RuntimeError(f"Missing completed singleton evidence for {job_id}: {run_dir}")
        records.append({"id": job_id, "name": name, "status": "completed_existing", "metrics": read_best_metrics(run_dir), "summary": json.loads(summary_path.read_text(encoding="utf-8"))})
    return records


def baseline_map95() -> float:
    """Read the same-protocol YOLOv8n scratch comparator from its CSV evidence."""
    return float(read_best_metrics(BASELINE_RUN)["map5095"])


def _report_row(record: dict, baseline: float) -> str:
    metrics, summary = record.get("metrics"), record.get("summary")
    if not (metrics and summary) or record.get("status") not in {"completed", "completed_existing"}:
        return f"| {record['name']} | {record.get('status', 'pending')} | — | — | — | — | — | — | — | — | — | — | — | — |"
    train_loss = "/".join(f"{float(metrics[key]):.4f}" for key in ("train_box", "train_cls", "train_dfl"))
    val_loss = "/".join(f"{float(metrics[key]):.4f}" for key in ("val_box", "val_cls", "val_dfl"))
    delta = float(metrics["map5095"]) - baseline
    status = "既有完成" if record["status"] == "completed_existing" else "completed"
    return (
        f"| {record['name']} | {status} | {int(float(metrics['epoch'])) + 1} | {float(metrics['precision']):.4f} | "
        f"{float(metrics['recall']):.4f} | {float(metrics['map50']):.4f} | {float(metrics['map5095']):.4f} | {delta:+.5f} | "
        f"{int(summary['parameters']):,} | {float(summary['weight_bytes']) / 1024 / 1024:.2f} MB | {train_loss} | {val_loss} | "
        f"{float(summary['flops_g']):.3f} | {float(summary['inference_ms_b1_640']):.3f} |"
    )


def report_section(state: dict) -> str:
    """Render all 15 unique subsets into one marker-bounded Chinese table."""
    baseline = baseline_map95()
    records = singleton_records() + list(state["jobs"])
    lines = [
        REPORT_START,
        "## YOLOv8n LGC / DR-SATA / QG-AFPN / DSR 全组合消融（15 个唯一子集）",
        "",
        "组合规则：4 个单模块采用既有完成结果；其余 11 个非单模块子集由本队列从零训练。LGC 与 DR-SATA 均作用于 C4/C5，联合时固定顺序为 `LGC → DR-SATA`；QG-AFPN 位于 Neck，DSR 位于 Detect。故不将无结构差异的排列重复计为实验。",
        "",
        "统一协议：5k_HRW、seed=42、从零训练且无预训练权重、batch=8、SGD、lr0=0.01、300 epoch、640×640。",
        "",
        "| 模型 | 状态 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 | 相对 YOLOv8n mAP50-95 | 参数量 | best.pt 大小 | train box/cls/dfl | val box/cls/dfl | FLOPs (G, 640) | 推理时间 (ms/图) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    lines.extend(_report_row(record, baseline) for record in records)
    lines.extend((
        "",
        "推理协议：RTX 4080 SUPER、CUDA、FP32、batch=1、640×640，50 次预热后 200 次同步前向均值；不含图像读取、预处理与 NMS。",
        "单一数据划分与单一 seed 的结果仅用于可重复的初筛比较，不构成统计显著性结论。",
        REPORT_END,
    ))
    return "\n".join(lines)


def refresh_standard_report(state: dict) -> None:
    """Replace only this 15-subset section, preserving every prior experiment result."""
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


def validate_models(config_paths: dict[str, Path]) -> None:
    """Build every combination graph with its real loss-model class before training starts."""
    for job_id, _name, _active in JOBS:
        if job_uses_dsr(job_id):
            DSRDetectionModel(str(config_paths[job_id]), nc=3, verbose=False)
        else:
            DetectionModel(str(config_paths[job_id]), nc=3, verbose=False)


def run_queue(resume: bool, dry_run: bool) -> None:
    """Run only pending combinations serially and stop transparently on the first failure."""
    config_paths = write_combo_yamls()
    validate_models(config_paths)
    state = load_state(config_paths)
    atomic_json(STATE_PATH, state)
    refresh_standard_report(state)
    if dry_run:
        return
    for job, (job_id, _name, _active) in zip(state["jobs"], JOBS):
        if job["status"] == "completed":
            continue
        run_dir = OUTPUT_ROOT / "native_runs" / job_id
        job.update({"status": "running", "run_dir": str(run_dir), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        state["running"] = job_id
        atomic_json(STATE_PATH, state)
        refresh_standard_report(state)
        try:
            result = YOLO(str(config_paths[job_id])).train(trainer=trainer_for(job_id), **train_kwargs(run_dir, resume=resume))
            final_dir = Path(result.save_dir)
            best = final_dir / "weights" / "best.pt"
            if not best.exists():
                raise RuntimeError(f"Missing expected best.pt: {best}")
            job["metrics"] = read_best_metrics(final_dir)
            job["summary"] = benchmark_checkpoint(best)
            (final_dir / "native_summary.json").write_text(
                json.dumps({"model": job_id, "status": "completed", **job["summary"]}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            job["status"] = "completed"
            job["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as exc:
            job.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            raise
        finally:
            state["running"] = None
            atomic_json(STATE_PATH, state)
            refresh_standard_report(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume the first unfinished job from last.pt when available.")
    parser.add_argument("--dry-run", action="store_true", help="Generate YAMLs, validate all 11 graphs and write the pending report only.")
    args = parser.parse_args()
    run_queue(resume=args.resume, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
