"""Serial, resumable seed-42 reproduction of the completed DSA-Det experiment family on 5k_HRW."""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from report_hrw_seed42_all_experiments import write_report


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("runs/hrw_seed42_all_experiments")
DATA_YAML = "ultralytics/cfg/datasets/5k-HRW.yaml"
SEED = 42
PRETRAINED = "yolo11n.pt"
PROTOCOL = {
    "epochs": 300,
    "patience": 100,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 8,
    "optimizer": "auto",
    "seed": SEED,
    "deterministic": True,
    "cos_lr": False,
    "close_mosaic": 10,
    "amp": True,
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "save": True,
    "save_period": -1,
    "val": True,
    "plots": True,
    "verbose": True,
}
PNR_CONFIG = {"alpha": 1.0, "lambda_pnr": 0.25, "scale_floor": 1.0, "normalize": True, "weight_mode": "joint"}
# The user requested that unfinished PNR mechanism ablations and every PNR
# stacking experiment be skipped. Completed cells remain auditable.
USER_SKIPPED_JOBS = {
    "norm_size",
    "norm_vertical",
    "pnr_sled",
    "pnr_bcsq",
    "pnr_afpn",
    "pnr_fadwc",
    "pnr_dyhead",
}
SKIP_REASON = "按用户要求跳过：不再执行 PNR 机制消融与 PNR 叠加验证。"


@dataclass(frozen=True)
class Job:
    """One explicitly named experiment cell; all output paths are private to this queue."""

    name: str
    display_name: str
    category: str
    kind: str
    yaml: str | None
    run_dir: str
    normalize: bool | None = None
    weight_mode: str | None = None


def build_jobs() -> list[Job]:
    """Return all prior research directions once under the requested seed-42 protocol."""
    root = OUTPUT_ROOT
    model = "ultralytics/cfg/models/11/"
    return [
        Job("baseline", "YOLO11n Baseline", "基线", "train", model + "yolo11n-baseline-matched.yaml", str(root / "baseline" / "seed42")),
        Job("pvrq_phase0", "PVR-Q Phase-0 诊断", "诊断", "phase0", None, str(root / "pvrq_phase0")),
        Job("pvrf", "PVRF", "单模块", "train", model + "yolo11n-pvrf.yaml", str(root / "pvrf" / "seed42")),
        Job("vbr", "VBR", "单模块", "train", model + "yolo11n-vbr.yaml", str(root / "vbr" / "seed42")),
        Job("bcsq", "BCSQ-Head", "单模块", "train", model + "yolo11n-bcsq.yaml", str(root / "bcsq" / "seed42")),
        Job("sled", "SLED-Head", "单模块", "train", model + "yolo11n-sled.yaml", str(root / "sled" / "seed42")),
        Job("pnr_dfl", "PNR-DFL（完整）", "单模块", "train", model + "yolo11n-pnr-dfl.yaml", str(root / "pnr_dfl" / "seed42"), True, "joint"),
        Job("sera", "SERA", "单模块", "train", model + "yolo11n-sera.yaml", str(root / "sera" / "seed42")),
        Job("hbpc", "HBPC-Head", "单模块", "train", model + "yolo11n-hbpc.yaml", str(root / "hbpc" / "seed42")),
        Job("raw_uniform", "PNR raw_uniform", "PNR 机制消融", "train", model + "yolo11n-pnr-dfl.yaml", str(root / "raw_uniform" / "seed42"), False, "uniform"),
        Job("norm_uniform", "PNR norm_uniform", "PNR 机制消融", "train", model + "yolo11n-pnr-dfl.yaml", str(root / "norm_uniform" / "seed42"), True, "uniform"),
        Job("norm_size", "PNR norm_size", "PNR 机制消融", "train", model + "yolo11n-pnr-dfl.yaml", str(root / "norm_size" / "seed42"), True, "size"),
        Job("norm_vertical", "PNR norm_vertical", "PNR 机制消融", "train", model + "yolo11n-pnr-dfl.yaml", str(root / "norm_vertical" / "seed42"), True, "vertical"),
        Job("composition", "Composition-Aware Auxiliary Learning", "早期方法复验", "composition", model + "yolo11n-comp.yaml", str(root / "composition" / "seed42")),
        Job("dsr", "DSR-Det", "早期方法复验", "dsr", model + "yolo11n-dsr.yaml", str(root / "dsr" / "seed42")),
        Job("pnr_sled", "PNR-DFL + SLED-Head", "PNR 叠加", "train", model + "yolo11n-pnr-sled.yaml", str(root / "pnr_sled" / "seed42")),
        Job("pnr_bcsq", "PNR-DFL + BCSQ-Head", "PNR 叠加", "train", model + "yolo11n-pnr-bcsq.yaml", str(root / "pnr_bcsq" / "seed42")),
        Job("pnr_afpn", "PNR-DFL + AFPN", "PNR 叠加", "train", model + "yolo11n-pnr-afpn.yaml", str(root / "pnr_afpn" / "seed42")),
        Job("pnr_fadwc", "PNR-DFL + FADWC-Head", "PNR 叠加", "train", model + "yolo11n-pnr-fadwc.yaml", str(root / "pnr_fadwc" / "seed42")),
        Job("pnr_dyhead", "PNR-DFL + 轻量 DyHead", "PNR 叠加", "train", model + "yolo11n-pnr-dyhead.yaml", str(root / "pnr_dyhead" / "seed42")),
    ]


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _gpu_memory_mib() -> int | None:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return int(result.stdout.splitlines()[0].strip()) if not result.returncode and result.stdout.strip() else None


def _train_complete(run_dir: Path) -> bool:
    return (run_dir / "results.csv").is_file() and (run_dir / "weights" / "best.pt").is_file()


def _phase0_complete(run_dir: Path) -> bool:
    return (run_dir / "phase0_summary.json").is_file() and (run_dir / "phase0_report.md").is_file()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _training_source(job: Job) -> str:
    """Build a self-contained training program with one of three documented trainer modes."""
    if job.kind == "dsr":
        baseline = (WORKSPACE / OUTPUT_ROOT / "baseline" / "seed42" / "weights" / "best.pt").as_posix()
        return f'''from ultralytics import YOLO
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer
model = YOLO("{job.yaml}")
model.load("{baseline}")
model.train(
    trainer=DSRDetectionTrainer, data="{DATA_YAML}", epochs=70, patience=30,
    imgsz=640, batch=16, device=0, workers=8, pretrained=False, optimizer="auto",
    seed={SEED}, deterministic=True, cos_lr=False, close_mosaic=9999, amp=True,
    lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    project="{Path(job.run_dir).parent.as_posix()}", name="{Path(job.run_dir).name}", exist_ok=False,
    save=True, save_period=-1, val=True, plots=True, verbose=True,
)
'''
    trainer = ""
    if job.kind == "composition":
        trainer = "from ultralytics.models.yolo.comp.train import CompositionTrainer\n"
    pnr_override = ""
    if job.normalize is not None and job.weight_mode is not None:
        config = {**PNR_CONFIG, "normalize": job.normalize, "weight_mode": job.weight_mode}
        pnr_override = f"model.model.yaml['pnr'] = {config!r}\nprint('PNR_CONFIG', model.model.yaml['pnr'], flush=True)\n"
    kwargs = {**PROTOCOL, "project": Path(job.run_dir).parent.as_posix(), "name": Path(job.run_dir).name, "exist_ok": False}
    trainer_arg = "trainer=CompositionTrainer, " if job.kind == "composition" else ""
    return f'''from ultralytics import YOLO
{trainer}model = YOLO("{job.yaml}")
model.load("{PRETRAINED}")
{pnr_override}model.train({trainer_arg}data="{DATA_YAML}", **{kwargs!r})
'''


def _phase0_source(job: Job) -> str:
    baseline = (WORKSPACE / OUTPUT_ROOT / "baseline" / "seed42" / "weights" / "best.pt").as_posix()
    return f'''from pvrq_diagnosis.run_phase0 import main
import sys
sys.argv = ["run_phase0", "--weights", "{baseline}", "--data", "{DATA_YAML}", "--output", "{job.run_dir}", "--imgsz", "640", "--device", "0"]
main()
'''


def _run_source(source: str, label: str, python: str, out_path: Path, err_path: Path) -> int:
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    command = [python, "-c", f"import base64;exec(compile(base64.b64decode('{encoded}'),'{label}','exec'))"]
    environment = os.environ | {"POLARS_SKIP_CPU_CHECK": "1"}
    with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
        return subprocess.run(command, cwd=WORKSPACE, stdout=stdout, stderr=stderr, env=environment, check=False).returncode


def _initial_state(wait_pid: int, jobs: list[Job]) -> dict:
    return {
        "benchmark": "5k_HRW all prior DSA-Det experiments, one seed (42)",
        "dataset": {"yaml": DATA_YAML, "seed": SEED, "class_names": ["class_0", "class_1", "class_2"]},
        "protocol": PROTOCOL,
        "wait_pid": wait_pid,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": {job.name: {**asdict(job), "status": "pending"} for job in jobs},
    }


def _save_state_and_report(output_root: Path, state: dict) -> None:
    _write_json(output_root / "queue_state.json", state)
    write_report(output_root)


def run_queue(wait_pid: int, python: str, max_gpu_memory_mib: int, poll_seconds: int) -> int:
    """Wait for the prior queue, run each fresh cell serially, and preserve an auditable state/report after every cell."""
    output_root = WORKSPACE / OUTPUT_ROOT
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    state_path = output_root / "queue_state.json"
    # Resume rather than recreate the state after a user-directed stop. This
    # preserves all completed outputs and makes skipped jobs explicit.
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else _initial_state(wait_pid, jobs)
    _write_json(output_root / "queue_manifest.json", {"jobs": [asdict(job) for job in jobs], "protocol": PROTOCOL})
    _save_state_and_report(output_root, state)
    while _pid_is_running(wait_pid):
        time.sleep(poll_seconds)
    while (used := _gpu_memory_mib()) is not None and used > max_gpu_memory_mib:
        time.sleep(poll_seconds)
    for job in jobs:
        job_state = state["jobs"][job.name]
        run_dir = WORKSPACE / job.run_dir
        if job.name in USER_SKIPPED_JOBS:
            job_state["status"] = "skipped_by_user"
            job_state["skip_reason"] = SKIP_REASON
            job_state["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _save_state_and_report(output_root, state)
            continue
        complete = _phase0_complete(run_dir) if job.kind == "phase0" else _train_complete(run_dir)
        if complete:
            job_state["status"] = "already_completed"
            _save_state_and_report(output_root, state)
            continue
        if run_dir.exists():
            job_state["status"] = "blocked_incomplete_output"
            _save_state_and_report(output_root, state)
            return 2
        job_state["status"] = "running"
        job_state["started_at"] = datetime.now().isoformat(timespec="seconds")
        _save_state_and_report(output_root, state)
        source = _phase0_source(job) if job.kind == "phase0" else _training_source(job)
        code = _run_source(source, f"hrw_{job.name}", python, logs / f"{job.name}.out", logs / f"{job.name}.err")
        complete = _phase0_complete(run_dir) if job.kind == "phase0" else _train_complete(run_dir)
        job_state["returncode"] = code
        job_state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        job_state["status"] = "completed" if code == 0 and complete else "failed"
        _save_state_and_report(output_root, state)
        if job_state["status"] == "failed":
            return code or 1
    state["status"] = "completed"
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _save_state_and_report(output_root, state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-gpu-memory-mib", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    return run_queue(args.wait_pid, args.python, args.max_gpu_memory_mib, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
