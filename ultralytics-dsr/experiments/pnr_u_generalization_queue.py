"""Local sequential paired three-seed PNR-DFL generalization queue for SCB-Dataset3-U."""

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
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("runs/pnr_u_generalization_3seed")
DATA_YAML = "ultralytics/cfg/datasets/SCB-Dataset3-U.yaml"


@dataclass(frozen=True)
class Job:
    """One fixed-protocol baseline or PNR training cell."""

    variant: str
    seed: int
    yaml: str
    run_dir: str


def build_jobs() -> list[Job]:
    """Return the complete paired six-class grid, including all three seeds."""
    variants = {
        "baseline": "ultralytics/cfg/models/11/yolo11n-baseline-matched.yaml",
        "pnr_dfl": "ultralytics/cfg/models/11/yolo11n-pnr-dfl.yaml",
    }
    return [
        Job(variant, seed, yaml, str(OUTPUT_ROOT / variant / f"seed{seed}"))
        for variant, yaml in variants.items()
        for seed in (42, 43, 44)
    ]


def _pid_is_running(pid: int) -> bool:
    """Check a Windows process ID without adding a dependency."""
    if pid <= 0:
        return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def _gpu_memory_mib() -> int | None:
    """Return used GPU 0 memory, or None when nvidia-smi is unavailable."""
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode or not completed.stdout.strip():
        return None
    return int(completed.stdout.splitlines()[0].strip())


def _is_complete(run_dir: Path) -> bool:
    """Accept a completed cell only when metrics and selected weights both exist."""
    return (run_dir / "results.csv").is_file() and (run_dir / "weights" / "best.pt").is_file()


def _training_source(job: Job) -> str:
    """Generate a self-contained training program for one U benchmark cell."""
    project = Path(job.run_dir).parent.as_posix()
    name = Path(job.run_dir).name
    return f'''from ultralytics import YOLO
model = YOLO("{job.yaml}")
model.load("yolo11n.pt")
model.train(
    data="{DATA_YAML}", epochs=300, patience=100,
    imgsz=640, batch=16, device=0, workers=8, pretrained=False, optimizer="auto",
    seed={job.seed}, deterministic=True, cos_lr=False, close_mosaic=10, amp=True,
    lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    project="{project}", name="{name}", exist_ok=False,
    save=True, save_period=-1, val=True, plots=True, verbose=True,
)
'''


def _write_state(path: Path, state: dict) -> None:
    """Atomically persist state to prevent an observer reading a partial JSON file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_queue(wait_pid: int, python: str, max_gpu_memory_mib: int, poll_seconds: int) -> int:
    """Run missing U cells serially, stopping at the first abnormal exit."""
    output_root = WORKSPACE / OUTPUT_ROOT
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "queue_state.json"
    jobs = build_jobs()
    state = {
        "benchmark": "SCB-Dataset3-U official six-class train/val split",
        "data_yaml": DATA_YAML,
        "wait_pid": wait_pid,
        "jobs": {f"{job.variant}_seed{job.seed}": {**asdict(job), "status": "pending"} for job in jobs},
    }
    _write_state(state_path, state)

    while _pid_is_running(wait_pid):
        time.sleep(poll_seconds)
    while (used := _gpu_memory_mib()) is not None and used > max_gpu_memory_mib:
        time.sleep(poll_seconds)

    for job in jobs:
        key = f"{job.variant}_seed{job.seed}"
        run_dir = WORKSPACE / job.run_dir
        if _is_complete(run_dir):
            state["jobs"][key]["status"] = "already_completed"
            _write_state(state_path, state)
            continue
        if run_dir.exists():
            state["jobs"][key]["status"] = "blocked_incomplete_output"
            _write_state(state_path, state)
            return 2

        state["jobs"][key]["status"] = "running"
        _write_state(state_path, state)
        encoded = base64.b64encode(_training_source(job).encode("utf-8")).decode("ascii")
        command = [python, "-c", f"import base64;exec(compile(base64.b64decode('{encoded}'),'pnr_u_{key}','exec'))"]
        environment = os.environ | {"POLARS_SKIP_CPU_CHECK": "1"}
        with (
            logs / f"{key}.out"
        ).open("w", encoding="utf-8") as stdout, (logs / f"{key}.err").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=WORKSPACE, stdout=stdout, stderr=stderr, env=environment, check=False)
        state["jobs"][key]["returncode"] = completed.returncode
        state["jobs"][key]["status"] = "completed" if completed.returncode == 0 and _is_complete(run_dir) else "failed"
        _write_state(state_path, state)
        if state["jobs"][key]["status"] == "failed":
            return completed.returncode or 1

    state["status"] = "completed"
    _write_state(state_path, state)
    return 0


def main() -> int:
    """Run from the workspace root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-gpu-memory-mib", type=int, default=4000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    return run_queue(args.wait_pid, args.python, args.max_gpu_memory_mib, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
