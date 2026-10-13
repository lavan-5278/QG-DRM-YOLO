"""Experiment matrix runner (v1.1 Phase 5).

Batch-launch comparison / ablation experiments through the official
``model.train()`` / ``model.val()`` entry points (no standalone train/val logic):

- reads a YAML matrix (default ``ultralytics/cfg/experiments/matrix.yaml``);
- for every experiment x seed, generates a per-run model yaml (DSR overrides
  applied, ``dsr.validator: true``) and a per-run cfg yaml, then trains via
  ``model.train()`` and validates via ``model.val(validator=DSRDetectionValidator)``;
- directory layout ``runs_root/<exp>/seed{...}/`` matches
  ``scripts/summarize_runs.py``, so multi-seed summary and paired bootstrap work
  directly on the same root;
- resumable: completed stages are skipped unless ``--force``; a failing run is
  isolated and the rest of the matrix continues; entries whose baseline weights
  are missing are reported as blocked instead of launched (run Stage A first);
  ``manifest.json`` records the status of every plan.

Usage::

    python scripts/run_experiments.py --matrix ultralytics/cfg/experiments/matrix.yaml
    python scripts/run_experiments.py --matrix ... --dry-run
    python scripts/run_experiments.py --matrix ... --only dsr_full ablate_r0_none --seeds 0
    python scripts/run_experiments.py --matrix ... --parallel 2
    python scripts/run_experiments.py --run-plan <plan.json>   # internal worker mode

Matrix schema (YAML)::

    global:
      runs_root: runs/matrix          # experiment root
      data: ultralytics/cfg/datasets/SCB-Dataset3-S.yaml
      imgsz: 640
      batch: 16
      device: 0
      workers: 8
      seeds: [0, 42, 3407]
      baseline_weights: runs/scb/yolo11n_baseline/weights/best.pt

    experiments:
      - name: dsr_full                  # unique experiment name
        cfg: ultralytics/cfg/experiments/scb_dsr_full.yaml   # template cfg
        model: ultralytics/cfg/models/11/yolo11n-dsr.yaml    # base model yaml
        trainer: dsr                    # "dsr" -> DSRDetectionTrainer, omit -> official
        load_weights: baseline          # "baseline" | explicit path | omit
        dsr: {relation_type: none}      # dsr block overrides (ablation switches)
        train: {epochs: 70}             # optional extra training args
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # worker subprocesses must import the vendored tree
# This conda env ships polars>=1.x whose CPU-flag parser raises a false-positive
# RuntimeError ("unknown feature flag: 'sse3'") on some CPUs, which would abort
# ultralytics trainer.save_model() at the end of every run. Bypass that check.
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
DEFAULT_MATRIX = ROOT / "ultralytics/cfg/experiments/matrix.yaml"
STATUS_NAME = "status.json"
MANIFEST_NAME = "manifest.json"
WORKER_LOG = "worker.log"


# --------------------------------------------------------------- yaml helpers
def _resolve(path) -> Path:
    """Resolve a path against the project root when it is relative."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


# -------------------------------------------------------------- status files
def _load_status(run_dir: Path) -> dict:
    p = run_dir / STATUS_NAME
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_status(run_dir: Path, status: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / STATUS_NAME).write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _is_done(run_dir: Path, force: bool = False) -> bool:
    """True when training and validation are both finished (unless --force)."""
    if force:
        return False
    s = _load_status(run_dir)
    return bool(
        s.get("train_done") and (s.get("val_done") or s.get("no_val")) and not s.get("error")
    )


# ----------------------------------------------------------------- plan build
def build_plan(exp: dict, seed: int, g: dict) -> dict:
    """Create the per-(experiment, seed) plan with all paths resolved."""
    name = str(exp["name"])
    run_dir = _resolve(Path(str(g["runs_root"])) / name / f"seed{seed}")
    base_model = _resolve(exp["model"])
    base_cfg = _resolve(exp["cfg"])
    load_weights = exp.get("load_weights")
    if load_weights is None:
        lw = None
    elif str(load_weights) == "baseline":
        # "baseline" may be a per-seed template, e.g.
        # runs/matrix/baseline/seed{seed}/weights/best.pt
        lw_tpl = str(g.get("baseline_weights", ""))
        lw = str(_resolve(lw_tpl.format(seed=seed) if "{seed}" in lw_tpl else lw_tpl))
    else:
        lw = str(_resolve(load_weights))
    return {
        "exp": name,
        "seed": int(seed),
        "run_dir": str(run_dir),
        "base_model": str(base_model),
        "base_cfg": str(base_cfg),
        "model_yaml": str(run_dir / f"{base_model.stem}_{name}.yaml"),
        "cfg_yaml": str(run_dir / "run_cfg.yaml"),
        "trainer": exp.get("trainer"),
        "load_weights": lw,
        "dsr": dict(exp.get("dsr") or {}),
        "train": dict(exp.get("train") or {}),
        "data": str(_resolve(g["data"])),
        "imgsz": int(g.get("imgsz", 640)),
        "batch": int(g.get("batch", 16)),
        "device": g.get("device", 0),
        "workers": int(g.get("workers", 8)),
        "project": str(_resolve(Path(str(g["runs_root"])) / name)),
        "name": f"seed{seed}",
    }


def build_plans(matrix_path, args) -> tuple[list[dict], dict]:
    """Load the matrix yaml and expand (experiment, seed) into a plan list."""
    matrix = _load_yaml(matrix_path)
    g = dict(matrix.get("global") or {})
    g["runs_root"] = str(g.get("runs_root", "runs/matrix"))
    if getattr(args, "device", None) is not None:
        g["device"] = args.device

    experiments = list(matrix.get("experiments") or [])
    only = set()
    for token in (getattr(args, "only", None) or []):
        only.update(t.strip() for t in str(token).split(",") if t.strip())
    if only:
        known = {e["name"] for e in experiments}
        missing = only - known
        if missing:
            raise ValueError(f"unknown experiments: {sorted(missing)} (have: {sorted(known)})")
        experiments = [e for e in experiments if e["name"] in only]

    seeds = [int(s) for s in (getattr(args, "seeds", None) or g.get("seeds") or [0])]
    plans = [build_plan(exp, seed, g) for exp in experiments for seed in seeds]
    for plan in plans:
        err = validate_plan(plan)
        if err:
            raise ValueError(err)
    return plans, g


def validate_plan(plan: dict) -> str | None:
    """Return an error message when a plan references missing base files."""
    tag = f"{plan['exp']}/seed{plan['seed']}"
    for key in ("base_model", "base_cfg"):
        if not Path(plan[key]).exists():
            return f"{tag}: missing {key}: {plan[key]}"
    return None


def plan_block_reason(plan: dict) -> str | None:
    """Return a reason when a plan cannot run yet (e.g. baseline weights missing)."""
    if plan["load_weights"] and not Path(plan["load_weights"]).exists():
        return f"missing baseline weights: {plan['load_weights']}"
    return None


# ------------------------------------------------------------ per-run configs
def write_run_files(plan: dict) -> dict:
    """Write the per-run model yaml (DSR overrides) and cfg yaml; returns plan."""
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if plan["trainer"] == "dsr":
        d = deepcopy(_load_yaml(plan["base_model"]))
        dsr = dict(d.get("dsr") or {})
        dsr.update(plan["dsr"])
        dsr.setdefault("validator", True)  # training-time val writes scenario reports
        d["dsr"] = dsr
        _dump_yaml(plan["model_yaml"], d)
    else:
        # official model yaml needs no copy
        plan["model_yaml"] = plan["base_model"]

    c = deepcopy(_load_yaml(plan["base_cfg"]))
    c.update(
        {
            "model": plan["model_yaml"],
            "data": plan["data"],
            "imgsz": plan["imgsz"],
            "batch": plan["batch"],
            "device": plan["device"],
            "workers": plan["workers"],
            "seed": plan["seed"],
            "project": plan["project"],
            "name": plan["name"],
            "exist_ok": True,
        }
    )
    c.update(plan["train"])
    _dump_yaml(plan["cfg_yaml"], c)
    return plan


# ---------------------------------------------------------------- worker mode
def _train(plan: dict) -> None:
    """Train one run through the official model.train() entry point."""
    from ultralytics import YOLO
    from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer

    model = YOLO(plan["model_yaml"])
    if plan["load_weights"]:
        model.load(plan["load_weights"])
    trainer = DSRDetectionTrainer if plan["trainer"] == "dsr" else None
    model.train(
        trainer=trainer,
        cfg=plan["cfg_yaml"],
        data=plan["data"],
        imgsz=plan["imgsz"],
        batch=plan["batch"],
        device=plan["device"],
        workers=plan["workers"],
        seed=plan["seed"],
        project=plan["project"],
        name=plan["name"],
        exist_ok=True,
    )


def _val(plan: dict) -> None:
    """Validate one run with DSRDetectionValidator so --compare has per-image data."""
    from ultralytics import YOLO
    from ultralytics.models.yolo.dsr.val import DSRDetectionValidator

    best = Path(plan["run_dir"]) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"no trained weights at {best}")
    model = YOLO(str(best))
    model.val(
        validator=DSRDetectionValidator,
        data=plan["data"],
        imgsz=plan["imgsz"],
        batch=plan["batch"],
        device=plan["device"],
        project=plan["project"],
        name=plan["name"],
        exist_ok=True,
        split="val",
    )


def run_worker(plan_path: Path) -> int:
    """Execute a single plan (train -> val), writing status.json incrementally."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if plan.get("force"):
        _save_status(run_dir, {})
    status = _load_status(run_dir)
    if _is_done(run_dir, force=plan.get("force", False)):
        print(f"[skip] {plan['exp']}/seed{plan['seed']} already done")
        return 0

    write_run_files(plan)
    try:
        if plan.get("force") or not status.get("train_done"):
            print(f"[train] {plan['exp']}/seed{plan['seed']} started")
            _train(plan)
            status["train_done"] = True
            _save_status(run_dir, status)
        if plan.get("no_val"):
            status["no_val"] = True
        elif plan.get("force") or not status.get("val_done"):
            print(f"[val] {plan['exp']}/seed{plan['seed']} started")
            _val(plan)
            status["val_done"] = True
            _save_status(run_dir, status)
        status["done"] = True
    except Exception as exc:  # noqa: BLE001 - isolate one run, keep the matrix going
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc()
        print(f"[error] {plan['exp']}/seed{plan['seed']}: {status['error']}", file=sys.stderr)
    _save_status(run_dir, status)
    return 0 if status.get("done") else 2


# ---------------------------------------------------------------- master mode
def _print_plan_table(plans: list[dict], force: bool) -> None:
    print(f"{'status':<8}{'experiment':<26}{'seed':>6}  note")
    for p in plans:
        if _is_done(Path(p["run_dir"]), force):
            tag, note = "SKIP", "already done"
        elif (reason := plan_block_reason(p)) and not force:
            tag, note = "BLOCK", reason
        else:
            tag, note = "RUN", p["model_yaml"]
        print(f"{tag:<8}{p['exp']:<26}{p['seed']:>6}  {note}")


def _print_summary(plans: list[dict]) -> None:
    print("\n=== matrix summary ===")
    print(f"{'experiment':<26}{'seed':>6}  {'train':<6}{'val':<6} status")
    for p in plans:
        s = _load_status(Path(p["run_dir"]))
        status = "ok" if _is_done(Path(p["run_dir"])) else (s.get("error") or "pending/failed")
        print(
            f"{p['exp']:<26}{p['seed']:>6}  "
            f"{str(bool(s.get('train_done'))):<6}{str(bool(s.get('val_done'))):<6} {status}"
        )


def write_manifest(runs_root: Path, plans: list[dict]) -> None:
    rows = []
    for p in plans:
        s = _load_status(Path(p["run_dir"]))
        reason = plan_block_reason(p)
        rows.append(
            {
                "experiment": p["exp"],
                "seed": p["seed"],
                "run_dir": p["run_dir"],
                "train_done": bool(s.get("train_done")),
                "val_done": bool(s.get("val_done")),
                "no_val": bool(s.get("no_val")),
                "error": s.get("error"),
                "blocked": reason,
                "done": _is_done(Path(p["run_dir"])),
            }
        )
    (runs_root / MANIFEST_NAME).write_text(
        json.dumps({"plans": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def launch(plans: list[dict], runs_root: Path, parallel: int, force: bool) -> None:
    """Spawn worker subprocesses (isolated per run), then summarize."""
    active: list[tuple[subprocess.Popen, Path]] = []
    done, failed, skipped, blocked = 0, 0, 0, 0

    def reap() -> int:
        """Remove finished workers from ``active`` and count them."""
        nonlocal done, failed
        n = 0
        for proc, plan_path in list(active):
            if proc.poll() is not None:
                active.remove((proc, plan_path))
                worker_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                s = _load_status(Path(worker_plan["run_dir"]))
                if s.get("error"):
                    failed += 1
                else:
                    done += 1
                n += 1
        return n

    def pump() -> None:
        """Wait until fewer than ``parallel`` workers are active (reaping in place)."""
        while len(active) >= parallel:
            time.sleep(5)
            reap()

    try:
        for plan in plans:
            run_dir = Path(plan["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            if _is_done(run_dir, force):
                skipped += 1
                print(f"[skip] {plan['exp']}/seed{plan['seed']} already done")
                continue
            reason = plan_block_reason(plan)
            if reason and not force:
                blocked += 1
                print(f"[blocked] {plan['exp']}/seed{plan['seed']}: {reason}")
                continue
            plan_path = run_dir / "plan.json"
            worker_plan = dict(plan)
            worker_plan["force"] = force
            plan_path.write_text(
                json.dumps(worker_plan, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log = open(run_dir / WORKER_LOG, "a", encoding="utf-8")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-plan", str(plan_path)]
            proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            active.append((proc, plan_path))
            print(f"[start] {plan['exp']}/seed{plan['seed']} -> {run_dir}")
            pump()

        while active:
            time.sleep(5)
            reap()
    except KeyboardInterrupt:
        print("\n[interrupt] terminating running workers...")
        for proc, _ in active:
            proc.terminate()
        raise
    finally:
        write_manifest(runs_root, plans)
    _print_summary(plans)
    print(f"\nfinished: {done} ok, {failed} failed, {skipped} skipped, {blocked} blocked "
          f"(manifest: {runs_root / MANIFEST_NAME})")


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX), help="matrix YAML path")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--only", nargs="*", help="experiment names to run (subset)")
    parser.add_argument("--seeds", nargs="*", type=int, help="override seeds")
    parser.add_argument("--device", default=None, help="override device (e.g. 0 or 'cpu')")
    parser.add_argument("--parallel", type=int, default=1, help="max concurrent workers")
    parser.add_argument("--force", action="store_true", help="rerun finished stages")
    parser.add_argument("--no-val", action="store_true", help="skip the validation stage")
    parser.add_argument("--run-plan", default=None, help="internal: execute one plan JSON")
    args = parser.parse_args(argv)

    if args.run_plan:
        return run_worker(Path(args.run_plan))

    try:
        plans, g = build_plans(args.matrix, args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    runs_root = _resolve(g["runs_root"])
    if args.dry_run:
        _print_plan_table(plans, args.force)
        return 0

    runs_root.mkdir(parents=True, exist_ok=True)
    launch(plans, runs_root, max(1, args.parallel), args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
