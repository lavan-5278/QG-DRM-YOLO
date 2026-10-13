"""Summarize and compare Ultralytics/DSR run outputs (v1.1 20.3 / 20.4).

Reads only artifacts produced by ``model.train()`` / ``model.val()``:

- ``results.csv`` (official training metrics, last epoch);
- ``per_image_metrics.json`` (written by DSRDetectionValidator) for paired
  image-level bootstrap 95% CI between two experiments.

Directory convention: ``runs/<exp>/seed{0,42,3407}/`` (or runs directly under
``runs/<exp>/`` when not seeding).

Usage:
    python scripts/summarize_runs.py [runs_dir]
    python scripts/summarize_runs.py [runs_dir] --compare EXP_A EXP_B [--iters 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

METRICS = ["metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/recall(B)", "metrics/precision(B)"]


def _load_run(run_dir: Path) -> dict | None:
    """Read the last results.csv row and per-image metrics of a run directory."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    last = None
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last = row
    per_image = None
    p = run_dir / "per_image_metrics.json"
    if p.exists():
        try:
            per_image = json.loads(p.read_text(encoding="utf-8"))["images"]
        except Exception:
            per_image = None
    return {"dir": run_dir, "metrics": last, "per_image": per_image}


def _discover(runs_dir: Path) -> dict[str, dict[str, dict]]:
    """Group runs as {experiment: {seed: run}}."""
    groups: dict[str, dict[str, dict]] = {}
    for csv_path in sorted(runs_dir.rglob("results.csv")):
        run_dir = csv_path.parent
        if run_dir.name.startswith("seed"):
            exp, seed = run_dir.parent.name, run_dir.name
        else:
            exp, seed = run_dir.name, "default"
        run = _load_run(run_dir)
        if run is not None:
            groups.setdefault(exp, {})[seed] = run
    return groups


def summarize(runs_dir: Path) -> dict[str, dict]:
    """Per-experiment last-epoch metrics with mean +/- std over seeds."""
    groups = _discover(runs_dir)
    out: dict[str, dict] = {}
    for exp, seeds in sorted(groups.items()):
        rows = []
        for seed, run in sorted(seeds.items()):
            row = {"seed": seed}
            for m in METRICS:
                try:
                    row[m] = float(run["metrics"][m])
                except (TypeError, ValueError, KeyError):
                    row[m] = None
            row["has_per_image"] = run["per_image"] is not None
            rows.append(row)
        summary: dict = {"seeds": rows, "n_seeds": len(rows)}
        for m in METRICS:
            vals = [r[m] for r in rows if r[m] is not None]
            if vals:
                summary[f"{m}_mean"] = round(float(np.mean(vals)), 5)
                summary[f"{m}_std"] = round(float(np.std(vals)), 5) if len(vals) > 1 else None
        out[exp] = summary
    return out


def _per_image_diff(run_a: dict, run_b: dict, key: str) -> list[float]:
    """Paired per-image differences (exp_a minus exp_b) for one seed pair."""
    pa, pb = run_a["per_image"], run_b["per_image"]
    if not pa or not pb:
        return []
    ma = {x["im_file"]: x[key] for x in pa}
    mb = {x["im_file"]: x[key] for x in pb}
    return [ma[k] - mb[k] for k in ma if k in mb]


def compare(runs_dir: Path, exp_a: str, exp_b: str, iters: int = 2000, seed: int = 0) -> dict:
    """Paired image-level bootstrap 95% CI for DSR vs baseline style comparisons."""
    groups = _discover(runs_dir)
    if exp_a not in groups or exp_b not in groups:
        raise ValueError(f"experiments not found (have: {sorted(groups)})")
    seeds = sorted(set(groups[exp_a]) & set(groups[exp_b]))
    rng = np.random.default_rng(seed)
    result: dict = {"exp_a": exp_a, "exp_b": exp_b, "n_seeds": len(seeds), "metrics": {}}
    for key in ("ap50_95", "recall05"):
        diffs = []
        for s in seeds:
            diffs += _per_image_diff(groups[exp_a][s], groups[exp_b][s], key)
        if not diffs:
            result["metrics"][key] = {"error": "per_image_metrics.json missing (run DSRDetectionValidator)"}
            continue
        d = np.asarray(diffs, dtype=float)
        boot = [float(np.mean(rng.choice(d, size=d.size, replace=True))) for _ in range(iters)]
        lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        result["metrics"][key] = {
            "n_images": int(d.size),
            "mean_diff_A_minus_B": round(float(d.mean()), 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "significant": not (lo <= 0.0 <= hi),
            "iters": iters,
        }
    return result


def _print_summary(summary: dict[str, dict]) -> None:
    if not summary:
        print("no results.csv found under the runs dir")
        return
    print(f"{'experiment':<24} {'seeds':>5} {'mAP50-95 mean':>14} {'std':>8} {'has_per_image':>14}")
    for exp, s in summary.items():
        m = s.get("metrics/mAP50-95(B)_mean")
        sd = s.get("metrics/mAP50-95(B)_std")
        print(
            f"{exp:<24} {s['n_seeds']:>5} "
            f"{(f'{m:.5f}' if m is not None else '-'):>14} "
            f"{(f'{sd:.5f}' if sd is not None else '-'):>8} "
            f"{str(any(r['has_per_image'] for r in s['seeds'])):>14}"
        )


def _print_compare(result: dict) -> None:
    print(f"\npaired image-level bootstrap: {result['exp_a']} vs {result['exp_b']} "
          f"({result['n_seeds']} seeds)")
    for key, v in result["metrics"].items():
        if "error" in v:
            print(f"  {key}: {v['error']}")
            continue
        print(
            f"  {key}: mean diff (A-B) = {v['mean_diff_A_minus_B']:+.5f}  "
            f"95% CI [{v['ci95'][0]:.5f}, {v['ci95'][1]:.5f}]  "
            f"n_images={v['n_images']}  significant={'YES' if v['significant'] else 'no'}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", nargs="?", default="runs", type=Path)
    parser.add_argument("--compare", nargs=2, metavar=("EXP_A", "EXP_B"))
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.runs_dir.exists():
        print(f"runs dir not found: {args.runs_dir}", file=sys.stderr)
        return 1
    if args.compare:
        try:
            _print_compare(compare(args.runs_dir, args.compare[0], args.compare[1], args.iters, args.seed))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        _print_summary(summarize(args.runs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
