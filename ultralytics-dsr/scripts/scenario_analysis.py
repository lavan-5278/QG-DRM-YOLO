"""Read DSRDetectionValidator scenario reports (v1.1 19.x helper).

The reports are produced by model.val() with DSRDetectionValidator; this script only
prints them (v1.1 entry constraint: no standalone evaluation logic here).

Usage:
    python scripts/scenario_analysis.py [runs_dir]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", nargs="?", default="runs", type=Path)
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"runs dir not found: {args.runs_dir}", file=sys.stderr)
        return 1

    found = 0
    for report in sorted(args.runs_dir.rglob("dsr_scenario.json")):
        found += 1
        data = json.loads(report.read_text(encoding="utf-8"))
        print(f"--- {report} ---")
        print(f"images_seen: {data.get('images_seen')}  size_quantiles: {data.get('size_quantiles')}")
        print("candidate recall (v1.1 7.2):")
        for conf, v in (data.get("candidate_recall") or {}).items():
            print(f"  conf={conf}: recall_proposal@0.5={v.get('recall_proposal@0.5')} "
                  f"(covered {v.get('covered_gt')}/{v.get('n_gt')})")
        print("strata AP50-95 / Recall@0.5:")
        for group, strata in (data.get("strata") or {}).items():
            for name, m in strata.items():
                if "AP50-95" in m:
                    print(f"  {group}/{name}: AP50-95={m['AP50-95']} Recall={m['Recall@0.5']} "
                          f"n_gt={m['n_gt']} CI={m.get('bootstrap_CI_AP50-95')}")
        print("oracle:", data.get("oracle"))
    if not found:
        print("no dsr_scenario.json found (Phase 5 fills these via DSRDetectionValidator)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
