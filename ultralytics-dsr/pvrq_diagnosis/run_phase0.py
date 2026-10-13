"""Command-line entry point for immutable PVR-Q Phase 0 baseline evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support both ``python -m pvrq_diagnosis.run_phase0`` and the documented
# ``python .\\pvrq_diagnosis\\run_phase0.py`` invocation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvrq_diagnosis.phase0_runner import run_phase0


def build_parser() -> argparse.ArgumentParser:
    """Build the stable Phase 0 command-line interface."""
    parser = argparse.ArgumentParser(description="Generate immutable PVR-Q Phase 0 instance records.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--min-oracle-gain", type=float, default=0.0)
    parser.add_argument("--reuse-records", action="store_true")
    return parser


def main() -> None:
    """Run Phase 0 record collection from parsed command-line arguments."""
    args = build_parser().parse_args()
    output_dir = run_phase0(args)
    print(output_dir)


if __name__ == "__main__":
    main()
