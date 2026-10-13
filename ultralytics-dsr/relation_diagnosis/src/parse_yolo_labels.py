"""Unified YOLO label parsing -> image-level instance tables.

Every instance is normalised to [0,1] coordinates:
    image_id, split, class_id, class_name, xc, yc, w, h, area
Malformed / out-of-range lines are skipped and counted as invalid.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(cfg_path: str | Path) -> dict:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_label_file(label_file: Path, image_id: str, split: str, nc: int, names: list[str]) -> list[dict]:
    rows: list[dict] = []
    try:
        text = label_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return rows
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            c = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        if c < 0 or c >= nc:
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
            continue
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "class_id": c,
                "class_name": names[c] if c < len(names) else str(c),
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
                "area": w * h,
            }
        )
    return rows


def load_dataset(cfg_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Return (instances DataFrame for train+val, dataset config dict)."""
    cfg = load_config(cfg_path)
    root = Path(cfg["path"])
    names: list[str] = list(cfg["names"])
    nc = int(cfg["nc"])
    all_rows: list[dict] = []
    for split in ("train", "val"):
        label_dir = root / "labels" / split
        if not label_dir.is_dir():
            continue
        for label_file in sorted(label_dir.glob("*.txt")):
            all_rows.extend(parse_label_file(label_file, label_file.stem, split, nc, names))
    df = pd.DataFrame(all_rows, columns=["image_id", "split", "class_id", "class_name", "xc", "yc", "w", "h", "area"])
    return df, cfg
