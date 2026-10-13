"""Phase 0 dataset statistics (v1.1 18.1 / Phase 0 acceptance).

Reads a YOLO-format dataset yaml and its labels (analysis only; no training/eval):
per-class instances, per-image object distribution, normalized area-ratio
distribution, empty/invalid label counts and train/val split confirmation.

Usage:
    python scripts/dataset_stats.py ultralytics/cfg/datasets/SCB-Dataset3-S.yaml [--save-json path]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _load_labels(lbl_dir: Path, stem: str) -> list[tuple[int, float, float, float, float]]:
    lbl = lbl_dir / f"{stem}.txt"
    rows: list[tuple[int, float, float, float, float]] = []
    if not lbl.exists():
        return rows
    for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            c, cx, cy, w, h = (float(x) for x in parts)
        except ValueError:
            continue
        rows.append((int(c), cx, cy, w, h))
    return rows


def dataset_stats(data_yaml: str, save_json: str | None = None) -> dict:
    """Compute dataset statistics for a YOLO-format dataset."""
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"])
    names: list[str] = list(cfg["names"])
    nc = int(cfg.get("nc", len(names)))
    out: dict = {"dataset": str(data_yaml), "path": str(root), "names": names, "nc": nc, "splits": {}}
    for split in ("train", "val"):
        img_rel = cfg.get(split)
        if not img_rel:
            continue
        img_dir = root / img_rel
        lbl_dir = root / str(img_rel).replace("images", "labels")
        imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        per_class = np.zeros(nc, dtype=int)
        per_image: list[int] = []
        areas: list[float] = []
        invalid = 0
        empty = 0
        for img in imgs:
            rows = _load_labels(lbl_dir, img.stem)
            n = 0
            for c, cx, cy, w, h in rows:
                if c >= nc or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    invalid += 1
                    continue
                per_class[c] += 1
                areas.append(w * h)
                n += 1
            if n == 0:
                empty += 1
            per_image.append(n)
        arr = np.asarray(per_image, dtype=float)
        area = np.asarray(areas, dtype=float)
        out["splits"][split] = {
            "images": len(imgs),
            "instances": int(per_class.sum()),
            "per_class_instances": {names[i]: int(per_class[i]) for i in range(nc)},
            "per_image": {
                "mean": round(float(arr.mean()), 2),
                "median": float(np.median(arr)),
                "max": int(arr.max()) if arr.size else 0,
                "p90": float(np.percentile(arr, 90)),
                "empty_images": empty,
            },
            "invalid_labels": invalid,
            "area_ratio": (
                {
                    "mean": round(float(area.mean()), 5),
                    "p25": round(float(np.percentile(area, 25)), 5),
                    "median": round(float(np.percentile(area, 50)), 5),
                    "p75": round(float(np.percentile(area, 75)), 5),
                }
                if area.size
                else None
            ),
        }
    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(out)
    return out


def _print(out: dict) -> None:
    print(f"dataset: {out['dataset']}  ({out['nc']} classes: {', '.join(out['names'])})")
    for split, s in out["splits"].items():
        print(f"  [{split}] images={s['images']} instances={s['instances']} "
              f"per_image(mean/median/max/p90)={s['per_image']['mean']}/{s['per_image']['median']}/"
              f"{s['per_image']['max']}/{s['per_image']['p90']} empty={s['per_image']['empty_images']} "
              f"invalid={s['invalid_labels']}")
        print(f"      per-class: {s['per_class_instances']}")
        if s["area_ratio"]:
            print(f"      area_ratio(p25/median/p75/mean)={s['area_ratio']['p25']}/"
                  f"{s['area_ratio']['median']}/{s['area_ratio']['p75']}/{s['area_ratio']['mean']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=str)
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args(argv)
    dataset_stats(args.data_yaml, args.save_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
