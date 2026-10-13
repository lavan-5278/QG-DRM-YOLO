# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Relation visualization (v1.1 19.5, Phase 3 "关系可视化接口"; paper Figure 2/4 material).

Runs the official ``model.predict()`` entry point on images and saves, per image:

- ``relation_stats.json``: scale fusion weights (classroom/region/neighbor), mean
  gate value and mean uncertainty over candidates;
- ``relation_<i>.png``: scale-weight bar + sorted gate-value curve.

Usage:
    from ultralytics.models.yolo.dsr.visualize import visualize_relation_stats
    visualize_relation_stats(model, "tests/data/smoke/images", save_dir="runs/relation_viz")
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import unwrap_model

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def visualize_relation_stats(
    model, source, save_dir="runs/relation_viz", imgsz=640, max_images=None, device=None
) -> list[dict]:
    """Predict images one-by-one (so the per-image MDRR aux is captured) and save stats/plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    paths = sorted(p for p in src.glob("*") if p.suffix.lower() in _IMG_SUFFIXES) if src.is_dir() else [src]
    if max_images is not None:
        paths = paths[: max_images]
    head = unwrap_model(model.model).model[-1]
    rows: list[dict] = []
    for i, p in enumerate(paths):
        results = model.predict(source=str(p), imgsz=imgsz, save=False, verbose=False, device=device)
        if not results:
            continue
        aux = getattr(head, "last_aux", None)
        mask = getattr(head, "last_cand_mask", None)
        if aux is None or mask is None or not mask.any():
            continue
        m = mask[0]
        scale_names = getattr(head.mdrr, "scale_names", ["g", "r", "n"])
        sw = aux["scale_weights"][0][m].mean(dim=0).cpu().numpy()
        gate = aux["gate"][0][m].mean(dim=1).cpu().numpy()
        unc = aux["uncertainty"][0][m].cpu().numpy()
        row = {
            "image": str(p),
            "n_candidates": int(m.sum()),
            "scale_weights": {name: round(float(v), 4) for name, v in zip(scale_names, sw.tolist())},
            "gate_mean": round(float(gate.mean()), 4),
            "uncertainty_mean": round(float(unc.mean()), 4),
        }
        rows.append(row)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(list(scale_names), sw.tolist(), color="steelblue")
        axes[0].set_title("scale fusion weights")
        axes[0].set_ylim(0, 1)
        axes[1].plot(np.sort(gate)[::-1])
        axes[1].set_title(f"gate values (n={len(gate)})")
        fig.suptitle(p.name)
        fig.tight_layout()
        fig.savefig(save_dir / f"relation_{i:03d}.png", dpi=120)
        plt.close(fig)
    (save_dir / "relation_stats.json").write_text(
        json.dumps({"images": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info(f"relation visualization saved to {save_dir} ({len(rows)} images)")
    return rows
