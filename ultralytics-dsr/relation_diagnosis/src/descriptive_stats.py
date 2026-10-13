"""Exp0 basic audit + Exp1 image entropy/dominance + Exp2 co-occurrence/PMI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_dataset(df: pd.DataFrame, cfg: dict) -> dict:
    """Exp0: basic data audit per split."""
    names = list(cfg["names"])
    out: dict[str, dict] = {}
    for split, g in df.groupby("split"):
        img_counts = g.groupby("image_id").size()
        n_img = len(img_counts)
        n_inst = len(g)
        per_class = g.groupby("class_id").size().reindex(range(len(names)), fill_value=0)
        counts = per_class.to_dict()
        freqs = {c: int((g["class_id"] == c).sum()) for c in range(len(names))}
        img_freq = g.drop_duplicates("image_id").groupby("class_id").size().reindex(range(len(names)), fill_value=0)
        n_empty = int((img_counts == 0).sum())
        n_single = int((img_counts == 1).sum())
        n_multi = int((img_counts >= 2).sum())
        areas = g["area"]
        wh = g["w"] / g["h"].replace(0, np.nan)
        imbalance = float(per_class.max() / max(1, per_class.min()))
        out[split] = {
            "images": n_img,
            "instances": n_inst,
            "classes": len(names),
            "per_class": counts,
            "per_class_freq": freqs,
            "class_image_freq": img_freq.to_dict(),
            "avg_per_image": float(img_counts.mean()),
            "median_per_image": float(img_counts.median()),
            "max_per_image": int(img_counts.max()),
            "p90_per_image": float(np.percentile(img_counts, 90)),
            "empty_images": n_empty,
            "single_object_ratio": n_single / max(1, n_img),
            "multi_object_ratio": n_multi / max(1, n_img),
            "relation_valid_ratio": n_multi / max(1, n_img),
            "imbalance_ratio": imbalance,
            "area_mean": float(areas.mean()),
            "area_median": float(areas.median()),
            "area_p25": float(areas.quantile(0.25)),
            "area_p75": float(areas.quantile(0.75)),
            "aspect_mean": float(wh.mean()),
            "aspect_median": float(wh.median()),
        }
    return out


def image_dominance_entropy(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Exp1: per-image max-class dominance D and normalised entropy H_norm."""
    nc = int(cfg["nc"])
    logc = np.log(max(2, nc))
    rows = []
    for (image_id, split), g in df.groupby(["image_id", "split"]):
        counts = g["class_id"].value_counts().reindex(range(nc), fill_value=0).to_numpy(dtype=float)
        n = counts.sum()
        if n == 0:
            continue
        p = counts / n
        dom = float(p.max())
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "n_objects": int(n),
                "dominance": dom,
                "entropy": ent,
                "entropy_norm": ent / logc,
            }
        )
    return pd.DataFrame(rows)


def entropy_summary(ent_df: pd.DataFrame) -> dict:
    """Table 2 aggregate: mean/median dominance, mean entropy, low-entropy ratio."""
    d = ent_df["dominance"]
    h = ent_df["entropy_norm"]
    return {
        "images": int(len(ent_df)),
        "mean_dominance": float(d.mean()),
        "median_dominance": float(d.median()),
        "mean_entropy_norm": float(h.mean()),
        "high_dominance_ratio_D>0.7": float((d > 0.7).mean()),
        "low_entropy_ratio_H<0.4": float((h < 0.4).mean()),
        "mixed_ratio_H>0.7": float((h > 0.7).mean()),
    }


def cooccurrence(df: pd.DataFrame, cfg: dict) -> dict:
    """Exp2: co-occurrence counts, conditional probabilities and PMI."""
    names = list(cfg["names"])
    nc = len(names)
    n_images = df["image_id"].nunique()
    img_classes = df.groupby("image_id")["class_id"].apply(lambda s: set(s.tolist()))
    C = np.zeros((nc, nc), dtype=float)
    for s in img_classes:
        for a in s:
            for b in s:
                C[a, b] += 1.0
    Pa = C.diagonal() / n_images  # P(a) = fraction of images containing a
    Pab = C / n_images
    Pba = np.where(Pa[:, None] > 0, Pab / np.maximum(Pa[:, None], 1e-12), 0.0)  # P(b | a)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(Pab / np.maximum(np.outer(Pa, Pa), 1e-12))
    pmi[np.isinf(pmi)] = np.nan
    return {
        "names": names,
        "count": C,
        "cond": Pba,
        "pmi": pmi,
        "p_class_image": Pa,
    }


def class_frequency_table(df: pd.DataFrame, cfg: dict, split: str) -> pd.DataFrame:
    names = list(cfg["names"])
    g = df[df["split"] == split]
    rows = []
    for c, name in enumerate(names):
        n = int((g["class_id"] == c).sum())
        nimg = int(g[g["class_id"] == c]["image_id"].nunique())
        rows.append({"class": name, "instances": n, "images": nimg})
    return pd.DataFrame(rows)


def layout_duplicate_check(df: pd.DataFrame, decimals: int = 2) -> dict:
    """Section 16 leakage check: fraction of val images whose box layout (rounded
    coordinates + class ids) exactly matches a train image layout."""
    def layout_hash(g: pd.DataFrame) -> tuple:
        arr = g[["class_id", "xc", "yc", "w", "h"]].round(decimals)
        return tuple(map(tuple, sorted(arr.to_numpy().tolist())))

    train_set = set()
    train_counts: dict[str, int] = {}
    for img_id, g in df[df["split"] == "train"].groupby("image_id"):
        h = layout_hash(g)
        train_set.add(h)
        train_counts[img_id] = 1
    val_total = 0
    val_match = 0
    matched_ids: list[str] = []
    for img_id, g in df[df["split"] == "val"].groupby("image_id"):
        val_total += 1
        if layout_hash(g) in train_set:
            val_match += 1
            matched_ids.append(img_id)
    return {
        "decimals": decimals,
        "val_images": val_total,
        "val_duplicate_layout": val_match,
        "val_duplicate_ratio": val_match / max(1, val_total),
        "example_ids": matched_ids[:10],
    }
