"""Exp7/8/11 + bootstrap: Neighbor-only prediction, GT social oracle,
class-wise RPG and image-level bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .relation_stats import _pair_distances


def build_instance_features(df: pd.DataFrame, cfg: dict, K: int = 5, tau: float = 0.1) -> dict:
    """Per-instance feature sets (geometry / global / KNN / full-neighbour oracle)."""
    nc = int(cfg["nc"])
    geom_dim, glob_dim, neigh_dim = 5, nc + 3, 2 * nc + 5
    rows: list[dict] = []
    for (image_id, split), g in df.groupby(["image_id", "split"]):
        labels = g["class_id"].to_numpy()
        centers = g[["xc", "yc"]].to_numpy()
        widths = g["w"].to_numpy()
        heights = g["h"].to_numpy()
        areas = g["area"].to_numpy()
        n = len(labels)
        if n < 2:
            continue
        dmat = _pair_distances(centers, heights, False)
        order = np.argsort(dmat, axis=1)
        counts_img = np.bincount(labels, minlength=nc)
        k = min(K, n - 1)
        for i in range(n):
            geom = np.array([centers[i, 0], centers[i, 1], widths[i], heights[i], areas[i]], dtype=np.float32)
            # global context excluding center
            other_counts = counts_img.astype(float).copy()
            other_counts[labels[i]] -= 1.0
            other_n = max(n - 1, 1)
            props = other_counts / other_n
            pos = props[props > 0]
            ent = float(-(pos * np.log(pos)).sum()) if len(pos) else 0.0
            dom = float(props.max()) if other_n else 0.0
            glob = np.concatenate(([float(n)], props, [ent / max(np.log(max(2, nc)), 1e-12), dom])).astype(np.float32)
            # KNN neighbours
            nb = order[i, :k]
            nb_labels = labels[nb]
            nb_d = dmat[i, nb]
            nb_areas = areas[nb]
            counts = np.bincount(nb_labels, minlength=nc).astype(np.float32)
            wcounts = np.zeros(nc, dtype=np.float32)
            for c in range(nc):
                wcounts[c] = float(np.exp(-nb_d[nb_labels == c] / tau).sum())
            scale_diff = float(np.mean(np.abs(np.log(np.maximum(widths[i] / np.maximum(widths[nb], 1e-6), 1e-4)))))
            neigh = np.concatenate(
                (
                    counts,
                    wcounts,
                    [
                        float(nb_d.min()),
                        float(nb_d.mean()),
                        float(nb_d.std()),
                        float(nb_areas.mean()),
                        scale_diff,
                    ],
                )
            ).astype(np.float32)
            # full-neighbour oracle (all other objects)
            nb_all = order[i, : n - 1]
            nb_all_labels = labels[nb_all]
            nb_all_d = dmat[i, nb_all]
            nb_all_areas = areas[nb_all]
            counts_all = np.bincount(nb_all_labels, minlength=nc).astype(np.float32)
            wcounts_all = np.zeros(nc, dtype=np.float32)
            for c in range(nc):
                wcounts_all[c] = float(np.exp(-nb_all_d[nb_all_labels == c] / tau).sum())
            scale_diff_all = float(
                np.mean(np.abs(np.log(np.maximum(widths[i] / np.maximum(widths[nb_all], 1e-6), 1e-4))))
            )
            oracle = np.concatenate(
                (
                    counts_all,
                    wcounts_all,
                    [
                        float(nb_all_d.min()),
                        float(nb_all_d.mean()),
                        float(nb_all_d.std()),
                        float(nb_all_areas.mean()),
                        scale_diff_all,
                    ],
                )
            ).astype(np.float32)
            rows.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "Y": int(labels[i]),
                    "geom": geom,
                    "global": glob,
                    "neigh": neigh,
                    "oracle": oracle,
                }
            )
    rec = pd.DataFrame(rows)
    feat = {
        "geom": np.stack(rec["geom"].to_numpy()),
        "global": np.stack(rec["global"].to_numpy()),
        "neigh": np.stack(rec["neigh"].to_numpy()),
    }
    feat["oracle"] = np.stack(rec["oracle"].to_numpy())
    return {"meta": rec[["image_id", "split", "Y"]], "feat": feat}


def _macro_f1_from_confusion(conf: np.ndarray) -> float:
    nc = conf.shape[0]
    f1s = []
    for c in range(nc):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        p = tp / max(tp + fp, 1e-12)
        r = tp / max(tp + fn, 1e-12)
        f1s.append(2 * p * r / max(p + r, 1e-12))
    return float(np.mean(f1s))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels_list: list[int]) -> dict:
    conf = confusion_matrix(y_true, y_pred, labels=labels_list)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)),
        "balanced_acc": float(recall_score(y_true, y_pred, average="macro", labels=labels_list, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": conf,
    }


def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    algo: str,
    seed: int = 0,
) -> tuple[object, np.ndarray]:
    if algo == "lr":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed
        )
    model.fit(X_train, y_train)
    return model, model.predict(X_val)


def run_prediction_pipeline(feats: dict, cfg: dict, seed: int = 0) -> dict:
    """B0-B6 + Oracle for both algorithms; returns metrics and per-image confusion."""
    nc = int(cfg["nc"])
    labels_list = list(range(nc))
    meta = feats["meta"]
    F = feats["feat"]
    train_mask = (meta["split"] == "train").to_numpy()
    val_mask = (meta["split"] == "val").to_numpy()
    y_train = meta["Y"].to_numpy()[train_mask]
    y_val = meta["Y"].to_numpy()[val_mask]
    train_ids = meta["image_id"].to_numpy()[train_mask]
    val_ids = meta["image_id"].to_numpy()[val_mask]

    prior = np.bincount(y_train, minlength=nc).astype(float)
    prior /= max(prior.sum(), 1e-12)
    rng = np.random.default_rng(seed)

    configs = {
        "B2_geometry": F["geom"],
        "B3_global": F["global"],
        "B4_neighbor": F["neigh"],
        "B5_geom_neighbor": np.concatenate([F["geom"], F["neigh"]], axis=1),
        "B6_geom_neighbor_global": np.concatenate([F["geom"], F["neigh"], F["global"]], axis=1),
        "Oracle_full": np.concatenate([F["geom"], F["oracle"], F["global"]], axis=1),
    }
    results: dict[str, dict] = {}
    for algo in ("lr", "tree"):
        for name, X in configs.items():
            model, pred_val = fit_predict(X[train_mask], y_train, X[val_mask], algo, seed)
            results[f"{name}_{algo}"] = {
                "name": name,
                "algo": algo,
                "pred": pred_val,
                "y_true": y_val,
            }
    # trivial baselines (algo-agnostic)
    maj = int(np.argmax(prior))
    for tag, pred in (
        ("B0_majority", np.full(len(y_val), maj)),
        ("B1_prior_random", rng.choice(nc, size=len(y_val), p=prior)),
    ):
        results[tag] = {"name": tag, "algo": "baseline", "pred": pred, "y_true": y_val}

    out: dict[str, dict] = {}
    for key, r in results.items():
        met = evaluate(r["y_true"], r["pred"], labels_list)
        # per-image confusion matrices
        conf_by_img: dict[str, np.ndarray] = {}
        for img_id in set(val_ids):
            m = val_ids == img_id
            conf_by_img[str(img_id)] = confusion_matrix(r["y_true"][m], r["pred"][m], labels=labels_list)
        out[key] = {"metrics": met, "per_image_conf": conf_by_img, "name": r["name"], "algo": r["algo"]}
    return {"results": out, "labels": labels_list, "val_ids": val_ids, "prior": prior, "nc": nc}


def bootstrap_macro_f1_diffs(pred_out: dict, name_a: str, name_b: str, B: int = 1000, seed: int = 0) -> dict:
    """Image-level bootstrap 95% CI of macro-F1(A) - macro-F1(B)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(list(pred_out["results"][name_a]["per_image_conf"].keys()))
    conf_a = np.stack([pred_out["results"][name_a]["per_image_conf"][i] for i in ids])
    conf_b = np.stack([pred_out["results"][name_b]["per_image_conf"][i] for i in ids])
    n_img = len(ids)
    diffs = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n_img, size=n_img)
        d = _macro_f1_from_confusion(conf_a[idx].sum(0)) - _macro_f1_from_confusion(conf_b[idx].sum(0))
        diffs[b] = d
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(not (lo <= 0 <= hi)),
        "B": B,
    }


def classwise_rpg(pred_out: dict, rel_name: str, geom_name: str) -> pd.DataFrame:
    """Exp11: per-class F1(relation) - F1(geometry)."""
    nc = pred_out["nc"]
    labels_list = list(range(nc))
    conf_rel = np.zeros((nc, nc))
    conf_geo = np.zeros((nc, nc))
    for img_id, c in pred_out["results"][rel_name]["per_image_conf"].items():
        conf_rel += c
    for img_id, c in pred_out["results"][geom_name]["per_image_conf"].items():
        conf_geo += c
    rows = []
    for c in labels_list:
        def f1(conf):
            tp = conf[c, c]
            fp = conf[:, c].sum() - tp
            fn = conf[c, :].sum() - tp
            p = tp / max(tp + fp, 1e-12)
            r = tp / max(tp + fn, 1e-12)
            return 2 * p * r / max(p + r, 1e-12)
        f1r, f1g = f1(conf_rel), f1(conf_geo)
        rows.append({"class": c, "F1_geometry": f1g, "F1_relation": f1r, "RPG_c": f1r - f1g})
    return pd.DataFrame(rows)
