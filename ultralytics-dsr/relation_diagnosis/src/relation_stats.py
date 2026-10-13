"""Exp3-6 / Exp10 / Exp12: KNN same-class rate + permutation, distance-behavior
curve, neighbor transition matrix + lift, assortativity, homophily/heterophily
and mutual information. All geometry is label-only (model-free)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def _image_arrays(df: pd.DataFrame):
    """Yield per-image (labels, centers, heights, distances, sorted neighbour idx)."""
    for (image_id, split), g in df.groupby(["image_id", "split"]):
        labels = g["class_id"].to_numpy()
        centers = g[["xc", "yc"]].to_numpy()
        heights = g["h"].to_numpy()
        n = len(labels)
        yield image_id, split, labels, centers, heights, n


def _pair_distances(centers, heights, geo: bool, lambda_s: float = 1.0) -> np.ndarray:
    n = len(centers)
    if geo:
        dx = centers[:, 0][:, None] - centers[:, 0][None, :]
        dy = centers[:, 1][:, None] - centers[:, 1][None, :]
        ratio = np.maximum(heights[:, None] / np.maximum(heights[None, :], 1e-6), 1e-3)
        logh = np.log(np.clip(ratio, 1e-3, 1e3))
        d = np.sqrt(dx * dx + dy * dy + (lambda_s * logh) ** 2)
    else:
        d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    return d


def knn_permutation(
    df: pd.DataFrame,
    cfg: dict,
    K_list: tuple[int, ...] = (1, 3, 5, 8),
    B: int = 1000,
    seed: int = 0,
    geo: bool = False,
    lambda_s: float = 1.0,
) -> dict:
    """Exp3: KNN same-class rate R_K vs within-image label permutation."""
    nc = int(cfg["nc"])
    rng = np.random.default_rng(seed)
    total_edges = {K: 0 for K in K_list}
    total_same = {K: 0 for K in K_list}
    perm_same = {K: np.zeros(B) for K in K_list}
    rows = []  # per-image contributions (for bootstrap)
    for image_id, split, labels, centers, heights, n in _image_arrays(df):
        rec = {"image_id": image_id, "split": split}
        if n < 2:
            for K in K_list:
                rec[f"e_{K}"] = 0
                rec[f"s_{K}"] = 0
                rec[f"q_{K}"] = 0.0
            rows.append(rec)
            continue
        dmat = _pair_distances(centers, heights, geo, lambda_s)
        counts = np.bincount(labels, minlength=nc)
        q = float((counts * (counts - 1)).sum() / (n * (n - 1)))  # P(same) under perm
        kmax = min(max(K_list), n - 1)
        sorted_idx = np.argsort(dmat, axis=1)[:, :kmax]
        idx = np.argsort(rng.random((B, n)), axis=1)
        lp = labels[idx]  # (B, n)
        for K in K_list:
            k = min(K, n - 1)
            if k < 1:
                rec[f"e_{K}"] = 0
                rec[f"s_{K}"] = 0
                rec[f"q_{K}"] = 0.0
                continue
            pi = np.repeat(np.arange(n), k)
            pj = sorted_idx[:, :k].ravel()
            e = n * k
            s = int((labels[pi] == labels[pj]).sum())
            same_b = (lp[:, pi] == lp[:, pj]).sum(axis=1)  # (B,)
            total_edges[K] += e
            total_same[K] += s
            perm_same[K] += same_b
            rec[f"e_{K}"] = e
            rec[f"s_{K}"] = s
            rec[f"q_{K}"] = q * e
        rows.append(rec)
    per_image = pd.DataFrame(rows)
    result = {}
    for K in K_list:
        e = total_edges[K]
        if e == 0:
            result[K] = None
            continue
        r_obs = total_same[K] / e
        rates_perm = perm_same[K] / e
        mu = float(rates_perm.mean())
        sigma = float(rates_perm.std())
        z = (r_obs - mu) / max(sigma, 1e-12)
        p = float((1.0 + (rates_perm >= r_obs).sum()) / (B + 1.0))
        result[K] = {
            "K": K,
            "split": sorted(set(per_image["split"])) if len(per_image) else [],
            "edges": int(e),
            "same_obs": int(total_same[K]),
            "R_obs": float(r_obs),
            "perm_mean": mu,
            "perm_std": sigma,
            "delta_R": float(r_obs - mu),
            "z_score": float(z),
            "p_value": p,
            "B": B,
            "geo": geo,
        }
    return {"result": result, "per_image": per_image, "K_list": list(K_list)}


def distance_curve(df: pd.DataFrame, cfg: dict, mode: str = "quantile", nbins: int = 5) -> pd.DataFrame:
    """Exp4: P(same | distance bin) with analytic within-image permutation baseline."""
    nc = int(cfg["nc"])
    rows_all: list[dict] = []
    pair_data = []
    for split, g in df.groupby("split"):
        all_d: list[float] = []
        bins: list[int] = []
        same: list[int] = []
        qs: list[float] = []
        for image_id, labels, centers, heights, n in (
            (i, l, c, h, n_) for i, _, l, c, h, n_ in _image_arrays(g)
        ):
            if n < 2:
                continue
            dmat = _pair_distances(centers, heights, False)
            counts = np.bincount(labels, minlength=nc)
            q = float((counts * (counts - 1)).sum() / (n * (n - 1)))
            ii, jj = np.nonzero(np.isfinite(dmat))
            if len(ii) == 0:
                continue
            ds = dmat[ii, jj]
            all_d.extend(ds.tolist())
            same.extend((labels[ii] == labels[jj]).tolist())
            qs.extend([q] * len(ii))
        if not all_d:
            continue
        all_d = np.asarray(all_d)
        same = np.asarray(same, dtype=float)
        qs = np.asarray(qs, dtype=float)
        if mode == "quantile":
            edges = np.quantile(all_d, np.linspace(0, 1, nbins + 1))
            edges[0] = -1e-9
            edges[-1] = np.inf
            bin_ids = np.searchsorted(edges, all_d, side="right") - 1
        else:  # fixed bins
            fixed = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, np.inf]
            bin_ids = np.searchsorted(fixed, all_d, side="right") - 1
            edges = np.asarray(fixed, dtype=float)
            nbins = len(fixed) - 1
        for b in range(nbins):
            m = bin_ids == b
            cnt = int(m.sum())
            if cnt == 0:
                continue
            rate = float(same[m].mean())
            null = float(qs[m].mean())
            rows_all.append(
                {
                    "split": split,
                    "mode": mode,
                    "bin": b,
                    "d_min": round(float(edges[b]), 4),
                    "d_max": round(float(edges[b + 1]), 4) if np.isfinite(edges[b + 1]) else float("inf"),
                    "n_pairs": cnt,
                    "P_same_obs": rate,
                    "P_same_perm_null": null,
                    "delta_P_same": rate - null,
                }
            )
    return pd.DataFrame(rows_all)


def _knn_edges_per_image(df: pd.DataFrame, K: int, geo: bool = False, lambda_s: float = 1.0):
    """Yield (center_labels, neighbor_labels) directed KNN edges."""
    for image_id, split, labels, centers, heights, n in _image_arrays(df):
        if n < 2:
            continue
        dmat = _pair_distances(centers, heights, geo, lambda_s)
        k = min(K, n - 1)
        sorted_idx = np.argsort(dmat, axis=1)[:, :k]
        pi = np.repeat(np.arange(n), k)
        pj = sorted_idx.ravel()
        yield labels[pi], labels[pj]


def transition_matrix(df: pd.DataFrame, cfg: dict, K: int = 5) -> dict:
    """Exp5: T_ab = P(neighbor=b | center=a) and Lift = T_ab / P(b)."""
    nc = int(cfg["nc"])
    names = list(cfg["names"])
    T = np.zeros((nc, nc), dtype=float)
    for ci, nj in _knn_edges_per_image(df, K):
        T[ci, nj] += 1.0
    prior = df["class_id"].value_counts().reindex(range(nc), fill_value=0).to_numpy(dtype=float)
    prior = prior / max(prior.sum(), 1e-12)
    row_sum = T.sum(axis=1, keepdims=True)
    Tn = np.where(row_sum > 0, T / np.maximum(row_sum, 1e-12), 0.0)
    lift = np.where(Tn > 0, Tn / np.maximum(prior[None, :], 1e-12), np.nan)
    return {"names": names, "count": T, "T": Tn, "lift": lift, "prior": prior}


def assortativity(df: pd.DataFrame, cfg: dict, K: int = 5) -> dict:
    """Exp6: categorical assortativity r over directed KNN graphs."""
    nc = int(cfg["nc"])
    e = np.zeros((nc, nc), dtype=float)
    for ci, nj in _knn_edges_per_image(df, K):
        e[ci, nj] += 1.0
    total = max(e.sum(), 1e-12)
    e = e / total
    num = float(np.trace(e) - (e * e).sum())
    den = float(1.0 - (e * e).sum())
    r = num / max(den, 1e-12)
    return {"K": K, "edges": int(total), "assortativity": r}


def homophily_heterophily(trans: dict) -> pd.DataFrame:
    """Exp12: H_c = T_cc - P(c); E_c = max_{b!=c}(T_cb - P(b))."""
    names = trans["names"]
    T = trans["T"]
    prior = trans["prior"]
    rows = []
    for c, name in enumerate(names):
        h = float(T[c, c] - prior[c])
        mask = np.ones(len(names), dtype=bool)
        mask[c] = False
        vals = T[c, mask] - prior[mask]
        e = float(vals.max()) if len(vals) else 0.0
        if h > e and h > 0.01:
            kind = "homophily"
        elif e > h and e > 0.01:
            kind = "heterophily"
        else:
            kind = "relation-neutral"
        rows.append({"class": name, "H_c": h, "E_c": e, "type": kind})
    return pd.DataFrame(rows)


def _per_instance_neighbor_labels(df: pd.DataFrame, K: int) -> pd.DataFrame:
    """Per-instance: KNN neighbour labels + distances + image stats (for MI)."""
    recs = []
    nc = int(df["class_id"].max()) + 1
    for (image_id, split), g in df.groupby(["image_id", "split"]):
        labels = g["class_id"].to_numpy()
        centers = g[["xc", "yc"]].to_numpy()
        n = len(labels)
        if n < 2:
            continue
        dmat = _pair_distances(centers, g["h"].to_numpy(), False)
        k = min(K, n - 1)
        sorted_idx = np.argsort(dmat, axis=1)[:, :k]
        counts = np.bincount(labels, minlength=nc)
        img_other = counts[None, :] - np.eye(nc, dtype=float)[labels]
        dom_other = img_other.argmax(axis=1)
        for i in range(n):
            nb_labels = labels[sorted_idx[i]]
            nb_counts = np.bincount(nb_labels, minlength=nc)
            recs.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "Y": int(labels[i]),
                    "nn_label": int(nb_labels[0]),
                    "knn_mode": int(np.argmax(nb_counts)),
                    "knn_hist": nb_counts,
                    "img_dom_other": int(dom_other[i]),
                }
            )
    return pd.DataFrame(recs)


def mutual_information(df: pd.DataFrame, K: int = 5) -> dict:
    """Exp10: I(Y; nn label), I(Y; KNN mode), I(Y; image dominant class excl. self)."""
    rec = _per_instance_neighbor_labels(df, K)
    out = {}
    for key in ("nn_label", "knn_mode", "img_dom_other"):
        ct = pd.crosstab(rec["Y"], rec[key])
        joint = np.array(ct.to_numpy(dtype=float), copy=True)
        joint /= joint.sum()
        py = joint.sum(axis=1, keepdims=True)
        px = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(joint * np.log2(joint / np.maximum(py @ px, 1e-12)))
        h_y = -np.nansum(py * np.log2(np.maximum(py, 1e-12)))
        h_x = -np.nansum(px * np.log2(np.maximum(px, 1e-12)))
        nmi = mi / max(0.5 * (h_y + h_x), 1e-12)
        out[key] = {"MI_bits": float(mi), "NMI": float(nmi)}
    return out
