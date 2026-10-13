"""Tables, figures, bootstrap CIs and the final CONTINUE / NARROW / STOP decision."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .predictive import bootstrap_macro_f1_diffs, classwise_rpg


def bootstrap_delta_r(knn_out: dict, K: int, B: int = 1000, seed: int = 0) -> dict:
    """Image-level bootstrap 95% CI of delta_R(K)."""
    rng = np.random.default_rng(seed)
    pi = knn_out["per_image"]
    e = pi[f"e_{K}"].to_numpy()
    s = pi[f"s_{K}"].to_numpy()
    q = pi[f"q_{K}"].to_numpy()
    keep = e > 0
    e, s, q = e[keep], s[keep], q[keep]
    n_img = len(e)
    deltas = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n_img, size=n_img)
        r_obs = s[idx].sum() / max(e[idx].sum(), 1e-12)
        r_null = q[idx].sum() / max(e[idx].sum(), 1e-12)
        deltas[b] = r_obs - r_null
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"mean": float(deltas.mean()), "ci_low": float(lo), "ci_high": float(hi), "B": B}


def decision_rule(ev: dict) -> str:
    """Section 26-27 decision."""
    n_strong = sum(ev[k] for k in ("A", "B", "C", "D"))
    if ev["A"] and ev["delta_r3"] < 0.02 and ev["rpg_local"] < 0.03 and ev["og"] < 0.03 and ev["perm_p"] >= 0.01:
        return "STOP"
    if n_strong >= 3:
        return "CONTINUE"
    if n_strong >= 1:
        return "NARROW"
    return "STOP"


def evaluate_evidence(
    knn: dict,
    dcurve: pd.DataFrame,
    pred: dict,
    algo: str,
    B_boot: int = 1000,
) -> dict:
    """Four core evidences (Section 26)."""
    r3 = knn["result"][3]
    delta_r3 = r3["delta_R"]
    perm_p = r3["p_value"]
    # distance effect: nearest vs farthest quantile bin net gain
    dc = dcurve[dcurve["mode"] == "quantile"].sort_values("bin")
    pooled = (
        dc.groupby("bin")
        .apply(
            lambda g: pd.Series(
                {
                    "P_same_obs": float((g["n_pairs"] * g["P_same_obs"]).sum() / g["n_pairs"].sum()),
                    "P_same_perm_null": float((g["n_pairs"] * g["P_same_perm_null"]).sum() / g["n_pairs"].sum()),
                    "n_pairs": int(g["n_pairs"].sum()),
                }
            )
        )
        .sort_index()
    )
    pooled["delta"] = pooled["P_same_obs"] - pooled["P_same_perm_null"]
    net_first = float(pooled.iloc[0]["delta"])
    net_last = float(pooled.iloc[-1]["delta"])
    trend = float(np.corrcoef(pooled.index.to_numpy(), pooled["delta"].to_numpy())[0, 1]) if len(pooled) > 2 else 0.0
    dist_effect = (net_first - net_last) >= 0.02 and trend <= 0.05
    # RPG
    key_b2 = f"B2_geometry_{algo}"
    key_b5 = f"B5_geom_neighbor_{algo}"
    key_b6 = f"B6_geom_neighbor_global_{algo}"
    key_or = f"Oracle_full_{algo}"
    rpg_local = pred["results"][key_b5]["metrics"]["macro_f1"] - pred["results"][key_b2]["metrics"]["macro_f1"]
    rpg_global = pred["results"][key_b3_global(algo)]["metrics"]["macro_f1"] - pred["results"][key_b2]["metrics"]["macro_f1"]
    rpg_all = pred["results"][key_b6]["metrics"]["macro_f1"] - pred["results"][key_b2]["metrics"]["macro_f1"]
    og = pred["results"][key_or]["metrics"]["macro_f1"] - pred["results"][key_b2]["metrics"]["macro_f1"]
    ev = {
        "delta_r3": delta_r3,
        "perm_p": perm_p,
        "dist_effect": dist_effect,
        "net_first_bin": net_first,
        "net_last_bin": net_last,
        "trend": trend,
        "rpg_local": rpg_local,
        "rpg_global": rpg_global,
        "rpg_all": rpg_all,
        "og": og,
        "A": delta_r3 >= 0.05 and perm_p < 0.01,
        "B": dist_effect,
        "C": rpg_local >= 0.05,
        "D": og >= 0.08,
    }
    return ev


def key_b3_global(algo: str) -> str:
    return f"B3_global_{algo}"


def write_tables(dataset: str, out_tables: Path, res: dict) -> None:
    """Write 01-13 CSVs following Section 31."""
    # 01 audit
    rows = []
    for split, a in res["audit"].items():
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "images": a["images"],
                "instances": a["instances"],
                "avg_per_image": a["avg_per_image"],
                "median_per_image": a["median_per_image"],
                "max_per_image": a["max_per_image"],
                "empty_images": a["empty_images"],
                "single_object_ratio": a["single_object_ratio"],
                "relation_valid_ratio": a["relation_valid_ratio"],
                "imbalance_ratio": a["imbalance_ratio"],
                "area_mean": a["area_mean"],
                "aspect_median": a["aspect_median"],
                **{f"class_{k}": v for k, v in a["per_class"].items()},
            }
        )
    pd.DataFrame(rows).to_csv(out_tables / "01_dataset_audit.csv", index=False, encoding="utf-8-sig")
    # 02 entropy
    res["entropy_df"].to_csv(out_tables / "02_image_entropy.csv", index=False, encoding="utf-8-sig")
    # 03 cooccurrence
    co = res["cooc"]
    rows = []
    names = co["names"]
    for a in range(len(names)):
        for b in range(len(names)):
            rows.append(
                {
                    "a": names[a],
                    "b": names[b],
                    "count": co["count"][a, b],
                    "P(b|a)": co["cond"][a, b],
                    "PMI": co["pmi"][a, b],
                }
            )
    pd.DataFrame(rows).to_csv(out_tables / "03_cooccurrence.csv", index=False, encoding="utf-8-sig")
    # 04 knn
    rows = []
    for K, r in res["knn"]["result"].items():
        if r is None:
            continue
        rows.append(
            {
                "K": K,
                "R_obs": r["R_obs"],
                "perm_mean": r["perm_mean"],
                "perm_std": r["perm_std"],
                "delta_R": r["delta_R"],
                "z_score": r["z_score"],
                "p_value": r["p_value"],
                "edges": r["edges"],
            }
        )
    pd.DataFrame(rows).to_csv(out_tables / "04_knn_relation.csv", index=False, encoding="utf-8-sig")
    # 05 distance
    res["dcurve"].to_csv(out_tables / "05_distance_relation.csv", index=False, encoding="utf-8-sig")
    # 06 transition
    tr = res["trans"]
    rows = []
    names = tr["names"]
    for a in range(len(names)):
        for b in range(len(names)):
            rows.append(
                {
                    "center": names[a],
                    "neighbor": names[b],
                    "count": tr["count"][a, b],
                    "T": tr["T"][a, b],
                    "lift": tr["lift"][a, b],
                }
            )
    pd.DataFrame(rows).to_csv(out_tables / "06_transition_matrix.csv", index=False, encoding="utf-8-sig")
    # 07 assortativity
    pd.DataFrame(res["assort"]).to_csv(out_tables / "07_assortativity.csv", index=False, encoding="utf-8-sig")
    # 08/09 prediction
    rows = []
    for key, r in res["pred"]["results"].items():
        rows.append(
            {
                "model": key,
                "algo": r["algo"],
                "macro_f1": r["metrics"]["macro_f1"],
                "balanced_acc": r["metrics"]["balanced_acc"],
                "accuracy": r["metrics"]["accuracy"],
            }
        )
    pd.DataFrame(rows).to_csv(out_tables / "08_neighbor_prediction.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "model": key,
                "algo": r["algo"],
                "confusion": np.array2string(r["metrics"]["confusion"]),
            }
            for key, r in res["pred"]["results"].items()
        ]
    ).to_csv(out_tables / "09_oracle_prediction.csv", index=False, encoding="utf-8-sig")
    # 10 permutation
    rows = []
    for K, r in res["knn"]["result"].items():
        if r is None:
            continue
        rows.append({"statistic": f"delta_R_{K}", "obs": r["delta_R"], "p_value": r["p_value"], "z": r["z_score"]})
    pd.DataFrame(rows).to_csv(out_tables / "10_permutation_test.csv", index=False, encoding="utf-8-sig")
    # 11 MI
    pd.DataFrame(
        [{"source": k, **v} for k, v in res["mi"].items()]
    ).to_csv(out_tables / "11_mutual_information.csv", index=False, encoding="utf-8-sig")
    # 12 classwise RPG
    res["class_rpg"].to_csv(out_tables / "12_classwise_rpg.csv", index=False, encoding="utf-8-sig")
    # 13 decision
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                **{f"{k}": v for k, v in res["evidence"].items()},
                "rpg_local_lr": res["evidence_lr"]["rpg_local"],
                "rpg_global_lr": res["evidence_lr"]["rpg_global"],
                "rpg_all_lr": res["evidence_lr"]["rpg_all"],
                "og_lr": res["evidence_lr"]["og"],
                "memorization_flag": res["memorization_flag"],
                "val_duplicate_layout_ratio": res["leakage"]["val_duplicate_ratio"],
                "decision": res["decision"],
                "decision_algo": res["decision_algo"],
            }
        ]
    ).to_csv(
        out_tables / "13_final_decision.csv", index=False, encoding="utf-8-sig"
    )


def _heatmap(ax, mat, names, title, cmap="viridis", annot=True, vmin=None, vmax=None):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title(title, fontsize=10)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white")
    return im


def write_figures(dataset: str, out_figs: Path, res: dict) -> None:
    d = res
    names = res["cfg"]["names"]
    df = res["df"]
    # Figure 1: class distribution + objects/image
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["class_name"].value_counts()
    axes[0].bar(range(len(names)), [counts.get(n, 0) for n in names], color="#4C72B0")
    axes[0].set_xticks(range(len(names)), names, rotation=30, fontsize=8)
    axes[0].set_title(f"{dataset} 类别实例数")
    per_img = df.groupby("image_id").size()
    axes[1].hist(per_img, bins=min(60, per_img.max()), color="#55A868", alpha=0.8)
    axes[1].set_title("每图目标数量分布")
    axes[1].set_xlabel("每图目标数")
    fig.tight_layout()
    fig.savefig(out_figs / "fig1_class_and_density.png", dpi=120)
    plt.close(fig)
    # Figure 2: dominance + entropy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(d["entropy_df"]["dominance"], bins=40, color="#C44E52", alpha=0.85)
    axes[0].axvline(0.7, color="k", ls="--", lw=1)
    axes[0].set_title("单图最大类别占比 D")
    axes[1].hist(d["entropy_df"]["entropy_norm"], bins=40, color="#8172B2", alpha=0.85)
    axes[1].axvline(0.4, color="k", ls="--", lw=1)
    axes[1].set_title("单图归一化行为熵 H_norm")
    fig.tight_layout()
    fig.savefig(out_figs / "fig2_dominance_entropy.png", dpi=120)
    plt.close(fig)
    # Figure 3: co-occurrence + PMI
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _heatmap(axes[0], d["cooc"]["count"], names, "类别共现 Count", cmap="Blues")
    _heatmap(axes[1], d["cooc"]["pmi"], names, "PMI", cmap="RdBu_r")
    fig.tight_layout()
    fig.savefig(out_figs / "fig3_cooccurrence_pmi.png", dpi=120)
    plt.close(fig)
    # Figure 4: distance curve
    dc = d["dcurve"][d["dcurve"]["mode"] == "quantile"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = dc["bin"]
    ax.plot(x, dc["P_same_obs"], "-o", label="Observed")
    ax.plot(x, dc["P_same_perm_null"], "-s", label="Permutation null")
    ax.plot(x, dc["delta_P_same"], "-^", label="Net relational effect")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("距离分位数分箱 (0=最近)")
    ax.set_ylabel("P(同类)")
    ax.set_title(f"{dataset} 距离—同类行为概率")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "fig4_distance_curve.png", dpi=120)
    plt.close(fig)
    # Figure 5/6: transition + lift
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _heatmap(axes[0], d["trans"]["T"], names, "Neighbor Transition T_ab", cmap="Blues")
    _heatmap(axes[1], d["trans"]["lift"], names, "Neighbor Lift", cmap="RdBu_r", vmin=0.0)
    fig.tight_layout()
    fig.savefig(out_figs / "fig5_transition_lift.png", dpi=120)
    plt.close(fig)
    # Figure 7: observed vs perm delta_R
    fig, ax = plt.subplots(figsize=(7, 4))
    ks, obs, perm = [], [], []
    for K, r in d["knn"]["result"].items():
        if r is None:
            continue
        ks.append(K)
        obs.append(r["R_obs"])
        perm.append(r["perm_mean"])
    w = 0.35
    ax.bar([k - w / 2 for k in ks], obs, w, label="Observed R_K", color="#4C72B0")
    ax.bar([k + w / 2 for k in ks], perm, w, label="Permutation mean", color="#DD8452")
    ax.set_xticks(ks)
    ax.set_ylabel("同类率")
    ax.set_title(f"{dataset} KNN 同类率 vs 置换基线")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "fig7_knn_permutation.png", dpi=120)
    plt.close(fig)
    # Figure 8: macro-F1 by feature set
    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = ["B2_geometry", "B3_global", "B4_neighbor", "B5_geom_neighbor", "B6_geom_neighbor_global", "Oracle_full"]
    labels = ["Geometry", "Global", "Neighbor", "Geom+Neigh", "Geom+Neigh+Global", "Oracle(full)"]
    for algo, color in (("lr", "#4C72B0"), ("tree", "#C44E52")):
        vals = [d["pred"]["results"][f"{n}_{algo}"]["metrics"]["macro_f1"] for n in order]
        ax.plot(labels, vals, "-o", label=f"Macro-F1 ({algo})", color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_ylabel("Macro-F1")
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.axhline(
        d["pred"]["results"]["B0_majority"]["metrics"]["macro_f1"], color="k", ls="--", lw=1, label="Majority"
    )
    ax.set_title(f"{dataset} Neighbor-only / Oracle Macro-F1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_figs / "fig8_prediction_macrof1.png", dpi=120)
    plt.close(fig)
    # Figure 9: classwise RPG
    cr = d["class_rpg"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f"class {c}" for c in cr["class"]], cr["RPG_c"], color=["#4C72B0" if v >= 0 else "#C44E52" for v in cr["RPG_c"]])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("RPG_c (F1 relation - F1 geometry)")
    ax.set_title(f"{dataset} 类别级 Relational Predictive Gain")
    fig.tight_layout()
    fig.savefig(out_figs / "fig9_classwise_rpg.png", dpi=120)
    plt.close(fig)


def write_summary(dataset: str, out_root: Path, res: dict) -> None:
    md = []
    ev = res["evidence_lr"] if res["memorization_flag"] else res["evidence"]
    decision_cn = {"CONTINUE": "继续 DSR", "NARROW": "收缩为选择性关系建模", "STOP": "放弃 SCB-DSR"}[res["decision"]]
    md.append(f"# {dataset} 关系可研究性诊断报告\n")
    md.append("> 依据《SCB-Dataset关系可研究性诊断实验方案》自动生成（模型无关，仅依赖 GT 标签）。\n")
    md.append("## 0. 数据质量检查（泄漏风险）\n")
    md.append(
        f"- train/val 框布局完全重复比例：{res['leakage']['val_duplicate_ratio']:.3f}"
        f"（{res['leakage']['val_duplicate_layout']}/{res['leakage']['val_images']} 张 val 图）\n"
    )
    md.append(
        f"- 几何特征 Tree vs LR Macro-F1 差距：{res['pred']['results']['B2_geometry_tree']['metrics']['macro_f1']:.3f} vs "
        f"{res['pred']['results']['B2_geometry_lr']['metrics']['macro_f1']:.3f}"
        f" → {'**存在记忆化风险，最终判定采用 LR 证据**' if res['memorization_flag'] else '正常'}\n"
    )
    md.append("## 1. 基础审计\n")
    rows = []
    for split, a in res["audit"].items():
        rows.append(
            f"| {split} | {a['images']} | {a['instances']} | {a['avg_per_image']:.2f} | "
            f"{a['median_per_image']} | {a['max_per_image']} | {a['relation_valid_ratio']:.3f} | {a['imbalance_ratio']:.2f} |"
        )
    md.append("| Split | Images | Instances | Avg/img | Median | Max | Relation-valid | Imbalance |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    md.extend(rows)
    md.append("")
    md.append(f"- 类别实例：{ {k: v for k, v in res['audit']['train']['per_class'].items()} }")
    md.append("")
    md.append("## 2. 群体结构（Exp1）\n")
    es = res["entropy_sum"]
    md.append(
        f"- Mean Dominance = {es['mean_dominance']:.3f}，Median = {es['median_dominance']:.3f}，"
        f"Mean H_norm = {es['mean_entropy_norm']:.3f}，低熵比例 (H<0.4) = {es['low_entropy_ratio_H<0.4']:.3f}\n"
    )
    md.append("## 3. KNN 同类率与置换检验（Exp3/9）\n")
    md.append("| K | R_obs | Perm Mean | ΔR | Z | p |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for K, r in res["knn"]["result"].items():
        if r is None:
            continue
        md.append(f"| {K} | {r['R_obs']:.4f} | {r['perm_mean']:.4f} | {r['delta_R']:.4f} | {r['z_score']:.2f} | {r['p_value']:.4f} |")
    md.append("")
    md.append("## 4. 距离效应（Exp4）\n")
    dc = res["dcurve"][res["dcurve"]["mode"] == "quantile"]
    md.append("| Bin | n_pairs | P_same_obs | Null | Net |")
    md.append("|---|---:|---:|---:|---:|")
    for _, r in dc.iterrows():
        md.append(
            f"| {int(r['bin'])} | {int(r['n_pairs'])} | {r['P_same_obs']:.4f} | {r['P_same_perm_null']:.4f} | {r['delta_P_same']:.4f} |"
        )
    md.append("")
    md.append("## 5. 同配/异配（Exp5/12）\n")
    md.append("| Class | T_cc | Lift(diag) | H_c | E_c | Type |")
    md.append("|---|---:|---:|---:|---:|---:|")
    names = res["trans"]["names"]
    for _, r in res["homo"].iterrows():
        c = names.index(r["class"])
        md.append(
            f"| {r['class']} | {res['trans']['T'][c, c]:.3f} | {res['trans']['lift'][c, c]:.3f} | "
            f"{r['H_c']:.3f} | {r['E_c']:.3f} | {r['type']} |"
        )
    md.append("")
    md.append("## 6. 预测性分析（Exp7/8/11）\n")
    md.append("| Model | LR Macro-F1 | Tree Macro-F1 |")
    md.append("|---|---:|---:|")
    for name in ["B0_majority", "B1_prior_random", "B2_geometry", "B3_global", "B4_neighbor", "B5_geom_neighbor", "B6_geom_neighbor_global", "Oracle_full"]:
        v = []
        for algo in ("lr", "tree"):
            if name == "B0_majority" or name == "B1_prior_random":
                v.append(res["pred"]["results"][name]["metrics"]["macro_f1"])
            else:
                v.append(res["pred"]["results"][f"{name}_{algo}"]["metrics"]["macro_f1"])
        md.append(f"| {name} | {v[0]:.4f} | {v[1]:.4f} |")
    md.append("")
    md.append("## 7. 核心证据与判定\n")
    md.append(
        f"- Evidence A（空间关联 ΔR3≥0.05 且 p<0.01）：**{ev['A']}**（ΔR3={ev['delta_r3']:.4f}, p={ev['perm_p']:.4f}）\n"
    )
    md.append(
        f"- Evidence B（距离效应）：**{ev['B']}**（最近箱净增益 {ev['net_first_bin']:.4f}，最远箱 {ev['net_last_bin']:.4f}）\n"
    )
    md.append(f"- Evidence C（RPG_local≥0.05）：**{ev['C']}**（RPG_local={ev['rpg_local']:.4f}）\n")
    md.append(f"- Evidence D（Oracle Gain≥0.08）：**{ev['D']}**（OG={ev['og']:.4f}）\n")
    md.append(f"- RPG_global={ev['rpg_global']:.4f}，RPG_all={ev['rpg_all']:.4f}\n")
    md.append(f"\n**最终判定：{decision_cn}（{res['decision']}；判定依据：{res['decision_algo']}）**\n")
    md.append("\n---\n")
    md.append("附表：Neighbor-only 详细指标见 `tables/08_neighbor_prediction.csv`，类别级 RPG 见 `tables/12_classwise_rpg.csv`，Bootstrap CI 见 `tables/13_final_decision.csv`。")
    (out_root / f"{dataset}_diagnosis.md").write_text("\n".join(md), encoding="utf-8")
