"""Run the full SCB relational-feasibility diagnosis for all configured datasets.

Usage:
    python relation_diagnosis/run_diagnosis.py
    python relation_diagnosis/run_diagnosis.py --configs relation_diagnosis/configs --outputs relation_diagnosis/outputs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.descriptive_stats import (
    audit_dataset,
    class_frequency_table,
    cooccurrence,
    entropy_summary,
    image_dominance_entropy,
    layout_duplicate_check,
)
from src.parse_yolo_labels import load_dataset
from src.predictive import (
    bootstrap_macro_f1_diffs,
    build_instance_features,
    classwise_rpg,
    run_prediction_pipeline,
)
from src.relation_stats import (
    assortativity,
    distance_curve,
    homophily_heterophily,
    knn_permutation,
    mutual_information,
    transition_matrix,
)
from src.report import (
    bootstrap_delta_r,
    decision_rule,
    evaluate_evidence,
    write_figures,
    write_summary,
    write_tables,
)


def run_dataset(cfg_path: Path, out_root: Path, B_perm: int = 1000, B_boot: int = 1000, seed: int = 0) -> dict:
    df, cfg = load_dataset(cfg_path)
    dataset = cfg["name"]
    print(f"\n===== {dataset} =====")

    audit = audit_dataset(df, cfg)
    leak = layout_duplicate_check(df, decimals=2)
    print(f"  layout-duplicate val ratio: {leak['val_duplicate_ratio']:.3f} ({leak['val_duplicate_layout']}/{leak['val_images']})")
    ent_df = image_dominance_entropy(df, cfg)
    ent_sum = entropy_summary(ent_df)
    cooc = cooccurrence(df, cfg)

    t0 = time.time()
    knn = knn_permutation(df, cfg, K_list=(1, 3, 5, 8), B=B_perm, seed=seed)
    print(f"  KNN permutation done in {time.time() - t0:.1f}s")
    knn_geo = knn_permutation(df, cfg, K_list=(1, 3, 5, 8), B=B_perm, seed=seed, geo=True)
    dcurve = distance_curve(df, cfg, mode="quantile", nbins=5)
    dcurve_fixed = distance_curve(df, cfg, mode="fixed")
    trans = transition_matrix(df, cfg, K=5)
    assort = [assortativity(df, cfg, K=3), assortativity(df, cfg, K=5)]
    homo = homophily_heterophily(trans)
    mi = mutual_information(df, K=5)

    t0 = time.time()
    feats = build_instance_features(df, cfg, K=5)
    print(f"  features built in {time.time() - t0:.1f}s")
    t0 = time.time()
    pred = run_prediction_pipeline(feats, cfg, seed=seed)
    print(f"  prediction pipeline done in {time.time() - t0:.1f}s")

    # bootstrap CIs
    boot = {}
    for K in (1, 3, 5, 8):
        boot[f"delta_r{K}"] = bootstrap_delta_r(knn, K, B=B_boot, seed=seed)
    for name in ("B3_global", "B5_geom_neighbor", "B6_geom_neighbor_global", "Oracle_full"):
        boot[f"{name}_tree_vs_B2"] = bootstrap_macro_f1_diffs(
            pred, f"{name}_tree", "B2_geometry_tree", B=B_boot, seed=seed
        )

    evidence_tree = evaluate_evidence(knn, dcurve, pred, algo="tree")
    evidence_lr = evaluate_evidence(knn, dcurve, pred, algo="lr")
    gap = pred["results"]["B2_geometry_tree"]["metrics"]["macro_f1"] - pred["results"]["B2_geometry_lr"]["metrics"]["macro_f1"]
    memorization_flag = bool(gap > 0.15)
    if memorization_flag:
        decision = decision_rule(evidence_lr)
        decision_algo = "lr (tree geometry-only gap suggests memorization)"
    else:
        decision = decision_rule(evidence_tree)
        decision_algo = "tree"

    class_rpg = classwise_rpg(pred, "B5_geom_neighbor_tree", "B2_geometry_tree")
    res = {
        "cfg": cfg,
        "df": df,
        "audit": audit,
        "entropy_df": ent_df,
        "entropy_sum": ent_sum,
        "cooc": cooc,
        "leakage": leak,
        "knn": knn,
        "knn_geo": knn_geo,
        "dcurve": pd.concat([dcurve, dcurve_fixed], ignore_index=True),
        "trans": trans,
        "assort": [{"K": a["K"], "assortativity": a["assortativity"]} for a in assort],
        "homo": homo,
        "mi": mi,
        "pred": pred,
        "boot": boot,
        "class_rpg": class_rpg,
        "evidence": evidence_tree,
        "evidence_lr": evidence_lr,
        "memorization_flag": memorization_flag,
        "decision": decision,
        "decision_algo": decision_algo,
    }

    out_tables = out_root / dataset / "tables"
    out_figs = out_root / dataset / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)
    write_tables(dataset, out_tables, res)
    write_figures(dataset, out_figs, res)
    write_summary(dataset, out_root, res)
    print(f"  decision: {decision}")
    return res


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for dataset, r in results.items():
        a = r["audit"]["train"]
        tr = r["trans"]
        ev = r["evidence"]
        boot = r["boot"]
        rows.append(
            {
                "Dataset": dataset,
                "Relation-valid Ratio": a["relation_valid_ratio"],
                "Mean Entropy (H_norm)": r["entropy_sum"]["mean_entropy_norm"],
                "dR K=3": r["knn"]["result"][3]["delta_R"],
                "dR K=5": r["knn"]["result"][5]["delta_R"],
                "Assortativity K=5": r["assort"][1]["assortativity"],
                "RPG-local": ev["rpg_local"],
                "RPG-global": ev["rpg_global"],
                "Oracle Gain": ev["og"],
                "Permutation p": ev["perm_p"],
                "dR3 95% CI": f"[{boot['delta_r3']['ci_low']:.3f}, {boot['delta_r3']['ci_high']:.3f}]",
                "RPG-local 95% CI": f"[{boot['B5_geom_neighbor_tree_vs_B2']['ci_low']:.3f}, {boot['B5_geom_neighbor_tree_vs_B2']['ci_high']:.3f}]",
                "OG 95% CI": f"[{boot['Oracle_full_tree_vs_B2']['ci_low']:.3f}, {boot['Oracle_full_tree_vs_B2']['ci_high']:.3f}]",
                "val_duplicate_ratio": r["leakage"]["val_duplicate_ratio"],
                "memorization_flag": r["memorization_flag"],
                "Decision": r["decision"],
                "decision_algo": r["decision_algo"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=str(ROOT / "configs"))
    ap.add_argument("--outputs", default=str(ROOT / "outputs"))
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of config names to run")
    ap.add_argument("--perm-b", type=int, default=1000)
    ap.add_argument("--boot-b", type=int, default=1000)
    args = ap.parse_args()

    cfg_dir = Path(args.configs)
    out_root = Path(args.outputs)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_files = sorted(cfg_dir.glob("*.yaml"))
    if args.datasets:
        cfg_files = [c for c in cfg_files if c.stem in args.datasets]

    results: dict[str, dict] = {}
    for cfg_path in cfg_files:
        results[cfg_path.stem] = run_dataset(cfg_path, out_root, B_perm=args.perm_b, B_boot=args.boot_b)

    if len(results) > 1:
        comp = comparison_table(results)
        (out_root / "tables").mkdir(parents=True, exist_ok=True)
        comp.to_csv(out_root / "tables" / "13b_dataset_comparison.csv", index=False, encoding="utf-8-sig")
        print("\n===== 三数据集关系可研究性总表 =====")
        print(comp.to_string(index=False))
        # Figure 10: radar / grouped bar
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        metrics = ["dR K=3", "dR K=5", "RPG-local", "Oracle Gain"]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(metrics))
        width = 0.25
        for i, (ds, r) in enumerate(results.items()):
            ev = r["evidence"]
            vals = [r["knn"]["result"][3]["delta_R"], r["knn"]["result"][5]["delta_R"], ev["rpg_local"], ev["og"]]
            ax.bar(x + (i - 1) * width, vals, width, label=ds)
        ax.set_xticks(x, metrics)
        ax.set_ylabel("数值")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend()
        ax.set_title("三版本 SCB 关系可研究性对比")
        fig.tight_layout()
        (out_root / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_root / "figures" / "fig10_dataset_comparison.png", dpi=120)
        plt.close(fig)

    print(f"\nDone. Outputs -> {out_root}")


if __name__ == "__main__":
    main()
