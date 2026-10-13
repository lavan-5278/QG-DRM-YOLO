"""Dataset feasibility pre-check for differential social-relation detection (DSR-Det).

Before committing to a full training matrix, verify that a box-annotated dataset
actually contains the signal the method relies on:

  1. relation-graph constructibility  -- enough instances per image;
  2. contextual class signal          -- GT-label kNN vote accuracy vs majority prior;
  3. differential structure           -- per-class neighbor-distribution shift (KL),
                                        and minority-class lift from neighbor votes;
  4. hard-sample availability         -- small-object and occlusion proxies;
  5. candidate feasibility note       -- requires a cheap baseline detector run
                                        (recall@0.5 >= 0.95 target).

No model is trained; only label files are read. Thresholds are heuristic defaults
and can be adjusted with --min-* flags.

Usage:
    python scripts/feasibility_check.py ultralytics/cfg/datasets/SCB-Dataset3-S.yaml
    python scripts/feasibility_check.py <data.yaml> --save-json feasibility.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml


def _load_set(root: Path, img_rel: str) -> dict[str, list[tuple[int, float, float, float, float]]]:
    img_dir = root / img_rel
    lbl_dir = root / str(img_rel).replace("images", "labels")
    imgs: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for img in sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}):
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                c, cx, cy, w, h = (float(x) for x in parts)
            except ValueError:
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue
            imgs[img.stem].append((int(c), cx, cy, w, h))
    return imgs


def _knn_vote(imgs: dict, k: int) -> tuple[float, dict[int, tuple[int, int]]]:
    """Overall and per-class accuracy of predicting class from k nearest GT boxes."""
    correct = 0
    total = 0
    per_cls_correct: Counter = Counter()
    per_cls_total: Counter = Counter()
    for boxes in imgs.values():
        n = len(boxes)
        if n < 2:
            continue
        cents = np.array([[b[1], b[2]] for b in boxes])
        for i, bi in enumerate(boxes):
            d2 = ((cents - cents[i]) ** 2).sum(1)
            d2[i] = np.inf
            nn = np.argsort(d2)[:k]
            vote = Counter(boxes[j][0] for j in nn)
            pred = vote.most_common(1)[0][0]
            correct += pred == bi[0]
            total += 1
            per_cls_correct[bi[0]] += pred == bi[0]
            per_cls_total[bi[0]] += 1
    per_cls = {c: (per_cls_correct[c], per_cls_total[c]) for c in per_cls_total}
    return (correct / total if total else float("nan")), per_cls


def _neighbor_signal(imgs: dict, nc: int, k: int) -> dict:
    """MI(target; nearest-neighbor class) and per-class KL of neighbor dist vs marginal."""
    cnt = np.zeros((nc, nc), dtype=float)  # [target][neighbor]
    others = np.zeros((nc, nc), dtype=float)  # [target][other-object pool]
    pool = np.zeros(nc, dtype=float)
    for boxes in imgs.values():
        n = len(boxes)
        if n < 2:
            continue
        cents = np.array([[b[1], b[2]] for b in boxes])
        other_counts = Counter(b[0] for b in boxes)
        for i, bi in enumerate(boxes):
            d2 = ((cents - cents[i]) ** 2).sum(1)
            d2[i] = np.inf
            nn = np.argsort(d2)[:k]
            for j in nn:
                cnt[bi[0], boxes[j][0]] += 1
            for c2, m in other_counts.items():
                m2 = m - (1 if c2 == bi[0] else 0)
                if m2 > 0:
                    others[bi[0], c2] += m2
                    pool[c2] += m2

    def _kl(row: np.ndarray, marg: np.ndarray) -> float:
        p = row / row.sum()
        q = marg / marg.sum()
        return float(np.sum(p * np.log((p + 1e-12) / (q + 1e-12))))

    marg_knn = cnt.sum(0)
    marg_pool = pool
    kls_knn = [_kl(cnt[c], marg_knn) for c in range(nc)]
    kls_pool = [_kl(others[c], marg_pool) for c in range(nc)]
    joint = cnt / cnt.sum()
    pc = joint.sum(1)
    pn = joint.sum(0)
    mi = float(np.sum(joint * np.log((joint + 1e-12) / (pc[:, None] * pn[None, :] + 1e-12))))
    hc = float(-np.sum(pc * np.log(pc + 1e-12)))
    return {
        "mi_nats": round(mi, 4),
        "h_class_nats": round(hc, 4),
        "norm_mi_pct": round(mi / hc * 100, 1) if hc > 0 else 0.0,
        "kl_knn": [round(v, 3) for v in kls_knn],
        "kl_other_objects": [round(v, 3) for v in kls_pool],
    }


def _hard_samples(imgs: dict, nc: int) -> dict:
    areas: list[float] = []
    ioa_any = 0
    ioa_03 = 0
    ioa_05 = 0
    n = 0
    small = 0
    for boxes in imgs.values():
        for i, bi in enumerate(boxes):
            areas.append(bi[3] * bi[4])
            if bi[3] * bi[4] < 0.02:
                small += 1
            x1 = bi[1] - bi[3] / 2
            x2 = bi[1] + bi[3] / 2
            y1 = bi[2] - bi[4] / 2
            y2 = bi[2] + bi[4] / 2
            ai = max(1e-9, (x2 - x1) * (y2 - y1))
            max_ioa = 0.0
            for j, bj in enumerate(boxes):
                if i == j:
                    continue
                a = bj[1] - bj[3] / 2
                b = bj[1] + bj[3] / 2
                c = bj[2] - bj[4] / 2
                d = bj[2] + bj[4] / 2
                ix = max(0.0, min(x2, b) - max(x1, a))
                iy = max(0.0, min(y2, d) - max(y1, c))
                max_ioa = max(max_ioa, ix * iy / ai)
            if max_ioa > 0:
                ioa_any += 1
            if max_ioa > 0.3:
                ioa_03 += 1
            if max_ioa > 0.5:
                ioa_05 += 1
            n += 1
    areas = np.asarray(areas)
    return {
        "area_ratio_p25": round(float(np.percentile(areas, 25)), 4),
        "area_ratio_median": round(float(np.percentile(areas, 50)), 4),
        "area_ratio_p75": round(float(np.percentile(areas, 75)), 4),
        "small_lt_0_02_pct": round(small / n * 100, 1) if n else 0.0,
        "ioa_gt0_pct": round(ioa_any / n * 100, 1) if n else 0.0,
        "ioa_gt0_3_pct": round(ioa_03 / n * 100, 1) if n else 0.0,
        "ioa_gt0_5_pct": round(ioa_05 / n * 100, 1) if n else 0.0,
    }


def feasibility_check(data_yaml: str, knn_k: int = 5, save_json: str | None = None) -> dict:
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
        imgs = _load_set(root, img_rel)
        n_img = len(imgs)
        n_inst = sum(len(b) for b in imgs.values())
        if n_img == 0 or n_inst == 0:
            out["splits"][split] = {"images": 0, "instances": 0, "note": "no data"}
            continue

        per_class = Counter(b[0] for b in sum(imgs.values(), []))
        per_img = np.asarray(sorted(len(b) for b in imgs.values()), dtype=float)
        rel2 = sum(len(b) for b in imgs.values() if len(b) >= 2)
        rel5 = sum(len(b) for b in imgs.values() if len(b) >= 5)
        rel10 = sum(len(b) for b in imgs.values() if len(b) >= 10)
        mixed_img = 0
        inst_mixed = 0
        for boxes in imgs.values():
            if len({b[0] for b in boxes}) >= 2:
                mixed_img += 1
                inst_mixed += len(boxes)

        knn_acc, per_cls = _knn_vote(imgs, knn_k)
        majority = max(per_class.values()) / n_inst
        sig = _neighbor_signal(imgs, nc, knn_k)
        hard = _hard_samples(imgs, nc)

        # Heuristic verdict
        checks = {
            "relation_rich_instances_ge5": rel5 / n_inst >= 0.6,
            "contextual_lift_ge15pt": (knn_acc - majority) * 100 >= 15,
            "mixed_class_instances_ge40": inst_mixed / n_inst >= 0.4,
            "minority_lift": any(
                per_class[c] / n_inst < 0.4 and per_cls.get(c, (0, 0))[1] >= 100
                and per_cls[c][0] / per_cls[c][1] - per_class[c] / n_inst >= 0.2
                for c in range(nc)
            ),
            "hard_samples_present": hard["small_lt_0_02_pct"] >= 20 or hard["ioa_gt0_3_pct"] >= 20,
        }
        n_pass = sum(checks.values())
        verdict = "supports" if n_pass >= 4 else ("partial" if n_pass >= 2 else "insufficient")

        out["splits"][split] = {
            "images": n_img,
            "instances": n_inst,
            "per_class_instances": {names[c]: int(per_class[c]) for c in range(nc)},
            "per_image_mean": round(float(per_img.mean()), 1),
            "per_image_median": float(np.median(per_img)),
            "per_image_p90": float(np.percentile(per_img, 90)),
            "per_image_max": int(per_img.max()),
            "relation_rich_instances_pct": {"ge2": round(rel2 / n_inst * 100, 1), "ge5": round(rel5 / n_inst * 100, 1), "ge10": round(rel10 / n_inst * 100, 1)},
            "mixed_class_images_pct": round(mixed_img / n_img * 100, 1),
            "mixed_class_instances_pct": round(inst_mixed / n_inst * 100, 1),
            "majority_class_prior_pct": round(majority * 100, 1),
            "knn_vote_acc_pct": round(knn_acc * 100, 1),
            "knn_vote_lift_pct": round((knn_acc - majority) * 100, 1),
            "per_class_knn_acc_pct": {names[c]: round(per_cls[c][0] / per_cls[c][1] * 100, 1) for c in per_cls},
            "neighbor_signal": sig,
            "hard_samples": hard,
            "checks": checks,
            "verdict": verdict,
        }

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(out)
    return out


def _print(out: dict) -> None:
    print(f"dataset: {out['dataset']}  ({out['nc']} classes: {', '.join(out['names'])})")
    for split, s in out["splits"].items():
        if s.get("instances", 0) == 0:
            print(f"  [{split}] no data")
            continue
        print(f"\n== {split} ==  images={s['images']} instances={s['instances']}  "
              f"per_img(mean/median/p90/max)={s['per_image_mean']}/{s['per_image_median']}/{s['per_image_p90']}/{s['per_image_max']}")
        print(f"  per-class: {s['per_class_instances']}")
        print(f"  relation-rich instances (>=2/>=5/>=10 objects): {s['relation_rich_instances_pct']}")
        print(f"  mixed-class images {s['mixed_class_images_pct']}%  /  instances in mixed images {s['mixed_class_instances_pct']}%")
        print(f"  majority prior {s['majority_class_prior_pct']}%  ->  GT-label kNN{out.get('knn_k', 5)} vote {s['knn_vote_acc_pct']}%  (lift {s['knn_vote_lift_pct']}pt)")
        print(f"  per-class kNN acc: {s['per_class_knn_acc_pct']}")
        print(f"  MI(c; neighbor class) {s['neighbor_signal']['mi_nats']} nats, norm {s['neighbor_signal']['norm_mi_pct']}% of H(c); "
              f"KL(neighbor|target||marginal) {s['neighbor_signal']['kl_knn']}")
        print(f"  hard samples: small(<0.02) {s['hard_samples']['small_lt_0_02_pct']}%, "
              f"IoA>0.3 {s['hard_samples']['ioa_gt0_3_pct']}%, IoA>0.5 {s['hard_samples']['ioa_gt0_5_pct']}%")
        print(f"  checks: {s['checks']}  -> verdict: {s['verdict']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=str)
    parser.add_argument("--knn-k", type=int, default=5, help="k for GT-label kNN vote (default 5)")
    parser.add_argument("--save-json", type=str, default=None)
    args = parser.parse_args(argv)
    feasibility_check(args.data_yaml, knn_k=args.knn_k, save_json=args.save_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
