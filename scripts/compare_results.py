"""Compile Ultralytics results.csv best mAP50-95 epochs into a comparison table.

Note: Ultralytics training-selected best.pt is based on fitness, not necessarily
simply argmax map50-95; explicit val(best.pt) is preferred for final numbers.
"""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='runs/ablation_s')
    a = p.parse_args()
    print('variant,epoch_of_best_map5095,P,R,mAP50,mAP50-95')
    for variant in ('a0','a1','a2','a3','a4'):
        path = Path(a.project) / variant / 'results.csv'
        if not path.exists():
            print(f'{variant},MISSING,,,,')
            continue
        with path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f'{variant},EMPTY,,,,')
            continue
        def v(row, key):
            return float(row[key].strip())
        best = max(rows, key=lambda r: v(r,'metrics/mAP50-95(B)'))
        print(','.join([variant, str(int(v(best,'epoch'))),
            f"{v(best,'metrics/precision(B)'):.4f}",
            f"{v(best,'metrics/recall(B)'):.4f}",
            f"{v(best,'metrics/mAP50(B)'):.4f}",
            f"{v(best,'metrics/mAP50-95(B)'):.4f}"]))


if __name__ == '__main__':
    main()
