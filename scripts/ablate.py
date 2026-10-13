"""Run the five progressive ablations of paper Table 3."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--project', default='runs/ablation_s')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    for variant in ('a0','a1','a2','a3','a4'):
        cmd = [sys.executable, '-m', 'scripts.train', '--data', args.data,
               '--device', args.device, '--project', args.project,
               '--epochs', str(args.epochs), '--batch', str(args.batch),
               '--workers', str(args.workers), '--variant', variant]
        print('RUN:', ' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
