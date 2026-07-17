"""Formal multi-seed discovery-to-training experiments."""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(arguments):
    command = [sys.executable, 'scripts/legacy/run_fast.py',
               '--steps-per-block', '100'] + arguments
    print('\nRUN:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-main', action='store_true')
    parser.add_argument('--skip-sweep', action='store_true')
    args = parser.parse_args()
    seeds = (0, 1, 2)
    if not args.skip_main:
        for region in ('centre', 'edge'):
            for source in ('none', 'analytic', 'discovered'):
                for seed in seeds:
                    run(['--region', region, '--seed', str(seed),
                         '--physics-source', source])
    if not args.skip_sweep:
        # Dimensionless normalized residuals make these values meaningful for
        # the new loss; this replaces the incompatible legacy sweep.
        for value in (0.001, 0.003, 0.01, 0.03, 0.1):
            for seed in seeds:
                run(['--region', 'centre', '--seed', str(seed),
                     '--physics-source', 'discovered',
                     '--lambda-phys', str(value)])


if __name__ == '__main__':
    main()
