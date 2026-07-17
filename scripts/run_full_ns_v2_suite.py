"""Run and aggregate the matched three-seed corrected full-NS experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ('low_enstrophy', 'high_enstrophy')
METHODS = ('none', 'analytic', 'discovered')
SEEDS = (42, 123, 456)


def run_id(method, region, seed):
    weight = '0' if method == 'none' else '0p01'
    return f'full_ns_v3_{method}_{region}_lambda{weight}_seed{seed}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--steps-per-block', type=int, default=100)
    parser.add_argument('--classifier-epochs', type=int, default=30)
    parser.add_argument('--pretrain-epochs', type=int, default=40)
    args = parser.parse_args()
    log_dir = ROOT/'outputs/runs/full_ns_v3_suite_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        for seed in SEEDS:
            # Each method has its own condition coefficients and therefore its
            # own shared encoder/decoder checkpoint.
            for method in ('discovered', 'analytic', 'none'):
                identifier = run_id(method, region, seed)
                metrics_path = ROOT/'outputs/runs'/identifier/'metrics.json'
                if metrics_path.exists() and not args.overwrite:
                    print('reuse', identifier, flush=True)
                    continue
                command = [
                    sys.executable, str(ROOT/'scripts/run_full_ns_v2.py'),
                    '--physics-source', method, '--region', region,
                    '--seed', str(seed),
                    '--steps-per-block', str(args.steps_per_block),
                    '--classifier-epochs', str(args.classifier_epochs),
                    '--pretrain-epochs', str(args.pretrain_epochs),
                    '--lambda-phys', '0.01',
                ]
                print('run', identifier, flush=True)
                with (log_dir/f'{identifier}.log').open('w', encoding='utf-8') as log:
                    completed = subprocess.run(
                        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f'{identifier} failed; inspect {log.name}')

    results = {region: {} for region in REGIONS}
    for region in REGIONS:
        for method in METHODS:
            records = [json.loads((ROOT/'outputs/runs'/run_id(method, region, seed)
                                   /'metrics.json').read_text(encoding='utf-8'))
                       for seed in SEEDS]
            def summary(key):
                values = np.asarray([record['test'][key] for record in records])
                return {'values': values.tolist(), 'mean': float(values.mean()),
                        'std': float(values.std(ddof=1))}
            results[region][method] = {
                'accuracy': summary('accuracy'),
                'eta_ns': summary('eta_ns'),
                'eta_div': summary('eta_div'),
                'block_seconds': {
                    'values': [record['block_train_seconds'] for record in records],
                    'mean': float(np.mean([record['block_train_seconds']
                                          for record in records])),
                },
                'peak_memory_mb': {
                    'values': [record['peak_memory_mb'] for record in records],
                    'mean': float(np.mean([record['peak_memory_mb']
                                          for record in records])),
                },
                'run_ids': [record['method'] + ':' + str(record['seed'])
                            for record in records],
            }
    artifact = {
        'schema_version': 3,
        'protocol': {
            'seeds': list(SEEDS), 'regions': list(REGIONS),
            'methods': list(METHODS),
            'trajectory_disjoint': True,
            'target': 'future local relative kinetic-energy decay quantile',
            'steps_per_block': args.steps_per_block,
        },
        'results': results,
    }
    output = ROOT/'outputs/aggregate/full_ns_v3_results.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == '__main__':
    main()
