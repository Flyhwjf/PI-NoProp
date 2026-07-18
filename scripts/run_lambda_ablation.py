"""Three-seed sweep of the local physics-loss weight with fixed NS input."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456)
LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)


def lambda_token(value):
    return f'{value:g}'.replace('.', 'p')


def run_id(value, seed):
    base = f'full_ns_v4_discovered_low_enstrophy_lambda{lambda_token(value)}_seed{seed}'
    return base if value == 0.01 else f'{base}_lambdaablation'


def main():
    for seed in SEEDS:
        for value in LAMBDAS:
            path = ROOT/'outputs/runs'/run_id(value, seed)/'metrics.json'
            if path.exists():
                print('reuse', value, seed, flush=True)
                continue
            completed = subprocess.run([
                sys.executable, str(ROOT/'scripts/run_experiment.py'),
                '--physics-source', 'discovered', '--region', 'low_enstrophy',
                '--seed', str(seed), '--relation-set', 'ns',
                '--lambda-phys', str(value), '--run-tag', 'lambdaablation'],
                cwd=ROOT)
            if completed.returncode:
                raise RuntimeError(f'lambda={value} seed={seed} failed')

    artifact = {'schema_version': 1, 'protocol': {
        'seeds': list(SEEDS), 'region': 'low_enstrophy',
        'trajectory_disjoint': True, 'relation_set': 'ns',
        'physics_condition': 'fixed discovered full-momentum coefficients',
        'varied_factor': 'local physics-loss weight only',
        'lambdas': list(LAMBDAS)}, 'results': {}}
    for value in LAMBDAS:
        records = [json.loads((ROOT/'outputs/runs'/run_id(value, seed)/
                              'metrics.json').read_text()) for seed in SEEDS]
        metrics = {}
        for key in ('accuracy', 'eta_ns', 'eta_div'):
            values = np.asarray([record['test'][key] for record in records])
            metrics[key] = {'values': values.tolist(),
                            'mean': float(values.mean()),
                            'std': float(values.std(ddof=1))}
        artifact['results'][str(value)] = metrics
    output = ROOT/'outputs/aggregate/full_ns_lambda_ablation.json'
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
