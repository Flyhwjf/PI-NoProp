"""Three-seed low-enstrophy ablation of artifact-derived physical relations."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456)
VARIANTS = ('ns', 'ns_pp', 'full')


def run_id(variant, seed):
    relation = '' if variant == 'ns' else f'_{variant}'
    return (f'full_ns_v3_discovered_low_enstrophy_lambda0p01_seed{seed}'
            f'{relation}_eqablation')


def main():
    for seed in SEEDS:
        for variant in VARIANTS:
            path = ROOT/'outputs/runs'/run_id(variant, seed)/'metrics.json'
            if not path.exists():
                print('run', variant, seed, flush=True)
                completed = subprocess.run([
                    sys.executable, str(ROOT/'scripts/run_full_ns_v2.py'),
                    '--physics-source', 'discovered', '--region', 'low_enstrophy',
                    '--seed', str(seed), '--relation-set', variant,
                    '--run-tag', 'eqablation'], cwd=ROOT)
                if completed.returncode:
                    raise RuntimeError(f'{variant} seed {seed} failed')
    artifact = {'schema_version': 1, 'protocol': {
        'seeds': list(SEEDS), 'region': 'low_enstrophy',
        'trajectory_disjoint': True,
        'relations': {
            'ns': 'discovered momentum + continuity',
            'ns_pp': 'NS + artifact-derived pressure-Poisson',
            'full': 'NS + pressure-Poisson + artifact-derived energy balance'}},
        'results': {}}
    for variant in VARIANTS:
        records = [json.loads((ROOT/'outputs/runs'/run_id(variant, seed)/
                              'metrics.json').read_text()) for seed in SEEDS]
        artifact['results'][variant] = {}
        for key in ('accuracy', 'eta_ns', 'eta_div', 'eta_pp', 'eta_energy'):
            values = np.asarray([record['test'][key] for record in records])
            artifact['results'][variant][key] = {
                'values': values.tolist(), 'mean': float(values.mean()),
                'std': float(values.std(ddof=1))}
    output = ROOT/'outputs/aggregate/full_ns_v3_equation_ablation.json'
    output.write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
