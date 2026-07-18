"""Matched three-seed temporal-decoder architecture ablation."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEEDS = (42, 123, 456)
ARCHITECTURES = ('linear', 'conv')


def run_id(architecture, seed):
    base = f'full_ns_v4_discovered_low_enstrophy_lambda0p01_seed{seed}'
    return base if architecture == 'conv' else f'{base}_decoderablation'


def main():
    for seed in SEEDS:
        for architecture in ARCHITECTURES:
            path = ROOT/'outputs/runs'/run_id(architecture, seed)/'metrics.json'
            if path.exists():
                print('reuse', architecture, seed, flush=True)
                continue
            completed = subprocess.run([
                sys.executable, str(ROOT/'scripts/run_experiment.py'),
                '--physics-source', 'discovered', '--region', 'low_enstrophy',
                '--seed', str(seed), '--relation-set', 'ns',
                '--decoder-architecture', architecture,
                '--run-tag', 'decoderablation'], cwd=ROOT)
            if completed.returncode:
                raise RuntimeError(f'{architecture} seed={seed} failed')

    artifact = {'schema_version': 1, 'protocol': {
        'seeds': list(SEEDS), 'region': 'low_enstrophy',
        'trajectory_disjoint': True, 'relation_set': 'ns',
        'lambda_weight': 0.01,
        'shared_condition_components_across_architectures': True,
        'architectures': list(ARCHITECTURES)}, 'results': {}}
    for architecture in ARCHITECTURES:
        records = [json.loads((ROOT/'outputs/runs'/run_id(architecture, seed)/
                              'metrics.json').read_text()) for seed in SEEDS]
        metrics = {}
        for key in ('accuracy', 'eta_ns', 'eta_div'):
            values = np.asarray([record['test'][key] for record in records])
            metrics[key] = {'values': values.tolist(),
                            'mean': float(values.mean()),
                            'std': float(values.std(ddof=1))}
        parameters = np.asarray([
            record.get('decoder_parameters', 0) for record in records])
        if not parameters.any():
            # Current convolutional runs predate explicit parameter logging.
            from src.decoder import TemporalPhysicsDecoder
            parameters[:] = sum(p.numel() for p in TemporalPhysicsDecoder().parameters())
        metrics['decoder_parameters'] = {
            'values': parameters.tolist(), 'mean': float(parameters.mean()),
            'std': float(parameters.std(ddof=1))}
        artifact['results'][architecture] = metrics
    output = ROOT/'outputs/aggregate/full_ns_decoder_ablation.json'
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
