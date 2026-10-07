"""Three-seed sweep of the local physics-loss weight with fixed NS input."""
from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_protocol import (ExperimentProtocol, add_protocol_arguments,
                                        load_run, protocol_from_args,
                                        protect_legacy_output)
SEEDS = (42, 123, 456)
LAMBDAS = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1)


def run_id(value, seed, protocol=None):
    protocol = protocol or ExperimentProtocol()
    return protocol.run_id('discovered', 'low_enstrophy', seed, weight=value,
                           tag='' if value == protocol.lambda_phys else 'lambdaablation')


def experiment_command(value, seed, protocol):
    command = [sys.executable, str(ROOT/'scripts/run_experiment.py'),
               '--physics-source', 'discovered', '--region', 'low_enstrophy',
               '--seed', str(seed), '--relation-set', 'ns',
               '--decoder-architecture', 'conv', *protocol.cli_args(weight=value)]
    if value != protocol.lambda_phys:
        command += ['--run-tag', 'lambdaablation']
    return command


def read_run(value, seed, protocol):
    return load_run(ROOT/'outputs/runs'/run_id(value, seed, protocol), protocol,
                    source='discovered', region='low_enstrophy', seed=seed,
                    weight=value)[2]


def build_parser():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    protocol = protocol_from_args(args, parser)
    for seed in SEEDS:
        for value in LAMBDAS:
            path = ROOT/'outputs/runs'/run_id(value, seed, protocol)/'metrics.json'
            if path.exists() and (value == protocol.lambda_phys or not args.overwrite):
                read_run(value, seed, protocol)
                print('reuse', value, seed, flush=True)
                continue
            protect_legacy_output(path, protocol)
            completed = subprocess.run(experiment_command(value, seed, protocol), cwd=ROOT)
            if completed.returncode:
                raise RuntimeError(f'lambda={value} seed={seed} failed')

    artifact = {'schema_version': 5, 'protocol': {**protocol.metadata(),
        'seeds': list(SEEDS), 'region': 'low_enstrophy',
        'trajectory_disjoint': True, 'relation_set': 'ns',
        'physics_condition': 'fixed discovered full-momentum coefficients',
        'varied_factor': 'local physics-loss weight only',
        'lambdas': list(LAMBDAS),
        'model_revision': protocol.model_revision,
        'readout': 'cosine similarity to frozen label embeddings'}, 'results': {}}
    for value in LAMBDAS:
        records = [read_run(value, seed, protocol) for seed in SEEDS]
        metrics = {}
        for key in ('accuracy', 'eta_ns', 'eta_div'):
            values = np.asarray([record['test'][key] for record in records])
            metrics[key] = {'values': values.tolist(),
                            'mean': float(values.mean()),
                            'std': float(values.std(ddof=1))}
        artifact['results'][str(value)] = metrics
    output = protocol.aggregate_path(ROOT, 'lambda_ablation')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
