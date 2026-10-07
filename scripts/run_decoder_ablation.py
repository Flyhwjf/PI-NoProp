"""Matched three-seed temporal-decoder architecture ablation."""
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
                                        assert_matched_condition, load_run,
                                        make_decoder, protocol_from_args,
                                        protect_legacy_output)
SEEDS = (42, 123, 456)
ARCHITECTURES = ('conv', 'linear')


def run_id(architecture, seed, protocol=None):
    protocol = protocol or ExperimentProtocol()
    return protocol.run_id('discovered', 'low_enstrophy', seed,
                           tag='' if architecture == 'conv' else 'decoderablation')


def experiment_command(architecture, seed, protocol):
    command = [sys.executable, str(ROOT/'scripts/run_experiment.py'),
               '--physics-source', 'discovered', '--region', 'low_enstrophy',
               '--seed', str(seed), '--relation-set', 'ns',
               '--decoder-architecture', architecture, *protocol.cli_args()]
    if architecture != 'conv':
        command += ['--run-tag', 'decoderablation']
    return command


def read_run(architecture, seed, protocol):
    return load_run(ROOT/'outputs/runs'/run_id(architecture, seed, protocol), protocol,
                    source='discovered', region='low_enstrophy', seed=seed,
                    architecture=architecture)


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
        reference = None
        for architecture in ARCHITECTURES:
            path = ROOT/'outputs/runs'/run_id(architecture, seed, protocol)/'metrics.json'
            if path.exists() and (architecture == 'conv' or not args.overwrite):
                print('reuse', architecture, seed, flush=True)
            else:
                protect_legacy_output(path, protocol)
                if architecture == 'linear':
                    shared = protocol.shared_path(ROOT, 'discovered', 'low_enstrophy', seed)
                    if not shared.exists():
                        raise FileNotFoundError(f'matched conv condition checkpoint required: {shared}')
                completed = subprocess.run(experiment_command(architecture, seed, protocol), cwd=ROOT)
                if completed.returncode:
                    raise RuntimeError(f'{architecture} seed={seed} failed')
            checkpoint, _, _ = read_run(architecture, seed, protocol)
            if architecture == 'conv':
                reference = checkpoint
            else:
                assert_matched_condition(reference, checkpoint)

    artifact = {'schema_version': 5, 'protocol': {**protocol.metadata(),
        'seeds': list(SEEDS), 'region': 'low_enstrophy',
        'trajectory_disjoint': True, 'relation_set': 'ns',
        'shared_condition_components_across_architectures': True,
        'condition_matching': 'bitwise checkpoint verification of full encoder, '
                              'physics condition, fusion, prototypes and normalization buffers',
        'architectures': list(ARCHITECTURES),
        'model_revision': protocol.model_revision,
        'readout': 'cosine similarity to frozen label embeddings'}, 'results': {}}
    for architecture in ARCHITECTURES:
        runs = [read_run(architecture, seed, protocol) for seed in SEEDS]
        records = [run[2] for run in runs]
        metrics = {}
        for key in ('accuracy', 'eta_ns', 'eta_div'):
            values = np.asarray([record['test'][key] for record in records])
            metrics[key] = {'values': values.tolist(),
                            'mean': float(values.mean()),
                            'std': float(values.std(ddof=1))}
        parameters = np.asarray([
            sum(parameter.numel() for parameter in make_decoder(run[1]).parameters())
            for run in runs])
        metrics['decoder_parameters'] = {
            'values': parameters.tolist(), 'mean': float(parameters.mean()),
            'std': float(parameters.std(ddof=1))}
        artifact['results'][architecture] = metrics
    output = protocol.aggregate_path(ROOT, 'decoder_ablation')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
