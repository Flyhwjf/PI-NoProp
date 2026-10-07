"""Three-seed low-enstrophy ablation of artifact-derived physical relations."""
from __future__ import annotations

import json
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_protocol import (ExperimentProtocol, add_protocol_arguments,
                                        assert_matched_condition, load_run, protocol_from_args,
                                        protect_legacy_output)
SEEDS = (42, 123, 456)
VARIANTS = ('ns', 'ns_pp', 'full')


def run_id(variant, seed, protocol=None):
    protocol = protocol or ExperimentProtocol()
    return protocol.run_id('discovered', 'low_enstrophy', seed,
                           relation=variant, tag='eqablation')


def experiment_command(variant, seed, protocol):
    return [sys.executable, str(ROOT/'scripts/run_experiment.py'),
            '--physics-source', 'discovered', '--region', 'low_enstrophy',
            '--seed', str(seed), '--relation-set', variant,
            '--decoder-architecture', 'conv', '--run-tag', 'eqablation',
            *protocol.cli_args()]


def read_run(variant, seed, protocol):
    return load_run(ROOT/'outputs/runs'/run_id(variant, seed, protocol), protocol,
                    source='discovered', region='low_enstrophy', seed=seed,
                    relation=variant)[2]


def assert_matched_shared_state(reference, candidate):
    """Only the local blocks and physical relation loss may vary within a seed."""
    try:
        assert_matched_condition(reference, candidate)
    except ValueError as error:
        raise ValueError(f'relation ablation condition/prototype/normalization mismatch: {error}') from error

    prefixes = ('encoder.', 'physics_encoder.', 'condition_fusion.', 'label_embed.')
    buffers = {'physics_coefficients', 'physics_mean', 'physics_std'}
    def condition_state(checkpoint):
        return {key: value for key, value in checkpoint['model_state_dict'].items()
                if key.startswith(prefixes) or key in buffers}

    for name, left, right in (
            ('condition/prototypes/normalization', condition_state(reference),
             condition_state(candidate)),
            ('decoder', reference.get('decoder_state_dict', {}),
             candidate.get('decoder_state_dict', {}))):
        if not left or left.keys() != right.keys():
            raise ValueError(f'relation ablation {name} state keys differ or are missing')
        changed = [key for key in left
                   if left[key].dtype != right[key].dtype
                   or left[key].shape != right[key].shape
                   or not torch.equal(
                       left[key].contiguous().reshape(-1).view(torch.uint8),
                       right[key].contiguous().reshape(-1).view(torch.uint8))]
        if changed:
            raise ValueError(f'relation ablation {name} state is not bitwise identical: {changed[:8]}')


def aggregate_runs(protocol):
    """Validate all completed checkpoints before building an aggregate."""
    records = {variant: [] for variant in VARIANTS}
    run_ids = {variant: [] for variant in VARIANTS}
    for seed in SEEDS:
        reference = None
        for variant in VARIANTS:
            run = ROOT/'outputs/runs'/run_id(variant, seed, protocol)
            checkpoint, _, record = load_run(
                run, protocol, source='discovered', region='low_enstrophy',
                seed=seed, relation=variant)
            if variant == 'ns':
                reference = checkpoint
            else:
                assert_matched_shared_state(reference, checkpoint)
            records[variant].append(record)
            run_ids[variant].append(run.name)

    artifact = {'schema_version': 5, 'protocol': {**protocol.metadata(),
        'seeds': list(SEEDS), 'region': 'low_enstrophy',
        'trajectory_disjoint': True,
        'model_revision': protocol.model_revision,
        'readout': 'cosine similarity to frozen label embeddings',
        'run_ids': run_ids,
        'shared_condition_and_decoder_verified': True,
        'condition_matching': 'bitwise checkpoint verification within each seed across '
                              'ns/ns_pp/full of full encoder, physics condition, fusion, '
                              'prototypes, normalization buffers and all decoder parameters/buffers',
        'ns_control': 'separately executed, same seeds/settings as main',
        'relations': {
            'ns': 'discovered momentum + continuity',
            'ns_pp': 'NS + artifact-derived pressure-Poisson',
            'full': 'NS + pressure-Poisson + artifact-derived energy balance'}},
        'results': {}}
    for variant in VARIANTS:
        artifact['results'][variant] = {}
        for key in ('accuracy', 'eta_ns', 'eta_div', 'eta_pp', 'eta_energy'):
            values = np.asarray([record['test'][key] for record in records[variant]])
            artifact['results'][variant][key] = {
                'values': values.tolist(), 'mean': float(values.mean()),
                'std': float(values.std(ddof=1))}
    return artifact


def build_parser():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='validate/reaggregate completed runs; never launch training')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.aggregate_only and args.overwrite:
        parser.error('--aggregate-only cannot be combined with --overwrite')
    protocol = protocol_from_args(args, parser)
    if not args.aggregate_only:
        for seed in SEEDS:
            for variant in VARIANTS:
                path = ROOT/'outputs/runs'/run_id(variant, seed, protocol)/'metrics.json'
                if args.overwrite or not path.exists():
                    protect_legacy_output(path, protocol)
                    print('run', variant, seed, flush=True)
                    completed = subprocess.run(experiment_command(variant, seed, protocol), cwd=ROOT)
                    if completed.returncode:
                        raise RuntimeError(f'{variant} seed {seed} failed')
                read_run(variant, seed, protocol)
    artifact = aggregate_runs(protocol)
    output = protocol.aggregate_path(ROOT, 'relation_ablation')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
