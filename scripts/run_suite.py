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


def protocol_prefix(input_size, target_size, context_encoder):
    if (input_size, target_size) == (16, 16):
        return 'full_ns_v4'
    return (f'full_ns_v5_input{input_size}_target{target_size}_'
            f'{context_encoder}')


def run_id(method, region, seed, prefix, lambda_phys):
    weight = '0' if method == 'none' else f'{lambda_phys:g}'.replace('.', 'p')
    return f'{prefix}_{method}_{region}_lambda{weight}_seed{seed}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--steps-per-block', type=int, default=100)
    parser.add_argument('--condition-epochs', '--classifier-epochs',
                        dest='condition_epochs', type=int, default=30,
                        help='epochs for condition pretraining')
    parser.add_argument('--pretrain-epochs', type=int, default=40)
    parser.add_argument('--lambda-phys', type=float, default=None)
    parser.add_argument('--context-adapt-epochs', type=int, default=20)
    parser.add_argument('--input-size', type=int, choices=[16, 32], default=32)
    parser.add_argument('--target-size', type=int, choices=[16, 32], default=16)
    parser.add_argument('--context-encoder',
                        choices=['single', 'dual_scale', 'residual_dual_scale',
                                 'residual_warmstart'],
                        default='residual_warmstart')
    args = parser.parse_args()
    lambda_phys = (args.lambda_phys if args.lambda_phys is not None else
                   (0.01 if (args.input_size, args.target_size) == (16, 16)
                    else 0.1))
    prefix = protocol_prefix(
        args.input_size, args.target_size,
        args.context_encoder if args.input_size > args.target_size else 'single')
    log_dir = ROOT/'outputs/runs'/f'{prefix}_suite_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        for seed in SEEDS:
            # Each method has its own condition coefficients and therefore its
            # own shared encoder/decoder checkpoint.
            for method in ('discovered', 'analytic', 'none'):
                identifier = run_id(method, region, seed, prefix, lambda_phys)
                metrics_path = ROOT/'outputs/runs'/identifier/'metrics.json'
                if metrics_path.exists() and not args.overwrite:
                    print('reuse', identifier, flush=True)
                    continue
                command = [
                    sys.executable, str(ROOT/'scripts/run_experiment.py'),
                    '--physics-source', method, '--region', region,
                    '--seed', str(seed),
                    '--steps-per-block', str(args.steps_per_block),
                    '--condition-epochs', str(args.condition_epochs),
                    '--pretrain-epochs', str(args.pretrain_epochs),
                    '--lambda-phys', str(lambda_phys),
                    '--input-size', str(args.input_size),
                    '--target-size', str(args.target_size),
                    '--context-encoder', args.context_encoder,
                    '--context-adapt-epochs', str(args.context_adapt_epochs),
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
            records = [json.loads((ROOT/'outputs/runs'/run_id(
                                       method, region, seed, prefix, lambda_phys)
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
                'run_ids': [run_id(method, region, seed, prefix, lambda_phys)
                            for seed in SEEDS],
            }
    artifact = {
        'schema_version': 4 if prefix == 'full_ns_v4' else 5,
        'protocol': {
            'seeds': list(SEEDS), 'regions': list(REGIONS),
            'methods': list(METHODS),
            'trajectory_disjoint': True,
            'target': 'future local relative kinetic-energy decay quantile',
            'steps_per_block': args.steps_per_block,
            'lambda_phys': lambda_phys,
            'input_spatial_size': args.input_size,
            'target_spatial_size': args.target_size,
            'spatial_context_mode': (
                args.context_encoder if args.input_size > args.target_size
                else 'single'),
            'model_revision': (
                'trainable-physics-condition-fusion-prototype-readout'
                if prefix == 'full_ns_v4' else
                'cached-residual-warmstart-physics-condition-prototype-readout'),
            'readout': 'cosine similarity to frozen label embeddings',
        },
        'results': results,
    }
    output = (ROOT/'outputs/aggregate/full_ns_results.json'
              if prefix == 'full_ns_v4' else
              ROOT/'outputs/aggregate'/f'{prefix}_results.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == '__main__':
    main()
