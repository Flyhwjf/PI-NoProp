"""Validate complete auxiliary experiments and their shared input/target protocol."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_protocol import (add_protocol_arguments, load_run,
                                        protocol_from_args)
from scripts.sync_paper_results import collect_values


def require(value, message):
    if not value:
        raise AssertionError(message)
    print('PASS:', message)


def main():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    args = parser.parse_args()
    protocol = protocol_from_args(args, parser)
    _, sources = collect_values(protocol)
    require(len(sources) == 7, 'all seven paper artifacts share the declared protocol')
    artifacts = {
        name: json.loads((ROOT/record['path']).read_text(encoding='utf-8'))
        for name, record in sources.items()}
    regions = ('low_enstrophy', 'high_enstrophy')
    methods = {'noprop_reference', 'noprop_ct', 'cnn_bp', 'global_physics_bp',
               'spider_rate_classifier', 'noprop_no_equation',
               'noprop_analytic_ns', 'pi_noprop'}
    for region in regions:
        require(set(artifacts['baselines']['results'][region]) == methods,
                f'{region} contains all eight comparison methods')
        for source in ('none', 'analytic', 'discovered'):
            for seed in (42, 123, 456):
                run = ROOT/'outputs/runs'/protocol.run_id(source, region, seed)
                checkpoint, config, metrics = load_run(
                    run, protocol, source=source, region=region, seed=seed)
                require(metrics['block_updates'] == [100]*10,
                        f'{run.name}: ten blocks each received 100 local updates')
                require(config.data.cache_dir == protocol.cache_dir,
                        f'{run.name}: size-matched input cache')
                require(metrics['readout'] == 'cosine similarity to frozen label embeddings',
                        f'{run.name}: frozen prototype readout')
                require(config.physics.physics_grid_size == protocol.target_size,
                        f'{run.name}: physics evaluated on centred target')
                require(all(np.isfinite(list(metrics['test'].values()))),
                        f'{run.name}: finite held-out metrics')
                del checkpoint

    relation = artifacts['relation_ablation']['results']
    require(set(relation) == {'ns', 'ns_pp', 'full'},
            'relation ablation contains all three declared relation sets')
    weights = artifacts['lambda_ablation']
    require(weights['protocol']['lambdas'] == [0.0, 0.001, 0.003, 0.01, 0.03, 0.1],
            'physics-weight sweep contains the six declared values')
    require(set(artifacts['decoder_ablation']['results']) == {'linear', 'conv'},
            'decoder comparison contains both architectures')
    require(artifacts['decoder_ablation']['protocol']
            ['shared_condition_components_across_architectures']
            and 'bitwise checkpoint verification' in artifacts['decoder_ablation']
            ['protocol']['condition_matching'],
            'decoder-only comparison verifies identical frozen condition state')

    noise = artifacts['noise']
    require(noise['protocol']['repetitions_per_seed'] == 5,
            'predictive noise averages five observation realisations per seed')
    require(noise['protocol']['varied_randomness'] == 'observation noise only',
            'predictive noise fixes inference randomness')
    require('Gaussian' in noise['protocol']['noise_definition'],
            'predictive observation corruption is explicitly Gaussian')
    levels = {'0.0', '0.01', '0.05', '0.1', '0.2', '0.5', '1.0'}
    for region in regions:
        for source in ('none', 'discovered'):
            require(set(noise['results'][region][source]) == levels,
                    f'{region}/{source}: all seven observation-noise levels')
            latent = artifacts['latent_analysis']['results'][region][source]
            require(latent['n_samples'] == (324 if region == 'low_enstrophy' else 252),
                    f'{region}/{source}: latent statistics use all three held-out seeds')

    spider = json.loads((ROOT/'outputs/aggregate/full_ns_spider_noise.json')
                        .read_text(encoding='utf-8'))
    require(spider.get('schema_version') == 3,
            'DNS-field discovery-noise protocol remains independently recorded')
    require(set(spider['levels']) ==
            {'0.0', '0.001', '0.005', '0.01', '0.02', '0.05', '0.1', '0.5', '1.0'},
            'SPIDER diagnostics cover all nine discovery-noise levels')
    require(spider['levels']['0.0']['terms'] ==
            ['time_derivative', 'convection', 'pressure_gradient', 'velocity_laplacian'],
            'independent clean SPIDER scan recovers complete momentum support')

    paper = (ROOT/'paper/Physics-Informed NoProp.tex').read_text(encoding='utf-8')
    for token in ('experiment_values.tex', 'eq:context-gate', 'tab:main-results',
                  'tab:equation-ablation', 'tab:efficiency', 'tab:spider-noise-appendix'):
        require(token in paper, f'manuscript contains {token}')
    print('Size-matched extended experiment validation complete.')


if __name__ == '__main__':
    main()
