"""Fail-fast audit for expanded baselines, relation ablation, and noise tables."""
import json
from pathlib import Path


def require(value, message):
    if not value: raise AssertionError(message)
    print('PASS:', message)


def main():
    baselines = json.loads(Path(
        'outputs/aggregate/full_ns_baselines.json').read_text())
    for region in ('low_enstrophy', 'high_enstrophy'):
        methods = baselines['results'][region]
        require(set(methods) == {'cnn_bp', 'global_physics_bp',
                'spider_rate_classifier', 'noprop_vanilla', 'pi_noprop'},
                f'{region} contains all five method baselines')
        for method in methods.values():
            require(len(method['accuracy']['values']) == 3,
                    f'{region} baseline has three seeds')
        require(methods['pi_noprop']['accuracy']['mean'] > 65,
                f'{region} PI-NoProp remains above 65%')

    ablation = json.loads(Path(
        'outputs/aggregate/full_ns_relation_ablation.json').read_text())
    require(set(ablation['results']) == {'ns', 'ns_pp', 'full'},
            'relation ablation contains NS, NS+PP and full variants')
    require(ablation['results']['full']['eta_pp']['mean']
            < ablation['results']['ns']['eta_pp']['mean'],
            'full relation set improves pressure-Poisson consistency')
    require(ablation['results']['full']['eta_energy']['mean']
            < ablation['results']['ns']['eta_energy']['mean'],
            'full relation set improves energy consistency')

    noise = json.loads(Path('outputs/aggregate/full_ns_noise.json').read_text())
    require(noise['protocol']['repetitions_per_seed'] == 5,
            'noise results average five realizations per seed')
    for region in noise['results'].values():
        for method in region.values():
            require(set(method) == {'0.0', '0.1', '0.5', '1.0'},
                    'predictive noise sweep contains all four levels')

    spider = json.loads(Path(
        'outputs/aggregate/full_ns_spider_noise.json').read_text())
    expected = ['time_derivative', 'convection', 'pressure_gradient',
                'velocity_laplacian']
    for record in spider['levels'].values():
        require(record['terms'] == expected,
                'noisy SPIDER run retains the four-term support')
    require(spider['levels']['0.0']['validation_passed_under_noise_protocol'],
            'clean large-domain SPIDER artifact passes noise protocol')
    require(not spider['levels']['0.5']['validation_passed_under_noise_protocol'],
            'high-noise coefficient failure is explicitly retained')

    paper = Path('paper/Physics-Informed NoProp.tex').read_text(encoding='utf-8')
    for token in ('tab:expanded-baselines', 'tab:equation-ablation',
                  'tab:noise-prediction', 'tab:spider-noise', 'fig_noise.png'):
        require(token in paper, f'paper contains {token}')
    print('Expanded experiment validation complete.')


if __name__ == '__main__':
    main()
