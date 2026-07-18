"""Fail-fast audit for expanded baselines, relation ablation, and noise tables."""
import json
from pathlib import Path


def require(value, message):
    if not value: raise AssertionError(message)
    print('PASS:', message)


def main():
    # Every retained run must point to the canonical current HIT dataset and
    # validated SPIDER artifact; stale v2 paths make results non-reproducible.
    run_files = list(Path('outputs/runs').glob('*/config.json'))
    require(run_files, 'run configurations are present')
    for config_path in run_files:
        config = json.loads(config_path.read_text())
        data_dir = config.get('data', {}).get('data_dir')
        cache_dir = config.get('data', {}).get('cache_dir')
        equation_path = config.get('physics', {}).get('discovered_artifact')
        require(data_dir == 'data/generated_hit_ns'
                and Path(data_dir).exists(),
                f'{config_path.parent.name} uses the canonical generated HIT dataset')
        require(cache_dir == 'data/cache_hit_ns'
                and Path(cache_dir).exists(),
                f'{config_path.parent.name} uses the canonical HIT cache')
        require(equation_path == 'outputs/spider/full_ns_equation.json'
                and Path(equation_path).exists(),
                f'{config_path.parent.name} uses the validated SPIDER artifact')

    baselines = json.loads(Path(
        'outputs/aggregate/full_ns_baselines.json').read_text())
    expected_baselines = {
        'noprop_reference', 'noprop_ct', 'cnn_bp', 'global_physics_bp',
        'spider_rate_classifier', 'noprop_no_equation',
        'noprop_analytic_ns', 'pi_noprop'}
    for region in ('low_enstrophy', 'high_enstrophy'):
        methods = baselines['results'][region]
        require(set(methods) == expected_baselines,
                f'{region} contains all eight comparison methods')
        for method in methods.values():
            require(len(method['accuracy']['values']) == 3,
                    f'{region} baseline has three seeds')
            require(len(method['parameters']['values']) == 3,
                    f'{region} baseline records parameter counts')
        require(methods['pi_noprop']['accuracy']['mean'] > 65,
                f'{region} PI-NoProp remains above 65%')
        require(methods['noprop_reference']['accuracy']['mean'] < 30,
                f'{region} reference NoProp result is honestly retained')
        require(methods['noprop_ct']['accuracy']['mean'] < 30,
                f'{region} NoProp-CT result is honestly retained')

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

    lambda_ablation = json.loads(Path(
        'outputs/aggregate/full_ns_lambda_ablation.json').read_text())
    require(lambda_ablation['protocol']['lambdas']
            == [0.0, 0.001, 0.003, 0.01, 0.03, 0.1],
            'lambda ablation contains the declared six weights')
    for result in lambda_ablation['results'].values():
        require(len(result['accuracy']['values']) == 3,
                'lambda ablation has three seeds per weight')
    require(lambda_ablation['results']['0.1']['eta_ns']['mean']
            < lambda_ablation['results']['0.0']['eta_ns']['mean'],
            'lambda sweep improves NS consistency over no physics loss')

    decoder_ablation = json.loads(Path(
        'outputs/aggregate/full_ns_decoder_ablation.json').read_text())
    require(set(decoder_ablation['results']) == {'linear', 'conv'},
            'decoder ablation contains linear and convolutional variants')
    for result in decoder_ablation['results'].values():
        require(len(result['accuracy']['values']) == 3,
                'decoder ablation has three seeds per architecture')
        require(result['decoder_parameters']['mean'] > 0,
                'decoder ablation records parameter counts')

    latent = json.loads(Path(
        'outputs/aggregate/full_ns_latent_analysis.json').read_text())
    require(set(latent['results']) == {'low_enstrophy', 'high_enstrophy'},
            'latent analysis contains both enstrophy groups')
    for region in latent['results'].values():
        require(set(region) == {'none', 'analytic', 'discovered'},
                'latent analysis contains all three matched variants')
        require(all('silhouette_128d' in value for value in region.values()),
                'latent analysis records original-space silhouette scores')
    require(latent['results']['low_enstrophy']['discovered']
            ['between_class_centroid_distance']
            > latent['results']['low_enstrophy']['none']
            ['between_class_centroid_distance'],
            'PI-NoProp improves low-enstrophy class separation')

    noise = json.loads(Path('outputs/aggregate/full_ns_noise.json').read_text())
    predictive_levels = {'0.0', '0.01', '0.05', '0.1', '0.2', '0.5', '1.0'}
    require(noise['protocol']['repetitions_per_seed'] == 5,
            'noise results average five realizations per seed')
    require(noise['protocol']['varied_randomness'] == 'observation noise only',
            'predictive noise sweep isolates observation noise')
    require('Gaussian' in noise['protocol']['noise_definition'],
            'predictive noise protocol records its Gaussian definition')
    for region in noise['results'].values():
        for method in region.values():
            require(set(method) == predictive_levels,
                    'predictive noise sweep contains all seven levels')
        require(region['discovered']['0.1']['accuracy']['mean'] > 70,
                'discovered PI-NoProp remains predictive at 10% channel noise')
        require(region['discovered']['0.5']['accuracy']['mean'] < 40,
                'predictive sweep honestly retains the 50% noise failure')

    spider = json.loads(Path(
        'outputs/aggregate/full_ns_spider_noise.json').read_text())
    require(spider.get('schema_version') == 3,
            'SPIDER noise artifact records the explicit tracking protocol')
    require(spider['protocol']['max_coefficient_relative_error'] == 2.0
            and spider['protocol']['max_validation_eta'] == 0.25
            and spider['protocol']['max_test_eta'] == 0.25
            and spider['protocol']['min_bootstrap_support'] == 0.5,
            'SPIDER corruption-sweep thresholds are machine readable')
    spider_levels = {'0.0', '0.001', '0.005', '0.01', '0.02',
                     '0.05', '0.1', '0.5', '1.0'}
    require(set(spider['levels']) == spider_levels,
            'SPIDER noise sweep contains all nine levels')
    require('Gaussian' in spider['noise_definition'],
            'SPIDER noise protocol records its Gaussian definition')
    expected = ['time_derivative', 'convection', 'pressure_gradient',
                'velocity_laplacian']
    require(spider['levels']['0.0']['terms'] == expected,
            'clean SPIDER run recovers the four-term support')
    require(spider['levels']['0.0']['validation_passed_under_noise_protocol'],
            'clean large-domain SPIDER artifact passes noise protocol')
    require(spider['levels']['0.005']['validation_passed_under_noise_protocol'],
            'SPIDER remains validated at 0.5% field noise')
    require(not spider['levels']['0.01']['validation_passed_under_noise_protocol'],
            'SPIDER support failure at 1% field noise is explicitly retained')
    require(not spider['levels']['0.5']['validation_passed_under_noise_protocol'],
            'high-noise coefficient failure is explicitly retained')

    paper = Path('paper/Physics-Informed NoProp.tex').read_text(encoding='utf-8')
    paper_v1 = Path('paper-v1/Physics-Informed NoProp.tex').read_text(encoding='utf-8')
    for token in ('tab:expanded-baselines', 'tab:equation-ablation',
                  'tab:noise-prediction', 'tab:spider-noise', 'fig_noise.png'):
        require(token in paper, f'paper contains {token}')
    require('fig_ablation.png' in paper_v1,
            'paper-v1 contains the 4.5 ablation figure')
    require('fig_latent_analysis.png' in paper_v1,
            'paper-v1 contains the 4.6 latent-analysis figure')
    require('tab:efficiency' in paper_v1 and 'RTX 4060' in paper_v1,
            'paper-v1 contains the measured 4.7 efficiency protocol')
    print('Expanded experiment validation complete.')


if __name__ == '__main__':
    main()
