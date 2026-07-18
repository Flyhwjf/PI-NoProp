"""Fail-fast scientific audit for the full-NS discovery-to-training chain."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print('PASS:', message)


def main():
    dataset = Path('data/generated_hit_ns')
    manifest = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['quality_passed'], 'all DNS trajectory quality gates passed')
    split_sets = {key: set(value) for key, value in manifest['splits'].items()}
    require(not split_sets['discovery'] & split_sets['validation'],
            'discovery and validation trajectories are disjoint')
    require(not split_sets['discovery'] & split_sets['test'],
            'discovery and test trajectories are disjoint')
    require(not split_sets['validation'] & split_sets['test'],
            'validation and test trajectories are disjoint')
    require(all(record['quality']['energy_monotone']
                for record in manifest['trajectories']),
            'kinetic energy decays monotonically in every unforced trajectory')
    require(max(record['quality']['max_divergence_rms']
                for record in manifest['trajectories']) < 1e-10,
            'spectral incompressibility is at machine precision')
    require(max(record['quality']['max_cfl']
                for record in manifest['trajectories']) < 0.5,
            'all trajectories satisfy the CFL quality limit')

    artifact = json.loads(Path('outputs/spider/full_ns_equation.json')
                          .read_text(encoding='utf-8'))
    require(artifact['validation']['passed'], 'full-NS artifact passed all gates')
    required_terms = ['time_derivative', 'convection', 'pressure_gradient',
                      'velocity_laplacian']
    require(artifact['equation']['terms'] == required_terms,
            'SPIDER selected exactly the four full momentum terms')
    require(artifact['metrics']['bootstrap_support_fraction'] >= 0.9,
            'trajectory-bootstrap support is at least 90%')
    require(artifact['metrics']['max_coefficient_relative_error'] <= 0.1,
            'all discovered coefficients are within 10% of DNS truth')
    require(artifact['metrics']['support_separation_ratio'] >= 1.25,
            'selected support is separated from the next candidate')

    cache = Path('data/cache_hit_ns')
    if cache.exists() and (cache/'trajectory_ids.npy').exists():
        trajectory_ids = np.load(cache/'trajectory_ids.npy')
        splits = np.load(cache/'splits.npy')
        groups = [set(trajectory_ids[splits == code].tolist()) for code in range(3)]
        require(not groups[0] & groups[1] and not groups[0] & groups[2]
                and not groups[1] & groups[2],
                'learning cache preserves trajectory-disjoint splits')
        metadata = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
        require(metadata['target'] ==
                'future local relative kinetic-energy decay quantile',
                'classification target is predictive and provenance-recorded')
        require((cache/'ns_terms.npy').exists(),
                'target-free first-frame NS energy terms are cached')

    audit_path = Path('outputs/aggregate/hit_predictability_audit.json')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for region, record in audit['regions'].items():
        require(record['ns_rate_target_correlation_test'] > 0.99,
                f'{region} discovered-NS rate predicts held-out target')
        require(record['first_frame_discovered_ns_accuracy']['test'] > 80,
                f'{region} first-frame NS oracle exceeds 80% test accuracy')

    aggregate = json.loads(Path('outputs/aggregate/full_ns_results.json')
                           .read_text(encoding='utf-8'))
    require(aggregate['protocol']['model_revision'] ==
            'trainable-physics-condition-fusion',
            'aggregate uses trained physics-condition fusion')
    for region, record in aggregate['results'].items():
        discovered_accuracy = record['discovered']['accuracy']['mean']
        none_accuracy = record['none']['accuracy']['mean']
        require(len(record['discovered']['accuracy']['values']) == 3,
                f'{region} has three discovered-equation seeds')
        require(discovered_accuracy > 65,
                f'{region} discovered PI-NoProp exceeds 65% mean accuracy')
        require(discovered_accuracy-none_accuracy > 45,
                f'{region} discovered equation improves accuracy by over 45 points')
    print('Full NS discovery-to-prediction validation complete.')


if __name__ == '__main__':
    main()
