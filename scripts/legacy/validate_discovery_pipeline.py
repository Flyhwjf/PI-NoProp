"""Fail-fast audit of discovery provenance, formal runs, and paper claims."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main():
    artifact = json.loads(Path('outputs/spider/hit_full_ns_v2.json').read_text())
    assert artifact['validation']['passed']
    assert artifact['equation']['terms'] == [
        'time_derivative', 'convection', 'pressure_gradient',
        'velocity_laplacian']
    assert artifact['metrics']['bootstrap_support_fraction'] == 1.0
    assert artifact['metrics']['support_separation_ratio'] >= 40.0
    assert artifact['metrics']['max_coefficient_relative_error'] < 0.005
    splits = {key: set(value) for key, value in
              artifact['data']['trajectory_splits'].items()}
    assert not splits['discovery'] & splits['validation']
    assert not splits['discovery'] & splits['test']
    assert not splits['validation'] & splits['test']

    results = json.loads(Path('outputs/aggregate/full_ns_v3_results.json').read_text())
    assert results['protocol']['seeds'] == [42, 123, 456]
    assert results['protocol']['trajectory_disjoint'] is True
    for region in ('low_enstrophy', 'high_enstrophy'):
        for source in ('none', 'analytic', 'discovered'):
            record = results['results'][region][source]
            assert len(record['accuracy']['values']) == 3
            assert len(record['eta_ns']['values']) == 3
        assert (results['results'][region]['discovered']['accuracy']['mean']
                - results['results'][region]['none']['accuracy']['mean']) > 45

    paper = Path('paper/Physics-Informed NoProp.tex').read_text(encoding='utf-8')
    required = ('1.003459', '0.00501841', '100\\%', '81.79', '73.41',
                'PI-NoProp (discovered NS)', 'fig_v2_spider.png')
    missing = [token for token in required if token not in paper]
    assert not missing, f'Paper is missing discovery claims: {missing}'
    forbidden = ('0.2929985', '60 archived', 'fig_spider_discovery.png')
    present = [token for token in forbidden if token.lower() in paper.lower()]
    assert not present, f'Paper retains superseded claims: {present}'
    print('Discovery pipeline audit: PASS')


if __name__ == '__main__':
    main()
