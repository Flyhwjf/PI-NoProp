"""Fail-fast audit of discovery provenance, formal runs, and paper claims."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.discovered_equation import load_validated_equation


def main():
    artifact_path = Path('outputs/spider/hit_spatial_equation.json')
    artifact = load_validated_equation(artifact_path)
    assert len(artifact['data']['train_files']) == 60
    assert len(artifact['data']['validation_files']) == 20
    assert not (set(artifact['data']['train_files']) &
                set(artifact['data']['validation_files']))
    assert artifact['metrics']['bootstrap_support_fraction'] >= 0.8
    assert artifact['metrics']['support_separation_ratio'] >= 1.5

    results = json.loads(Path('outputs/aggregate/discovery_results.json').read_text())
    for region in ('centre', 'edge'):
        for source in ('none', 'analytic', 'spider'):
            record = results['main'][region][source]
            assert record['n_runs'] == 3
            assert record['seeds'] == [0, 1, 2]
    assert len(results['lambda_sweep']) == 5
    assert all(record['n_runs'] == 3
               for record in results['lambda_sweep'].values())

    paper = Path('paper/Physics-Informed NoProp.tex').read_text(encoding='utf-8')
    required = ('0.2929985', '100\\% bootstrap',
                'Discovery-informed NoProp (SPIDER)',
                'fig_spider_discovery.png')
    missing = [token for token in required if token not in paper]
    assert not missing, f'Paper is missing discovery claims: {missing}'
    forbidden = ('current SPIDER output does not directly recover',
                 'lambda has not yet been swept', '65.6\\%')
    present = [token for token in forbidden if token.lower() in paper.lower()]
    assert not present, f'Paper retains superseded claims: {present}'
    print('Discovery pipeline audit: PASS')


if __name__ == '__main__':
    main()
