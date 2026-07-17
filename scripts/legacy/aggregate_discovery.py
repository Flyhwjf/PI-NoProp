"""Aggregate formal discovery-to-training runs and the new lambda sweep."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    runs = []
    for path in Path('outputs/runs').glob('local_fast_*_lambda*_seed*_steps100/metrics.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('seed') in (0, 1, 2) and data.get('physics_source'):
            runs.append(data)

    main_groups = defaultdict(list)
    sweep_groups = defaultdict(list)
    for run in runs:
        config_path = Path('outputs/runs') / run['run_id'] / 'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        value = float(config['physics']['lambda_weight'])
        source = run['physics_source']
        if abs(value - 0.01) < 1e-12:
            main_groups[(run['region'], source)].append(run)
        if run['region'] == 'centre' and source == 'spider':
            sweep_groups[value].append(run)

    def summarize(group):
        accuracy = np.asarray([item['test']['accuracy'] for item in group])
        def mean_metric(name):
            values = [item['test'].get(name, np.nan) for item in group]
            return float(np.nanmean(values))
        return {
            'n_runs': len(group),
            'seeds': sorted(item['seed'] for item in group),
            'accuracy_mean': float(accuracy.mean()),
            'accuracy_std': float(accuracy.std(ddof=1)) if len(group) > 1 else 0.0,
            'eta_div_mean': mean_metric('eta_div'),
            'eta_pp_discovered_mean': mean_metric('eta_pp_discovered'),
            'eta_pp_analytic_mean': mean_metric('eta_pp_analytic'),
            'train_seconds_mean': float(np.mean(
                [item['block_train_seconds'] for item in group])),
            'peak_memory_mb_mean': float(np.mean(
                [item['peak_memory_mb'] for item in group])),
        }

    output = {
        'protocol': {
            'seeds': [0, 1, 2],
            'steps_per_block': 100,
            'common_pp_metric': 'eta_pp_discovered',
            'equation_artifact': 'outputs/spider/hit_spatial_equation.json',
        },
        'main': {region: {source: summarize(group)
                          for (group_region, source), group in main_groups.items()
                          if group_region == region}
                 for region in ('centre', 'edge')},
        'lambda_sweep': {str(value): summarize(group)
                         for value, group in sorted(sweep_groups.items())},
    }
    path = Path('outputs/aggregate/discovery_results.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
