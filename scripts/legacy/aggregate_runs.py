"""Aggregate immutable optimized run records into paper-ready JSON."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    official_seeds = {0, 1, 2}
    grouped = defaultdict(list)
    for path in Path('outputs/runs').glob('*/metrics.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('run_id', '').endswith('_smoke'):
            continue
        if data.get('seed') not in official_seeds:
            continue
        key = (data['method'], data['region'], data['run_id'].split('_seed')[0])
        grouped[key].append(data)
    aggregate = {}
    for (method, region, variant), runs in grouped.items():
        accuracies = np.asarray([run['test']['accuracy'] for run in runs], dtype=float)
        aggregate.setdefault(region, {})[variant] = {
            'method': method,
            'n_runs': len(runs),
            'accuracy_mean': float(accuracies.mean()),
            'accuracy_std': float(accuracies.std(ddof=1)) if len(runs) > 1 else 0.0,
            'eta_div_mean': float(np.mean([run['test'].get('eta_div', np.nan)
                                           for run in runs])),
            'eta_pp_mean': float(np.mean([run['test'].get('eta_pp', np.nan)
                                          for run in runs])),
            'train_seconds_mean': float(np.mean([run['block_train_seconds']
                                                  for run in runs])),
            'peak_memory_mb_mean': float(np.mean([run['peak_memory_mb']
                                                   for run in runs])),
            'seeds': [run['seed'] for run in runs],
        }
    output = Path('outputs/aggregate/optimized_results.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
