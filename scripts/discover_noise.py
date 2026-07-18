"""Weak-SPIDER equation recovery across controlled observation noise."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spider_ns import NSSPIDERConfig, discover


def main():
    levels = (0.0, 0.1, 0.5, 1.0)
    artifact = {'schema_version': 1,
                'noise_definition': 'Gaussian fraction of per-field DNS standard deviation',
                'levels': {}}
    for level in levels:
        print('discover noise', level, flush=True)
        result = discover(NSSPIDERConfig(
            noise_level=level, time_window=17, domain_size=32,
            windows_per_trajectory=1, domains_per_window=4,
            bootstrap_repeats=30, selection_eta=0.01,
            max_coefficient_relative_error=2.0,
            max_validation_eta=0.25, max_test_eta=0.25,
            min_bootstrap_support=0.5))
        artifact['levels'][str(level)] = {
            'terms': result['equation']['terms'],
            'coefficients': result['equation']['coefficients'],
            'discovery_eta': result['metrics']['discovery_eta'],
            'validation_eta': result['metrics']['validation_eta'],
            'test_eta': result['metrics']['test_eta'],
            'bootstrap_support_fraction': result['metrics']['bootstrap_support_fraction'],
            'max_coefficient_relative_error': result['metrics']['max_coefficient_relative_error'],
            'validation_passed_under_noise_protocol': result['validation']['passed'],
            'failure_reasons': result['validation']['failure_reasons'],
        }
    output = ROOT/'outputs/aggregate/full_ns_spider_noise.json'
    output.write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2))


if __name__ == '__main__':
    main()
