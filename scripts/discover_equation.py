"""Run full Navier--Stokes discovery on the decaying-HIT dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spider_ns import NSSPIDERConfig, discover


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/generated_hit_ns')
    parser.add_argument('--output', default='outputs/spider/full_ns_equation.json')
    parser.add_argument('--domain-size', type=int, default=16)
    parser.add_argument('--time-window', type=int, default=9)
    parser.add_argument('--windows-per-trajectory', type=int, default=2)
    parser.add_argument('--domains-per-window', type=int, default=4)
    parser.add_argument('--test-functions', type=int, default=8)
    parser.add_argument('--bootstrap', type=int, default=100)
    parser.add_argument('--selection-eta', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = NSSPIDERConfig(
        dataset_dir=args.dataset,
        domain_size=args.domain_size,
        time_window=args.time_window,
        windows_per_trajectory=args.windows_per_trajectory,
        domains_per_window=args.domains_per_window,
        n_test_functions=args.test_functions,
        bootstrap_repeats=args.bootstrap,
        selection_eta=args.selection_eta,
        seed=args.seed,
    )
    artifact = discover(config)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    print('Equation:', dict(zip(artifact['equation']['terms'],
                                artifact['equation']['coefficients'])))
    print('Metrics:', json.dumps(artifact['metrics'], indent=2))
    print('Validation:', artifact['validation'])
    print('Artifact:', output)
    if not artifact['validation']['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
