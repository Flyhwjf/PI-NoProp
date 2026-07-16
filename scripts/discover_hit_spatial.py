"""Discover and independently validate a spatial HIT equation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.spider_hit import HITSPIDERConfig, discover, save_artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/spider/hit_spatial_equation.json')
    parser.add_argument('--domains-per-snapshot', type=int, default=4)
    parser.add_argument('--bootstrap-repeats', type=int, default=100)
    args = parser.parse_args()
    config = HITSPIDERConfig(domains_per_snapshot=args.domains_per_snapshot,
                             bootstrap_repeats=args.bootstrap_repeats)
    artifact = discover(config)
    save_artifact(artifact, args.output)
    print(json.dumps({
        'output': args.output,
        'equation': artifact['equation'],
        'metrics': artifact['metrics'],
        'validation': artifact['validation'],
    }, indent=2))
    if not artifact['validation']['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
