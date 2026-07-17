"""Build the trajectory-disjoint prediction cache for full-NS PI-NoProp."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hit_ns_v2 import build_learning_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/generated_hit_ns_v2')
    parser.add_argument('--cache', default='data/cache_hit_ns_v2')
    parser.add_argument('--samples-per-trajectory', type=int, default=64)
    parser.add_argument('--spatial-size', type=int, default=16)
    parser.add_argument('--time-window', type=int, default=9)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    metadata = build_learning_cache(
        args.dataset, args.cache, args.samples_per_trajectory,
        args.spatial_size, args.time_window, args.classes,
        overwrite=args.overwrite)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
