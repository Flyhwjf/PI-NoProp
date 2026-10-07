"""Build the trajectory-disjoint prediction cache for full-NS PI-NoProp."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hit_dataset import build_learning_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/generated_hit_ns')
    parser.add_argument('--cache', default=None)
    parser.add_argument('--samples-per-trajectory', type=int, default=64)
    parser.add_argument('--input-size', '--spatial-size', dest='input_size',
                        type=int, default=32,
                        help='spatial size of the model input context')
    parser.add_argument('--target-size', type=int, default=16)
    parser.add_argument('--time-window', type=int, default=9)
    parser.add_argument('--classes', type=int, default=5)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    cache = (args.cache or
             ('data/cache_hit_ns'
              if (args.input_size, args.target_size) == (16, 16)
              else f'data/cache_hit_ns_input{args.input_size}_target{args.target_size}'))
    metadata = build_learning_cache(
        args.dataset, cache, args.samples_per_trajectory,
        args.input_size, args.time_window, args.classes,
        overwrite=args.overwrite, target_spatial_size=args.target_size)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
