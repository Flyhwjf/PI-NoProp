"""Build the first-frame classification cache without regenerating HIT."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PINoPropConfig
from src.data.dataset import build_first_frame_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    cfg = PINoPropConfig()
    build_first_frame_cache(
        cfg.data.data_dir, cfg.data.cache_dir, cfg.data.regions,
        cfg.data.n_subdomains, cfg.data.split_seed,
        cfg.data.n_train, cfg.data.n_val, cfg.data.n_test,
        overwrite=args.overwrite,
    )
    print(f'Cache ready: {cfg.data.cache_dir}')


if __name__ == '__main__':
    main()
