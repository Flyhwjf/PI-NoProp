"""Generate trajectory-disjoint decaying HIT data for full-NS discovery."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hit_dns_v2 import DecayingHITDNS, HITDNSConfig


def split_name(index: int, count: int) -> str:
    """Deterministic 60/20/20 trajectory split with no temporal leakage."""
    train_end = max(1, int(round(0.6 * count)))
    validation_end = max(train_end + 1, int(round(0.8 * count)))
    if index < train_end:
        return 'discovery'
    if index < validation_end:
        return 'validation'
    return 'test'


def generate_trajectory(root: Path, index: int, args) -> dict:
    trajectory_dir = root / f'trajectory_{index:03d}'
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = trajectory_dir / 'metadata.json'
    if args.resume and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        expected = trajectory_dir / f'frame_{args.frames - 1:03d}.npz'
        if metadata.get('complete') and expected.exists():
            print(f'[{index + 1}/{args.trajectories}] reuse {trajectory_dir.name}')
            return metadata

    config = HITDNSConfig(
        grid_size=args.grid_size,
        box_length=args.box_length,
        viscosity=args.viscosity,
        time_step=args.time_step,
        initial_energy=args.initial_energy,
        spectrum_peak=args.spectrum_peak,
        seed=args.seed + index,
    )
    dns = DecayingHITDNS(config)
    started = time.perf_counter()
    dns.step(args.warmup_steps)
    diagnostics = []
    for frame in range(args.frames):
        if frame:
            dns.step(args.save_stride)
        snapshot = dns.snapshot()
        path = trajectory_dir / f'frame_{frame:03d}.npz'
        np.savez(path, **snapshot)
        stats = dns.diagnostics()
        stats['frame'] = frame
        diagnostics.append(stats)
    energy = np.asarray([item['kinetic_energy'] for item in diagnostics])
    max_divergence = max(item['divergence_rms'] for item in diagnostics)
    max_cfl = max(item['cfl'] for item in diagnostics)
    metadata = {
        'schema_version': 2,
        'complete': True,
        'trajectory_id': index,
        'split': split_name(index, args.trajectories),
        'equation': 'unforced incompressible Navier-Stokes',
        'config': config.to_dict(),
        'sampling': {
            'warmup_steps': args.warmup_steps,
            'frames': args.frames,
            'save_stride': args.save_stride,
            'snapshot_dt': args.time_step * args.save_stride,
        },
        'quality': {
            'max_divergence_rms': max_divergence,
            'max_cfl': max_cfl,
            'energy_monotone': bool(np.all(np.diff(energy) <= 1e-10)),
            'energy_drop_fraction': float((energy[0] - energy[-1]) / energy[0]),
        },
        'diagnostics': diagnostics,
        'elapsed_seconds': time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[{index + 1}/{args.trajectories}] {trajectory_dir.name} '
          f'{metadata["split"]}: {metadata["elapsed_seconds"]:.1f}s, '
          f'div={max_divergence:.2e}, CFL={max_cfl:.3f}, '
          f'dE={metadata["quality"]["energy_drop_fraction"]:.2%}')
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/generated_hit_ns_v2')
    parser.add_argument('--grid-size', type=int, default=64)
    parser.add_argument('--box-length', type=float, default=2*np.pi)
    parser.add_argument('--viscosity', type=float, default=5e-3)
    parser.add_argument('--time-step', type=float, default=2e-3)
    parser.add_argument('--initial-energy', type=float, default=0.5)
    parser.add_argument('--spectrum-peak', type=float, default=4.0)
    parser.add_argument('--trajectories', type=int, default=15)
    parser.add_argument('--warmup-steps', type=int, default=32)
    parser.add_argument('--frames', type=int, default=17)
    parser.add_argument('--save-stride', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260717)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.trajectories < 5:
        raise ValueError('At least five trajectories are required for trajectory-disjoint splits')
    if args.frames < 7:
        raise ValueError('At least seven dense frames are required for temporal weak forms')

    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    records = [generate_trajectory(root, index, args)
               for index in range(args.trajectories)]
    failed = [record['trajectory_id'] for record in records
              if (not record['quality']['energy_monotone']
                  or record['quality']['max_divergence_rms'] > 1e-8
                  or record['quality']['max_cfl'] > 0.5)]
    manifest = {
        'schema_version': 2,
        'dataset': 'trajectory-disjoint decaying HIT for full-NS discovery',
        'generator': 'scripts/generate_hit_ns_v2.py',
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'arguments': vars(args),
        'splits': {
            name: [record['trajectory_id'] for record in records
                   if record['split'] == name]
            for name in ('discovery', 'validation', 'test')
        },
        'quality_passed': not failed,
        'failed_trajectories': failed,
        'trajectories': records,
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2),
                                        encoding='utf-8')
    if failed:
        raise RuntimeError(f'DNS quality checks failed for trajectories {failed}')
    print(f'Dataset complete: {root}')
    print('Splits:', manifest['splits'])


if __name__ == '__main__':
    main()
