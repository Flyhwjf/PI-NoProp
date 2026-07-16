"""Weak-form spatial SPIDER discovery for the archived forced-HIT run.

The classification model consumes one snapshot.  This module therefore
discovers a spatial pressure relation rather than manufacturing a time axis.
Discovery and validation use disjoint DNS snapshots.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TERM_NAMES = (
    'pressure_laplacian',
    'convection_divergence',
    'kinetic_energy_laplacian',
)


@dataclass
class HITSPIDERConfig:
    snapshot_dir: str = 'data/generated/snapshots'
    grid_size: int = 64
    box_length: float = 2.0 * math.pi
    snapshot_dt: float = 0.05
    domain_size: int = 16
    n_test_functions: int = 8
    domains_per_snapshot: int = 4
    beta: float = 6.0
    train_snapshots: int = 60
    validation_snapshots: int = 20
    max_terms: int = 4
    bootstrap_repeats: int = 100
    seed: int = 42
    # The archived generator's pressure projection and unscaled 3/2 padding
    # leave a measurable discretisation residual.  A relation must also beat
    # the next-best same-dimension support by a wide margin (checked below).
    max_validation_eta: float = 0.35
    min_bootstrap_support: float = 0.8
    max_coefficient_cv: float = 0.25


def _test_functions(config):
    size = config.domain_size
    dx = config.box_length / config.grid_size
    coords = (np.arange(size) - (size - 1) / 2) * dx
    half_width = max(abs(coords[0]), dx)
    normalized = coords / half_width
    x, y, z = np.meshgrid(normalized, normalized, normalized, indexing='ij')
    envelope = np.clip((1 - x*x) * (1 - y*y) * (1 - z*z), 0, None) ** config.beta
    functions = []
    for index in range(config.n_test_functions):
        frequency = index // 3 + 1
        axis = index % 3
        coord = (x, y, z)[axis]
        value = envelope * (1 + 0.35 * np.cos(frequency * np.pi * coord))
        functions.append(value / max(np.max(np.abs(value)), 1e-12))
    w = np.stack(functions)
    grad = np.stack(np.gradient(w, dx, dx, dx, axis=(1, 2, 3), edge_order=2), axis=1)
    hessian = np.empty((len(w), 3, 3, size, size, size), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            hessian[:, i, j] = np.gradient(grad[:, i], dx, axis=j + 1, edge_order=2)
    lap = hessian[:, 0, 0] + hessian[:, 1, 1] + hessian[:, 2, 2]
    return w.astype(np.float64), grad, hessian, lap


def _rows_from_snapshot(velocity, pressure, config, rng, tests):
    w, grad_w, hessian_w, lap_w = tests
    n = velocity.shape[-1]
    size = config.domain_size
    rows = []
    for _ in range(config.domains_per_snapshot):
        starts = [int(rng.integers(0, n - size + 1)) for _ in range(3)]
        sl = tuple(slice(start, start + size) for start in starts)
        u = velocity[(slice(None),) + sl].astype(np.float64, copy=False)
        p = pressure[sl].astype(np.float64, copy=False)
        u_sq = np.einsum('ixyz,ixyz->xyz', u, u)
        # Pure weak form: for incompressible flow,
        # div((u.grad)u) = d_i d_j (u_i u_j).
        pressure_lap = np.einsum('xyz,kxyz->k', p, lap_w)
        convection_div = np.einsum('ixyz,jxyz,kijxyz->k', u, u, hessian_w)
        values = np.stack([
            pressure_lap,
            convection_div,
            np.einsum('xyz,kxyz->k', u_sq, lap_w),
        ], axis=1)
        volume = max(np.sum(np.abs(w[0])), 1e-12)
        rows.append(values / volume)
    return np.concatenate(rows, axis=0)


def build_feature_matrices(config):
    files = sorted(Path(config.snapshot_dir).glob('snap_*.npz'))
    required = config.train_snapshots + config.validation_snapshots
    if len(files) < required:
        raise RuntimeError(f'Need {required} snapshots, found {len(files)}')
    rng = np.random.default_rng(config.seed)
    tests = _test_functions(config)
    train_rows, validation_rows = [], []
    for index, path in enumerate(files[:required]):
        with np.load(path) as data:
            rows = _rows_from_snapshot(data['velocity'], data['pressure'],
                                       config, rng, tests)
        (train_rows if index < config.train_snapshots else validation_rows).append(rows)
    return np.concatenate(train_rows), np.concatenate(validation_rows), files[:required]


def _fit_null_vector(matrix, subset, scales=None):
    active = matrix[:, subset]
    if scales is None:
        scales = np.sqrt(np.mean(active * active, axis=0)).clip(1e-14)
    normalized = active / scales
    _, _, vh = np.linalg.svd(normalized, full_matrices=False)
    coefficients = vh[-1] / scales
    coefficients /= np.linalg.norm(coefficients)
    return coefficients, scales


def _eta(matrix, subset, coefficients):
    contributions = matrix[:, subset] * coefficients[None, :]
    residual = contributions.sum(axis=1)
    denominator = np.abs(contributions).sum(axis=1).clip(1e-14)
    return float(np.sqrt(np.mean((residual / denominator) ** 2)))


def discover(config=None):
    config = config or HITSPIDERConfig()
    train, validation, files = build_feature_matrices(config)
    def select(matrix, score_matrix):
        candidates = []
        for size in range(2, min(config.max_terms, len(TERM_NAMES)) + 1):
            for subset in itertools.combinations(range(len(TERM_NAMES)), size):
                coefficients, scales = _fit_null_vector(matrix, subset)
                candidates.append({
                    'subset': subset,
                    'coefficients': coefficients,
                    'scales': scales,
                    'train_eta': _eta(matrix, subset, coefficients),
                    'validation_eta': _eta(score_matrix, subset, coefficients),
                })
        passing = [item for item in candidates
                   if item['validation_eta'] <= config.max_validation_eta]
        pool = passing or candidates
        min_size = min(len(item['subset']) for item in pool)
        best = min((item for item in pool if len(item['subset']) == min_size),
                   key=lambda item: item['validation_eta'])
        return best, candidates

    # Select support without consulting the held-out snapshots.  Their only
    # role is the final acceptance test below.
    best, candidates = select(train, train)
    best['validation_eta'] = _eta(validation, best['subset'], best['coefficients'])
    for candidate in candidates:
        candidate['validation_eta'] = _eta(
            validation, candidate['subset'], candidate['coefficients'])

    rng = np.random.default_rng(config.seed + 1)
    boot_coefficients = []
    matching_support = 0
    for _ in range(config.bootstrap_repeats):
        indices = rng.integers(0, len(train), len(train))
        boot_best, _ = select(train[indices], train)
        if boot_best['subset'] == best['subset']:
            matching_support += 1
            coefficients = boot_best['coefficients']
            if np.dot(coefficients, best['coefficients']) < 0:
                coefficients = -coefficients
            boot_coefficients.append(coefficients)
    boot = np.stack(boot_coefficients)
    mean = boot.mean(axis=0)
    std = boot.std(axis=0)
    coefficient_cv = float(np.max(std / np.abs(mean).clip(1e-12)))

    selected_terms = [TERM_NAMES[index] for index in best['subset']]
    required_terms = {'pressure_laplacian', 'convection_divergence'}
    support_fraction = matching_support / config.bootstrap_repeats
    competing = sorted(item['validation_eta'] for item in candidates
                       if item['subset'] != best['subset']
                       and len(item['subset']) == len(best['subset']))
    next_best_eta = competing[0] if competing else float('inf')
    separation_ratio = next_best_eta / max(best['validation_eta'], 1e-12)
    failure_reasons = []
    if set(selected_terms) != required_terms:
        failure_reasons.append(f'selected support is {selected_terms}, expected the two PP terms')
    if best['validation_eta'] > config.max_validation_eta:
        failure_reasons.append(
            f'validation eta {best["validation_eta"]:.4g} exceeds {config.max_validation_eta}')
    if support_fraction < config.min_bootstrap_support:
        failure_reasons.append('bootstrap support is unstable')
    if coefficient_cv > config.max_coefficient_cv:
        failure_reasons.append(
            f'coefficient CV {coefficient_cv:.4g} exceeds {config.max_coefficient_cv}')
    if separation_ratio < 1.5:
        failure_reasons.append(
            f'next-best support separation {separation_ratio:.3g} is below 1.5')

    coefficients = best['coefficients'].copy()
    if 'pressure_laplacian' in selected_terms:
        anchor = coefficients[selected_terms.index('pressure_laplacian')]
        coefficients /= anchor
        boot /= anchor
    artifact = {
        'schema_version': 1,
        'method': 'SPIDER weak-form sparse nullspace discovery',
        'source': 'archived forced-HIT DNS snapshots',
        'config': asdict(config),
        'data': {
            'train_files': [path.name for path in files[:config.train_snapshots]],
            'validation_files': [path.name for path in files[config.train_snapshots:]],
            'train_rows': len(train),
            'validation_rows': len(validation),
            'dx': config.box_length / config.grid_size,
            'snapshot_dt': config.snapshot_dt,
        },
        'candidate_terms': list(TERM_NAMES),
        'equation': {
            'terms': selected_terms,
            'coefficients': coefficients.tolist(),
            'normalization': 'pressure_laplacian coefficient fixed to one',
        },
        'metrics': {
            'train_eta': best['train_eta'],
            'validation_eta': best['validation_eta'],
            'bootstrap_coefficient_mean': mean.tolist(),
            'bootstrap_coefficient_std': std.tolist(),
            'bootstrap_coefficient_cv_max': coefficient_cv,
            'bootstrap_support_fraction': support_fraction,
            'next_best_validation_eta': next_best_eta,
            'support_separation_ratio': separation_ratio,
        },
        'validation': {
            'passed': not failure_reasons,
            'failure_reasons': failure_reasons,
        },
        'provenance': {
            'discovered_not_prescribed': True,
            'validation_split_disjoint': True,
            'silent_analytic_fallback': False,
        },
    }
    return artifact


def save_artifact(artifact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding='utf-8')
