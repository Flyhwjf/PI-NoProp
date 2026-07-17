"""Trajectory-disjoint 4-D weak-form SPIDER for the full momentum equation."""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TERM_NAMES = (
    'time_derivative',
    'convection',
    'pressure_gradient',
    'velocity_laplacian',
    'kinetic_energy_gradient',
)
EXPECTED_SUPPORT = TERM_NAMES[:4]


@dataclass
class NSSPIDERConfig:
    dataset_dir: str = 'data/generated_hit_ns_v2'
    domain_size: int = 16
    time_window: int = 9
    windows_per_trajectory: int = 2
    domains_per_window: int = 4
    n_test_functions: int = 8
    beta: float = 4.0
    max_terms: int = 5
    selection_eta: float = 0.01
    max_validation_eta: float = 0.02
    max_test_eta: float = 0.02
    min_support_separation: float = 1.25
    min_bootstrap_support: float = 0.90
    max_coefficient_relative_error: float = 0.10
    bootstrap_repeats: int = 100
    noise_level: float = 0.0
    noise_seed: int = 2718
    seed: int = 42


def _profiles(config: NSSPIDERConfig, dx: float, dt: float):
    s, nt = config.domain_size, config.time_window
    rs = np.linspace(-1.0, 1.0, s)
    rt = np.linspace(-1.0, 1.0, nt)

    def compact(coordinate):
        base = np.clip(1-coordinate*coordinate, 0, None)
        value = base**config.beta
        first = -2*config.beta*coordinate*base**(config.beta-1)
        second = (-2*config.beta*base**(config.beta-1)
                  + 4*config.beta*(config.beta-1)*coordinate**2
                  * base**(config.beta-2))
        return value, first, second

    q, dq, d2q = compact(rs)
    qt, dqt, _ = compact(rt)
    spatial_scale = 2.0 / ((s-1)*dx)
    temporal_scale = 2.0 / ((nt-1)*dt)
    spatial, temporal = [], []
    for index in range(config.n_test_functions):
        frequency = index // 3 + 1
        axis = index % 3
        factors, firsts, seconds = [], [], []
        for dimension in range(3):
            if dimension == axis:
                phase = frequency*np.pi*rs
                modulation = 1 + 0.25*np.cos(phase)
                dmod = -0.25*frequency*np.pi*np.sin(phase)
                d2mod = -0.25*(frequency*np.pi)**2*np.cos(phase)
                factors.append(q*modulation)
                firsts.append((dq*modulation + q*dmod)*spatial_scale)
                seconds.append((d2q*modulation + 2*dq*dmod + q*d2mod)
                               * spatial_scale**2)
            else:
                factors.append(q)
                firsts.append(dq*spatial_scale)
                seconds.append(d2q*spatial_scale**2)
        ws = np.einsum('x,y,z->xyz', *factors)
        grad = np.stack([
            np.einsum('x,y,z->xyz',
                      *(firsts[d] if j == d else factors[j] for j in range(3)))
            for d in range(3)])
        lap = sum(
            np.einsum('x,y,z->xyz',
                      *(seconds[d] if j == d else factors[j] for j in range(3)))
            for d in range(3))
        spatial_norm = max(np.max(np.abs(ws)), 1e-15)
        ws, grad, lap = ws/spatial_norm, grad/spatial_norm, lap/spatial_norm

        time_frequency = index % 3 + 1
        phase_t = time_frequency*np.pi*rt
        modulation_t = 1 + 0.20*np.cos(phase_t)
        dmod_t = -0.20*time_frequency*np.pi*np.sin(phase_t)
        wt = qt*modulation_t
        dwt = (dqt*modulation_t + qt*dmod_t)*temporal_scale
        temporal_norm = max(np.max(np.abs(wt)), 1e-15)
        wt, dwt = wt/temporal_norm, dwt/temporal_norm
        spatial.append((ws, grad, lap))
        temporal.append((wt, dwt))
    return spatial, temporal


def _weak_rows(velocity: np.ndarray, pressure: np.ndarray,
               spatial, temporal) -> np.ndarray:
    """Return K*3 rows for one compact time-space domain.

    Integration by parts gives, componentwise,
    ``[-u w_t, -u_i u_j w_j, -p w_i, u_i lap(w), -|u|^2 w_i]``.
    The first four columns therefore satisfy coefficients ``(1,1,1,-nu)``.
    """
    rows = []
    points = np.prod(velocity.shape[2:])
    for (ws, grad, lap), (wt, dwt) in zip(spatial, temporal):
        kinetic = np.sum(velocity * velocity, axis=1)
        for component in range(3):
            time_term = -np.sum(
                velocity[:, component] * dwt[:, None, None, None] * ws[None])
            convection = -sum(np.sum(
                velocity[:, component] * velocity[:, axis]
                * wt[:, None, None, None] * grad[axis][None])
                for axis in range(3))
            pressure_gradient = -np.sum(
                pressure * wt[:, None, None, None] * grad[component][None])
            velocity_laplacian = np.sum(
                velocity[:, component] * wt[:, None, None, None] * lap[None])
            kinetic_gradient = -np.sum(
                kinetic * wt[:, None, None, None] * grad[component][None])
            rows.append(np.asarray([
                time_term, convection, pressure_gradient,
                velocity_laplacian, kinetic_gradient,
            ], dtype=np.float64) / points)
    return np.stack(rows)


def _load_trajectory(path: Path, noise_level=0.0, noise_seed=0):
    files = sorted(path.glob('frame_*.npz'))
    velocity, pressure, times = [], [], []
    for file in files:
        with np.load(file) as data:
            velocity.append(data['velocity'].astype(np.float64))
            pressure.append(data['pressure'].astype(np.float64))
            times.append(float(data['time']))
    velocity, pressure = np.stack(velocity), np.stack(pressure)
    if noise_level > 0:
        rng = np.random.default_rng(noise_seed)
        velocity_scale = velocity.std(axis=(0, 2, 3, 4), keepdims=True)
        pressure_scale = pressure.std()
        velocity = velocity+noise_level*velocity_scale*rng.standard_normal(
            velocity.shape)
        pressure = pressure+noise_level*pressure_scale*rng.standard_normal(
            pressure.shape)
    return velocity, pressure, np.asarray(times), files


def build_feature_matrices(config: NSSPIDERConfig):
    root = Path(config.dataset_dir)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(config.seed)
    matrices, row_trajectories, provenance = {}, {}, {}
    for split, trajectory_ids in manifest['splits'].items():
        split_rows, split_ids, split_files = [], [], []
        for trajectory_id in trajectory_ids:
            path = root / f'trajectory_{trajectory_id:03d}'
            velocity, pressure, times, files = _load_trajectory(
                path, config.noise_level,
                config.noise_seed+int(trajectory_id))
            if len(files) < config.time_window:
                raise RuntimeError(f'{path} has too few frames')
            dt_values = np.diff(times)
            if not np.allclose(dt_values, dt_values[0], rtol=1e-8, atol=1e-12):
                raise RuntimeError(f'{path} does not have uniform dense sampling')
            n = velocity.shape[-1]
            if config.domain_size > n:
                raise ValueError('domain_size exceeds the DNS grid')
            dx = float(manifest['trajectories'][trajectory_id]['config']['box_length']) / n
            profiles = _profiles(config, dx, float(dt_values[0]))
            max_t_start = len(files) - config.time_window
            t_starts = rng.integers(0, max_t_start + 1,
                                    size=config.windows_per_trajectory)
            trajectory_rows = []
            for t_start in t_starts:
                t_slice = slice(t_start, t_start + config.time_window)
                for _ in range(config.domains_per_window):
                    starts = rng.integers(0, n-config.domain_size+1, size=3)
                    xyz = tuple(slice(int(start), int(start)+config.domain_size)
                                for start in starts)
                    u = velocity[(t_slice, slice(None)) + xyz]
                    p = pressure[(t_slice,) + xyz]
                    trajectory_rows.append(_weak_rows(u, p, *profiles))
            trajectory_rows = np.concatenate(trajectory_rows)
            split_rows.append(trajectory_rows)
            split_ids.extend([trajectory_id] * len(trajectory_rows))
            split_files.extend(str(file.relative_to(root)) for file in files)
        matrices[split] = np.concatenate(split_rows)
        row_trajectories[split] = np.asarray(split_ids)
        provenance[split] = split_files
    viscosity = float(manifest['trajectories'][0]['config']['viscosity'])
    return matrices, row_trajectories, provenance, manifest, viscosity


def _fit(matrix: np.ndarray, subset: tuple[int, ...]) -> np.ndarray:
    if 0 not in subset:
        raise ValueError('time derivative must anchor the candidate equation')
    others = [index for index in subset if index != 0]
    scales = np.sqrt(np.mean(matrix[:, others]**2, axis=0)).clip(1e-14)
    design = matrix[:, others] / scales
    normalized, *_ = np.linalg.lstsq(design, -matrix[:, 0], rcond=1e-12)
    coefficients = np.zeros(len(subset), dtype=np.float64)
    coefficients[subset.index(0)] = 1.0
    for local, scale, value in zip(others, scales, normalized):
        coefficients[subset.index(local)] = value / scale
    return coefficients


def _eta(matrix: np.ndarray, subset: tuple[int, ...], coefficients: np.ndarray) -> float:
    contributions = matrix[:, subset] * coefficients[None]
    numerator = np.sqrt(np.mean(np.sum(contributions, axis=1)**2))
    denominator = np.sqrt(np.mean(np.sum(contributions**2, axis=1))).clip(1e-15)
    return float(numerator / denominator)


def _select(matrix: np.ndarray, config: NSSPIDERConfig):
    candidates = []
    for size in range(2, min(config.max_terms, len(TERM_NAMES)) + 1):
        for tail in itertools.combinations(range(1, len(TERM_NAMES)), size-1):
            subset = (0,) + tail
            coefficients = _fit(matrix, subset)
            candidates.append({
                'subset': subset,
                'coefficients': coefficients,
                'train_eta': _eta(matrix, subset, coefficients),
            })
    passing = [candidate for candidate in candidates
               if candidate['train_eta'] <= config.selection_eta]
    if passing:
        smallest = min(len(candidate['subset']) for candidate in passing)
        pool = [candidate for candidate in passing
                if len(candidate['subset']) == smallest]
    else:
        pool = candidates
    return min(pool, key=lambda candidate: candidate['train_eta']), candidates


def discover(config: NSSPIDERConfig | None = None) -> dict:
    config = config or NSSPIDERConfig()
    matrices, row_ids, provenance, manifest, viscosity = build_feature_matrices(config)
    best, candidates = _select(matrices['discovery'], config)
    subset = best['subset']
    coefficients = best['coefficients']
    for candidate in candidates:
        candidate['validation_eta'] = _eta(
            matrices['validation'], candidate['subset'], candidate['coefficients'])
    validation_eta = _eta(matrices['validation'], subset, coefficients)
    test_eta = _eta(matrices['test'], subset, coefficients)

    competitors = sorted(candidate['validation_eta'] for candidate in candidates
                         if candidate['subset'] != subset
                         and len(candidate['subset']) == len(subset))
    next_best = competitors[0] if competitors else float('inf')
    separation = next_best / max(validation_eta, 1e-15)

    rng = np.random.default_rng(config.seed + 1)
    trajectory_ids = np.unique(row_ids['discovery'])
    matching, boot_coefficients = 0, []
    for _ in range(config.bootstrap_repeats):
        sampled = rng.choice(trajectory_ids, size=len(trajectory_ids), replace=True)
        indices = np.concatenate([
            np.flatnonzero(row_ids['discovery'] == trajectory_id)
            for trajectory_id in sampled])
        boot_best, _ = _select(matrices['discovery'][indices], config)
        if boot_best['subset'] == subset:
            matching += 1
            boot_coefficients.append(boot_best['coefficients'])
    support_fraction = matching / config.bootstrap_repeats
    boot = np.stack(boot_coefficients) if boot_coefficients else np.empty((0, len(subset)))

    selected_terms = [TERM_NAMES[index] for index in subset]
    expected_by_name = {
        'time_derivative': 1.0,
        'convection': 1.0,
        'pressure_gradient': 1.0,
        'velocity_laplacian': -viscosity,
    }
    relative_errors = {}
    for term, coefficient in zip(selected_terms, coefficients):
        if term in expected_by_name:
            expected = expected_by_name[term]
            relative_errors[term] = abs(coefficient-expected) / max(abs(expected), 1e-15)
    max_error = max(relative_errors.values(), default=float('inf'))
    failures = []
    if tuple(selected_terms) != EXPECTED_SUPPORT:
        failures.append(f'selected support {selected_terms} is not the full NS support')
    if validation_eta > config.max_validation_eta:
        failures.append(f'validation eta {validation_eta:.4g} exceeds threshold')
    if test_eta > config.max_test_eta:
        failures.append(f'test eta {test_eta:.4g} exceeds threshold')
    if separation < config.min_support_separation:
        failures.append(f'support separation {separation:.3g} is too small')
    if support_fraction < config.min_bootstrap_support:
        failures.append(f'bootstrap support {support_fraction:.1%} is unstable')
    if max_error > config.max_coefficient_relative_error:
        failures.append(f'maximum coefficient error {max_error:.1%} exceeds threshold')

    return {
        'schema_version': 2,
        'method': 'trajectory-disjoint 4-D weak-form SPIDER',
        'source': manifest['dataset'],
        'config': asdict(config),
        'data': {
            'dataset_manifest': str(Path(config.dataset_dir) / 'manifest.json'),
            'trajectory_splits': manifest['splits'],
            'rows': {key: len(value) for key, value in matrices.items()},
            'files': provenance,
            'viscosity_used_only_for_post_discovery_validation': viscosity,
        },
        'candidate_terms': list(TERM_NAMES),
        'equation': {
            'terms': selected_terms,
            'coefficients': coefficients.tolist(),
            'normalization': 'time-derivative coefficient fixed to one',
        },
        'expected_equation_for_post_discovery_audit': {
            'terms': list(EXPECTED_SUPPORT),
            'coefficients': [1.0, 1.0, 1.0, -viscosity],
        },
        'metrics': {
            'discovery_eta': best['train_eta'],
            'validation_eta': validation_eta,
            'test_eta': test_eta,
            'next_best_validation_eta': next_best,
            'support_separation_ratio': separation,
            'bootstrap_support_fraction': support_fraction,
            'bootstrap_coefficient_mean': boot.mean(axis=0).tolist() if len(boot) else [],
            'bootstrap_coefficient_std': boot.std(axis=0).tolist() if len(boot) else [],
            'coefficient_relative_errors': relative_errors,
            'max_coefficient_relative_error': max_error,
        },
        'validation': {'passed': not failures, 'failure_reasons': failures},
        'candidates': [{
            'terms': [TERM_NAMES[index] for index in candidate['subset']],
            'coefficients': candidate['coefficients'].tolist(),
            'discovery_eta': candidate['train_eta'],
            'validation_eta': candidate['validation_eta'],
        } for candidate in candidates],
    }
