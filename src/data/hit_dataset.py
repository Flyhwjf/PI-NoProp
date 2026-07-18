"""Trajectory-disjoint learning cache for the decaying-HIT dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SPLIT_CODE = {'discovery': 0, 'validation': 1, 'test': 2}
REGION_NAMES = ('low_enstrophy', 'high_enstrophy')


def _ns_energy_terms(field: np.ndarray, dx: float) -> np.ndarray:
    """Return first-frame energy-rate terms without using future labels.

    For a relation ``u_t + c_a (u.grad)u + c_p grad(p) + c_v lap(u)=0``,
    ``[c_a,c_p,c_v] @ terms`` is the instantaneous relative kinetic-energy
    decay rate.  Keeping the three unweighted terms lets analytic and
    discovered equations use the same target-free cache.
    """
    velocity = field[:3].astype(np.float64, copy=False)
    pressure = field[3].astype(np.float64, copy=False)
    gradients = np.stack([
        np.gradient(velocity, dx, axis=axis+1, edge_order=2)
        for axis in range(3)], axis=1)
    convection = np.einsum('jxyz,ijxyz->ixyz', velocity, gradients)
    pressure_gradient = np.stack([
        np.gradient(pressure, dx, axis=axis, edge_order=2)
        for axis in range(3)])
    laplacian = sum(
        np.gradient(np.gradient(velocity, dx, axis=axis+1, edge_order=2),
                    dx, axis=axis+1, edge_order=2)
        for axis in range(3))
    energy = 0.5*np.mean(np.sum(velocity**2, axis=0))
    denominator = max(float(energy), 1e-12)
    return np.asarray([
        np.mean(np.sum(velocity*term, axis=0))/denominator
        for term in (convection, pressure_gradient, laplacian)
    ], dtype=np.float32)


def _local_enstrophy(velocity: np.ndarray, dx: float) -> float:
    gradients = np.stack([
        np.gradient(velocity, dx, axis=axis+1, edge_order=2)
        for axis in range(3)], axis=1)  # component, derivative, x, y, z
    omega_x = gradients[2, 1] - gradients[1, 2]
    omega_y = gradients[0, 2] - gradients[2, 0]
    omega_z = gradients[1, 0] - gradients[0, 1]
    return float(np.mean(omega_x**2 + omega_y**2 + omega_z**2))


def _quantile_edges(values: np.ndarray, n_classes: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0, 1, n_classes+1))
    # Degenerate adjacent quantiles are expanded by machine-scale increments.
    for index in range(1, len(edges)):
        if edges[index] <= edges[index-1]:
            edges[index] = np.nextafter(edges[index-1], np.inf)
    margin = max(float(np.ptp(values)), abs(float(values.mean())), 1.0) * 1e-9
    edges[0], edges[-1] = float(values.min())-margin, float(values.max())+margin
    return edges


def build_learning_cache(dataset_dir='data/generated_hit_ns',
                         cache_dir='data/cache_hit_ns',
                         samples_per_trajectory=64, spatial_size=16,
                         time_window=9, n_classes=5, seed=31415,
                         overwrite=False):
    source = Path(dataset_dir)
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / 'metadata.json'
    required = [destination/name for name in (
        'fields.npy', 'sequences.npy', 'labels.npy', 'splits.npy',
        'regions.npy', 'trajectory_ids.npy', 'ns_terms.npy', 'stats.npz')]
    if not overwrite and metadata_path.exists() and all(path.exists() for path in required):
        return json.loads(metadata_path.read_text(encoding='utf-8'))

    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(seed)
    samples = []
    for record in manifest['trajectories']:
        trajectory_id = int(record['trajectory_id'])
        trajectory_dir = source/f'trajectory_{trajectory_id:03d}'
        files = sorted(trajectory_dir.glob('frame_*.npz'))
        if len(files) < time_window:
            raise RuntimeError(f'{trajectory_dir} has fewer than {time_window} frames')
        velocity, pressure = [], []
        for file in files:
            with np.load(file) as data:
                velocity.append(data['velocity'])
                pressure.append(data['pressure'])
        velocity, pressure = np.stack(velocity), np.stack(pressure)
        n = velocity.shape[-1]
        dx = float(record['config']['box_length'])/n
        for _ in range(samples_per_trajectory):
            t0 = int(rng.integers(0, len(files)-time_window+1))
            starts = rng.integers(0, n-spatial_size+1, size=3)
            xyz = tuple(slice(int(start), int(start)+spatial_size) for start in starts)
            u = velocity[(slice(t0, t0+time_window), slice(None))+xyz]
            p = pressure[(slice(t0, t0+time_window),)+xyz]
            sequence = np.concatenate([u, p[:, None]], axis=1).astype(np.float32)
            initial_energy = 0.5*np.mean(np.sum(u[0]**2, axis=0))
            final_energy = 0.5*np.mean(np.sum(u[-1]**2, axis=0))
            relative_decay = float((initial_energy-final_energy)
                                   / max(initial_energy, 1e-12))
            samples.append({
                'sequence': sequence,
                'enstrophy': _local_enstrophy(u[0].astype(np.float64), dx),
                'ns_terms': _ns_energy_terms(sequence[0], dx),
                'target': relative_decay,
                'split': SPLIT_CODE[record['split']],
                'trajectory_id': trajectory_id,
            })

    discovery = np.asarray([sample['split'] == SPLIT_CODE['discovery']
                            for sample in samples])
    enstrophy_values = np.asarray([sample['enstrophy'] for sample in samples])
    threshold = float(np.median(enstrophy_values[discovery]))
    regions = (enstrophy_values > threshold).astype(np.int8)
    targets = np.asarray([sample['target'] for sample in samples])
    labels = np.empty(len(samples), dtype=np.int64)
    class_edges = {}
    for region, name in enumerate(REGION_NAMES):
        train_mask = discovery & (regions == region)
        edges = _quantile_edges(targets[train_mask], n_classes)
        labels[regions == region] = np.digitize(
            targets[regions == region], edges[1:-1], right=False)
        class_edges[name] = edges.tolist()

    sequences = np.stack([sample['sequence'] for sample in samples])
    ns_terms = np.stack([sample['ns_terms'] for sample in samples])
    fields = sequences[:, 0]
    train_sequences = sequences[discovery].astype(np.float64)
    means = train_sequences.mean(axis=(0, 1, 3, 4, 5)).astype(np.float32)
    stds = train_sequences.std(axis=(0, 1, 3, 4, 5)).astype(np.float32)
    stds = np.maximum(stds, 1e-8)
    np.save(destination/'fields.npy', fields)
    np.save(destination/'sequences.npy', sequences)
    np.save(destination/'labels.npy', labels)
    np.save(destination/'splits.npy', np.asarray([s['split'] for s in samples], dtype=np.int8))
    np.save(destination/'regions.npy', regions)
    np.save(destination/'trajectory_ids.npy',
            np.asarray([s['trajectory_id'] for s in samples], dtype=np.int16))
    np.save(destination/'ns_terms.npy', ns_terms)
    term_means = ns_terms[discovery].mean(axis=0, dtype=np.float64)
    term_covariance = np.cov(ns_terms[discovery].astype(np.float64), rowvar=False)
    np.savez(destination/'stats.npz', means=means, stds=stds,
             enstrophy_threshold=threshold,
             target_values=targets.astype(np.float32),
             ns_term_means=term_means.astype(np.float32),
             ns_term_covariance=term_covariance.astype(np.float32))

    metadata = {
        'schema_version': 2,
        'source_manifest': str((source/'manifest.json').resolve()),
        'n_samples': len(samples),
        'samples_per_trajectory': samples_per_trajectory,
        'spatial_size': spatial_size,
        'time_window': time_window,
        'n_classes': n_classes,
        'input': 'first velocity-pressure frame',
        'target': 'future local relative kinetic-energy decay quantile',
        'region_definition': 'discovery-set median initial local enstrophy',
        'enstrophy_threshold': threshold,
        'class_edges': class_edges,
        'split_counts': {
            name: int(np.sum(np.asarray([s['split'] for s in samples]) == code))
            for name, code in SPLIT_CODE.items()},
        'region_split_class_counts': {
            f'{REGION_NAMES[region]}_{split}': [
                int(np.sum((regions == region)
                           & (np.asarray([s['split'] for s in samples]) == code)
                           & (labels == label)))
                for label in range(n_classes)]
            for region in range(2) for split, code in SPLIT_CODE.items()},
        'seed': seed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


class HITNSSequenceDataset(Dataset):
    def __init__(self, cache_dir, region, split, in_memory=True):
        root = Path(cache_dir)
        region_code = REGION_NAMES.index(region)
        split_code = SPLIT_CODE[split]
        regions = np.load(root/'regions.npy', mmap_mode='r')
        splits = np.load(root/'splits.npy', mmap_mode='r')
        self.indices = np.flatnonzero((regions == region_code) & (splits == split_code))
        stats = np.load(root/'stats.npz')
        self.means = np.asarray(stats['means'], dtype=np.float32)
        self.stds = np.asarray(stats['stds'], dtype=np.float32)
        fields = np.load(root/'fields.npy', mmap_mode='r')
        sequences = np.load(root/'sequences.npy', mmap_mode='r')
        labels = np.load(root/'labels.npy', mmap_mode='r')
        trajectories = np.load(root/'trajectory_ids.npy', mmap_mode='r')
        ns_terms = np.load(root/'ns_terms.npy', mmap_mode='r')
        if in_memory:
            field_values = np.asarray(fields[self.indices], dtype=np.float32)
            sequence_values = np.asarray(sequences[self.indices], dtype=np.float32)
            self.fields = torch.from_numpy(np.ascontiguousarray(
                (field_values-self.means[None, :, None, None, None])
                / self.stds[None, :, None, None, None]))
            self.sequences = torch.from_numpy(np.ascontiguousarray(
                (sequence_values-self.means[None, None, :, None, None, None])
                / self.stds[None, None, :, None, None, None]))
            self.labels = torch.from_numpy(np.asarray(labels[self.indices]).copy())
            self.trajectories = torch.from_numpy(
                np.asarray(trajectories[self.indices], dtype=np.int64).copy())
            self.ns_terms = torch.from_numpy(
                np.asarray(ns_terms[self.indices], dtype=np.float32).copy())
        else:
            raise NotImplementedError('the current pipeline uses a bounded in-memory cache')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        field = self.fields[index]
        return {
            'field': field,
            'velocity': field[:3],
            'pressure': field[3],
            'sequence': self.sequences[index],
            'label': self.labels[index],
            'idx': int(self.indices[index]),
            'trajectory_id': self.trajectories[index],
            'ns_terms': self.ns_terms[index],
        }


def create_hit_dataloaders(config):
    build_learning_cache(
        config.data.data_dir, config.data.cache_dir,
        samples_per_trajectory=config.data.n_subdomains,
        spatial_size=config.data.subdomain_size,
        time_window=config.physics.n_time,
        n_classes=config.data.n_classes,
    )
    stats = np.load(Path(config.data.cache_dir)/'stats.npz')
    config.data.ns_term_means = np.asarray(
        stats['ns_term_means'], dtype=np.float64).tolist()
    config.data.ns_term_covariance = np.asarray(
        stats['ns_term_covariance'], dtype=np.float64).tolist()
    result = {}
    for region in config.data.regions:
        result[region] = {}
        for source_split, loader_split in (
                ('discovery', 'train'), ('validation', 'val'), ('test', 'test')):
            dataset = HITNSSequenceDataset(config.data.cache_dir, region, source_split)
            result[region][loader_split] = DataLoader(
                dataset, batch_size=config.data.batch_size,
                shuffle=(source_split == 'discovery'), num_workers=0,
                pin_memory=torch.cuda.is_available())
    return result
