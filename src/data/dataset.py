"""Turbulence datasets and fast first-frame cache for PI-NoProp.

The original NPZ files contain 32 time frames plus coordinate arrays.  The
classification task only consumes frame zero, so repeatedly opening the NPZ
files wastes most of the input bandwidth.  This module keeps the legacy NPZ
dataset and adds a contiguous NPY cache used by the optimized training path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import AddNoise, Normalize, ToTensor


class TurbulenceDataset(Dataset):
    """Legacy NPZ loader retained for result reproduction."""

    def __init__(self, data_dir, region='centre', n_classes=10,
                 n_subdomains=256, transform=None):
        self.data_dir = Path(data_dir) / region
        self.n_classes = n_classes
        self.transform = transform
        self.subdomain_files = sorted(self.data_dir.glob('sub_*.npz'))[:n_subdomains]
        assert self.subdomain_files, f'No NPZ files found in {self.data_dir}'
        self._load_or_generate_labels()

    def _load_or_generate_labels(self):
        try:
            with np.load(self.subdomain_files[0]) as data:
                has_label = 'label' in data
            if has_label:
                labels = []
                for path in self.subdomain_files:
                    with np.load(path) as data:
                        labels.append(int(data['label']))
                self.labels = np.asarray(labels, dtype=np.int64)
                return
        except (OSError, ValueError, KeyError):
            pass
        means = []
        for path in self.subdomain_files:
            with np.load(path) as data:
                velocity = data['velocity']
                means.append(float(velocity[0, ..., 0].mean() if velocity.ndim == 5
                                   else velocity[0].mean()))
        bins = np.percentile(means, np.linspace(0, 100, self.n_classes + 1))
        bins[-1] += 1e-6
        self.labels = np.clip(np.digitize(means, bins[:-1]) - 1,
                              0, self.n_classes - 1).astype(np.int64)

    def __len__(self):
        return len(self.subdomain_files)

    def __getitem__(self, idx):
        with np.load(self.subdomain_files[idx]) as data:
            velocity = data['velocity'].astype(np.float32)
            pressure = data['pressure'].astype(np.float32)
        if velocity.ndim == 5:
            velocity = velocity[..., 0]
        if pressure.ndim == 4:
            pressure = pressure[..., 0]
        sample = {
            'velocity': velocity,
            'pressure': pressure,
            'label': self.labels[idx],
            'idx': idx,
        }
        return self.transform(sample) if self.transform else sample


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def _split_indices(n_samples: int, n_train: int, n_val: int, n_test: int,
                   seed: int) -> Dict[str, np.ndarray]:
    requested = n_train + n_val + n_test
    if requested != n_samples:
        if n_val + n_test >= n_samples:
            raise ValueError('Validation and test sets leave no training samples')
        n_train = n_samples - n_val - n_test
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    return {
        'train': order[:n_train],
        'val': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:n_train + n_val + n_test],
    }


def build_first_frame_cache(data_dir='data/generated', cache_dir='data/cache',
                            regions: Iterable[str] = ('centre', 'edge'),
                            n_subdomains: int = 256, split_seed: int = 42,
                            n_train: int = 192, n_val: int = 32,
                            n_test: int = 32, overwrite: bool = False):
    """Extract frame zero into contiguous NPY files and persist fixed splits.

    The cache is deliberately raw.  Channel statistics are computed using the
    training split only, preventing validation/test leakage.
    """
    source_root = Path(data_dir)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for region in regions:
        fields_path = cache_root / f'{region}_fields.npy'
        labels_path = cache_root / f'{region}_labels.npy'
        split_path = cache_root / f'{region}_split.npz'
        stats_path = cache_root / f'{region}_stats.npz'
        if (not overwrite and fields_path.exists() and labels_path.exists()
                and split_path.exists() and stats_path.exists()):
            continue

        files = sorted((source_root / region).glob('sub_*.npz'))[:n_subdomains]
        if not files:
            raise FileNotFoundError(f'No NPZ files found for {region}')

        first = np.load(files[0])
        spatial_shape = first['velocity'].shape[1:4]
        first.close()
        fields = np.lib.format.open_memmap(
            fields_path, mode='w+', dtype=np.float32,
            shape=(len(files), 4, *spatial_shape),
        )
        labels = np.lib.format.open_memmap(
            labels_path, mode='w+', dtype=np.int64, shape=(len(files),),
        )
        for index, path in enumerate(files):
            with np.load(path) as data:
                velocity = data['velocity']
                pressure = data['pressure']
                fields[index, :3] = velocity[..., 0] if velocity.ndim == 5 else velocity
                fields[index, 3] = pressure[..., 0] if pressure.ndim == 4 else pressure
                labels[index] = int(data['label'])
        fields.flush()
        labels.flush()

        splits = _split_indices(len(files), n_train, n_val, n_test, split_seed)
        np.savez(split_path, **splits, seed=np.asarray(split_seed))

        train_fields = np.asarray(fields[splits['train']], dtype=np.float64)
        means = train_fields.mean(axis=(0, 2, 3, 4)).astype(np.float32)
        stds = train_fields.std(axis=(0, 2, 3, 4)).astype(np.float32)
        stds = np.maximum(stds, 1e-8)
        np.savez(stats_path, means=means, stds=stds)

        metadata = {
            'region': region,
            'n_samples': len(files),
            'shape': [len(files), 4, *spatial_shape],
            'source': str((source_root / region).resolve()),
            'split_seed': split_seed,
            'splits': {key: len(value) for key, value in splits.items()},
        }
        (cache_root / f'{region}_cache.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8')


class CachedTurbulenceDataset(Dataset):
    """Fast dataset backed by the contiguous first-frame cache."""

    def __init__(self, cache_dir, region, split, noise_level=0.0,
                 in_memory=True, noise_seed=42):
        root = Path(cache_dir)
        raw_fields = np.load(root / f'{region}_fields.npy', mmap_mode='r')
        raw_labels = np.load(root / f'{region}_labels.npy', mmap_mode='r')
        split_data = np.load(root / f'{region}_split.npz')
        self.indices = np.asarray(split_data[split], dtype=np.int64)
        stats = np.load(root / f'{region}_stats.npz')
        self.means = np.asarray(stats['means'], dtype=np.float32)[:, None, None, None]
        self.stds = np.asarray(stats['stds'], dtype=np.float32)[:, None, None, None]
        self.noise_level = float(noise_level)
        self.noise_seed = int(noise_seed)
        self.in_memory = bool(in_memory)
        if self.in_memory:
            # Normalize each split exactly once.  The previous implementation
            # repeated ~8 million floating-point operations for every sample
            # on every epoch, leaving the GPU idle between local updates.
            selected = np.asarray(raw_fields[self.indices], dtype=np.float32)
            selected = (selected - self.means[None]) / self.stds[None]
            self.fields = torch.from_numpy(np.ascontiguousarray(selected))
            self.labels = torch.from_numpy(
                np.asarray(raw_labels[self.indices], dtype=np.int64).copy())
        else:
            self.fields = raw_fields
            self.labels = raw_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        source_index = int(self.indices[item])
        if self.in_memory:
            field = self.fields[item]
            label = self.labels[item]
            if self.noise_level <= 0:
                return {
                    'velocity': field[:3], 'pressure': field[3], 'field': field,
                    'label': label, 'idx': source_index,
                }
            field = field.clone()
        else:
            field = np.asarray(self.fields[source_index], dtype=np.float32).copy()
            field = torch.from_numpy((field - self.means) / self.stds)
            label = torch.tensor(int(self.labels[source_index]), dtype=torch.long)
        if self.noise_level > 0:
            rng = np.random.default_rng(self.noise_seed + source_index)
            noise = torch.from_numpy(
                rng.uniform(-1.0, 1.0, tuple(field.shape)).astype(np.float32))
            field += self.noise_level * noise
        return {
            'velocity': field[:3],
            'pressure': field[3],
            'field': field,
            'label': label,
            'idx': source_index,
        }


def _loader(dataset, batch_size, shuffle, config):
    workers = int(config.data.num_workers)
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': workers,
        'pin_memory': bool(config.data.pin_memory and torch.cuda.is_available()),
    }
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def create_cached_dataloaders(config):
    build_first_frame_cache(
        data_dir=config.data.data_dir,
        cache_dir=config.data.cache_dir,
        regions=config.data.regions,
        n_subdomains=config.data.n_subdomains,
        split_seed=config.data.split_seed,
        n_train=config.data.n_train,
        n_val=config.data.n_val,
        n_test=config.data.n_test,
    )
    result = {}
    for region in config.data.regions:
        result[region] = {}
        for split in ('train', 'val', 'test'):
            dataset = CachedTurbulenceDataset(
                config.data.cache_dir, region, split,
                noise_level=config.data.noise_level if split == 'train' else 0.0,
                in_memory=config.data.cache_in_memory,
                noise_seed=config.seed,
            )
            result[region][split] = _loader(
                dataset, config.data.batch_size, split == 'train', config)
    return result


class DeviceTensorLoader:
    """Batch directly from a small dataset resident on the GPU.

    The complete train split is about 96 MB, far below the 8 GB device budget.
    Keeping it resident avoids copying the same 32 MB batches over PCIe for
    thousands of local block updates.
    """

    def __init__(self, dataset, batch_size, shuffle, device):
        if not isinstance(dataset, CachedTurbulenceDataset) or not dataset.in_memory:
            raise TypeError('DeviceTensorLoader requires an in-memory cached dataset')
        if dataset.noise_level > 0:
            raise ValueError('On-device cache currently supports clean fields only')
        self.fields = dataset.fields.to(device)
        self.labels = dataset.labels.to(device)
        self.indices = torch.as_tensor(dataset.indices, device=device)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.device = torch.device(device)

    def __len__(self):
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.labels), device=self.device)
        else:
            order = torch.arange(len(self.labels), device=self.device)
        for start in range(0, len(order), self.batch_size):
            selected = order[start:start + self.batch_size]
            field = self.fields[selected]
            yield {
                'field': field,
                'velocity': field[:, :3],
                'pressure': field[:, 3],
                'label': self.labels[selected],
                'idx': self.indices[selected],
            }


def cache_loaders_on_device(loaders, device):
    """Replace CPU DataLoaders with resident tensor batch iterators."""
    result = {}
    for region, split_loaders in loaders.items():
        result[region] = {}
        for split, loader in split_loaders.items():
            result[region][split] = DeviceTensorLoader(
                loader.dataset, loader.batch_size, split == 'train', device)
    return result


def create_dataloaders(config):
    """Create loaders; optimized runs use the first-frame cache by default."""
    if getattr(config.data, 'use_cache', False):
        return create_cached_dataloaders(config)

    datasets = {}
    for region in config.data.regions:
        transform = Compose([
            Normalize(), AddNoise(sigma=config.data.noise_level), ToTensor(),
        ])
        full_dataset = TurbulenceDataset(
            config.data.data_dir, region, config.data.n_classes,
            config.data.n_subdomains, transform,
        )
        n_val = max(1, int(len(full_dataset) * config.data.val_split))
        n_train = len(full_dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.data.split_seed),
        )
        datasets[region] = {
            'train': _loader(train_ds, config.data.batch_size, True, config),
            'val': _loader(val_ds, config.data.batch_size, False, config),
        }
    return datasets
