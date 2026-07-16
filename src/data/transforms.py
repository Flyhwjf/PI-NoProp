"""Data transforms for turbulence subdomains."""
import torch
import numpy as np


class Normalize:
    """Normalize fields to zero mean and unit variance per channel."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def __call__(self, sample):
        vel = sample['velocity']    # (3, 16, 16, 16) or (3, 32, 32, 32)
        pres = sample['pressure']   # (16, 16, 16) or (32, 32, 32)

        if self.means is None:
            self.means = {
                'velocity': vel.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32),
                'pressure': np.array(pres.mean(), dtype=np.float32),
            }
            self.stds = {
                'velocity': vel.std(axis=(1, 2, 3), keepdims=True).astype(np.float32) + 1e-8,
                'pressure': np.array(pres.std(), dtype=np.float32) + 1e-8,
            }

        sample['velocity'] = (vel - self.means['velocity']) / self.stds['velocity']
        sample['pressure'] = (pres - self.means['pressure']) / self.stds['pressure']
        return sample


class AddNoise:
    """Add uniform noise: f_σ = f + σ * ξ * s_f  (eq.4.12)"""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def __call__(self, sample):
        if self.sigma <= 0:
            return sample
        rng = np.random.RandomState()
        for key in ['velocity', 'pressure']:
            field = sample[key]
            s_f = max(field.std(), 1e-8)
            noise = self.sigma * rng.uniform(-1, 1, field.shape).astype(np.float32) * s_f
            sample[key] = field + noise
        return sample


class ToTensor:
    """Convert numpy arrays to torch tensors."""

    def __call__(self, sample):
        sample['velocity'] = torch.from_numpy(sample['velocity']).float()
        sample['pressure'] = torch.from_numpy(sample['pressure']).float()
        sample['label'] = torch.tensor(sample['label'], dtype=torch.long)
        return sample
