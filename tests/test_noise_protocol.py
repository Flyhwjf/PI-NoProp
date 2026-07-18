import unittest

import numpy as np
import torch

from scripts.run_noise import ns_terms_from_standardized
from src.data.hit_dataset import _ns_energy_terms


class TestNoiseProtocol(unittest.TestCase):
    def test_clean_terms_match_training_cache_construction(self):
        rng = np.random.default_rng(20260718)
        physical = rng.standard_normal((2, 4, 8, 8, 8)).astype(np.float32)
        means = np.asarray([0.2, -0.1, 0.3, 0.05], dtype=np.float32)
        stds = np.asarray([1.2, 0.8, 1.5, 0.6], dtype=np.float32)
        standardized = ((physical-means[None, :, None, None, None])
                        / stds[None, :, None, None, None])
        dx = 2*np.pi/64

        actual = ns_terms_from_standardized(
            torch.from_numpy(standardized),
            torch.from_numpy(means).view(1, 4, 1, 1, 1),
            torch.from_numpy(stds).view(1, 4, 1, 1, 1), dx).numpy()
        expected = np.stack([_ns_energy_terms(field, dx) for field in physical])

        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


if __name__ == '__main__':
    unittest.main()
