import unittest

import numpy as np

from src.spider_ns_v2 import (
    EXPECTED_SUPPORT, NSSPIDERConfig, TERM_NAMES, _eta, _select,
)


class TestFullNSSPIDER(unittest.TestCase):
    def test_sparse_selection_recovers_full_ns_and_rejects_distractor(self):
        rng = np.random.default_rng(19)
        convection = rng.normal(size=1000)
        pressure = rng.normal(size=1000)
        laplacian = rng.normal(size=1000)
        distractor = rng.normal(size=1000)
        viscosity = 0.005
        time = -convection - pressure + viscosity*laplacian
        matrix = np.column_stack([
            time, convection, pressure, laplacian, distractor,
        ])
        config = NSSPIDERConfig(selection_eta=1e-10)
        best, _ = _select(matrix, config)
        terms = tuple(TERM_NAMES[index] for index in best['subset'])
        self.assertEqual(terms, EXPECTED_SUPPORT)
        np.testing.assert_allclose(best['coefficients'],
                                   [1.0, 1.0, 1.0, -viscosity],
                                   rtol=1e-10, atol=1e-12)
        self.assertLess(_eta(matrix, best['subset'], best['coefficients']), 1e-12)

    def test_missing_viscosity_cannot_pass_strict_threshold(self):
        rng = np.random.default_rng(23)
        columns = rng.normal(size=(500, 3))
        time = -columns[:, 0] - columns[:, 1] + 0.05*columns[:, 2]
        matrix = np.column_stack([time, columns, rng.normal(size=500)])
        subset = (0, 1, 2)
        # The omitted independent viscous contribution remains measurable.
        from src.spider_ns_v2 import _fit
        self.assertGreater(_eta(matrix, subset, _fit(matrix, subset)), 0.01)


if __name__ == '__main__':
    unittest.main()
