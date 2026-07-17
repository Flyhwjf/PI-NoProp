import unittest

import numpy as np

from src.hit_dns_v2 import DecayingHITDNS, HITDNSConfig


class TestHITDNSV2(unittest.TestCase):
    def setUp(self):
        self.dns = DecayingHITDNS(HITDNSConfig(
            grid_size=16, viscosity=1e-2, time_step=1e-3, seed=7))

    def test_padding_preserves_physical_amplitude(self):
        x = np.arange(self.dns.n) * self.dns.length / self.dns.n
        field = np.sin(2*x)[:, None, None] * np.ones((1, self.dns.n, self.dns.n))
        coefficients = np.fft.fftn(field)
        padded_field = np.fft.ifftn(self.dns._pad(coefficients)).real
        xd = np.arange(self.dns.nd) * self.dns.length / self.dns.nd
        expected = np.sin(2*xd)[:, None, None] * np.ones((1, self.dns.nd, self.dns.nd))
        self.assertLess(np.max(np.abs(padded_field - expected)), 1e-12)

    def test_truncation_round_trip(self):
        coefficients = self.dns._u_hat.copy()
        nyquist = self.dns.n // 2
        coefficients[..., nyquist, :, :] = 0
        coefficients[..., :, nyquist, :] = 0
        coefficients[..., :, :, nyquist] = 0
        recovered = self.dns._truncate(self.dns._pad(coefficients))
        self.assertLess(np.max(np.abs(recovered - coefficients)), 1e-10)

    def test_projection_and_step_remain_divergence_free(self):
        self.dns.step(2)
        diagnostics = self.dns.diagnostics()
        self.assertLess(diagnostics['divergence_rms'], 1e-11)
        self.assertLess(diagnostics['cfl'], 0.2)

    def test_pressure_poisson_identity_has_unit_coefficient(self):
        nonlinear = self.dns.nonlinear_hat()
        pressure = self.dns.pressure()
        pressure_hat = np.fft.fftn(pressure)
        lap_pressure = -self.dns.k2 * pressure_hat
        div_nonlinear = 1j * (
            self.dns.kx * nonlinear[0] + self.dns.ky * nonlinear[1]
            + self.dns.kz * nonlinear[2])
        relative = np.linalg.norm(lap_pressure - div_nonlinear) / np.linalg.norm(div_nonlinear)
        self.assertLess(relative, 1e-11)


if __name__ == '__main__':
    unittest.main()
