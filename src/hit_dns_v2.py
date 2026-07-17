"""Validated pseudo-spectral DNS for decaying homogeneous isotropic turbulence.

This solver is deliberately independent from the archived v1 generator.  It
uses a correctly normalized 3/2 de-aliasing transform, a divergence-free RK4
integrator, and an unforced momentum equation so that every term required for
blind Navier--Stokes discovery is observable from saved velocity and pressure
snapshots.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.fft import fftn, fftshift, ifftn, ifftshift


@dataclass(frozen=True)
class HITDNSConfig:
    grid_size: int = 64
    box_length: float = 2.0 * np.pi
    viscosity: float = 5.0e-3
    time_step: float = 2.0e-3
    initial_energy: float = 0.5
    spectrum_peak: float = 4.0
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DecayingHITDNS:
    """Fourier--Galerkin solver for incompressible decaying HIT.

    Fourier coefficients use SciPy's default convention: the forward transform
    is unnormalised and the inverse carries ``1 / N**3``.  Moving coefficients
    between the base and 3/2 grids therefore requires explicit volume factors.
    Omitting these factors was the source of the effective 0.293 pressure
    coefficient in the archived v1 data.
    """

    def __init__(self, config: HITDNSConfig):
        self.config = config
        self.n = int(config.grid_size)
        if self.n < 12 or self.n % 2:
            raise ValueError('grid_size must be an even integer >= 12')
        self.nd = 3 * self.n // 2
        self.length = float(config.box_length)
        self.nu = float(config.viscosity)
        self.dt = float(config.time_step)
        self.time = 0.0
        self.step_count = 0
        self.rng = np.random.default_rng(config.seed)

        self.k = 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.length / self.n)
        self.kd = 2.0 * np.pi * np.fft.fftfreq(self.nd, d=self.length / self.nd)
        self.kx, self.ky, self.kz = np.meshgrid(
            self.k, self.k, self.k, indexing='ij', sparse=True)
        self.kxd, self.kyd, self.kzd = np.meshgrid(
            self.kd, self.kd, self.kd, indexing='ij', sparse=True)
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        self.k2_safe = np.where(self.k2 > 0, self.k2, 1.0)
        self._embed_start = (self.nd - self.n) // 2
        self._u_hat = self._initial_condition()

    def _project(self, vector_hat: np.ndarray) -> np.ndarray:
        k_dot = (self.kx * vector_hat[0] + self.ky * vector_hat[1]
                 + self.kz * vector_hat[2]) / self.k2_safe
        result = np.empty_like(vector_hat)
        result[0] = vector_hat[0] - self.kx * k_dot
        result[1] = vector_hat[1] - self.ky * k_dot
        result[2] = vector_hat[2] - self.kz * k_dot
        result[:, 0, 0, 0] = 0.0
        return result

    def _initial_condition(self) -> np.ndarray:
        physical = self.rng.standard_normal((3, self.n, self.n, self.n))
        spectrum = fftn(physical, axes=(-3, -2, -1), workers=-1)
        spectrum = self._project(spectrum)
        k_mag = np.sqrt(self.k2)
        peak = max(float(self.config.spectrum_peak), 1.0)
        # Smooth large-scale initial spectrum; the zero and unresolved tail are
        # removed before rescaling to the requested kinetic energy.
        spectral_filter = (k_mag / peak) ** 2 * np.exp(-(k_mag / peak) ** 2)
        spectrum *= spectral_filter[None]
        spectrum[:, k_mag >= self.n / 3] = 0.0
        spectrum = self._project(spectrum)
        velocity = ifftn(spectrum, axes=(-3, -2, -1), workers=-1).real
        energy = 0.5 * np.mean(np.sum(velocity * velocity, axis=0))
        spectrum *= np.sqrt(self.config.initial_energy / max(energy, 1e-30))
        return spectrum

    def _pad(self, coefficients: np.ndarray) -> np.ndarray:
        """Embed base-grid coefficients on the 3/2 grid with FFT scaling."""
        # An even grid has one unpaired Nyquist plane per dimension.  It is
        # spectrally unresolved for differentiation and must not be embedded
        # as if it had a distinct positive-frequency partner.
        coefficients = coefficients.copy()
        nyquist = self.n // 2
        coefficients[..., nyquist, :, :] = 0.0
        coefficients[..., :, nyquist, :] = 0.0
        coefficients[..., :, :, nyquist] = 0.0
        shifted = fftshift(coefficients, axes=(-3, -2, -1))
        padded = np.zeros(coefficients.shape[:-3] + (self.nd,) * 3,
                          dtype=coefficients.dtype)
        start = self._embed_start
        padded[..., start:start + self.n, start:start + self.n,
               start:start + self.n] = shifted
        padded = ifftshift(padded, axes=(-3, -2, -1))
        return padded * (self.nd / self.n) ** 3

    def _truncate(self, coefficients: np.ndarray) -> np.ndarray:
        """Return 3/2-grid coefficients to the base FFT convention."""
        shifted = fftshift(coefficients, axes=(-3, -2, -1))
        start = self._embed_start
        cropped = shifted[..., start:start + self.n, start:start + self.n,
                          start:start + self.n]
        cropped = ifftshift(cropped, axes=(-3, -2, -1))
        cropped = cropped * (self.n / self.nd) ** 3
        # The positive and negative Nyquist modes collapse onto the same base
        # grid samples.  Zeroing those planes avoids an ambiguous merge and
        # preserves Hermitian symmetry of every retained real field.
        nyquist = self.n // 2
        cropped[..., nyquist, :, :] = 0.0
        cropped[..., :, nyquist, :] = 0.0
        cropped[..., :, :, nyquist] = 0.0
        return cropped

    def nonlinear_hat(self, velocity_hat: np.ndarray | None = None) -> np.ndarray:
        """Return ``-(u . grad) u`` with correctly normalized 3/2 de-aliasing."""
        velocity_hat = self._u_hat if velocity_hat is None else velocity_hat
        padded = self._pad(velocity_hat)
        velocity = ifftn(padded, axes=(-3, -2, -1), workers=-1).real
        gradients = (
            ifftn(1j * self.kxd * padded, axes=(-3, -2, -1), workers=-1).real,
            ifftn(1j * self.kyd * padded, axes=(-3, -2, -1), workers=-1).real,
            ifftn(1j * self.kzd * padded, axes=(-3, -2, -1), workers=-1).real,
        )
        nonlinear = np.empty_like(velocity)
        for component in range(3):
            nonlinear[component] = -sum(
                velocity[axis] * gradients[axis][component]
                for axis in range(3))
        nonlinear_hat_d = fftn(nonlinear, axes=(-3, -2, -1), workers=-1)
        return self._truncate(nonlinear_hat_d)

    def rhs(self, velocity_hat: np.ndarray) -> np.ndarray:
        nonlinear = self.nonlinear_hat(velocity_hat)
        return self._project(nonlinear) - self.nu * self.k2[None] * velocity_hat

    def step(self, count: int = 1) -> None:
        """Advance with classical RK4 and re-project round-off divergence."""
        for _ in range(int(count)):
            u0 = self._u_hat
            k1 = self.rhs(u0)
            k2 = self.rhs(u0 + 0.5 * self.dt * k1)
            k3 = self.rhs(u0 + 0.5 * self.dt * k2)
            k4 = self.rhs(u0 + self.dt * k3)
            self._u_hat = self._project(
                u0 + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0)
            self.time += self.dt
            self.step_count += 1

    def velocity(self) -> np.ndarray:
        return ifftn(self._u_hat, axes=(-3, -2, -1), workers=-1).real

    def pressure(self) -> np.ndarray:
        """Pressure consistent with the projected nonlinear acceleration."""
        nonlinear = self.nonlinear_hat(self._u_hat)
        divergence_hat = 1j * (self.kx * nonlinear[0]
                               + self.ky * nonlinear[1]
                               + self.kz * nonlinear[2])
        pressure_hat = -divergence_hat / self.k2_safe
        pressure_hat[0, 0, 0] = 0.0
        pressure = ifftn(pressure_hat, workers=-1).real
        return pressure - pressure.mean()

    def diagnostics(self) -> dict[str, float]:
        velocity = self.velocity()
        energy = 0.5 * np.mean(np.sum(velocity * velocity, axis=0))
        divergence_hat = 1j * (self.kx * self._u_hat[0]
                               + self.ky * self._u_hat[1]
                               + self.kz * self._u_hat[2])
        divergence = ifftn(divergence_hat, workers=-1).real
        gradients_sq = sum(
            np.mean(ifftn(1j * wave * self._u_hat[component], workers=-1).real**2)
            for component in range(3) for wave in (self.kx, self.ky, self.kz))
        dissipation = self.nu * gradients_sq
        u_rms = np.sqrt(2.0 * energy / 3.0)
        max_speed = float(np.max(np.sqrt(np.sum(velocity * velocity, axis=0))))
        cfl = max_speed * self.dt / (self.length / self.n)
        taylor_re = (np.sqrt(15.0) * u_rms**2
                     / np.sqrt(max(self.nu * dissipation, 1e-30)))
        return {
            'time': float(self.time),
            'kinetic_energy': float(energy),
            'dissipation': float(dissipation),
            'u_rms': float(u_rms),
            're_lambda': float(taylor_re),
            'divergence_rms': float(np.sqrt(np.mean(divergence**2))),
            'cfl': float(cfl),
        }

    def snapshot(self) -> dict[str, np.ndarray]:
        return {
            'velocity': self.velocity().astype(np.float32),
            'pressure': self.pressure().astype(np.float32),
            'time': np.asarray(self.time, dtype=np.float64),
            'viscosity': np.asarray(self.nu, dtype=np.float64),
            'box_length': np.asarray(self.length, dtype=np.float64),
        }
