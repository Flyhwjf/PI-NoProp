"""Fast spatial weak-form constraints for single-snapshot HIT training.

Classification consumes one 3-D snapshot, so a duplicated artificial time
axis is neither necessary nor physically meaningful.  This module evaluates
continuity and pressure-Poisson residuals using compact 3-D test functions.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _spatial_test_functions(n_functions: int, size: int, beta: float):
    coordinates = torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
    x, y, z = torch.meshgrid(coordinates, coordinates, coordinates, indexing='ij')
    envelope = ((1 - x.square()).clamp_min(0)
                * (1 - y.square()).clamp_min(0)
                * (1 - z.square()).clamp_min(0)).pow(beta)
    functions = []
    for k in range(n_functions):
        # Smooth deterministic modulations produce distinct tests while the
        # shared envelope keeps every function zero at the domain boundary.
        frequency = k // 3 + 1
        axis = k % 3
        coord = (x, y, z)[axis]
        modulation = 1.0 + 0.35 * torch.cos(frequency * math.pi * coord)
        w = envelope * modulation
        functions.append(w / w.abs().amax().clamp_min(1e-8))
    w = torch.stack(functions)
    spacing = 2.0 / max(size - 1, 1)
    gradients = torch.gradient(w, spacing=(spacing, spacing, spacing), dim=(1, 2, 3))
    laplacian = torch.zeros_like(w)
    for dimension in range(3):
        first = torch.gradient(w, spacing=spacing, dim=dimension + 1)[0]
        laplacian += torch.gradient(first, spacing=spacing, dim=dimension + 1)[0]
    return w, torch.stack(gradients, dim=1), laplacian


def _central_difference_kernels(spacing: float):
    kernels = torch.zeros(3, 1, 3, 3, 3, dtype=torch.float32)
    kernels[0, 0, 0, 1, 1] = -0.5 / spacing
    kernels[0, 0, 2, 1, 1] = 0.5 / spacing
    kernels[1, 0, 1, 0, 1] = -0.5 / spacing
    kernels[1, 0, 1, 2, 1] = 0.5 / spacing
    kernels[2, 0, 1, 1, 0] = -0.5 / spacing
    kernels[2, 0, 1, 1, 2] = 0.5 / spacing
    return kernels


class SpatialPhysicsLoss(nn.Module):
    """Continuity + pressure-Poisson weak loss on a 3-D integration domain."""

    def __init__(self, config, discovered_artifact=None):
        super().__init__()
        physics = config.physics
        self.domain_size = int(getattr(physics, 'physics_grid_size',
                                       physics.domain_size))
        self.use_continuity = bool(physics.use_continuity)
        self.use_pressure_poisson = bool(physics.use_pressure_poisson)
        self.pp_weight = float(getattr(physics, 'pressure_poisson_weight', 0.5))
        self.physics_source = 'spider_discovered' if discovered_artifact else 'analytic'
        pp_coefficients = {'pressure_laplacian': 1.0,
                           'convection_divergence': 1.0}
        if discovered_artifact:
            equation = discovered_artifact['equation']
            pp_coefficients.update(dict(zip(equation['terms'],
                                            equation['coefficients'])))
        self.register_buffer('pp_coefficients', torch.tensor([
            pp_coefficients['pressure_laplacian'],
            pp_coefficients['convection_divergence'],
        ], dtype=torch.float32))

        # The decoder reconstructs standardized cache fields.  Governing
        # equation coefficients live in physical units, so undo that transform
        # before evaluating either analytic or discovered constraints.
        regions = list(getattr(config.data, 'regions', []))
        stats_path = Path(config.data.cache_dir) / f'{regions[0]}_stats.npz' if regions else None
        if stats_path and stats_path.exists():
            stats = np.load(stats_path)
            means = stats['means'].astype(np.float32)
            stds = stats['stds'].astype(np.float32)
        else:
            means = np.zeros(4, dtype=np.float32)
            stds = np.ones(4, dtype=np.float32)
        self.register_buffer('field_means', torch.from_numpy(means).view(1, 4, 1, 1, 1))
        self.register_buffer('field_stds', torch.from_numpy(stds).view(1, 4, 1, 1, 1))
        w, grad_w, lap_w = _spatial_test_functions(
            physics.n_test_functions, self.domain_size, physics.beta)
        self.register_buffer('w', w)
        self.register_buffer('grad_w', grad_w)
        self.register_buffer('lap_w', lap_w)
        spacing = 2.0 / max(self.domain_size - 1, 1)
        self.register_buffer('derivative_kernels', _central_difference_kernels(spacing))

    def _crop(self, fields):
        size = self.domain_size
        if fields.shape[-3:] == (size, size, size):
            return fields
        starts = [(length - size) // 2 for length in fields.shape[-3:]]
        return fields[..., starts[0]:starts[0] + size,
                      starts[1]:starts[1] + size,
                      starts[2]:starts[2] + size]

    def velocity_gradient(self, velocity):
        """Return du_i/dx_j with shape (B, 3, 3, H, H, H)."""
        batch, channels = velocity.shape[:2]
        padded = F.pad(velocity, (1, 1, 1, 1, 1, 1), mode='replicate')
        kernels = self.derivative_kernels.to(dtype=velocity.dtype)
        # One grouped convolution computes all three spatial derivatives for
        # each velocity channel. Output ordering is channel-major.
        weight = kernels.repeat(channels, 1, 1, 1, 1)
        gradient = F.conv3d(padded, weight, groups=channels)
        return gradient.reshape(batch, channels, 3, *velocity.shape[-3:])

    def _residual_terms(self, fields):
        fields = self._crop(fields)
        fields = fields.float() * self.field_stds + self.field_means
        # Tensor-core operations may run in FP16, but weak integrations and
        # small residuals are accumulated in FP32 for numerical stability.
        velocity = fields[:, :3].float()
        pressure = fields[:, 3].float()
        grad_w = self.grad_w.float()
        lap_w = self.lap_w.float()

        # Weak continuity: integral(w div u) = -integral(u . grad w).
        n_points = velocity[0, 0].numel()
        div_terms = -torch.einsum('bixyz,kixyz->bki', velocity, grad_w) / n_points
        r_div = div_terms.sum(dim=-1)

        # conv_i = u_j * d_j u_i, evaluated in a single batched contraction.
        velocity_gradient = self.velocity_gradient(velocity)
        convection = torch.einsum('bjxyz,bijxyz->bixyz', velocity, velocity_gradient)
        pressure_term = torch.einsum('bxyz,kxyz->bk', pressure, lap_w) / n_points
        convection_terms = torch.einsum(
            'bixyz,kixyz->bki', convection, grad_w) / n_points
        r_pp = (self.pp_coefficients[0] * pressure_term
                - self.pp_coefficients[1] * convection_terms.sum(dim=-1))
        return r_div, r_pp, div_terms, pressure_term, convection_terms

    def residuals(self, fields):
        r_div, r_pp, _, _, _ = self._residual_terms(fields)
        return r_div, r_pp

    def normalized_residuals(self, fields):
        r_div, r_pp, div_terms, pressure_term, convection_terms = self._residual_terms(fields)
        eps = 1e-6
        div_scale = div_terms.abs().sum(dim=-1).clamp_min(eps)
        pp_scale = (pressure_term.abs() + convection_terms.abs().sum(dim=-1)).clamp_min(eps)
        return r_div / div_scale, r_pp / pp_scale, r_div, r_pp

    def forward(self, fields):
        eta_div_values, eta_pp_values, r_div, r_pp = self.normalized_residuals(fields)
        loss_div = eta_div_values.square().mean()
        loss_pp = eta_pp_values.square().mean()
        total = torch.zeros((), device=fields.device, dtype=torch.float32)
        if self.use_continuity:
            total = total + loss_div
        if self.use_pressure_poisson:
            total = total + self.pp_weight * loss_pp
        metrics = {
            'loss_div': loss_div.detach(),
            'loss_pp': loss_pp.detach(),
            'eta_div': eta_div_values.square().mean().sqrt().detach(),
            'eta_pp': eta_pp_values.square().mean().sqrt().detach(),
            'raw_div': r_div.square().mean().sqrt().detach(),
            'raw_pp': r_pp.square().mean().sqrt().detach(),
            'pp_pressure_coefficient': self.pp_coefficients[0].detach(),
            'pp_convection_coefficient': self.pp_coefficients[1].detach(),
        }
        return total, metrics

    @torch.no_grad()
    def evaluate_metrics(self, fields):
        _, metrics = self(fields)
        return {key: float(value.cpu()) for key, value in metrics.items()}
