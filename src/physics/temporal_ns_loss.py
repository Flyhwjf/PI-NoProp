"""Differentiable 4-D weak full-Navier--Stokes loss for local NoProp blocks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.spider_ns import NSSPIDERConfig, _profiles


REQUIRED_TERMS = (
    'time_derivative', 'convection', 'pressure_gradient', 'velocity_laplacian')


class TemporalNSPhysicsLoss(nn.Module):
    def __init__(self, config, discovered_artifact):
        super().__init__()
        if isinstance(discovered_artifact, (str, Path)):
            discovered_artifact = json.loads(
                Path(discovered_artifact).read_text(encoding='utf-8'))
        if not discovered_artifact.get('validation', {}).get('passed', False):
            raise ValueError('full-NS training requires a validated SPIDER artifact')
        equation = discovered_artifact['equation']
        if tuple(equation['terms']) != REQUIRED_TERMS:
            raise ValueError(f'expected full NS terms {REQUIRED_TERMS}, '
                             f'got {equation["terms"]}')
        self.register_buffer(
            'coefficients', torch.tensor(equation['coefficients'], dtype=torch.float32))

        size = int(config.physics.physics_grid_size)
        n_time = int(config.physics.n_time)
        dns_grid = int(config.physics.dns_grid_size)
        dx = float(config.physics.box_length)/dns_grid
        dt = float(config.physics.snapshot_dt)
        profile_config = NSSPIDERConfig(
            domain_size=size, time_window=n_time,
            n_test_functions=int(config.physics.n_test_functions),
            beta=float(config.physics.beta))
        spatial, temporal = _profiles(profile_config, dx, dt)
        self.register_buffer('w_space', torch.tensor(
            np.stack([value[0] for value in spatial]), dtype=torch.float32))
        self.register_buffer('grad_space', torch.tensor(
            np.stack([value[1] for value in spatial]), dtype=torch.float32))
        self.register_buffer('lap_space', torch.tensor(
            np.stack([value[2] for value in spatial]), dtype=torch.float32))
        hessians = []
        for _, gradient, _ in spatial:
            hessians.append(np.stack([
                np.gradient(gradient[i], dx, axis=j, edge_order=2)
                for i in range(3) for j in range(3)]).reshape(
                    3, 3, size, size, size))
        self.register_buffer('hessian_space', torch.tensor(
            np.stack(hessians), dtype=torch.float32))
        self.register_buffer('w_time', torch.tensor(
            np.stack([value[0] for value in temporal]), dtype=torch.float32))
        self.register_buffer('dt_weight', torch.tensor(
            np.stack([value[1] for value in temporal]), dtype=torch.float32))

        stats = np.load(Path(config.data.cache_dir)/'stats.npz')
        means = torch.from_numpy(stats['means'].astype(np.float32))
        stds = torch.from_numpy(stats['stds'].astype(np.float32))
        self.register_buffer('field_means', means.view(1, 1, 4, 1, 1, 1))
        self.register_buffer('field_stds', stds.view(1, 1, 4, 1, 1, 1))
        self.size = size
        self.divergence_weight = float(config.physics.divergence_weight)
        self.use_pressure_poisson = bool(config.physics.use_pressure_poisson)
        self.use_energy = bool(config.physics.use_energy)
        self.pressure_poisson_weight = float(
            config.physics.pressure_poisson_weight)
        self.energy_weight = float(config.physics.energy_weight)
        self.dx = dx

    def _crop(self, fields):
        if fields.shape[-3:] == (self.size,)*3:
            return fields
        starts = [(length-self.size)//2 for length in fields.shape[-3:]]
        return fields[..., starts[0]:starts[0]+self.size,
                      starts[1]:starts[1]+self.size,
                      starts[2]:starts[2]+self.size]

    def residual_terms(self, standardized_fields):
        fields = self._crop(standardized_fields).float()
        fields = fields*self.field_stds + self.field_means
        velocity, pressure = fields[:, :, :3], fields[:, :, 3]
        points = velocity.shape[1]*velocity.shape[-1]**3
        time_term = -torch.einsum(
            'btixyz,kt,kxyz->bki', velocity, self.dt_weight, self.w_space)/points
        convection = -torch.einsum(
            'btixyz,btjxyz,kt,kjxyz->bki', velocity, velocity,
            self.w_time, self.grad_space)/points
        pressure_gradient = -torch.einsum(
            'btxyz,kt,kixyz->bki', pressure, self.w_time,
            self.grad_space)/points
        velocity_laplacian = torch.einsum(
            'btixyz,kt,kxyz->bki', velocity, self.w_time,
            self.lap_space)/points
        return torch.stack([
            time_term, convection, pressure_gradient, velocity_laplacian,
        ], dim=-1)

    def normalized_residuals(self, fields):
        terms = self.residual_terms(fields)
        contributions = terms*self.coefficients.view(1, 1, 1, -1)
        residual = contributions.sum(dim=-1)
        scale = contributions.abs().sum(dim=-1).clamp_min(1e-7)
        eta_ns = residual/scale

        physical = self._crop(fields).float()*self.field_stds + self.field_means
        velocity = physical[:, :, :3]
        div_components = -torch.einsum(
            'btixyz,kixyz->btki', velocity, self.grad_space)
        div_residual = div_components.sum(dim=-1)
        div_scale = div_components.abs().sum(dim=-1).clamp_min(1e-7)
        eta_div = div_residual/div_scale
        return eta_ns, eta_div, residual, terms

    def additional_residuals(self, fields):
        """Artifact-derived pressure-Poisson and kinetic-energy balances."""
        physical = self._crop(fields).float()*self.field_stds + self.field_means
        velocity, pressure = physical[:, :, :3], physical[:, :, 3]
        points = velocity.shape[1]*velocity.shape[-1]**3
        c_time, c_conv, c_pressure, c_laplacian = self.coefficients

        pp_pressure = torch.einsum(
            'btxyz,kt,kxyz->bk', pressure, self.w_time,
            self.lap_space)/points
        pp_convection = torch.einsum(
            'btixyz,btjxyz,kt,kijxyz->bk', velocity, velocity,
            self.w_time, self.hessian_space)/points
        pp_parts = torch.stack([
            c_pressure*pp_pressure, c_conv*pp_convection], dim=-1)
        eta_pp = pp_parts.sum(-1)/pp_parts.abs().sum(-1).clamp_min(1e-7)

        kinetic = 0.5*velocity.square().sum(dim=2)
        energy_time = -torch.einsum(
            'btxyz,kt,kxyz->bk', kinetic, self.dt_weight,
            self.w_space)/points
        energy_convection = -torch.einsum(
            'btxyz,btixyz,kt,kixyz->bk', kinetic, velocity,
            self.w_time, self.grad_space)/points
        energy_pressure = -torch.einsum(
            'btxyz,btixyz,kt,kixyz->bk', pressure, velocity,
            self.w_time, self.grad_space)/points
        energy_laplacian = torch.einsum(
            'btxyz,kt,kxyz->bk', kinetic, self.w_time,
            self.lap_space)/points
        gradients = torch.stack([
            torch.gradient(velocity, spacing=self.dx, dim=axis+3)[0]
            for axis in range(3)], dim=3)
        dissipation = torch.einsum(
            'btijxyz,kt,kxyz->bk', gradients.square(), self.w_time,
            self.w_space)/points
        energy_parts = torch.stack([
            c_time*energy_time,
            c_conv*energy_convection,
            c_pressure*energy_pressure,
            c_laplacian*energy_laplacian,
            -c_laplacian*dissipation,
        ], dim=-1)
        eta_energy = (energy_parts.sum(-1)
                      / energy_parts.abs().sum(-1).clamp_min(1e-7))
        return eta_pp, eta_energy

    def forward(self, fields):
        eta_ns, eta_div, residual, _ = self.normalized_residuals(fields)
        loss_ns = eta_ns.square().mean()
        loss_div = eta_div.square().mean()
        total = loss_ns + self.divergence_weight*loss_div
        eta_pp, eta_energy = self.additional_residuals(fields)
        loss_pp = eta_pp.square().mean()
        loss_energy = eta_energy.square().mean()
        if self.use_pressure_poisson:
            total = total+self.pressure_poisson_weight*loss_pp
        if self.use_energy:
            total = total+self.energy_weight*loss_energy
        return total, {
            'loss_ns': loss_ns.detach(),
            'loss_div': loss_div.detach(),
            'eta_ns': eta_ns.square().mean().sqrt().detach(),
            'eta_div': eta_div.square().mean().sqrt().detach(),
            'loss_pp': loss_pp.detach(),
            'loss_energy': loss_energy.detach(),
            'eta_pp': eta_pp.square().mean().sqrt().detach(),
            'eta_energy': eta_energy.square().mean().sqrt().detach(),
            'raw_ns': residual.square().mean().sqrt().detach(),
        }

    @torch.no_grad()
    def evaluate_metrics(self, fields):
        _, metrics = self(fields)
        return {key: float(value.cpu()) for key, value in metrics.items()}
