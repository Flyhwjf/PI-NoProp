"""Reference 3-D NoProp baselines without physics-specific conditioning."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.noprop.blocks import NoPropBlock
from src.noprop.diffusion import NoiseSchedule


class LocalFieldEncoder(nn.Module):
    """Compact per-learner encoder for four-channel 16^3 or 32^3 fields."""

    def __init__(self, in_channels=4, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(64*2*2*2, output_dim),
        )

    def forward(self, fields):
        # Pool the full context to the same feature grid. Existing 16^3
        # checkpoint keys and parameter counts remain unchanged.
        hidden = self.net[:6](fields)
        hidden = F.adaptive_avg_pool3d(hidden, (2, 2, 2))
        return self.net[6:](hidden)


class ReferenceNoPropStep(nn.Module):
    """One independently trainable reference NoProp denoising learner."""

    def __init__(self, n_classes=5, condition_dim=128, hidden_dim=256):
        super().__init__()
        self.encoder = LocalFieldEncoder(4, condition_dim)
        self.denoiser = NoPropBlock(
            n_classes, condition_dim, hidden_dim=hidden_dim,
            n_hidden_layers=3, activation='relu')

    def forward(self, latent, fields):
        return self.denoiser(latent, self.encoder(fields))


class ReferenceNoProp3D(nn.Module):
    """One-hot NoProp adapted to 3-D fields with strictly local blocks."""

    def __init__(self, n_classes=5, n_blocks=10, eta=0.1):
        super().__init__()
        self.n_classes = int(n_classes)
        self.n_blocks = int(n_blocks)
        self.eta = float(eta)
        self.schedule = NoiseSchedule(T=n_blocks)
        self.steps = nn.ModuleList([
            ReferenceNoPropStep(n_classes=n_classes) for _ in range(n_blocks)])

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.schedule.to(device)
        return result

    def target(self, labels):
        return F.one_hot(labels, self.n_classes).to(torch.float32)

    def local_loss(self, fields, labels, block_index):
        target = self.target(labels)
        t = int(block_index)
        signal = self.schedule.get_input_signal(t)
        latent = (signal.sqrt()*target
                  + (1-signal).sqrt()*torch.randn_like(target))
        prediction = self.steps[t](latent, fields)
        return (self.n_blocks/2)*self.eta*F.mse_loss(prediction, target)

    def forward(self, fields, initial_latent=None):
        if initial_latent is None:
            latent = torch.randn(
                fields.shape[0], self.n_classes, device=fields.device)
        else:
            latent = initial_latent
        for t, step in enumerate(self.steps):
            a_t, b_t, _ = self.schedule.get_coeffs(t)
            latent = a_t*step(latent, fields)+b_t*latent
        return latent


class ContinuousNoProp3D(nn.Module):
    """Continuous-time one-hot transport trained by conditional flow matching."""

    def __init__(self, n_classes=5, condition_dim=128, hidden_dim=256):
        super().__init__()
        self.n_classes = int(n_classes)
        self.encoder = LocalFieldEncoder(4, condition_dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.velocity = nn.Sequential(
            nn.Linear(n_classes+condition_dim+32, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def vector_field(self, time, latent, condition):
        if time.ndim == 0:
            time = time.expand(latent.shape[0], 1)
        elif time.ndim == 1:
            time = time[:, None]
        encoded_time = self.time_encoder(time.to(latent.dtype))
        return self.velocity(torch.cat([latent, condition, encoded_time], dim=-1))

    def flow_matching_loss(self, fields, labels):
        target = F.one_hot(labels, self.n_classes).to(torch.float32)
        initial = torch.randn_like(target)
        time = torch.rand(target.shape[0], 1, device=target.device)
        latent = (1-time)*initial+time*target
        desired_velocity = target-initial
        condition = self.encoder(fields)
        prediction = self.vector_field(time, latent, condition)
        return F.mse_loss(prediction, desired_velocity)

    def integrate(self, fields, initial_latent=None, steps=10, adjoint=True):
        try:
            from torchdiffeq import odeint, odeint_adjoint
        except ImportError as error:
            raise RuntimeError(
                'NoProp-CT requires torchdiffeq; install requirements.txt') from error

        condition = self.encoder(fields)
        if initial_latent is None:
            initial_latent = torch.randn(
                fields.shape[0], self.n_classes, device=fields.device)

        class AugmentedDynamics(nn.Module):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent

            def forward(self, time, state):
                latent, fixed_condition = state
                return (self.parent.vector_field(
                    time, latent, fixed_condition), torch.zeros_like(fixed_condition))

        dynamics = AugmentedDynamics(self)
        times = torch.tensor([0.0, 1.0], device=fields.device)
        solver = odeint_adjoint if adjoint else odeint
        options = {'step_size': 1.0/int(steps)}
        kwargs = dict(method='rk4', options=options)
        if adjoint:
            kwargs.update(adjoint_method='rk4', adjoint_options=options)
        latent_path, _ = solver(
            dynamics, (initial_latent, condition), times, **kwargs)
        return latent_path[-1]
