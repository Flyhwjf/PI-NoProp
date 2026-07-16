"""NoProp noise schedule and SNR computation.

Implements the cosine noise schedule from DDPM / NoProp paper.
Provides alpha_bar, SNR, and the coefficients a_t, b_t, c_t for inference.
"""
import torch
import numpy as np


class NoiseSchedule:
    """Cosine noise schedule for NoProp diffusion.

    alpha_bar_t = f(t) / f(0),  f(t) = cos^2((t/T + s) / (1 + s) * pi/2)
    SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
    """

    def __init__(self, T=10, s=0.008):
        self.T = T
        self.s = s

        t = torch.linspace(0, T, T + 1)
        f = torch.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_bar = f / f[0]
        self.alpha_bar = alpha_bar[1:]  # T values, remove t=0
        self.alpha = self.alpha_bar / torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])

        # SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
        self.snr = self.alpha_bar / (1 - self.alpha_bar + 1e-8)

        # Inference coefficients (eq.3.1)
        # z_t = a_t * u_hat + b_t * z_{t-1} + sqrt(c_t) * eps
        self.a = torch.sqrt(1.0 - self.alpha)
        self.b = torch.sqrt(self.alpha)
        self.c = 1.0 - self.alpha

    def get_coeffs(self, t):
        """Get (a_t, b_t, c_t) for step t (0-indexed)."""
        return self.a[t], self.b[t], self.c[t]

    def get_snr_weight(self, t):
        """Get |SNR(t) - SNR(t-1)| for diffusion loss weighting (eq.3.2).

        The absolute value ensures a positive weight since SNR decreases with t.
        """
        if t == 0:
            return self.snr[0]
        return torch.abs(self.snr[t] - self.snr[t - 1])

    def q_sample(self, u_y, t):
        """Sample z_t ~ q(z_t | y) = N(sqrt(alpha_bar_t) * u_y, 1 - alpha_bar_t).

        Uses reparameterization.
        """
        eps = torch.randn_like(u_y)
        alpha_bar_t = self.alpha_bar[t]
        z_t = torch.sqrt(alpha_bar_t) * u_y + torch.sqrt(1 - alpha_bar_t) * eps
        return z_t, eps

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha = self.alpha.to(device)
        self.snr = self.snr.to(device)
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.c = self.c.to(device)
        return self
