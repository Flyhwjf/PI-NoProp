"""Noise-to-label cosine interpolation used by the local NoProp blocks.

The important invariant is that block ``t`` sees the same marginal at train
and inference time: block zero starts from pure noise and the final block ends
at a clean label representation.  The previous implementation used a
clean-to-noise schedule for training but traversed it noise-to-clean during
inference.
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
        remaining_noise = (
            torch.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2)
        remaining_noise = remaining_noise / remaining_noise[0]
        # Signal level at the input/output boundaries of T blocks.
        self.signal = (1.0-remaining_noise).clamp(0.0, 1.0)
        self.signal[0] = 0.0
        self.signal[-1] = 1.0
        # Backwards-compatible public names used by diagnostics.
        self.alpha_bar = self.signal[1:]
        self.snr = self.alpha_bar / (1-self.alpha_bar).clamp_min(1e-8)

        # Deterministic DDIM interpolation.  If z_t has marginal
        # sqrt(g_t)u + sqrt(1-g_t)e, this update has the g_{t+1} marginal
        # whenever u_hat=u.  Hence train and inference distributions agree.
        g0, g1 = self.signal[:-1], self.signal[1:]
        self.b = torch.sqrt((1-g1)/(1-g0).clamp_min(1e-8))
        self.a = torch.sqrt(g1)-self.b*torch.sqrt(g0)
        self.c = torch.zeros_like(self.a)
        self.alpha = self.b.square()

    def get_coeffs(self, t):
        """Get (a_t, b_t, c_t) for step t (0-indexed)."""
        return self.a[t], self.b[t], self.c[t]

    def get_snr_weight(self, t):
        """Uniform local weighting avoids singular endpoint SNR weights."""
        return torch.ones((), device=self.signal.device)

    def get_input_signal(self, t):
        """Signal fraction expected at the input of block ``t``."""
        return self.signal[t]

    def q_sample(self, u_y, t):
        """Sample z_t ~ q(z_t | y) = N(sqrt(alpha_bar_t) * u_y, 1 - alpha_bar_t).

        Uses reparameterization.
        """
        eps = torch.randn_like(u_y)
        signal_t = self.signal[t+1]
        z_t = torch.sqrt(signal_t) * u_y + torch.sqrt(1-signal_t) * eps
        return z_t, eps

    def to(self, device):
        self.alpha_bar = self.alpha_bar.to(device)
        self.alpha = self.alpha.to(device)
        self.snr = self.snr.to(device)
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        self.c = self.c.to(device)
        self.signal = self.signal.to(device)
        return self
