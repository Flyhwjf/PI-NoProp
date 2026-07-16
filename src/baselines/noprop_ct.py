"""NoProp-CT: Continuous-time NoProp baseline using Ornstein-Uhlenbeck SDE."""
import math
import torch
import torch.nn as nn


class NoPropCTModel(nn.Module):
    """Continuous-time NoProp using Ornstein-Uhlenbeck SDE.

    Forward SDE:  dZ_t = -0.5 * beta(t) * Z_t dt + sqrt(beta(t)) dW_t
    Reverse SDE:  dZ_t = [-0.5*beta(t)*Z_t - beta(t)*score(Z_t,x,t)] dt + sqrt(beta(t)) dŴ_t

    Inference: Euler-Maruyama over N_steps.

    Trainer interface:
        model(x)                        -> logits
        model(x, return_all_latents=T)  -> (logits, z_all)
        model.label_embed(labels)       -> u_y embeddings
        model.compute_diffusion_loss(x, u_y, t) -> scalar
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_cfg = config.noprop

        self.d_latent = d_cfg.embedding_dim
        self.condition_dim = d_cfg.condition_dim
        self.n_classes = config.data.n_classes
        self.N_steps = config.diffusion.T

        self.beta_min = 0.1
        self.beta_max = 20.0
        self.t_eps = 0.01

        self.T_emb_dim = 128

        # ---------- x encoder: 3D CNN, output flat condition vector ----------
        self.encoder = nn.Sequential(
            nn.Conv3d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4 * 4, self.condition_dim),
        )

        # ---------- score network ----------
        score_in = self.d_latent + self.condition_dim + self.T_emb_dim
        self.score_net = nn.Sequential(
            nn.Linear(score_in, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.d_latent),
        )

        # ---------- classifier ----------
        self.classifier = nn.Linear(self.d_latent, self.n_classes)

        # ---------- label embedding ----------
        self.label_embed = nn.Embedding(self.n_classes, self.d_latent)

    # ------------------------------------------------------------------ #
    #  OU process helpers
    # ------------------------------------------------------------------ #

    def _beta(self, t):
        """beta(t) = beta_min + t * (beta_max - beta_min)."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _beta_integral(self, t):
        """∫_0^t beta(s) ds = beta_min*t + 0.5*(beta_max-beta_min)*t^2."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * (t ** 2)

    def _mu_sigma(self, t):
        """OU transition kernel: N(mu_t * u_y, sigma_t^2 I).

        mu_t    = exp(-0.5 * ∫_0^t beta(s) ds)
        sigma_t = sqrt(1 - mu_t^2)
        """
        integral = self._beta_integral(t)
        mu = torch.exp(-0.5 * integral)
        sigma = torch.sqrt(torch.clamp(1.0 - mu ** 2, min=0.0) + 1e-8)
        return mu, sigma

    # ------------------------------------------------------------------ #
    #  Time embedding
    # ------------------------------------------------------------------ #

    def _time_embedding(self, t):
        """Sinusoidal positional encoding (diffusion-style).

        Args:
            t: (batch_size,) float tensor
        Returns:
            (batch_size, T_emb_dim)
        """
        half = self.T_emb_dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.unsqueeze(-1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # ------------------------------------------------------------------ #
    #  Trainer interface
    # ------------------------------------------------------------------ #

    def forward(self, x, return_all_latents=False):
        """Inference: integrate the reverse SDE with Euler-Maruyama.

        z_n ~ N(0, I), then N_steps of
        Δz = [-0.5*beta(t)*z - beta(t)*score(z,x,t)] * dt + sqrt(beta(t)*dt) * eps
        finishing at z_N (near u_y), then classify.

        Args:
            x: (B, 4, H, H, H)
            return_all_latents: return list of all intermediate z
        Returns:
            logits, or (logits, z_all) when return_all_latents=True
        """
        batch_size = x.shape[0]
        device = x.device

        x_cond = self.encoder(x)

        z = torch.randn(batch_size, self.d_latent, device=device)
        z_all = [z] if return_all_latents else None

        dt = 1.0 / self.N_steps

        for n in range(self.N_steps):
            t_val = 1.0 - n * dt - self.t_eps
            t_batch = torch.full((batch_size,), t_val, device=device)

            beta_t = self._beta(t_val)
            t_emb = self._time_embedding(t_batch)
            score = self.score_net(torch.cat([z, x_cond, t_emb], dim=-1))

            eps = torch.randn_like(z) if self.training else torch.zeros_like(z)

            # Reverse SDE with positive dt (integrating backward in original time)
            z = z + (0.5 * beta_t * z + beta_t * score) * dt \
                + torch.sqrt(torch.tensor(beta_t * dt, device=device)) * eps

            if return_all_latents:
                z_all.append(z.clone())

        logits = self.classifier(z)

        if return_all_latents:
            return logits, z_all
        return logits

    def compute_diffusion_loss(self, x, u_y, t):
        """Denoising score matching loss.

        t is a block-index placeholder (ignored); a random continuous time
        is sampled inside each call.

        Loss (epsilon-prediction form):
            L = E_{t~U(eps,1-eps)} [ beta(t)*sigma_t^2 * ||score - target||^2 ]

        where target = -noise / sigma_t and z_t = mu_t*u_y + sigma_t*noise.

        Args:
            x: (B, 4, H, H, H)
            u_y: (B, d_latent) target embedding
            t: ignored (continuous time sampled internally)
        Returns:
            loss: scalar
        """
        batch_size = x.shape[0]
        device = x.device

        x_cond = self.encoder(x)

        t_cont = torch.rand(batch_size, device=device) * (1.0 - 2 * self.t_eps) + self.t_eps

        mu_t, sigma_t = self._mu_sigma(t_cont)

        noise = torch.randn_like(u_y)
        z_t = mu_t.unsqueeze(-1) * u_y + sigma_t.unsqueeze(-1) * noise

        t_emb = self._time_embedding(t_cont)
        score_pred = self.score_net(torch.cat([z_t, x_cond, t_emb], dim=-1))

        target = -noise / (sigma_t.unsqueeze(-1) + 1e-8)

        beta_t = self._beta(t_cont)
        weight = beta_t * (sigma_t ** 2)

        loss = (weight.unsqueeze(-1) * (score_pred - target) ** 2).mean()
        return loss
