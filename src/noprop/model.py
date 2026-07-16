"""Complete NoProp model: T blocks + classifier + diffusion losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import NoiseSchedule
from .embedding import LabelEmbedding
from .blocks import NoPropBlock
from .classifier import ClassifierHead


class NoPropModel(nn.Module):
    """Full NoProp model with T sequential denoising blocks.

    Inference (eq.3.1):
        z_0 ~ N(0, I)
        z_t = a_t * û_θt(z_{t-1}, x) + b_t * z_{t-1} + sqrt(c_t) * ε_t

    Diffusion loss (eq.3.2):
        L_diff^(t) = (T/2) * η * (SNR(t)-SNR(t-1)) * ||û_θt(z_{t-1}, x) - u_y||²
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_cfg = config.noprop

        spatial_size = config.data.subdomain_size
        in_features = config.data.n_channels * (spatial_size ** 3)

        # Diffusion schedule (moved to device in forward/training)
        self.noise_schedule = NoiseSchedule(T=config.diffusion.T, s=config.diffusion.s)

        # Label embedding
        self.label_embed = LabelEmbedding(
            n_classes=config.data.n_classes,
            embedding_dim=d_cfg.embedding_dim,
            learnable=True,
        )

        # Input encoder: (4, H, H, H) → condition_dim
        # AdaptiveAvgPool3d reduces spatial size before flat Linear
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, d_cfg.condition_dim),
        )

        # NoProp blocks
        self.blocks = nn.ModuleList([
            NoPropBlock(
                input_dim=d_cfg.embedding_dim,
                condition_dim=d_cfg.condition_dim,
                hidden_dim=d_cfg.hidden_dim,
                n_hidden_layers=d_cfg.n_hidden_layers,
                activation=d_cfg.activation,
            )
            for _ in range(config.diffusion.T)
        ])

        # Classifier head
        self.classifier = ClassifierHead(d_cfg.embedding_dim, config.data.n_classes)

    def encode_condition(self, x):
        condition = self.encoder(x)
        if getattr(self.config.noprop, 'normalize_condition', False):
            condition = F.normalize(condition, dim=-1) * (condition.shape[-1] ** 0.5)
        return condition

    def forward(self, x, return_all_latents=False):
        """Inference: denoise from z_0 through all blocks.

        Args:
            x: (batch_size, 4, H, H, H) input field
            return_all_latents: return list of all intermediate z_t
        Returns:
            logits or (logits, z_all)
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode input to condition vector
        x_cond = self.encode_condition(x)  # (batch_size, condition_dim)

        # z_0 ~ N(0, I)
        z = torch.randn(batch_size, self.config.noprop.embedding_dim, device=device)

        if return_all_latents:
            z_all = [z]

        for t in range(self.config.diffusion.T):
            a_t, b_t, c_t = self.noise_schedule.get_coeffs(t)
            u_hat = self.blocks[t](z, x_cond)

            eps = torch.randn_like(z) if self.training else 0
            z = a_t * u_hat + b_t * z + torch.sqrt(c_t) * eps

            if return_all_latents:
                z_all.append(z)

        logits = self.classifier(z)

        if return_all_latents:
            return logits, z_all
        return logits

    def compute_diffusion_loss(self, x, u_y, t):
        """Compute diffusion loss for block t (eq.3.2).

        Args:
            x: (batch_size, 4, H, H, H)
            u_y: (batch_size, embedding_dim) target embedding
            t: block index (0..T-1)
        Returns:
            loss: scalar
        """
        batch_size = x.shape[0]
        device = x.device
        T = self.config.diffusion.T
        eta = self.config.diffusion.eta

        x_cond = self.encode_condition(x)

        # Sample z_{t-1} ~ q(z_{t-1}|y) = N(sqrt(alpha_bar_{t-1}) * u_y, 1 - alpha_bar_{t-1})
        alpha_bar = self.noise_schedule.alpha_bar
        if t == 0:
            alpha_bar_tm1 = torch.tensor(1.0, device=device)
        else:
            alpha_bar_tm1 = alpha_bar[t - 1]

        noise = torch.randn_like(u_y)
        z_tm1 = torch.sqrt(alpha_bar_tm1) * u_y + torch.sqrt(1 - alpha_bar_tm1) * noise

        # Predict clean embedding
        u_hat = self.blocks[t](z_tm1, x_cond)

        # Weighted MSE (eq.3.2)
        snr_weight = self.noise_schedule.get_snr_weight(t)
        loss = (T / 2) * eta * snr_weight * torch.mean((u_hat - u_y) ** 2)

        return loss
