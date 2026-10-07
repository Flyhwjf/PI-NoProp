"""Complete NoProp model: T local blocks, prototype readout, and losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import NoiseSchedule
from .embedding import LabelEmbedding
from .blocks import NoPropBlock
from .classifier import PrototypeReadout


def _make_spatial_backbone(input_channels, condition_dim):
    return nn.Sequential(
        nn.Conv3d(input_channels, 16, 3, padding=1), nn.GELU(),
        nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GELU(),
        nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GELU(),
        nn.AdaptiveAvgPool3d((2, 2, 2)), nn.Flatten(),
        nn.Linear(64 * 2 * 2 * 2, condition_dim),
    )


class DualScaleSpatialEncoder(nn.Module):
    """Encode a centred target cube together with its larger context.

    Both scales share the 3-D backbone.  The fusion layer starts as an exact
    pass-through of the centred target feature, so expanding the field of view
    does not initially dilute the signal that defines the prediction target.
    Supervised condition pretraining can then learn how much surrounding
    context to add.
    """

    def __init__(self, input_channels, condition_dim, target_size):
        super().__init__()
        self.target_size = int(target_size)
        self.backbone = _make_spatial_backbone(input_channels, condition_dim)
        self.fusion = nn.Linear(2*condition_dim, condition_dim)
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, :condition_dim].copy_(
                torch.eye(condition_dim))

    def forward(self, fields):
        starts = [(length-self.target_size)//2
                  for length in fields.shape[-3:]]
        local = fields[..., starts[0]:starts[0]+self.target_size,
                       starts[1]:starts[1]+self.target_size,
                       starts[2]:starts[2]+self.target_size]
        local_feature = self.backbone(local)
        context_feature = self.backbone(fields)
        return self.fusion(torch.cat([local_feature, context_feature], dim=-1))


class ResidualContextSpatialEncoder(nn.Module):
    """Add bounded large-cube context to an undisturbed centre pathway.

    The centre and context use independent backbones, so gradients produced by
    the surrounding 32^3 field cannot overwrite filters used for the 16^3
    target cube.  A zero-initialized, bounded feature-wise gate makes the
    initial representation exactly the centre feature and limits context to a
    residual correction during supervised condition pretraining.
    """

    def __init__(self, input_channels, condition_dim, target_size,
                 max_context_scale=0.25):
        super().__init__()
        self.target_size = int(target_size)
        self.max_context_scale = float(max_context_scale)
        self.context_enabled = True
        self.local_backbone = _make_spatial_backbone(
            input_channels, condition_dim)
        self.context_backbone = _make_spatial_backbone(
            input_channels, condition_dim)
        self.context_projection = nn.Linear(
            condition_dim, condition_dim, bias=False)
        self.context_gate = nn.Parameter(torch.zeros(condition_dim))

    def forward(self, fields):
        starts = [(length-self.target_size)//2
                  for length in fields.shape[-3:]]
        local = fields[..., starts[0]:starts[0]+self.target_size,
                       starts[1]:starts[1]+self.target_size,
                       starts[2]:starts[2]+self.target_size]
        local_feature = self.local_backbone(local)
        if not self.context_enabled:
            return local_feature
        context_feature = self.context_projection(
            self.context_backbone(fields))
        context_feature = F.normalize(context_feature, dim=-1)
        local_scale = local_feature.norm(dim=-1, keepdim=True).detach()
        gate = self.max_context_scale * torch.tanh(self.context_gate)
        return local_feature + gate * context_feature * local_scale


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

        # Diffusion schedule (moved to device in forward/training)
        self.noise_schedule = NoiseSchedule(T=config.diffusion.T, s=config.diffusion.s)

        # Label embedding
        self.label_embed = LabelEmbedding(
            n_classes=config.data.n_classes,
            embedding_dim=d_cfg.embedding_dim,
            learnable=True,
        )

        # Preserve local derivatives before pooling.  The previous direct
        # 16^3 -> 4^3 average erased precisely the pressure/velocity-gradient
        # signal that controls short-horizon energy change.
        target_size = (config.data.target_subdomain_size
                       or config.data.subdomain_size)
        if config.data.subdomain_size > target_size:
            if d_cfg.spatial_context_mode == 'dual_scale':
                self.encoder = DualScaleSpatialEncoder(
                    config.data.n_channels, d_cfg.condition_dim, target_size)
            elif d_cfg.spatial_context_mode in (
                    'residual_dual_scale', 'residual_warmstart'):
                self.encoder = ResidualContextSpatialEncoder(
                    config.data.n_channels, d_cfg.condition_dim, target_size)
            else:
                self.encoder = _make_spatial_backbone(
                    config.data.n_channels, d_cfg.condition_dim)
        else:
            self.encoder = _make_spatial_backbone(
                config.data.n_channels, d_cfg.condition_dim)
        coefficients = config.physics.condition_coefficients
        if coefficients is None:
            self.register_buffer('physics_coefficients', torch.zeros(3))
            self.physics_condition_enabled = False
        else:
            self.register_buffer(
                'physics_coefficients', torch.as_tensor(coefficients, dtype=torch.float32))
            self.physics_condition_enabled = True
        means = torch.as_tensor(config.data.ns_term_means, dtype=torch.float32)
        covariance = torch.as_tensor(config.data.ns_term_covariance, dtype=torch.float32)
        physics_mean = self.physics_coefficients @ means
        physics_variance = self.physics_coefficients @ covariance @ self.physics_coefficients
        self.register_buffer('physics_mean', physics_mean)
        self.register_buffer('physics_std', physics_variance.clamp_min(1e-12).sqrt())
        self.physics_encoder = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, d_cfg.condition_dim))
        self.condition_fusion = nn.Linear(2*d_cfg.condition_dim, d_cfg.condition_dim)

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

        # Parameter-free classifier: final latents are read against the
        # frozen class prototypes stored in label_embed.
        self.classifier = PrototypeReadout(
            d_cfg.embedding_dim, config.data.n_classes)

    def _label_prototypes(self):
        embedding = self.label_embed.embed
        if isinstance(embedding, nn.Embedding):
            return embedding.weight
        return embedding

    def encode_condition(self, x, ns_terms=None):
        spatial = self.encoder(x)
        if self.physics_condition_enabled and ns_terms is not None:
            rate = ns_terms @ self.physics_coefficients
            rate = ((rate-self.physics_mean)/self.physics_std).unsqueeze(-1)
            physical = self.physics_encoder(rate)
        else:
            physical = torch.zeros_like(spatial)
        condition = self.condition_fusion(torch.cat([spatial, physical], dim=-1))
        if getattr(self.config.noprop, 'normalize_condition', False):
            condition = F.normalize(condition, dim=-1) * (condition.shape[-1] ** 0.5)
        return condition

    def forward(self, x, ns_terms=None, return_all_latents=False):
        """Inference: denoise from z_0 through all blocks.

        Args:
            x: (batch_size, 4, H, H, H) input field
            return_all_latents: return list of all intermediate z_t
        Returns:
            cosine-readout logits or (logits, z_all)
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode input to condition vector
        x_cond = self.encode_condition(x, ns_terms)  # (batch_size, condition_dim)

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

        logits = self.classifier(z, self._label_prototypes())

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
        alpha_bar_tm1 = self.noise_schedule.get_input_signal(t)

        noise = torch.randn_like(u_y)
        z_tm1 = torch.sqrt(alpha_bar_tm1) * u_y + torch.sqrt(1 - alpha_bar_tm1) * noise

        # Predict clean embedding
        u_hat = self.blocks[t](z_tm1, x_cond)

        # Weighted MSE (eq.3.2)
        snr_weight = self.noise_schedule.get_snr_weight(t)
        loss = (T / 2) * eta * snr_weight * torch.mean((u_hat - u_y) ** 2)

        return loss
