"""Decoder D_phi: maps latent z_t to physical fields (ũ_t, p̃_t) on 32^3 grid."""
import torch
import torch.nn as nn


class FieldDecoder(nn.Module):
    """Decoder shared across all NoProp blocks.

    Maps latent z_t ∈ R^d to physical fields:
        (ũ_t, p̃_t): (batch, 4, 32, 32, 32)
    where channels = [u_x, u_y, u_z, p].

    Architecture:  Linear → Reshape → 3× ConvTranspose3d → Conv3d
        latent(128) → FC → (base_c, 4, 4, 4)
           → UpBlock(base_c, base_c)       → (base_c, 8, 8, 8)
           → UpBlock(base_c, base_c//2)    → (base_c//2, 16, 16, 16)
           → UpBlock(base_c//2, base_c//4) → (base_c//4, 32, 32, 32)
           → Conv3d(base_c//4, 4)          → (4, 32, 32, 32)
    """

    def __init__(self, latent_dim=128, base_channels=32, output_channels=4):
        super().__init__()
        c = base_channels

        # FC → initial 3D feature map (c, 4, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, c * 4 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # First upsampling: 4 → 8
        self.up1 = self._make_upblock(c, c, use_bn=True)
        # Second upsampling: 8 → 16
        self.up2 = self._make_upblock(c, c // 2, use_bn=True)
        # Third upsampling: 16 → 32
        self.up3 = self._make_upblock(c // 2, c // 4, use_bn=True)

        self.out_conv = nn.Conv3d(c // 4, output_channels, kernel_size=3, padding=1)

    @staticmethod
    def _make_upblock(in_ch, out_ch, use_bn=True):
        layers = [
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        ]
        if use_bn:
            layers.append(nn.BatchNorm3d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, z_t):
        """Decode latent to physical fields.

        Args:
            z_t: (batch_size, latent_dim)
        Returns:
            fields: (batch_size, 4, 32, 32, 32)
        """
        x = self.fc(z_t)
        x = x.view(x.shape[0], -1, 4, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        fields = self.out_conv(x)
        return fields


class LinearDecoder(nn.Module):
    """Linear decoder for ablation study.
    Maps latent z_t directly to flattened field via single linear layer.
    """

    def __init__(self, latent_dim=128, output_channels=4, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.out_features = output_channels * grid_size ** 3
        self.linear = nn.Linear(latent_dim, self.out_features)

    def forward(self, z_t):
        x = self.linear(z_t)
        return x.view(x.shape[0], -1, self.grid_size, self.grid_size, self.grid_size)


class PhysicsFieldDecoder(nn.Module):
    """Compact latent-to-field decoder for the 16^3 weak integration domain.

    The legacy decoder computes a 32^3 field and immediately discards seven
    eighths of it through a centre crop.  This head stops at 16^3 and is used
    only by the optimized local training path.
    """

    def __init__(self, latent_dim=128, base_channels=32, output_channels=4):
        super().__init__()
        c = base_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, c * 4 * 4 * 4), nn.ReLU(inplace=True),
        )
        self.up1 = FieldDecoder._make_upblock(c, c, use_bn=True)
        self.up2 = FieldDecoder._make_upblock(c, c // 2, use_bn=True)
        self.out_conv = nn.Conv3d(c // 2, output_channels, kernel_size=3, padding=1)

    def forward(self, z_t):
        value = self.fc(z_t).view(z_t.shape[0], -1, 4, 4, 4)
        value = self.up1(value)
        value = self.up2(value)
        return self.out_conv(value)
