"""Temporal decoder from a local latent to velocity-pressure trajectories."""
import torch
import torch.nn as nn


def _make_upblock(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class TemporalPhysicsDecoder(nn.Module):
    """Decode one local latent into a compact velocity-pressure time window.

    Full momentum residuals require a temporal field.  Channels are produced
    jointly and reshaped to ``(batch, time, 4, 16, 16, 16)`` so the decoder
    remains a frozen shared map and does not introduce gradients between
    NoProp blocks.
    """

    def __init__(self, latent_dim=128, base_channels=32, n_time=9,
                 output_channels=4):
        super().__init__()
        self.n_time = int(n_time)
        self.output_channels = int(output_channels)
        c = base_channels
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, c*4*4*4), nn.ReLU(inplace=True),
        )
        self.up1 = _make_upblock(c, c)
        self.up2 = _make_upblock(c, c//2)
        self.out_conv = nn.Conv3d(
            c//2, self.n_time*self.output_channels, kernel_size=3, padding=1)

    def forward(self, z_t):
        value = self.fc(z_t).view(z_t.shape[0], -1, 4, 4, 4)
        value = self.up2(self.up1(value))
        value = self.out_conv(value)
        return value.view(value.shape[0], self.n_time, self.output_channels,
                          *value.shape[-3:])
