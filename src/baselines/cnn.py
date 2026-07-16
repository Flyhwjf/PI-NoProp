"""Simple 3D CNN baseline for 3D velocity field classification."""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Small 3D CNN for 10-class classification of 3D flow fields.

    Args:
        in_channels: number of input channels (4 for [u_x, u_y, u_z, p])
        n_classes: number of output classes (10)
        grid_size: spatial size of input (default 16, after 3 MaxPool → //8)
    """

    def __init__(self, in_channels=4, n_classes=10, grid_size=16):
        super().__init__()
        final_size = grid_size // 8
        self.features = nn.Sequential(
            # 16^3 → 8^3, 16ch
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # 8^3 → 4^3, 32ch
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            # 4^3 → 2^3, 64ch
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (final_size ** 3), 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
