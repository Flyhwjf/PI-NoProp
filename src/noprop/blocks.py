"""NoPropBlock: individual denoising block."""
import torch
import torch.nn as nn


class NoPropBlock(nn.Module):
    """A single NoProp denoising block.

    Input: z_{t-1} (latent) and x_cond (conditioning embedding)
    Output: û_t = û_θt(z_{t-1}, x_cond), the predicted clean label embedding.

    Architecture: MLP with configurable depth and width.
    """

    def __init__(self, input_dim, condition_dim, hidden_dim=256,
                 n_hidden_layers=3, activation='relu'):
        super().__init__()

        total_input_dim = input_dim + condition_dim
        act = nn.ReLU if activation == 'relu' else nn.GELU

        layers = []
        curr_dim = total_input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(act())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z_prev, x_cond):
        """Forward pass.

        Args:
            z_prev: (batch_size, input_dim)
            x_cond: (batch_size, condition_dim)
        Returns:
            u_hat: (batch_size, input_dim)
        """
        inp = torch.cat([z_prev, x_cond], dim=-1)
        return self.net(inp)
