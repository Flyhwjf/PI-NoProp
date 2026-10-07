"""Parameter-free prototype readout for NoProp."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeReadout(nn.Module):
    """Classify the final latent by cosine similarity to label prototypes.

    The NoProp target embedding table already contains one prototype per
    class.  Reusing that table removes a separately trained output head while
    keeping the readout in the same representation space as the local blocks.
    """

    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_classes = int(n_classes)

    def forward(self, z_T: torch.Tensor, prototypes: torch.Tensor):
        """Return cosine-similarity logits against class prototypes."""
        if z_T.shape[-1] != self.latent_dim:
            raise ValueError(f'expected latent dimension {self.latent_dim}, '
                             f'got {z_T.shape[-1]}')
        if prototypes.shape != (self.n_classes, self.latent_dim):
            raise ValueError('prototype table has an incompatible shape')
        return F.normalize(z_T, dim=-1) @ F.normalize(prototypes, dim=-1).t()


# Keep the old public name importable for downstream scripts.  The current
# implementation is now parameter-free and performs prototype readout.
ClassifierHead = PrototypeReadout
