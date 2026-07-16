"""Final classifier head for NoProp."""
import torch.nn as nn


class ClassifierHead(nn.Module):
    """Classifier on top of final latent z_T.

    p̂(y|z_T) = softmax(Linear(z_T))
    """

    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(latent_dim, n_classes)

    def forward(self, z_T):
        """Returns logits (before softmax)."""
        return self.linear(z_T)
