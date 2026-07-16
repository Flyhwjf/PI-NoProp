"""Label embedding for NoProp."""
import torch
import torch.nn as nn


class LabelEmbedding(nn.Module):
    """Maps class index to dense embedding vector u_y.

    Supports both learnable and fixed one-hot embeddings.
    """

    def __init__(self, n_classes, embedding_dim, learnable=True):
        super().__init__()
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.learnable = learnable

        if learnable:
            self.embed = nn.Embedding(n_classes, embedding_dim)
        else:
            self.register_buffer('embed', torch.eye(n_classes))

    def forward(self, labels):
        """Get embedding u_y for class labels.

        Args:
            labels: (batch_size,) long tensor
        Returns:
            u_y: (batch_size, embedding_dim)
        """
        if self.learnable:
            return self.embed(labels)
        else:
            return self.embed[labels]
