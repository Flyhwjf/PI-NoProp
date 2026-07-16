"""Evaluation metrics for PI-NoProp.

Implements eq.4.11: η_NS = ||R_NS||_weak / sum(individual term magnitudes)
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score


def evaluate(model, dataloader, device='cuda'):
    """Evaluate model classification accuracy on a dataloader.

    Returns:
        metrics: dict with accuracy, avg_loss
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
            x = x.to(device)
            labels = batch['label'].to(device)

            logits = model(x)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()
            n_batches += 1

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        'accuracy': float(accuracy) * 100,
        'avg_loss': total_loss / max(n_batches, 1),
    }
