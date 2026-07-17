"""Reusable one-time pretraining for frozen local-NoProp components."""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _field(batch, device):
    value = batch.get('field')
    if value is None:
        value = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
    return value.to(device, non_blocking=True)


def _ns_terms(batch, device):
    value = batch.get('ns_terms')
    return None if value is None else value.to(device, non_blocking=True)


def _match_spatial_size(target, prediction):
    if target.shape[-3:] == prediction.shape[-3:]:
        return target
    starts = [(source - destination) // 2
              for source, destination in zip(target.shape[-3:], prediction.shape[-3:])]
    sizes = prediction.shape[-3:]
    return target[..., starts[0]:starts[0] + sizes[0],
                  starts[1]:starts[1] + sizes[1],
                  starts[2]:starts[2] + sizes[2]]


def _autocast(config, device):
    enabled = bool(config.training.use_amp and device.type == 'cuda')
    if not enabled:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if config.training.amp_dtype == 'bfloat16' else torch.float16
    return torch.autocast('cuda', dtype=dtype)


def pretrain_encoder(model, dataloader, config, epochs=None, val_loader=None,
                     patience=6):
    """Supervised pretraining for the shared condition encoder."""
    device = torch.device(config.device)
    epochs = int(epochs or config.training.classifier_epochs)
    head = nn.Linear(config.noprop.condition_dim, config.data.n_classes).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(head.parameters()),
        lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(config.training.use_amp
                                                   and device.type == 'cuda'))
    model.encoder.train()
    best_state = None
    best_accuracy = -1.0
    stale_epochs = 0
    for epoch in range(epochs):
        correct = total = 0
        for batch in dataloader:
            fields = _field(batch, device)
            labels = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                logits = head(model.encode_condition(fields, _ns_terms(batch, device)))
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += int((logits.argmax(-1) == labels).sum())
            total += labels.shape[0]
        train_accuracy = 100 * correct / max(total, 1)
        val_accuracy = train_accuracy
        if val_loader is not None:
            model.encoder.eval()
            head.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    fields = _field(batch, device)
                    labels = batch['label'].to(device, non_blocking=True)
                    logits = head(model.encode_condition(
                        fields, _ns_terms(batch, device)))
                    val_correct += int((logits.argmax(-1) == labels).sum())
                    val_total += labels.shape[0]
            val_accuracy = 100 * val_correct / max(val_total, 1)
            model.encoder.train()
            head.train()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = (copy.deepcopy(model.encoder.state_dict()),
                          copy.deepcopy(head.state_dict()))
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            print(f'encoder pretrain {epoch+1}/{epochs}: train={train_accuracy:.1f}% '
                  f'val={val_accuracy:.1f}%')
        if val_loader is not None and stale_epochs >= patience:
            print(f'encoder early stop at {epoch+1}; best val={best_accuracy:.1f}%')
            break
    if best_state is not None:
        model.encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    return head


@torch.no_grad()
def align_label_embeddings_to_encoder(model, dataloader, config):
    """Set each target embedding to its encoder-feature class centroid."""
    device = torch.device(config.device)
    model.encoder.eval()
    sums = torch.zeros(config.data.n_classes, config.noprop.embedding_dim,
                       device=device)
    counts = torch.zeros(config.data.n_classes, device=device)
    for batch in dataloader:
        fields = _field(batch, device)
        labels = batch['label'].to(device, non_blocking=True)
        features = model.encode_condition(fields, _ns_terms(batch, device))
        sums.index_add_(0, labels, features)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
    centroids = sums / counts.clamp_min(1).unsqueeze(1)
    centroids = F.normalize(centroids, dim=-1) * (centroids.shape[-1] ** 0.5)
    embedding = getattr(model.label_embed, 'embed', None)
    if not isinstance(embedding, nn.Embedding):
        raise TypeError('Optimized training requires an nn.Embedding target table')
    embedding.weight.copy_(centroids)
    embedding.weight.requires_grad_(False)
    return centroids


def pretrain_decoder(model, decoder, dataloader, config, epochs=None):
    """Pretrain a field auto-decoder on frozen encoder features."""
    device = torch.device(config.device)
    epochs = int(epochs or config.training.n_pretrain_epochs)
    model.encoder.eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.label_embed.eval()
    for parameter in model.label_embed.parameters():
        parameter.requires_grad_(False)
    decoder.train()
    for parameter in decoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=config.training.pretrain_lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(config.training.use_amp
                                                   and device.type == 'cuda'))
    for epoch in range(epochs):
        loss_sum = samples = 0
        for batch in dataloader:
            fields = _field(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = model.encode_condition(fields, _ns_terms(batch, device))
                # Mild latent noise makes the frozen decoder useful around,
                # not only exactly on, the encoder manifold.
                z = z + 0.02 * torch.randn_like(z)
            with _autocast(config, device):
                reconstruction = decoder(z)
                if reconstruction.ndim == 6:
                    target = batch['sequence'].to(device, non_blocking=True)
                else:
                    target = fields
                loss = F.mse_loss(
                    reconstruction, _match_spatial_size(target, reconstruction))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * fields.shape[0]
            samples += fields.shape[0]
        if epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(f'decoder pretrain {epoch+1}/{epochs}: '
                  f'mse={loss_sum/max(samples,1):.5f}')


def save_shared_components(model, decoder, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'encoder': model.encoder.state_dict(),
        'label_embed': model.label_embed.state_dict(),
        'decoder': decoder.state_dict(),
    }, path)


def load_shared_components(model, decoder, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.label_embed.load_state_dict(checkpoint['label_embed'])
    decoder.load_state_dict(checkpoint['decoder'])
