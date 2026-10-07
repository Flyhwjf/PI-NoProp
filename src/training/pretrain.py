"""Reusable one-time pretraining for frozen local-NoProp components."""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _condition_modules(model):
    """Shared modules that turn fields and equation rates into a condition."""
    return (model.encoder, model.physics_encoder, model.condition_fusion)


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


@torch.no_grad()
def build_condition_cache(model, dataloader, config):
    """Encode every indexed sample once after the condition path is frozen.

    HIT datasets expose their stable global sample index as ``idx``.  A dense
    lookup table is tiny (roughly 0.5 MiB for 960 x 128 float32 values) and
    removes repeated 3-D encoder passes during decoder and local-block
    training.  Generic loaders without stable indices transparently fall back
    to on-demand encoding.
    """
    if not bool(getattr(config.training, 'cache_conditions', False)):
        return None
    dataset_indices = getattr(dataloader.dataset, 'indices', None)
    if dataset_indices is None or len(dataset_indices) == 0:
        return None

    device = torch.device(config.device)
    cache_device = (device if bool(getattr(config.data, 'cache_on_device', False))
                    else torch.device('cpu'))
    max_index = int(np.asarray(dataset_indices).max())
    values = torch.empty(
        max_index+1, config.noprop.condition_dim, dtype=torch.float32,
        device=cache_device,
        pin_memory=(cache_device.type == 'cpu' and device.type == 'cuda'))
    valid = torch.zeros(max_index+1, dtype=torch.bool, device=cache_device)

    for module in _condition_modules(model):
        module.eval()
    rng_devices = ([torch.cuda.current_device()]
                   if device.type == 'cuda' else [])
    # Iterating a shuffled loader consumes RNG state; restore it so enabling
    # the cache does not alter the subsequent local-training schedule.
    with torch.random.fork_rng(devices=rng_devices):
        for batch in dataloader:
            if 'idx' not in batch:
                return None
            fields = _field(batch, device)
            conditions = model.encode_condition(
                fields, _ns_terms(batch, device)).detach().float()
            indices = batch['idx'].long().to(cache_device)
            values.index_copy_(0, indices, conditions.to(cache_device))
            valid.index_fill_(0, indices, True)

    expected = torch.as_tensor(dataset_indices, dtype=torch.long,
                               device=cache_device)
    if not bool(valid.index_select(0, expected).all()):
        raise RuntimeError('condition cache is incomplete')
    return {'values': values, 'valid': valid}


def lookup_condition_cache(cache, batch, device):
    """Return cached conditions for one indexed batch."""
    if cache is None or 'idx' not in batch:
        return None
    values, valid = cache['values'], cache['valid']
    indices = batch['idx'].long().to(values.device)
    if not bool(valid.index_select(0, indices).all()):
        raise KeyError('batch contains an index absent from the condition cache')
    conditions = values.index_select(0, indices)
    return conditions.to(device, non_blocking=True)


def pretrain_encoder(model, dataloader, config, epochs=None, val_loader=None,
                     patience=6):
    """Supervised pretraining for the shared condition encoder."""
    device = torch.device(config.device)
    epochs = int(epochs or config.training.classifier_epochs)
    head = nn.Linear(config.noprop.condition_dim, config.data.n_classes).to(device)
    condition_modules = _condition_modules(model)
    for module in condition_modules:
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [parameter for module in condition_modules for parameter in module.parameters()]
        + list(head.parameters()),
        lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(config.training.use_amp
                                                   and device.type == 'cuda'))
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
            for module in condition_modules:
                module.eval()
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
            for module in condition_modules:
                module.train()
            head.train()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = {
                'encoder': copy.deepcopy(model.encoder.state_dict()),
                'physics_encoder': copy.deepcopy(model.physics_encoder.state_dict()),
                'condition_fusion': copy.deepcopy(model.condition_fusion.state_dict()),
                'head': copy.deepcopy(head.state_dict()),
            }
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
        model.encoder.load_state_dict(best_state['encoder'])
        model.physics_encoder.load_state_dict(best_state['physics_encoder'])
        model.condition_fusion.load_state_dict(best_state['condition_fusion'])
        head.load_state_dict(best_state['head'])
    return head


@torch.no_grad()
def align_label_embeddings_to_encoder(model, dataloader, config):
    """Set each target embedding to its encoder-feature class centroid."""
    device = torch.device(config.device)
    for module in _condition_modules(model):
        module.eval()
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


def initialize_residual_context_from_local(model):
    """Copy the trained centre backbone into a zero-gated context branch."""
    encoder = model.encoder
    encoder.context_backbone.load_state_dict(encoder.local_backbone.state_dict())
    with torch.no_grad():
        encoder.context_gate.zero_()
        encoder.context_projection.weight.copy_(torch.eye(
            encoder.context_projection.out_features,
            device=encoder.context_projection.weight.device,
            dtype=encoder.context_projection.weight.dtype))
    encoder.context_enabled = True


def load_residual_context_warmstart(model, decoder, path, device,
                                    load_decoder=True):
    """Initialize a residual 32^3 encoder from a matched 16^3 pathway."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    encoder = model.encoder
    required_attributes = ('local_backbone', 'context_backbone',
                           'context_projection', 'context_gate')
    missing_attributes = [name for name in required_attributes
                          if not hasattr(encoder, name)]
    if missing_attributes:
        raise TypeError(
            'residual warm-start requires a residual context encoder; missing '
            f'{missing_attributes}')
    encoder.local_backbone.load_state_dict(checkpoint['encoder'])
    encoder.context_backbone.load_state_dict(checkpoint['encoder'])
    model.physics_encoder.load_state_dict(checkpoint['physics_encoder'])
    model.condition_fusion.load_state_dict(checkpoint['condition_fusion'])
    model.label_embed.load_state_dict(checkpoint['label_embed'])
    if load_decoder:
        decoder.load_state_dict(checkpoint['decoder'])
    initialize_residual_context_from_local(model)


@torch.no_grad()
def _context_prototype_metrics(model, dataloader, device, temperature):
    model.eval()
    prototypes = F.normalize(model._label_prototypes().detach(), dim=-1)
    correct = total = 0
    loss_sum = 0.0
    for batch in dataloader:
        fields = _field(batch, device)
        labels = batch['label'].to(device, non_blocking=True)
        conditions = model.encode_condition(fields, _ns_terms(batch, device))
        logits = F.normalize(conditions, dim=-1) @ prototypes.t() / temperature
        loss_sum += float(F.cross_entropy(logits, labels)) * labels.shape[0]
        correct += int((logits.argmax(-1) == labels).sum())
        total += labels.shape[0]
    return {'accuracy': 100.0*correct/max(total, 1),
            'loss': loss_sum/max(total, 1)}


def adapt_residual_context(model, train_loader, val_loader, config, epochs=20,
                           patience=6, temperature=0.1):
    """Fit only the bounded context residual, retaining the best val state.

    The zero-gate centre model is evaluated before the first update and is a
    valid early-stopping candidate.  Context is therefore retained only when
    it improves validation accuracy, or validation loss at equal accuracy.
    """
    device = torch.device(config.device)
    encoder = model.encoder
    for module in _condition_modules(model):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    for module in (encoder.context_backbone, encoder.context_projection):
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    encoder.context_gate.requires_grad_(True)
    trainable = [parameter for parameter in encoder.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=min(config.training.lr, 3e-4),
        weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler(
        'cuda', enabled=(config.training.use_amp and device.type == 'cuda'))
    baseline = _context_prototype_metrics(
        model, val_loader, device, temperature)
    best_metrics = dict(baseline)
    best_state = copy.deepcopy(encoder.state_dict())
    stale_epochs = 0
    prototypes = F.normalize(model._label_prototypes().detach(), dim=-1)
    for epoch in range(int(epochs)):
        encoder.context_backbone.train()
        encoder.context_projection.train()
        for batch in train_loader:
            fields = _field(batch, device)
            labels = batch['label'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                conditions = model.encode_condition(
                    fields, _ns_terms(batch, device))
                logits = (F.normalize(conditions, dim=-1)
                          @ prototypes.t() / temperature)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        metrics = _context_prototype_metrics(
            model, val_loader, device, temperature)
        improved = (
            metrics['accuracy'] > best_metrics['accuracy']
            or (metrics['accuracy'] == best_metrics['accuracy']
                and metrics['loss'] < best_metrics['loss']))
        if improved:
            best_metrics = dict(metrics)
            best_state = copy.deepcopy(encoder.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == int(epochs)-1 or (epoch+1) % 5 == 0:
            print(f'context adapt {epoch+1}/{epochs}: '
                  f'val={metrics["accuracy"]:.1f}% loss={metrics["loss"]:.4f}')
        if stale_epochs >= patience:
            print(f'context adaptation early stop at {epoch+1}')
            break
    encoder.load_state_dict(best_state)
    for module in _condition_modules(model):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return {
        'baseline_validation': baseline,
        'selected_validation': best_metrics,
        'gate_norm': float(encoder.context_gate.detach().norm()),
    }


def pretrain_decoder(model, decoder, dataloader, config, epochs=None):
    """Pretrain a field auto-decoder on frozen encoder features."""
    device = torch.device(config.device)
    epochs = int(epochs or config.training.n_pretrain_epochs)
    for module in _condition_modules(model):
        module.eval()
        for parameter in module.parameters():
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
    condition_cache = build_condition_cache(model, dataloader, config)
    for epoch in range(epochs):
        loss_sum = samples = 0
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = lookup_condition_cache(condition_cache, batch, device)
                if z is None:
                    fields = _field(batch, device)
                    z = model.encode_condition(fields, _ns_terms(batch, device))
                # Mild latent noise makes the frozen decoder useful around,
                # not only exactly on, the encoder manifold.
                z = z + 0.02 * torch.randn_like(z)
            with _autocast(config, device):
                reconstruction = decoder(z)
                if reconstruction.ndim == 6:
                    target = batch['sequence'].to(device, non_blocking=True)
                else:
                    fields = _field(batch, device)
                    target = fields
                loss = F.mse_loss(
                    reconstruction, _match_spatial_size(target, reconstruction))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * z.shape[0]
            samples += z.shape[0]
        if epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            print(f'decoder pretrain {epoch+1}/{epochs}: '
                  f'mse={loss_sum/max(samples,1):.5f}')
    return condition_cache


def save_shared_components(model, decoder, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'schema_version': 2,
        'encoder': model.encoder.state_dict(),
        'physics_encoder': model.physics_encoder.state_dict(),
        'condition_fusion': model.condition_fusion.state_dict(),
        'label_embed': model.label_embed.state_dict(),
        'decoder': decoder.state_dict(),
    }, path)


def load_shared_components(model, decoder, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {'encoder', 'physics_encoder', 'condition_fusion',
                'label_embed', 'decoder'}
    missing = sorted(required-checkpoint.keys())
    if missing:
        raise ValueError(
            f'Legacy shared checkpoint {path} lacks trainable-condition state: '
            f'{missing}. Rebuild it with the v4 pipeline.')
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.physics_encoder.load_state_dict(checkpoint['physics_encoder'])
    model.condition_fusion.load_state_dict(checkpoint['condition_fusion'])
    model.label_embed.load_state_dict(checkpoint['label_embed'])
    decoder.load_state_dict(checkpoint['decoder'])


def load_shared_condition_components(model, path, device):
    """Reuse the exact condition pathway while changing decoder architecture."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    required = {'encoder', 'physics_encoder', 'condition_fusion', 'label_embed'}
    missing = sorted(required-checkpoint.keys())
    if missing:
        raise ValueError(f'Shared checkpoint {path} lacks condition state: {missing}')
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.physics_encoder.load_state_dict(checkpoint['physics_encoder'])
    model.condition_fusion.load_state_dict(checkpoint['condition_fusion'])
    model.label_embed.load_state_dict(checkpoint['label_embed'])
