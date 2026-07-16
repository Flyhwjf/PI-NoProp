"""Strictly local, time-sampled training for Physics-Informed NoProp."""
from __future__ import annotations

import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def configure_torch(config):
    """Apply safe RTX/CUDA performance settings in one place."""
    if torch.cuda.is_available():
        enabled = bool(config.training.use_tf32)
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')


class LocalNoPropTrainer:
    """Train each denoising block with an independent local objective.

    No computation graph crosses block boundaries.  A uniformly sampled block
    gives an unbiased estimate of the sum over all diffusion/physics losses.
    """

    def __init__(self, model, decoder, physics_loss, config):
        self.model = model
        self.decoder = decoder
        self.physics_loss = physics_loss
        self.config = config
        self.device = torch.device(config.device)
        self.T = config.diffusion.T
        configure_torch(config)
        self.model.noise_schedule.to(self.device)

        self.amp_enabled = bool(config.training.use_amp and self.device.type == 'cuda')
        self.amp_dtype = (torch.bfloat16 if config.training.amp_dtype == 'bfloat16'
                          else torch.float16)
        self.scalers = [torch.amp.GradScaler('cuda', enabled=self.amp_enabled)
                        for _ in range(self.T)]

        optimizer_kwargs = dict(lr=config.training.lr,
                                weight_decay=config.training.weight_decay)
        if config.training.fused_optimizer and self.device.type == 'cuda':
            optimizer_kwargs['fused'] = True
        self.block_optimizers = []
        for block in self.model.blocks:
            try:
                optimizer = torch.optim.AdamW(block.parameters(), **optimizer_kwargs)
            except (TypeError, RuntimeError):
                optimizer_kwargs.pop('fused', None)
                optimizer = torch.optim.AdamW(block.parameters(), **optimizer_kwargs)
            self.block_optimizers.append(optimizer)

        self.classifier_optimizer = torch.optim.AdamW(
            self.model.classifier.parameters(), lr=config.training.lr,
            weight_decay=config.training.weight_decay)
        self.block_updates = torch.zeros(self.T, dtype=torch.long)
        self._freeze_shared_modules()

    def _freeze_shared_modules(self):
        for module in (self.model.encoder, self.model.label_embed, self.decoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _autocast(self):
        if not self.amp_enabled:
            return contextlib.nullcontext()
        return torch.autocast('cuda', dtype=self.amp_dtype)

    def _batch(self, batch):
        field = batch.get('field')
        if field is None:
            field = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
        return (field.to(self.device, non_blocking=True),
                batch['label'].to(self.device, non_blocking=True))

    @staticmethod
    def _match_spatial_size(target, prediction):
        if target.shape[-3:] == prediction.shape[-3:]:
            return target
        starts = [(source - destination) // 2
                  for source, destination in zip(target.shape[-3:], prediction.shape[-3:])]
        sizes = prediction.shape[-3:]
        return target[..., starts[0]:starts[0] + sizes[0],
                      starts[1]:starts[1] + sizes[1],
                      starts[2]:starts[2] + sizes[2]]

    @torch.no_grad()
    def _condition_and_target(self, fields, labels):
        condition = self.model.encode_condition(fields)
        target = self.model.label_embed(labels)
        return condition, target

    def train_local_step(self, batch, block_index):
        """Update exactly one block and return scalar diagnostics."""
        fields_true, labels = self._batch(batch)
        condition, target = self._condition_and_target(fields_true, labels)
        t = int(block_index)
        schedule = self.model.noise_schedule
        if t == 0:
            alpha_bar_prev = torch.ones((), device=self.device)
        else:
            alpha_bar_prev = schedule.alpha_bar[t - 1]
        z_prev = (alpha_bar_prev.sqrt() * target
                  + (1 - alpha_bar_prev).sqrt() * torch.randn_like(target))

        block = self.model.blocks[t]
        optimizer = self.block_optimizers[t]
        scaler = self.scalers[t]
        optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            prediction = block(z_prev, condition)
            snr_weight = schedule.get_snr_weight(t)
            diffusion = ((self.T / 2) * self.config.diffusion.eta
                         * snr_weight * F.mse_loss(prediction, target))
            a_t, b_t, _ = schedule.get_coeffs(t)
            z_t = a_t * prediction + b_t * z_prev
            needs_decoder = (
                self.config.training.local_rec_weight > 0
                or (self.config.physics.lambda_weight > 0
                    and (self.config.physics.use_continuity
                         or self.config.physics.use_pressure_poisson))
            )
            reconstructed = self.decoder(z_t) if needs_decoder else None
        if reconstructed is not None:
            # Spatial weak integration explicitly accumulates in FP32.
            physics, physics_metrics = self.physics_loss(reconstructed)
            reconstruction = F.mse_loss(
                reconstructed.float(),
                self._match_spatial_size(fields_true, reconstructed).float())
        else:
            physics = torch.zeros((), device=self.device)
            reconstruction = torch.zeros((), device=self.device)
            physics_metrics = {
                'eta_div': torch.zeros((), device=self.device),
                'eta_pp': torch.zeros((), device=self.device),
            }
        local_loss = self.T * (
            diffusion.float()
            + self.config.physics.lambda_weight * physics
            + self.config.training.local_rec_weight * reconstruction
        )
        scaler.scale(local_loss).backward()
        if self.config.training.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(block.parameters(),
                                           self.config.training.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        self.block_updates[t] += 1
        return {
            'loss': float(local_loss.detach()),
            'diff': float(diffusion.detach()),
            'phys': float(physics.detach()),
            'rec': float(reconstruction.detach()),
            'eta_div': float(physics_metrics['eta_div']),
            'eta_pp': float(physics_metrics['eta_pp']),
            'block': t,
        }

    def train_local_epoch(self, dataloader, start_offset=0):
        """Use a shuffled balanced block schedule over an epoch."""
        self.model.train()
        self._freeze_shared_modules()
        totals = {key: 0.0 for key in ('loss', 'diff', 'phys', 'rec',
                                       'eta_div', 'eta_pp')}
        count = 0
        order = torch.randperm(self.T).tolist()
        for batch_index, batch in enumerate(dataloader):
            if batch_index % self.T == 0 and batch_index:
                order = torch.randperm(self.T).tolist()
            block_index = order[(batch_index + start_offset) % self.T]
            stats = self.train_local_step(batch, block_index)
            for key in totals:
                totals[key] += stats[key]
            count += 1
        return {key: value / max(count, 1) for key, value in totals.items()}

    def train_blocks(self, dataloader, steps_per_block=None, log_interval=100):
        target = int(steps_per_block or self.config.training.local_steps_per_block)
        iterator = iter(dataloader)
        history = []
        start = time.perf_counter()
        while int(self.block_updates.min()) < target:
            eligible = torch.where(self.block_updates < target)[0]
            t = int(eligible[torch.randint(len(eligible), ())])
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            stats = self.train_local_step(batch, t)
            history.append(stats)
            total_updates = int(self.block_updates.sum())
            if log_interval and total_updates % log_interval == 0:
                elapsed = time.perf_counter() - start
                print(f'local steps={total_updates} min/block={int(self.block_updates.min())} '
                      f'loss={stats["loss"]:.4f} {total_updates/elapsed:.1f} step/s')
        return history

    def train_classifier_epoch(self, dataloader):
        for block in self.model.blocks:
            block.eval()
            for parameter in block.parameters():
                parameter.requires_grad_(False)
        self.model.classifier.train()
        total_loss = correct = total = 0
        for batch in dataloader:
            fields, labels = self._batch(batch)
            self.classifier_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                condition = self.model.encode_condition(fields)
                z = torch.randn(labels.shape[0], self.config.noprop.embedding_dim,
                                device=self.device)
                for t, block in enumerate(self.model.blocks):
                    a_t, b_t, _ = self.model.noise_schedule.get_coeffs(t)
                    z = a_t * block(z, condition) + b_t * z
            with self._autocast():
                logits = self.model.classifier(z.detach())
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            self.classifier_optimizer.step()
            total_loss += float(loss.detach()) * labels.shape[0]
            correct += int((logits.argmax(-1) == labels).sum())
            total += labels.shape[0]
        return {'loss': total_loss / max(total, 1),
                'accuracy': 100.0 * correct / max(total, 1)}

    @torch.no_grad()
    def evaluate(self, dataloader, include_physics=False):
        devices = ([torch.cuda.current_device()]
                   if self.device.type == 'cuda' else [])
        # NoProp starts inference from Gaussian noise.  Forking the RNG makes
        # validation repeatable without perturbing the subsequent train RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed + 10_000)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed_all(self.config.seed + 10_000)
            self.model.eval()
            self.decoder.eval()
            correct = total = 0
            loss_sum = eta_div = eta_pp = 0.0
            batches = 0
            for batch in dataloader:
                fields, labels = self._batch(batch)
                logits, latents = self.model(fields, return_all_latents=True)
                loss_sum += float(F.cross_entropy(logits, labels)) * labels.shape[0]
                correct += int((logits.argmax(-1) == labels).sum())
                total += labels.shape[0]
                if include_physics:
                    reconstructed = self.decoder(latents[-1])
                    metrics = self.physics_loss.evaluate_metrics(reconstructed)
                    eta_div += metrics['eta_div']
                    eta_pp += metrics['eta_pp']
                    batches += 1
            result = {'loss': loss_sum / max(total, 1),
                      'accuracy': 100.0 * correct / max(total, 1)}
            if include_physics:
                result.update(eta_div=eta_div / max(batches, 1),
                              eta_pp=eta_pp / max(batches, 1))
            return result

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'block_updates': self.block_updates,
            'config': self.config,
            'extra': extra or {},
        }, path)
