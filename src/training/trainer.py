"""Training loop for Physics-Informed NoProp (Algorithm 1)."""
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


class Trainer:
    """Implements Algorithm 1: training of Physics-Informed NoProp.

    Total loss:
        L_total = L_cls + Σ_t (L_diff^(t) + λ_t · L_phys^(t) + β · L_rec^(t))
    """

    def __init__(self, model, decoder, physics_loss, config):
        self.model = model
        self.decoder = decoder
        self.physics_loss = physics_loss
        self.config = config

        # Joint optimizer for all components
        params = (
            list(model.parameters()) +
            list(decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(
            params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        self.lambda_phys = config.physics.lambda_weight
        self.T = config.diffusion.T
        self.device = torch.device(config.device)

    def train_epoch(self, dataloader):
        """Train for one epoch. Returns average losses."""
        self.model.train()
        self.decoder.train()

        total_loss = 0.0
        total_cls = 0.0
        total_diff = 0.0
        total_phys = 0.0
        total_rec = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc='Training')
        for batch in pbar:
            # Move data to device
            x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
            x = x.to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through NoProp (get all intermediate latents)
            logits, z_all = self.model(x, return_all_latents=True)

            # 1. Classification loss
            loss_cls = nn.functional.cross_entropy(logits, labels)

            # 2. Diffusion losses sum over all blocks
            loss_diff = 0.0
            u_y = self.model.label_embed(labels)
            for t in range(self.T):
                loss_diff = loss_diff + self.model.compute_diffusion_loss(x, u_y, t)

            # 3. Physics loss + reconstruction loss summed over all blocks
            loss_phys = 0.0
            loss_rec = 0.0
            for z_t in z_all[1:]:  # skip z_0 (pure noise)
                fields = self.decoder(z_t)
                loss_block, _ = self.physics_loss(fields)
                loss_phys = loss_phys + loss_block
                loss_rec = loss_rec + nn.functional.mse_loss(fields, x)

            # Total loss
            rec_weight = getattr(self.config.training, 'rec_weight', 1.0)
            loss = loss_cls + loss_diff + self.lambda_phys * loss_phys + rec_weight * loss_rec

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls += loss_cls.item()
            total_diff += loss_diff.item()
            total_phys += loss_phys.item()
            total_rec += loss_rec.item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'cls': f'{loss_cls.item():.3f}',
                'phys': f'{loss_phys.item():.4e}',
            })

        return {
            'loss': total_loss / max(n_batches, 1),
            'cls': total_cls / max(n_batches, 1),
            'diff': total_diff / max(n_batches, 1),
            'phys': total_phys / max(n_batches, 1),
            'rec': total_rec / max(n_batches, 1),
        }

    def pretrain_decoder(self, dataloader, n_epochs=20):
        """Pre-train decoder as autoencoder via q(z_t|y).

        Paper sec 4.5: 'The decoder can be pre-trained on a reconstruction
        task (e.g., autoencoding) before end-to-end training.'

        For each batch:
            z_t = √α̅_t · u_y + √(1-α̅_t) · ε   (diffusion posterior)
            decoder(z_t) → fields
            loss = MSE(fields, x)
        """
        self.model.eval()    # model is not trained in this phase
        self.decoder.train()

        optim = torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.config.training.pretrain_lr,
        )

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(dataloader, desc=f'Pre-train decoder epoch {epoch+1}/{n_epochs}')
            for batch in pbar:
                x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
                x = x.to(self.device)
                labels = batch['label'].to(self.device)

                optim.zero_grad()

                # Get label embedding
                u_y = self.model.label_embed(labels)

                # Pick random t (on CPU for indexing alpha_bar which is CPU)
                t = torch.randint(0, self.T, (labels.shape[0],), device='cpu')

                # Sample z_t ~ q(z_t|y) using the noise schedule
                alpha_bar_t = self.model.noise_schedule.alpha_bar[t].to(self.device)
                noise = torch.randn_like(u_y)
                z_t = torch.sqrt(alpha_bar_t.unsqueeze(1)) * u_y + \
                      torch.sqrt(1 - alpha_bar_t.unsqueeze(1)) * noise

                # Decode and compute reconstruction loss
                fields = self.decoder(z_t)
                loss = nn.functional.mse_loss(fields, x)

                loss.backward()
                optim.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({'mse': f'{loss.item():.6f}'})

            avg_loss = total_loss / max(n_batches, 1)
            print(f'  Avg MSE: {avg_loss:.6f}')

        print('Decoder pretraining done.')

    def validate(self, dataloader):
        """Validation loop."""
        self.model.eval()
        self.decoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x = torch.cat([batch['velocity'], batch['pressure'].unsqueeze(1)], dim=1)
                x = x.to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(x)
                loss = nn.functional.cross_entropy(logits, labels)

                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
                total_loss += loss.item()
                n_batches += 1

        accuracy = correct / total * 100
        avg_loss = total_loss / max(n_batches, 1)

        return {'accuracy': accuracy, 'loss': avg_loss}

    def save_checkpoint(self, epoch, path):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        print(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Checkpoint loaded from {path}')
        return checkpoint['epoch']
