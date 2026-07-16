"""Standard Physics-Informed Neural Network baseline for 3D Navier-Stokes.

Maps (x,y,z,t) -> (u,v,w,p) via MLP with Fourier feature encoding.
Trained with data loss + NS residual penalty using backprop.
Classification accuracy derived from mean(u_x) quantile binning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class Sin(nn.Module):
    """Periodic activation — well-suited for PDE solution representation."""

    def forward(self, x):
        return torch.sin(x)


class PINNModel(nn.Module):
    """Standard PINN for 3D incompressible Navier-Stokes.

    Maps (x,y,z,t) -> (u,v,w,p) via MLP with Fourier feature encoding.
    Trained with data loss + NS residual penalty using backprop.

    Args:
        config: PINoPropConfig instance
        nu: kinematic viscosity (overrides config.physics.viscosity if given)
    """

    def __init__(self, config, nu=None):
        super().__init__()
        # Fourier feature: gamma(v) = [sin(2pi*B*v), cos(2pi*B*v)]
        #   B ~ N(0, sigma^2), sigma=1.0, m=64 features
        m = 64
        sigma = 1.0
        self.register_buffer('B', torch.randn(m, 4) * sigma)

        # MLP: [m*2, 128, 128, 128, 128, 4]  — reduced for memory
        self.mlp = nn.Sequential(
            nn.Linear(m * 2, 128), Sin(),
            nn.Linear(128, 128), Sin(),
            nn.Linear(128, 128), Sin(),
            nn.Linear(128, 128), Sin(),
            nn.Linear(128, 4),
        )

        self.nu = nu if nu is not None else config.physics.viscosity
        self.subdomain_size = config.data.subdomain_size
        self.n_timesteps = config.data.n_timesteps
        self.n_classes = config.data.n_classes

        # Adaptive loss weights (initialised during first fit step)
        self.lambda_data = 1.0
        self.lambda_phys = 1.0
        self._weights_initialized = False

    def forward(self, coords):
        """Forward pass with Fourier feature encoding.

        Args:
            coords: (N, 4) normalized to [-1, 1], order (x,y,z,t)

        Returns:
            (N, 4) = (u, v, w, p)
        """
        proj = 2.0 * torch.pi * coords @ self.B.T          # (N, m)
        gamma = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2m)
        return self.mlp(gamma)

    def compute_ns_residual(self, coords):
        """Compute NS momentum and continuity residuals via autograd.

        Navier-Stokes (incompressible):
            du/dt + (u·grad)u + grad(p) - nu * laplacian(u) = 0
            div(u) = 0

        Args:
            coords: (N, 4) collocation points [x,y,z,t]

        Returns:
            loss_ns: scalar MSE of momentum residuals
            loss_cont: scalar MSE of continuity residual
        """
        coords = coords.clone().detach().requires_grad_(True)
        out = self(coords)                   # (N, 4)
        u, v, w, p = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

        # ---- first derivatives ----
        grad_u = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        u_x, u_y, u_z, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2], grad_u[:, 3]

        grad_v = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
        v_x, v_y, v_z, v_t = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2], grad_v[:, 3]

        grad_w = torch.autograd.grad(w.sum(), coords, create_graph=True)[0]
        w_x, w_y, w_z, w_t = grad_w[:, 0], grad_w[:, 1], grad_w[:, 2], grad_w[:, 3]

        grad_p = torch.autograd.grad(p.sum(), coords, create_graph=True)[0]
        p_x, p_y, p_z = grad_p[:, 0], grad_p[:, 1], grad_p[:, 2]

        # ---- second derivatives (diagonal Laplacian) ----
        u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
        u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]

        v_xx = torch.autograd.grad(v_x.sum(), coords, create_graph=True)[0][:, 0]
        v_yy = torch.autograd.grad(v_y.sum(), coords, create_graph=True)[0][:, 1]
        v_zz = torch.autograd.grad(v_z.sum(), coords, create_graph=True)[0][:, 2]

        w_xx = torch.autograd.grad(w_x.sum(), coords, create_graph=True)[0][:, 0]
        w_yy = torch.autograd.grad(w_y.sum(), coords, create_graph=True)[0][:, 1]
        w_zz = torch.autograd.grad(w_z.sum(), coords, create_graph=True)[0][:, 2]

        # ---- momentum residuals ----
        # du/dt + (u·∇)u + ∇p - ν∇²u
        r_u = u_t + u * u_x + v * u_y + w * u_z + p_x - self.nu * (u_xx + u_yy + u_zz)
        r_v = v_t + u * v_x + v * v_y + w * v_z + p_y - self.nu * (v_xx + v_yy + v_zz)
        r_w = w_t + u * w_x + v * w_y + w * w_z + p_z - self.nu * (w_xx + w_yy + w_zz)

        loss_ns = torch.mean(r_u ** 2) + torch.mean(r_v ** 2) + torch.mean(r_w ** 2)

        # ---- continuity residual ----
        r_cont = u_x + v_y + w_z
        loss_cont = torch.mean(r_cont ** 2)

        return loss_ns, loss_cont

    def _sample_collocation(self, vel, pres, n_col):
        """Sample n_col collocation points from a batch of subdomains.

        Args:
            vel: (B, 3, H, H, H)
            pres: (B, H, H, H)
            n_col: number of points per sample

        Returns:
            coords: (B*n_col, 4) normalized [-1, 1]
            true: (B*n_col, 4) ground truth (u,v,w,p)
        """
        device = vel.device
        B, _, H, _, _ = vel.shape

        idx = torch.stack([
            torch.randperm(H * H * H, device=device)[:n_col]
            for _ in range(B)
        ])  # (B, n_col)

        z_i = idx % H
        y_i = (idx // H) % H
        x_i = idx // (H * H)

        x_n = 2.0 * x_i.float() / (H - 1) - 1.0
        y_n = 2.0 * y_i.float() / (H - 1) - 1.0
        z_n = 2.0 * z_i.float() / (H - 1) - 1.0
        t_raw = torch.rand(B, n_col, device=device)  # [0, 1]
        t_n = 2.0 * t_raw - 1.0  # normalize to [-1, 1]

        coords = torch.stack([x_n, y_n, z_n, t_n], dim=-1).reshape(-1, 4)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_col).reshape(-1)

        u_t = vel[b_idx, 0, x_i.reshape(-1), y_i.reshape(-1), z_i.reshape(-1)]
        v_t = vel[b_idx, 1, x_i.reshape(-1), y_i.reshape(-1), z_i.reshape(-1)]
        w_t = vel[b_idx, 2, x_i.reshape(-1), y_i.reshape(-1), z_i.reshape(-1)]
        p_t = pres[b_idx, x_i.reshape(-1), y_i.reshape(-1), z_i.reshape(-1)]
        true = torch.stack([u_t, v_t, w_t, p_t], dim=-1)

        return coords, true

    def fit(self, dataloader, n_epochs=150, lr=1e-3, n_col=2048, device='cuda',
            disable_physics=True):
        """Train the PINN with Adam and cosine annealing.

        Loss:  L = lambda_data * MSE(pred, true) +
                     lambda_phys * (L_ns + L_cont)

        Adaptive weights are updated every 50 steps by balancing the
        running magnitudes of data and physics losses.

        Args:
            dataloader: training DataLoader
            n_epochs: number of epochs
            lr: learning rate
            n_col: collocation points per subdomain per step
            device: torch device string
            disable_physics: if True, set lambda_phys=0 (for normalized data
                where the NS equation no longer holds dimensionally)
        """
        self.to(device)
        self.train()

        # Extract quantile bins from the full (un-split) training dataset
        # for classification evaluation.  The dataset stored in the dataloader
        # (after random_split) may be a Subset, so walk back to the full dataset.
        ds = dataloader.dataset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        means = []
        for fpath in ds.subdomain_files:
            data = np.load(fpath)
            vel = data['velocity'].astype(np.float32)
            means.append(float(vel[0, ..., 0].mean()))
        means = np.array(means)
        bins = np.percentile(means, np.linspace(0, 100, self.n_classes + 1))
        bins[-1] += 1e-6
        self.register_buffer('_train_bins', torch.from_numpy(bins[:-1]).float())

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs,
        )

        last_layer = self.mlp[-1].weight
        ema_data = None
        ema_phys = None
        global_step = 0

        pbar = tqdm(range(n_epochs), desc='PINN training', unit='epoch')
        for epoch in pbar:
            epoch_data = 0.0
            epoch_phys = 0.0
            epoch_loss = 0.0
            n_batches = 0

            for batch in dataloader:
                vel = batch['velocity'].to(device)    # (B, 3, H, H, H)
                pres = batch['pressure'].to(device)    # (B, H, H, H)
                B = vel.shape[0]

                coords, true = self._sample_collocation(vel, pres, n_col)

                pred = self(coords)                    # (B*n_col, 4)
                data_loss = F.mse_loss(pred, true)

                if not disable_physics:
                    loss_ns, loss_cont = self.compute_ns_residual(coords)
                    phys_loss = loss_ns + loss_cont

                    if not self._weights_initialized:
                        self.lambda_phys = data_loss.item() / (phys_loss.item() + 1e-8)
                        self._weights_initialized = True
                    elif global_step % 50 == 0:
                        with torch.no_grad():
                            ratio = data_loss.item() / (phys_loss.item() + 1e-8)
                            self.lambda_phys = 0.7 * self.lambda_phys + 0.3 * ratio

                    if global_step % 200 == 0 and global_step > 0:
                        g_data = torch.autograd.grad(
                            data_loss, last_layer, retain_graph=True, create_graph=True)[0]
                        g_phys = torch.autograd.grad(
                            phys_loss, last_layer, retain_graph=True, create_graph=True)[0]
                        g_data_norm = g_data.norm()
                        g_phys_norm = g_phys.norm()
                        g_mean = (g_data_norm + g_phys_norm) / 2.0
                        with torch.no_grad():
                            grad_lambda_phys = torch.sign(g_phys_norm - g_mean) * 0.1
                            self.lambda_phys = max(0.01, self.lambda_phys - grad_lambda_phys.item())

                    total_loss = self.lambda_data * data_loss + self.lambda_phys * phys_loss
                else:
                    phys_loss = torch.tensor(0.0, device=device)
                    total_loss = self.lambda_data * data_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_data += data_loss.item()
                epoch_phys += phys_loss.item()
                epoch_loss += total_loss.item()
                n_batches += 1
                global_step += 1

            scheduler.step()
            avg_data = epoch_data / max(n_batches, 1)
            avg_phys = epoch_phys / max(n_batches, 1)
            pbar.set_postfix({
                'loss': f'{epoch_loss / max(n_batches, 1):.4f}',
                'data': f'{avg_data:.4f}',
                'phys': f'{avg_phys:.4f}',
                'λ_phys': f'{self.lambda_phys:.2f}',
            })

    @torch.no_grad()
    def evaluate(self, dataloader, device='cuda'):
        """Evaluate classification accuracy by quantile binning of mean(u_x).

        Predicts the full velocity field on the 3D grid, computes mean(u_x),
        and maps to 10 classes via quantile binning.

        Args:
            dataloader: validation/test DataLoader
            device: torch device string

        Returns:
            dict with keys 'accuracy' (percentage) and 'avg_loss'
        """
        self.eval()
        H = self.subdomain_size

        all_pred_means = []
        all_true_labels = []
        total_mse = 0.0
        n_samples = 0

        for batch in dataloader:
            vel = batch['velocity']                   # (B, 3, H, H, H)
            pres = batch['pressure']                  # (B, H, H, H)
            labels = batch['label']                   # (B,)
            B = vel.shape[0]

            for i in range(B):
                # build full-grid normalised coordinates
                lin = torch.linspace(-1, 1, H, device=device)
                X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
                coords = torch.stack([
                    X.reshape(-1), Y.reshape(-1), Z.reshape(-1),
                    torch.zeros(H * H * H, device=device),
                ], dim=-1)                             # (H^3, 4)

                pred = self(coords)                   # (H^3, 4)
                u_pred = pred[:, 0:3]                 # (H^3, 3)

                # MSE against ground truth
                u_true = vel[i].reshape(3, -1).T.to(device)   # (H^3, 3)
                total_mse += F.mse_loss(u_pred, u_true).item()

                all_pred_means.append(pred[:, 0].mean().item())
                all_true_labels.append(labels[i].item())
                n_samples += 1

        # classification via stored training-data quantile bins
        all_pred_means = np.array(all_pred_means)
        all_true_labels = np.array(all_true_labels)

        bins = self._train_bins.cpu().numpy()
        pred_labels = np.clip(np.digitize(all_pred_means, bins) - 1, 0, self.n_classes - 1)
        accuracy = float((pred_labels == all_true_labels).mean()) * 100.0

        return {
            'accuracy': accuracy,
            'avg_loss': total_mse / max(n_samples, 1),
        }

    @torch.no_grad()
    def predict_field(self, H=None, device='cuda'):
        """Predict (u,v,w,p) on a full normalised 3D grid.

        Args:
            H: grid size (defaults to subdomain_size)
            device: torch device

        Returns:
            fields: (H, H, H, 4) numpy array [u, v, w, p]
        """
        H = H or self.subdomain_size
        self.eval()

        lin = torch.linspace(-1, 1, H, device=device)
        X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
        coords = torch.stack([
            X.reshape(-1), Y.reshape(-1), Z.reshape(-1),
            torch.zeros(H * H * H, device=device),
        ], dim=-1)

        # process in chunks to avoid OOM on large grids
        chunk = 4096
        preds = []
        for i in range(0, coords.shape[0], chunk):
            preds.append(self(coords[i:i + chunk]).cpu())
        pred = torch.cat(preds, dim=0)

        return pred.reshape(H, H, H, 4).numpy()
