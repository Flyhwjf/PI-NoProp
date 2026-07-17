"""
HIT DNS — Pseudo-Spectral Solver for Homogeneous Isotropic Turbulence
======================================================================
Incompressible Navier-Stokes with triply-periodic boundary conditions.
Pure FFT spectral method — no matrices, no Chebyshev, no LU.

Method:
  Spatial: Fourier-Galerkin on [0, 2pi]^3
  Time:    RK4 (4th order explicit)
  Dealias: 3/2 rule in all directions
  Pressure: Fourier-space projection P(k) = I - kk^T / |k|^2
  Forcing:  Stochastic at low wavenumbers (|k| <= k_f)

Output:  256 subdomains of 32^3 x 32 timesteps per region
         (centre = low enstrophy, edge = high enstrophy)
"""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from pathlib import Path
import os, sys, time, json, glob

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kw):
        return iterable


# ============================================================================
#  HIT DNS Solver
# ============================================================================

class HIT_DNS:
    """Pseudo-spectral HIT solver on [0, L]^3."""

    def __init__(self, N=128, nu=0.001, dt=0.001, L=2.0 * np.pi,
                 k_f=2.5, force_amp=0.1, seed=42):
        self.N = N
        self.nu = nu
        self.dt = dt
        self.L = L
        self.k_f = k_f
        self.force_amp = force_amp
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Dealiased grid size
        self.N_d = 3 * N // 2

        # Wavenumbers
        k1d = 2.0 * np.pi * fftfreq(N, L / N)
        k1d_d = 2.0 * np.pi * fftfreq(self.N_d, L / self.N_d)

        KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing='ij')
        self.K2 = KX ** 2 + KY ** 2 + KZ ** 2  # (N, N, N)
        self.KX = KX
        self.KY = KY
        self.KZ = KZ

        KX_d, KY_d, KZ_d = np.meshgrid(k1d_d, k1d_d, k1d_d, indexing='ij')
        self.K2_d = KX_d ** 2 + KY_d ** 2 + KZ_d ** 2
        self.KX_d = KX_d
        self.KY_d = KY_d
        self.KZ_d = KZ_d

        # Forcing mask: |k| <= k_f
        k_mag = np.sqrt(self.K2)
        self._kf_mask = (k_mag > 0) & (k_mag <= k_f)

        # Precompute viscous decay factor exp(-nu * k^2 * dt)
        self._vis_dt = np.exp(-nu * self.K2 * dt)

        # Initialise velocity field with random spectrum
        self._init_field()

    def _init_field(self):
        """Initialise divergence-free velocity with target kinetic energy."""
        N = self.N
        # White noise in physical space -> FFT (Hermitian by construction)
        u_phys = self._rng.randn(3, N, N, N)
        u_hat = np.zeros((3, N, N, N), dtype=np.complex128)
        for c in range(3):
            u_hat[c] = fftn(u_phys[c])

        # Project to divergence-free
        u_hat = self._project_div_free(u_hat, self.KX, self.KY, self.KZ)

        # Scale to target kinetic energy
        ke_current = 0.5 * np.sum(np.abs(u_hat) ** 2) / (N ** 6)
        target_ke = 0.5
        scale = np.sqrt(target_ke / (ke_current + 1e-16))
        self._u_hat = u_hat * scale

        # AB2 state: previous RHS for Adams-Bashforth
        self._rhs_prev = np.zeros_like(self._u_hat)
        self._step_count = 0

    def _project_div_free(self, u_hat, kx, ky, kz):
        """Project onto divergence-free subspace: P(k) = I - kk^T/|k|^2."""
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k2_safe = np.where(k2 > 0, k2, 1.0)

        k_dot_u = (kx * u_hat[0] + ky * u_hat[1] + kz * u_hat[2]) / k2_safe
        result = np.zeros_like(u_hat)
        result[0] = u_hat[0] - kx * k_dot_u
        result[1] = u_hat[1] - ky * k_dot_u
        result[2] = u_hat[2] - kz * k_dot_u
        # Zero out k=0 mode
        result[:, 0, 0, 0] = 0.0
        return result

    def _pad(self, u_hat):
        """Pad Fourier coefficients to 3/2 grid (zero high frequencies)."""
        N, N_d = self.N, self.N_d
        out = np.zeros((3, N_d, N_d, N_d), dtype=np.complex128)
        # Low frequencies go in the corners
        half = N // 2
        for c in range(3):
            out[c, :half, :half, :half] = u_hat[c, :half, :half, :half]
            out[c, :half, :half, -half:] = u_hat[c, :half, :half, -half:]
            out[c, :half, -half:, :half] = u_hat[c, :half, -half:, :half]
            out[c, :half, -half:, -half:] = u_hat[c, :half, -half:, -half:]
            out[c, -half:, :half, :half] = u_hat[c, -half:, :half, :half]
            out[c, -half:, :half, -half:] = u_hat[c, -half:, :half, -half:]
            out[c, -half:, -half:, :half] = u_hat[c, -half:, -half:, :half]
            out[c, -half:, -half:, -half:] = u_hat[c, -half:, -half:, -half:]
        return out

    def _trunc(self, u_hat_d):
        """Truncate from 3/2 grid back to original resolution."""
        N, N_d = self.N, self.N_d
        half = N // 2
        out = np.zeros((3, N, N, N), dtype=np.complex128)
        for c in range(3):
            out[c, :half, :half, :half] = u_hat_d[c, :half, :half, :half]
            out[c, :half, :half, -half:] = u_hat_d[c, :half, :half, -half:]
            out[c, :half, -half:, :half] = u_hat_d[c, :half, -half:, :half]
            out[c, :half, -half:, -half:] = u_hat_d[c, :half, -half:, -half:]
            out[c, -half:, :half, :half] = u_hat_d[c, -half:, :half, :half]
            out[c, -half:, :half, -half:] = u_hat_d[c, -half:, :half, -half:]
            out[c, -half:, -half:, :half] = u_hat_d[c, -half:, -half:, :half]
            out[c, -half:, -half:, -half:] = u_hat_d[c, -half:, -half:, -half:]
        return out

    def _compute_nonlinear(self, u_hat):
        """N = -(u·∇)u with 3/2 dealiasing. Returns N_hat on original grid."""
        u_hat_d = self._pad(u_hat)

        # IFFT to physical space on dealiased grid
        u_phys = np.zeros((3, self.N_d, self.N_d, self.N_d), dtype=np.float64)
        for c in range(3):
            u_phys[c] = ifftn(u_hat_d[c]).real

        # Compute derivatives in spectral space (on dealiased grid)
        du_dx = np.zeros_like(u_hat_d)
        du_dy = np.zeros_like(u_hat_d)
        du_dz = np.zeros_like(u_hat_d)
        for c in range(3):
            du_dx[c] = 1j * self.KX_d * u_hat_d[c]
            du_dy[c] = 1j * self.KY_d * u_hat_d[c]
            du_dz[c] = 1j * self.KZ_d * u_hat_d[c]

        # IFFT gradients to physical
        for c in range(3):
            du_dx[c] = ifftn(du_dx[c])
            du_dy[c] = ifftn(du_dy[c])
            du_dz[c] = ifftn(du_dz[c])
        # du_dx[c] is now physical (complex but real-valued)

        # Nonlinear products in physical space
        Nx = -(u_phys[0] * du_dx[0].real + u_phys[1] * du_dy[0].real + u_phys[2] * du_dz[0].real)
        Ny = -(u_phys[0] * du_dx[1].real + u_phys[1] * du_dy[1].real + u_phys[2] * du_dz[1].real)
        Nz = -(u_phys[0] * du_dx[2].real + u_phys[1] * du_dy[2].real + u_phys[2] * du_dz[2].real)

        # FFT back
        N_hat_d = np.zeros_like(u_hat_d)
        N_hat_d[0] = fftn(Nx)
        N_hat_d[1] = fftn(Ny)
        N_hat_d[2] = fftn(Nz)

        return self._trunc(N_hat_d)

    def _add_forcing(self):
        """Stochastic forcing: physical noise → FFT → project to div-free."""
        f_phys = self._rng.randn(3, self.N, self.N, self.N) * self.force_amp
        f_hat = np.zeros((3, self.N, self.N, self.N), dtype=np.complex128)
        for c in range(3):
            f_hat[c] = fftn(f_phys[c])
        f_hat = self._project_div_free(f_hat, self.KX, self.KY, self.KZ)
        return f_hat

    def _rhs(self, u_hat):
        """Right-hand side: du/dt = -ν k² u + N(u) + f."""
        N_hat = self._compute_nonlinear(u_hat)
        f_hat = self._add_forcing()
        return -self.nu * self.K2 * u_hat + N_hat + f_hat

    def step(self):
        """One AB2 time step (2nd order Adams-Bashforth).
        First step uses Euler, subsequent steps use AB2.
        """
        rhs = self._rhs(self._u_hat)

        if self._step_count == 0:
            # Euler for first step
            self._u_hat += self.dt * rhs
        else:
            # AB2: u^{n+1} = u^n + dt * (1.5*rhs^n - 0.5*rhs^{n-1})
            self._u_hat += self.dt * (1.5 * rhs - 0.5 * self._rhs_prev)

        self._u_hat = self._project_div_free(self._u_hat, self.KX, self.KY, self.KZ)
        self._rhs_prev = rhs
        self._step_count += 1

    def get_physical(self):
        """Return (u, v, w) in physical space, shape (3, N, N, N)."""
        phys = np.zeros((3, self.N, self.N, self.N), dtype=np.float64)
        for c in range(3):
            phys[c] = ifftn(self._u_hat[c]).real
        return phys

    def get_pressure(self):
        """Compute pressure from velocity via spectral Poisson equation.
        p_hat(k) = -(i/k^2) * k_j * N_j(k)   (from taking div of NS)
        Simplified: we compute the nonlinear term, then pressure.
        """
        N_hat = self._compute_nonlinear(self._u_hat)
        k2 = self.K2
        k2_safe = np.where(k2 > 0, k2, 1.0)
        k_dot_N = (self.KX * N_hat[0] + self.KY * N_hat[1] + self.KZ * N_hat[2])
        p_hat = -1j * k_dot_N / k2_safe
        p_hat[0, 0, 0] = 0.0
        p = ifftn(p_hat).real
        # Subtract mean
        p -= p.mean()
        return p

    def get_stats(self):
        """Return diagnostic statistics from physical space."""
        u = self.get_physical()  # (3, N, N, N)
        ke = 0.5 * np.mean(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

        # Enstrophy via finite differences in physical space
        # omega = curl u
        dux_dy, dux_dz = np.gradient(u[0], axis=(1, 2))
        duy_dx, duy_dz = np.gradient(u[1], axis=(0, 2))
        duz_dx, duz_dy = np.gradient(u[2], axis=(0, 1))
        om_x = duz_dy - duy_dz
        om_y = dux_dz - duz_dx
        om_z = duy_dx - dux_dy
        omega_sq = np.mean(om_x ** 2 + om_y ** 2 + om_z ** 2)

        eps = 2.0 * self.nu * omega_sq
        Re_lambda = np.sqrt(20.0 / 3.0 * ke ** 2 / (self.nu * eps + 1e-30))

        return {
            'ke': float(ke),
            'enstrophy': float(omega_sq),
            'epsilon': float(eps),
            'Re_lambda': float(Re_lambda),
            'u_rms': float(np.sqrt(2.0 * ke / 3.0)),
        }


# ============================================================================
#  Subdomain extraction
# ============================================================================

def extract_subdomains(snapshot_dir, output_centre, output_edge,
                       n_subdomains=256, sub_size=32, n_t=32):
    """Extract labelled 32^3 x 32 subdomains from saved DNS snapshots.

    Regions:
      'centre' = low local enstrophy windows
      'edge'   = high local enstrophy windows
    """
    snap_files = sorted(glob.glob(os.path.join(snapshot_dir, 'snap_*.npz')))
    if len(snap_files) < n_t:
        raise ValueError(f"Need >= {n_t} snapshots, got {len(snap_files)}")

    print(f"  Found {len(snap_files)} snapshots. Loading...")
    sys.stdout.flush()

    # Load all snapshots
    all_vel = []   # list of (3, N, N, N) float32
    all_prs = []   # list of (N, N, N) float32
    for f in tqdm(snap_files, desc="  Loading"):
        d = np.load(f)
        all_vel.append(d['velocity'].astype(np.float32))
        all_prs.append(d['pressure'].astype(np.float32))

    N = all_vel[0].shape[1]  # grid size
    n_t_avail = len(snap_files) - n_t + 1

    # Compute local enstrophy for all snapshots (averaged over 32^3 windows)
    print("  Computing enstrophy maps...")
    sys.stdout.flush()

    # Sample enstrophy at random window centres across all snapshots
    rng = np.random.RandomState(42)
    n_samples = n_subdomains * 8  # oversample
    enstrophy_vals = []
    window_specs = []

    for _ in tqdm(range(n_samples), desc="  Sampling enstrophy"):
        it = rng.randint(0, n_t_avail)
        ix = rng.randint(0, N - sub_size + 1)
        iy = rng.randint(0, N - sub_size + 1)
        iz = rng.randint(0, N - sub_size + 1)
        # Compute enstrophy from curl of velocity
        vel = all_vel[it][:, ix:ix + sub_size, iy:iy + sub_size, iz:iz + sub_size]
        # Approximate enstrophy = |curl u|^2 using finite differences
        omega_sq = 0.0
        for c in range(3):
            dv_dx = np.gradient(vel[c], axis=0)
            dv_dy = np.gradient(vel[c], axis=1)
            dv_dz = np.gradient(vel[c], axis=2)
            omega_sq += np.mean(dv_dx ** 2 + dv_dy ** 2 + dv_dz ** 2)
        enstrophy_vals.append(omega_sq)
        window_specs.append((it, ix, iy, iz))

    enstrophy_vals = np.array(enstrophy_vals)
    median_ens = np.median(enstrophy_vals)

    # Split into centre (low ens) and edge (high ens)
    centre_mask = enstrophy_vals <= median_ens
    edge_mask = enstrophy_vals > median_ens

    print(f"  Enstrophy median: {median_ens:.4f}")
    print(f"  Centre candidates: {centre_mask.sum()}, Edge candidates: {edge_mask.sum()}")
    sys.stdout.flush()

    for region, mask, out_dir in [('centre', centre_mask, output_centre),
                                   ('edge', edge_mask, output_edge)]:
        candidates = [(window_specs[i], enstrophy_vals[i])
                      for i in range(n_samples) if mask[i]]

        # Randomly select n_subdomains
        selected = rng.choice(len(candidates), min(n_subdomains, len(candidates)),
                              replace=False)
        os.makedirs(out_dir, exist_ok=True)

        raw_data = []
        u_means = []

        for idx in tqdm(selected, desc=f"  {region} extract"):
            (it, ix, iy, iz), _ = candidates[idx]

            vel_chunks = np.zeros((3, sub_size, sub_size, sub_size, n_t), dtype=np.float32)
            prs_chunks = np.zeros((sub_size, sub_size, sub_size, n_t), dtype=np.float32)
            for dt_idx in range(n_t):
                vel_chunks[..., dt_idx] = all_vel[it + dt_idx][:, ix:ix + sub_size,
                                                               iy:iy + sub_size,
                                                               iz:iz + sub_size]
                prs_chunks[..., dt_idx] = all_prs[it + dt_idx][ix:ix + sub_size,
                                                               iy:iy + sub_size,
                                                               iz:iz + sub_size]
            u_mean = float(np.mean(vel_chunks[0]))
            u_means.append(u_mean)
            raw_data.append((vel_chunks, prs_chunks, it, ix, iy, iz))

        # Class labels by quantile binning
        u_means = np.array(u_means)
        q = np.percentile(u_means, np.linspace(0, 100, 11))
        labels = np.clip(np.digitize(u_means, q[:-1]) - 1, 0, 9)

        # Channel statistics
        all_v = np.stack([r[0] for r in raw_data], axis=0)
        all_p = np.stack([r[1] for r in raw_data], axis=0)
        v_mean = np.mean(all_v, axis=(0, 2, 3, 4, 5))
        v_std = np.std(all_v, axis=(0, 2, 3, 4, 5)) + 1e-8
        p_mean = float(np.mean(all_p))
        p_std = float(np.std(all_p)) + 1e-8
        stats_arr = np.array([[v_mean[i], v_std[i]] for i in range(3)] +
                             [[p_mean, p_std]], dtype=np.float32)

        # Write files
        for fi, ((vel, prs, it, ix, iy, iz), label) in enumerate(zip(raw_data, labels)):
            coords = np.zeros((4, 2, sub_size, sub_size, sub_size, n_t), dtype=np.float32)
            coords[0, 0] = ix / N
            coords[0, 1] = ix / N
            coords[1, 0] = iy / N
            coords[1, 1] = iy / N
            coords[2, 0] = iz / N
            coords[2, 1] = iz / N
            for dt_idx in range(n_t):
                coords[3, 0, :, :, :, dt_idx] = (it + dt_idx) / max(n_t_avail, 1)
                coords[3, 1, :, :, :, dt_idx] = (it + dt_idx) / max(n_t_avail, 1)

            np.savez_compressed(
                os.path.join(out_dir, f'sub_{fi:04d}.npz'),
                velocity=vel, pressure=prs,
                label=np.array(label, dtype=np.int32),
                coords=coords, stats=stats_arr,
            )

        meta = {
            'region': region, 'n_subdomains': len(raw_data),
            'subdomain_size': sub_size, 'n_timesteps': n_t,
            'n_classes': 10, 'class_quantiles': q.tolist(),
            'label_counts': [int(np.sum(labels == c)) for c in range(10)],
        }
        with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  {region}: saved {len(raw_data)} subdomains")
        print(f"  Class distribution: {meta['label_counts']}")

    # Free memory
    del all_vel, all_prs


# ============================================================================
#  Main
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description='HIT DNS + Subdomain Extraction')
    p.add_argument('--N', type=int, default=128)
    p.add_argument('--nu', type=float, default=0.001)
    p.add_argument('--dt', type=float, default=0.001)
    p.add_argument('--k_f', type=float, default=2.5)
    p.add_argument('--force_amp', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_steady', type=int, default=2000, help='Steps to steady state')
    p.add_argument('--n_sample', type=int, default=6400, help='Sampling steps')
    p.add_argument('--save_every', type=int, default=32, help='Save snapshot every N steps')
    p.add_argument('--snapshot_dir', type=str, default='data/generated/snapshots')
    p.add_argument('--out_centre', type=str, default='data/generated/centre')
    p.add_argument('--out_edge', type=str, default='data/generated/edge')
    p.add_argument('--n_subdomains', type=int, default=256)
    p.add_argument('--n_t', type=int, default=32)
    p.add_argument('--skip_sim', action='store_true')
    args = p.parse_args()

    for d in [args.snapshot_dir, args.out_centre, args.out_edge]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if not args.skip_sim:
        print("=" * 60)
        print(f"HIT DNS  |  N={args.N}^3  nu={args.nu}  dt={args.dt}")
        print(f"  Steady: {args.n_steady} steps  |  Sampling: {args.n_sample} steps")
        print("=" * 60)
        sys.stdout.flush()

        dns = HIT_DNS(N=args.N, nu=args.nu, dt=args.dt,
                      k_f=args.k_f, force_amp=args.force_amp,
                      seed=args.seed)

        # --- Steady-state phase ---
        print("Phase 1: Reaching statistical steady state...")
        sys.stderr.flush()
        pbar = tqdm(total=args.n_steady, desc="Steady", unit="step",
                     dynamic_ncols=True, file=sys.stderr)
        for step in range(1, args.n_steady + 1):
            dns.step()
            if step % 50 == 0:
                s = dns.get_stats()
                pbar.set_postfix_str(f"Re_l={s['Re_lambda']:.0f} u_rms={s['u_rms']:.3f}")
                pbar.update(50)
                sys.stderr.flush()
        pbar.close()
        sys.stderr.flush()

        # --- Sampling phase ---
        print("Phase 2: Sampling...")
        sys.stderr.flush()
        n_snap = args.n_sample // args.save_every
        pbar = tqdm(total=args.n_sample, desc="Sample", unit="step",
                     dynamic_ncols=True, file=sys.stderr)
        snap_count = 0
        for step in range(1, args.n_sample + 1):
            dns.step()
            if step % 50 == 0:
                pbar.update(50)
                sys.stderr.flush()
            if step % args.save_every == 0:
                u_phys = dns.get_physical().astype(np.float32)
                p_phys = dns.get_pressure().astype(np.float32)
                np.savez_compressed(
                    os.path.join(args.snapshot_dir, f'snap_{snap_count:04d}.npz'),
                    velocity=u_phys, pressure=p_phys)
                snap_count += 1
                pbar.set_postfix_str(f"snaps={snap_count}")
        pbar.close()
        sys.stderr.flush()

        print(f"Saved {snap_count} snapshots.")
        s = dns.get_stats()
        print(f"Final: Re_lambda={s['Re_lambda']:.1f}, u_rms={s['u_rms']:.4f}")

    # --- Extract subdomains ---
    print("\n" + "=" * 60)
    print("Extracting subdomains...")
    extract_subdomains(args.snapshot_dir, args.out_centre, args.out_edge,
                       n_subdomains=args.n_subdomains, n_t=args.n_t)
    print("Done.")


if __name__ == '__main__':
    main()
