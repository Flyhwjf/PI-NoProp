"""
Channel Flow DNS — Chebyshev-Fourier Spectral Solver
=====================================================
Incompressible Navier-Stokes, Re_tau=180 (Kim-Moin-Moser benchmark).

Numerical method:
  Spatial: Chebyshev-Galerkin (y) × Fourier-Galerkin (x, z)
  Time:    3rd-order semi-implicit (AB3 explicit / CN implicit)
  Pressure: Projection method (Chorin-Temam)
  Dealias: 3/2 Fourier padding + 2/3 Chebyshev low-pass filter
  Flow control: PI controller for constant bulk velocity

Output: 256 subdomains of 32^3 x 4 timesteps per region (centre + edge)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftn, ifftn, dct, idct
from scipy.linalg import lu_factor, lu_solve
import json
import os
import sys
import time
import glob
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================================
#  Chebyshev utilities
# ============================================================================

def cheb_points(N):
    """Chebyshev-Gauss-Lobatto: y_j = cos(pi*j/N), j=0..N"""
    return np.cos(np.pi * np.arange(N + 1) / N)


def cheb_D1(y):
    """First-derivative Chebyshev collocation matrix (Ny, Ny)."""
    N = len(y) - 1
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (y[i] - y[j])
    for i in range(1, N):
        D[i, i] = -y[i] / (2.0 * (1.0 - y[i] ** 2))
    D[0, 0] = (2.0 * N ** 2 + 1.0) / 6.0
    D[-1, -1] = -D[0, 0]
    return D


def phys_to_cheb(f_phys):
    """Physical values at CGL points → Chebyshev expansion coefficients.
    f_phys: (..., Ny) — evaluated at CGL collocation points.
    Returns: (..., Ny) — coefficients a_k such that f = sum a_k T_k.
    """
    N = f_phys.shape[-1] - 1
    d = dct(f_phys, type=1, axis=-1) / N
    d[..., 0] /= 2.0
    d[..., -1] /= 2.0
    return d


def cheb_to_phys(coeffs):
    """Chebyshev coefficients → physical values at CGL points.
    coeffs: (..., Ny) — Chebyshev expansion coefficients.
    Returns: (..., Ny) — f evaluated at CGL points.
    """
    N = coeffs.shape[-1] - 1
    x = coeffs.copy()
    x[..., 0] *= (2.0 * N)
    x[..., 1:-1] *= N
    x[..., -1] *= (2.0 * N)
    return idct(x, type=1, axis=-1)


def cheb_pad(coeffs, Ny_new):
    """Pad Chebyshev coefficients with zeros (increase polynomial degree).
    coeffs: (..., Ny_old)
    Returns: (..., Ny_new) — coefficients up to new degree, higher modes = 0.
    """
    Ny_old = coeffs.shape[-1]
    shape = coeffs.shape[:-1] + (Ny_new,)
    out = np.zeros(shape, dtype=coeffs.dtype)
    out[..., :Ny_old] = coeffs
    return out


def cheb_trunc(coeffs, Ny_target):
    """Truncate Chebyshev coefficients (reduce polynomial degree)."""
    return coeffs[..., :Ny_target]


def cheb_filter(coeffs, frac=2.0 / 3.0, order=2):
    """Exponential low-pass filter for Chebyshev dealiasing.

    sigma(k) = exp(-alpha * ((k - k_c) / (N - k_c))^order)  for k > k_c,
    sigma(k) = 1  for k <= k_c.
    alpha = -ln(eps) to drive sigma(N) to machine precision.
    """
    N = coeffs.shape[-1] - 1
    k = np.arange(N + 1)
    k_c = int(frac * N)
    sigma = np.ones(N + 1)
    mask = k > k_c
    alpha = -np.log(np.finfo(np.float64).eps)
    sigma[mask] = np.exp(-alpha * ((k[mask] - k_c) / (N - k_c)) ** order)
    return coeffs * sigma


# ============================================================================
#  ChannelFlowDNS
# ============================================================================

class ChannelFlowDNS:
    """Chebyshev-Fourier channel flow DNS."""

    def __init__(self, Nx=128, Ny=129, Nz=128, Re_tau=180.0,
                 Lx=4.0 * np.pi, Lz=2.0 * np.pi, dt=0.002):
        self.Nx = Nx
        self.Ny = Ny          # collocation points (polynomial degree = Ny-1)
        self.Nz = Nz
        self.Ny_deg = Ny - 1  # polynomial degree
        self.Re_tau = Re_tau
        self.nu = 1.0 / Re_tau
        self.Lx = Lx
        self.Lz = Lz
        self.dt = dt

        # Dealias sizes (3/2 rule)
        self.Nx_d = 3 * Nx // 2
        self.Nz_d = 3 * Nz // 2
        self.Ny_d = 3 * (Ny - 1) // 2 + 1  # not used; we use filter instead

        # AB3 coefficients
        self._ab3 = np.array([23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0])

        self._build_fourier()
        self._build_chebyshev()
        self._build_dealias_indices()
        self._build_solvers()
        self._init_state()

        self._dPdx = -2.0 / self.Re_tau  # laminar equilibrium
        self._ubulk_target = 15.6         # KMM Re_tau=180 turbulent bulk velocity
        self._dPdx_integral = 0.0

    # -------- Grids --------

    def _build_fourier(self):
        kx_raw = 2.0 * np.pi * fftfreq(self.Nx, self.Lx / self.Nx)
        kz_raw = 2.0 * np.pi * fftfreq(self.Nz, self.Lz / self.Nz)
        self.kx = kx_raw
        self.kz = kz_raw
        KKX, KKZ = np.meshgrid(kx_raw, kz_raw, indexing='ij')
        self.K2 = KKX ** 2 + KKZ ** 2   # (Nx, Nz)
        self.KX = KKX
        self.KZ = KKZ

    def _build_chebyshev(self):
        self.y = cheb_points(self.Ny_deg)        # (Ny,)
        self.D1 = cheb_D1(self.y)                # (Ny, Ny)
        self.D2 = self.D1 @ self.D1              # (Ny, Ny)
        self.i_int = slice(1, self.Ny - 1)       # interior indices

    def _build_dealias_indices(self):
        """Precompute index arrays for 3/2 Fourier padding/truncation."""
        Nx, Nz = self.Nx, self.Nz
        Nx_d, Nz_d = self.Nx_d, self.Nz_d

        # Pad indices: where each original FFT index goes in the padded FFT
        # N = 128 → N_d = 192
        #  0..63  → 0..63
        #  64     → 64
        #  65..127 → 129..191  
        # Padded indices 65..128 are zeroed (new high-freq positive modes)
        self._pad_x_idx = np.zeros(Nx, dtype=int)
        self._pad_x_idx[:Nx // 2] = np.arange(Nx // 2)
        self._pad_x_idx[Nx // 2] = Nx // 2
        self._pad_x_idx[Nx // 2 + 1:] = np.arange(Nx_d - Nx // 2 + 1, Nx_d)

        self._pad_z_idx = np.zeros(Nz, dtype=int)
        self._pad_z_idx[:Nz // 2] = np.arange(Nz // 2)
        self._pad_z_idx[Nz // 2] = Nz // 2
        self._pad_z_idx[Nz // 2 + 1:] = np.arange(Nz_d - Nz // 2 + 1, Nz_d)

        # Truncate indices: where each original index comes from in padded FFT
        self._trunc_x_idx = np.zeros(Nx, dtype=int)
        self._trunc_x_idx[:Nx // 2 + 1] = np.arange(Nx // 2 + 1)
        self._trunc_x_idx[Nx // 2 + 1:] = np.arange(Nx_d - Nx // 2 + 1, Nx_d)

        self._trunc_z_idx = np.zeros(Nz, dtype=int)
        self._trunc_z_idx[:Nz // 2 + 1] = np.arange(Nz // 2 + 1)
        self._trunc_z_idx[Nz // 2 + 1:] = np.arange(Nz_d - Nz // 2 + 1, Nz_d)

    # -------- Build linear solvers --------

    def _build_solvers(self):
        """Precompute LU decompositions and group modes by k^2 value."""
        Ny = self.Ny
        nu = self.nu
        dt = self.dt

        k2_unique = np.unique(np.round(self.K2.ravel(), decimals=12))
        print(f"  Building solvers for {len(k2_unique)} unique k^2 values...")
        sys.stdout.flush()

        # Group (i,k) indices by k^2
        self._k2_groups = {}
        for k2 in tqdm(k2_unique, desc="  Grouping modes"):
            indices = np.where(np.round(self.K2, 12) == k2)
            self._k2_groups[k2] = (indices[0], indices[1])

        # Helmholtz: (D^2 - (k^2 + 2/nu/dt)) u = f, u(+-1)=0
        self._helm_lu = {}
        for k2 in tqdm(k2_unique, desc="  Helmholtz LU"):
            lam = k2 + 2.0 / (nu * dt)
            A = self.D2[self.i_int, self.i_int] - lam * np.eye(Ny - 2)
            self._helm_lu[k2] = lu_factor(A)

        # Poisson: (D^2 - k^2) p = f, Dp(+-1)=0
        self._pois_lu = {}
        for k2 in tqdm(k2_unique, desc="  Poisson LU"):
            A = self.D2 - k2 * np.eye(Ny)
            if k2 < 1e-12:
                A_bc = A.copy()
                A_bc[0, :] = self.D1[0, :]
                A_bc[-1, :] = self.D1[-1, :]
                mid = Ny // 2
                A_bc[mid, :] = 0.0
                A_bc[mid, mid] = 1.0
            else:
                A_bc = A.copy()
                A_bc[0, :] = self.D1[0, :]
                A_bc[-1, :] = self.D1[-1, :]
            self._pois_lu[k2] = lu_factor(A_bc)

    # -------- State initialisation --------

    def _init_state(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        u_lam = (1.0 - self.y ** 2).astype(np.float64)
        self.u_hat = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        self.v_hat = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        self.w_hat = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        self.p_hat = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

        for j in range(Ny):
            self.u_hat[0, j, 0] = u_lam[j] * Nx * Nz

        self._add_perturbations(amplitude=0.05)

        # Populate nonlinear term history (use same value for all 3)
        print("  Computing initial nonlinear terms..."); sys.stdout.flush()
        Nx0, Ny0, Nz0 = self._compute_nonlinear()
        self._Nx_hist = [Nx0.copy(), Nx0.copy(), Nx0.copy()]
        self._Ny_hist = [Ny0.copy(), Ny0.copy(), Ny0.copy()]
        self._Nz_hist = [Nz0.copy(), Nz0.copy(), Nz0.copy()]
        print("  Initialisation complete."); sys.stdout.flush()

    # -------- Batched solves --------

    def _solve_helmholtz(self, rhs_hat):
        """Batched Helmholtz solve for all (kx,kz) modes via LU."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        nu, dt = self.nu, self.dt
        factor = -2.0 / (nu * dt)
        result = np.zeros_like(rhs_hat)

        for k2, (ix, iz) in self._k2_groups.items():
            lu, piv = self._helm_lu[k2]
            b = factor * rhs_hat[ix, self.i_int, iz].real  # (n_modes, Ny-2)
            if b.ndim == 1:
                b = b.reshape(1, -1)
            sol = np.array([lu_solve((lu, piv), bi) for bi in b])
            result[ix, 1:-1, iz] = sol

        return result

    def _solve_poisson(self, rhs_hat):
        """Batched Poisson solve for all (kx,kz) modes via LU."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        result = np.zeros_like(rhs_hat)

        for k2, (ix, iz) in self._k2_groups.items():
            lu, piv = self._pois_lu[k2]
            b = rhs_hat[ix, :, iz].real.copy()
            if b.ndim == 1:
                b = b.reshape(1, -1)
            if k2 < 1e-12:
                b[:, 0] = 0.0
                b[:, -1] = 0.0
                b[:, Ny // 2] = 0.0
            else:
                b[:, 0] = 0.0
                b[:, -1] = 0.0
            sol = np.array([lu_solve((lu, piv), bi) for bi in b])
            result[ix, :, iz] = sol

        return result

    def _laplacian(self, f_hat):
        """nabla^2 f in Fourier-Chebyshev space. Vectorized over (kx,kz)."""
        result = np.zeros_like(f_hat)
        for k2, (ix, iz) in self._k2_groups.items():
            result[ix, :, iz] = (self.D2 @ f_hat[ix, :, iz].T).T - k2 * f_hat[ix, :, iz]
        return result

    def _divergence(self, u_hat, v_hat, w_hat):
        """nabla.u in Fourier-Chebyshev space. Vectorized over (kx,kz)."""
        div = np.zeros_like(u_hat)
        for k2, (ix, iz) in self._k2_groups.items():
            kx_v = self.KX[ix, iz]
            kz_v = self.KZ[ix, iz]
            div[ix, :, iz] = (1j * kx_v[:, None] * u_hat[ix, :, iz] +
                              (self.D1 @ v_hat[ix, :, iz].T).T +
                              1j * kz_v[:, None] * w_hat[ix, :, iz])
        return div

    # -------- Nonlinear term --------

    def _add_perturbations(self, amplitude=0.05):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        rng = np.random.RandomState(42)

        Ax = rng.randn(Nx, Ny, Nz) * amplitude
        Ay = rng.randn(Nx, Ny, Nz) * amplitude
        Az = rng.randn(Nx, Ny, Nz) * amplitude

        wall_mask = (1.0 - self.y ** 2)[np.newaxis, :, np.newaxis]
        Ax *= wall_mask
        Ay *= wall_mask
        Az *= wall_mask

        Ax_hat = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        Ay_hat = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        Az_hat = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
        for j in range(Ny):
            for arr, out in [(Ax, Ax_hat), (Ay, Ay_hat), (Az, Az_hat)]:
                tmp = fft(arr[:, j, :], axis=0)
                tmp = fft(tmp, axis=1)
                out[:, j, :] = tmp

        for k2, (ix, iz) in self._k2_groups.items():
            kx_v = self.KX[ix, iz]
            kz_v = self.KZ[ix, iz]
            self.u_hat[ix, :, iz] += ((self.D1 @ Az_hat[ix, :, iz].T).T -
                                        1j * kz_v[:, None] * Ay_hat[ix, :, iz])
            self.v_hat[ix, :, iz] += (1j * kz_v[:, None] * Ax_hat[ix, :, iz] -
                                       1j * kx_v[:, None] * Az_hat[ix, :, iz])
            self.w_hat[ix, :, iz] += (1j * kx_v[:, None] * Ay_hat[ix, :, iz] -
                                       (self.D1 @ Ax_hat[ix, :, iz].T).T)

        for hat in [self.u_hat, self.v_hat, self.w_hat]:
            hat[:, 0, :] = 0.0
            hat[:, -1, :] = 0.0

    def _compute_nonlinear(self):
        """N = -(u.nabla)u with 3/2 Fourier dealiasing + Chebyshev filter."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        Nx_d, Nz_d = self.Nx_d, self.Nz_d

        # Pad Fourier coefficients via fancy indexing
        def pad_fourier(f_hat):
            out = np.zeros((Nx_d, Ny, Nz_d), dtype=np.complex128)
            out[np.ix_(self._pad_x_idx, np.arange(Ny), self._pad_z_idx)] = f_hat
            return out

        u_pad = pad_fourier(self.u_hat)
        v_pad = pad_fourier(self.v_hat)
        w_pad = pad_fourier(self.w_hat)

        kx_d = 2.0 * np.pi * fftfreq(Nx_d, self.Lx / Nx_d)
        kz_d = 2.0 * np.pi * fftfreq(Nz_d, self.Lz / Nz_d)

        def to_phys(pad_hat):
            phys = ifftn(pad_hat, axes=(0, 2)).real
            coeffs = phys_to_cheb(phys)
            coeffs_f = cheb_filter(coeffs, frac=2.0 / 3.0)
            return cheb_to_phys(coeffs_f)

        u_phys = to_phys(u_pad)
        v_phys = to_phys(v_pad)
        w_phys = to_phys(w_pad)

        # Compute Fourier derivatives via fftn
        def compute_grads(phys):
            tmp = fftn(phys, axes=(0, 2))
            du_dx = ifftn(1j * kx_d[:, np.newaxis, np.newaxis] * tmp, axes=(0, 2)).real
            du_dz = ifftn(1j * kz_d[np.newaxis, np.newaxis, :] * tmp, axes=(0, 2)).real
            return du_dx, du_dz

        du_dx, du_dz = compute_grads(u_phys)
        dv_dx, dv_dz = compute_grads(v_phys)
        dw_dx, dw_dz = compute_grads(w_phys)

        # Chebyshev derivative (reuse self.D1, already computed)
        du_dy = np.tensordot(self.D1, u_phys, axes=([1], [1])).transpose(1, 0, 2)
        dv_dy = np.tensordot(self.D1, v_phys, axes=([1], [1])).transpose(1, 0, 2)
        dw_dy = np.tensordot(self.D1, w_phys, axes=([1], [1])).transpose(1, 0, 2)

        # Nonlinear products
        Nx_phys = -(u_phys * du_dx + v_phys * du_dy + w_phys * du_dz)
        Ny_phys = -(u_phys * dv_dx + v_phys * dv_dy + w_phys * dv_dz)
        Nz_phys = -(u_phys * dw_dx + v_phys * dw_dy + w_phys * dw_dz)

        # Truncate back via fftn + fancy indexing
        def trunc_fourier(phys):
            tmp = fftn(phys, axes=(0, 2))
            out = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
            out[:, :, :] = tmp[np.ix_(self._trunc_x_idx, np.arange(Ny), self._trunc_z_idx)]
            return out

        Nx_hat = trunc_fourier(Nx_phys)
        Ny_hat = trunc_fourier(Ny_phys)
        Nz_hat = trunc_fourier(Nz_phys)

        return Nx_hat, Ny_hat, Nz_hat

    # -------- Time step --------

    def step(self):
        """One full AB3/CN fractional-step."""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        nu, dt = self.nu, self.dt

        # 1. Nonlinear term
        Nx_n, Ny_n, Nz_n = self._compute_nonlinear()

        # 2. AB3 extrapolation + mean pressure gradient
        Nx_exp = (self._ab3[0] * Nx_n +
                  self._ab3[1] * self._Nx_hist[-1] +
                  self._ab3[2] * self._Nx_hist[-2])
        Ny_exp = (self._ab3[0] * Ny_n +
                  self._ab3[1] * self._Ny_hist[-1] +
                  self._ab3[2] * self._Ny_hist[-2])
        Nz_exp = (self._ab3[0] * Nz_n +
                  self._ab3[1] * self._Nz_hist[-1] +
                  self._ab3[2] * self._Nz_hist[-2])

        Nx_exp[0, :, 0] += -self._dPdx

        self._Nx_hist.append(Nx_n)
        self._Ny_hist.append(Ny_n)
        self._Nz_hist.append(Nz_n)

        # 3. RHS = (1 + νΔt/2 ∇²) u^n + Δt N_exp
        rhs_u = self.u_hat + 0.5 * nu * dt * self._laplacian(self.u_hat) + dt * Nx_exp
        rhs_v = self.v_hat + 0.5 * nu * dt * self._laplacian(self.v_hat) + dt * Ny_exp
        rhs_w = self.w_hat + 0.5 * nu * dt * self._laplacian(self.w_hat) + dt * Nz_exp
        for r in [rhs_u, rhs_v, rhs_w]:
            r[:, 0, :] = 0.0
            r[:, -1, :] = 0.0

        # 4. Helmholtz → intermediate velocity u*
        us = self._solve_helmholtz(rhs_u)
        vs = self._solve_helmholtz(rhs_v)
        ws = self._solve_helmholtz(rhs_w)

        # 5. Divergence of u*
        div = self._divergence(us, vs, ws) / dt

        # 6. Pressure Poisson
        self.p_hat = self._solve_poisson(div)

        # 7. Velocity correction u^{n+1} = u* - dt ∇p  [vectorized]
        for k2, (ix, iz) in self._k2_groups.items():
            kx_v = self.KX[ix, iz]
            kz_v = self.KZ[ix, iz]
            dp = (self.D1 @ self.p_hat[ix, :, iz].T).T
            self.u_hat[ix, :, iz] = us[ix, :, iz] - dt * 1j * kx_v[:, None] * self.p_hat[ix, :, iz]
            self.v_hat[ix, :, iz] = vs[ix, :, iz] - dt * dp
            self.w_hat[ix, :, iz] = ws[ix, :, iz] - dt * 1j * kz_v[:, None] * self.p_hat[ix, :, iz]

        # No-slip
        for h in [self.u_hat, self.v_hat, self.w_hat]:
            h[:, 0, :] = 0.0
            h[:, -1, :] = 0.0

        # 8. Flow rate control
        self._update_dPdx()

    def _update_dPdx(self):
        """PI controller for constant bulk velocity.

        dP/dx is the mean pressure gradient.  A negative dP/dx drives the
        flow forward (+x).  When the flow is too fast, we make dP/dx less
        negative (increase it toward zero) to reduce the driving force.
        """
        u_00 = self.u_hat[0, :, 0].real / (self.Nx * self.Nz)
        w = self._cc_weights()
        u_bulk = 0.5 * np.sum(w * u_00)
        error = u_bulk - self._ubulk_target
        Kp = 1.0 / self.dt
        Ki = 0.5 / self.dt
        self._dPdx += Kp * error + Ki * self._dPdx_integral
        self._dPdx_integral += error * self.dt

    def _cc_weights(self):
        """Clenshaw-Curtis quadrature weights for integral over [-1, 1].
        Returns array w of length Ny such that sum(w * f) ≈ ∫_{-1}^{1} f(y) dy.
        """
        N = self.Ny_deg
        if not hasattr(self, '_w_cc'):
            w = np.zeros(N + 1)
            for j in range(N + 1):
                theta = np.pi * j / N
                s = 0.0
                for k in range(1, N // 2 + 1):
                    s += np.cos(2 * k * theta) / (4 * k ** 2 - 1)
                w[j] = 1.0 - 2.0 * s
            # Interior c_j = 2, endpoints c_j = 1
            w[0] /= N
            w[-1] /= N
            w[1:-1] *= 2.0 / N
            self._w_cc = w
        return self._w_cc

    # -------- Diagnostics --------

    def get_physical_fields(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        u = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        v = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        w = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        p = np.zeros((Nx, Ny, Nz), dtype=np.float64)
        for j in range(Ny):
            u[:, j, :] = ifft(ifft(self.u_hat[:, j, :], axis=0), axis=1).real
            v[:, j, :] = ifft(ifft(self.v_hat[:, j, :], axis=0), axis=1).real
            w[:, j, :] = ifft(ifft(self.w_hat[:, j, :], axis=0), axis=1).real
            p[:, j, :] = ifft(ifft(self.p_hat[:, j, :], axis=0), axis=1).real
        return u, v, w, p

    def get_stats(self):
        u, v, w, p = self.get_physical_fields()
        u_00 = self.u_hat[0, :, 0].real / (self.Nx * self.Nz)
        w_cc = self._cc_weights()
        u_bulk = 0.5 * np.sum(w_cc * u_00)
        tau_w_top = self.nu * (self.D1[0, :] @ u_00)
        tau_w_bot = self.nu * (-self.D1[-1, :] @ u_00)
        u_tau = np.sqrt(0.5 * (np.abs(tau_w_top) + np.abs(tau_w_bot)))
        return {
            'u_bulk': float(u_bulk),
            'tau_w': float(0.5 * (tau_w_top + tau_w_bot)),
            'u_tau': float(u_tau),
            'Re_tau_act': float(u_tau / self.nu),
            'dPdx': float(self._dPdx),
            'u_rms': float(np.std(u)),
            'v_rms': float(np.std(v)),
            'w_rms': float(np.std(w)),
        }


# ============================================================================
#  Subdomain extraction
# ============================================================================

def extract_subdomains(snapshot_dir, output_dir, region, n_subdomains=256,
                       sub_size=32, n_t=4):
    """Extract labelled 32³×4 subdomains from saved DNS snapshots."""
    snap_files = sorted(glob.glob(os.path.join(snapshot_dir, 'snap_*.npz')))
    if len(snap_files) < n_t:
        raise ValueError(f"Need >= {n_t} snapshots, got {len(snap_files)}")

    # Load metadata from first snapshot
    d0 = np.load(snap_files[0])
    _, Nx, Ny_full, Nz = d0['velocity'].shape
    y = d0['y']

    if region == 'centre':
        y_lo, y_hi = -0.35, 0.35
    else:
        y_lo, y_hi = 0.6, 0.95

    j_lo = np.searchsorted(y[::-1], y_hi)
    j_hi = Ny_full - np.searchsorted(y, y_lo)
    j_lo = max(0, Ny_full - j_hi)
    j_hi = min(Ny_full, Ny_full - j_lo)

    y_avail = j_hi - j_lo
    if y_avail < sub_size:
        j_mid = (j_lo + j_hi) // 2
        j_lo = max(0, j_mid - sub_size // 2)
        j_hi = min(Ny_full, j_lo + sub_size)
    if j_hi - j_lo < sub_size:
        raise ValueError(f"Not enough y-points in {region}")

    print(f"  {region}: y-range [{y[j_lo]:.3f}, {y[j_hi-1]:.3f}], "
          f"available {j_hi - j_lo} points")

    n_t_avail = len(snap_files) - n_t + 1
    rng = np.random.RandomState(42)
    os.makedirs(output_dir, exist_ok=True)

    # Load snapshots into memory for speed
    print(f"  Loading {len(snap_files)} snapshots...")
    all_snaps = []
    for f in tqdm(snap_files, desc="  Loading snaps", unit="file"):
        d = np.load(f)
        all_snaps.append({
            'v': d['velocity'].astype(np.float32),
            'p': d['pressure'].astype(np.float32),
            't': float(d['t']),
        })

    # Extract raw subdomains
    raw_data = []
    u_means = []
    for idx in tqdm(range(n_subdomains), desc=f"  {region} extract", unit="sub"):
        ix = rng.randint(0, Nx - sub_size + 1)
        iy = rng.randint(j_lo, j_hi - sub_size + 1)
        iz = rng.randint(0, Nz - sub_size + 1)
        it = rng.randint(0, n_t_avail)

        vel_chunks = np.zeros((3, sub_size, sub_size, sub_size, n_t), dtype=np.float32)
        prs_chunks = np.zeros((sub_size, sub_size, sub_size, n_t), dtype=np.float32)
        for dt_idx in range(n_t):
            s = all_snaps[it + dt_idx]
            vel_chunks[..., dt_idx] = s['v'][:, ix:ix + sub_size,
                                             iy:iy + sub_size,
                                             iz:iz + sub_size]
            prs_chunks[..., dt_idx] = s['p'][ix:ix + sub_size,
                                             iy:iy + sub_size,
                                             iz:iz + sub_size]
        u_mean = float(np.mean(vel_chunks[0]))
        u_means.append(u_mean)
        raw_data.append((vel_chunks, prs_chunks, ix, iy, iz, it))

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
    for idx, ((vel, prs, ix, iy, iz, it), label) in enumerate(
            tqdm(zip(raw_data, labels), desc=f"  {region} save", unit="file", total=n_subdomains)):
        coords = np.zeros((4, 2, sub_size, sub_size, sub_size, n_t), dtype=np.float32)
        coords[0, 0] = ix / Nx
        coords[0, 1] = ix / Nx
        coords[1, 0] = iy / Ny_full
        coords[1, 1] = iy / Ny_full
        coords[2, 0] = iz / Nz
        coords[2, 1] = iz / Nz
        for dt_idx in range(n_t):
            coords[3, 0, :, :, :, dt_idx] = (it + dt_idx) / max(n_t_avail, 1)
            coords[3, 1, :, :, :, dt_idx] = (it + dt_idx) / max(n_t_avail, 1)

        np.savez_compressed(
            os.path.join(output_dir, f'sub_{idx:04d}.npz'),
            velocity=vel, pressure=prs,
            label=np.array(label, dtype=np.int32),
            coords=coords, stats=stats_arr,
        )

    meta = {
        'region': region, 'n_subdomains': n_subdomains,
        'subdomain_size': sub_size, 'n_timesteps': n_t,
        'n_classes': 10, 'class_quantiles': q.tolist(),
        'label_counts': [int(np.sum(labels == c)) for c in range(10)],
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {n_subdomains} subdomains → {output_dir}")
    print(f"  Class distribution: {meta['label_counts']}")


# ============================================================================
#  Main
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description='Channel Flow DNS + Subdomain Extraction')
    p.add_argument('--Nx', type=int, default=128)
    p.add_argument('--Ny', type=int, default=129)
    p.add_argument('--Nz', type=int, default=128)
    p.add_argument('--Re_tau', type=float, default=180.0)
    p.add_argument('--dt', type=float, default=0.0015)
    p.add_argument('--n_steps', type=int, default=40000)
    p.add_argument('--save_every', type=int, default=200)
    p.add_argument('--snapshot_dir', type=str, default='data/generated/snapshots')
    p.add_argument('--out_centre', type=str, default='data/generated/centre')
    p.add_argument('--out_edge', type=str, default='data/generated/edge')
    p.add_argument('--n_subdomains', type=int, default=256)
    p.add_argument('--skip_sim', action='store_true')
    args = p.parse_args()

    for d in [args.snapshot_dir, args.out_centre, args.out_edge]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if not args.skip_sim:
        print("=" * 60)
        print(f"Channel Flow DNS  |  Re_tau={args.Re_tau}  "
              f"Grid={args.Nx}x{args.Ny}x{args.Nz}  dt={args.dt}")
        print(f"  nu = {1.0 / args.Re_tau:.6f},  Total steps = {args.n_steps}")
        print("=" * 60)
        sys.stdout.flush()

        dns = ChannelFlowDNS(Nx=args.Nx, Ny=args.Ny, Nz=args.Nz,
                             Re_tau=args.Re_tau, dt=args.dt)
        t0 = time.time()
        last_report = t0

        pbar = tqdm(total=args.n_steps, desc="DNS", unit="step",
                     dynamic_ncols=True, file=sys.stdout)
        for step in range(1, args.n_steps + 1):
            dns.step()

            if step % 100 == 0 or step <= 5:
                s = dns.get_stats()
                elapsed = time.time() - t0
                pbar.set_postfix_str(
                    f"u_bulk={s['u_bulk']:.2f} Re_tau={s['Re_tau_act']:.0f} "
                    f"urms={s['u_rms']:.3f} dPdx={s['dPdx']:.3f}"
                )
                pbar.update(100 if step > 5 else 1)
                if step % 100 == 0:
                    sys.stdout.flush()
            else:
                pbar.update(1)

            if step % args.save_every == 0:
                pbar.write(f"  [snapshot at step {step}]")
                u, v, w, p = dns.get_physical_fields()
                vel = np.stack([u, v, w], axis=0).astype(np.float32)
                np.savez_compressed(
                    os.path.join(args.snapshot_dir, f'snap_{step:06d}.npz'),
                    velocity=vel, pressure=p.astype(np.float32),
                    y=dns.y.astype(np.float32), t=np.float32(step * args.dt))
                sys.stdout.flush()

        pbar.close()
        print(f"Simulation complete. Total time: {time.time() - t0:.1f}s")
        sys.stdout.flush()

    # Extract
    print("\n" + "=" * 60)
    print("Extracting subdomains...")
    for reg, out in [('centre', args.out_centre), ('edge', args.out_edge)]:
        extract_subdomains(args.snapshot_dir, out, reg,
                           n_subdomains=args.n_subdomains)
    print("Done.")


if __name__ == '__main__':
    main()
