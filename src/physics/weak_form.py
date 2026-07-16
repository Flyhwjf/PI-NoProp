"""Weak-form integration engine for PDE residuals.

Computes weak-form integrals of PDE terms against test functions.
Integration by parts shifts derivatives from fields to test functions,
avoiding numerical differentiation of potentially noisy data.
"""
import torch


class WeakFormIntegrator:
    """Compute weak-form integrals of NS equation terms.

    For each test function w_k, computes:
        r_time:  ∫ w · ∂_t u      → -∫ u · ∂_t w          (IBP in time)
        r_conv:  ∫ w · (u·∇)u     → -∫ u_c u_j ∂w/∂x_j    (IBP + ∇·u=0)
        r_press: ∫ w · ∇p         → -∫ p · ∇w             (IBP in space)
        r_visc:  ∫ w · (-ν∇²u)    → -ν∫ u · ∇²w           (IBP twice)
        r_cont:  ∫ w · (∇·u)      → -∫ u · ∇w             (IBP in space)
    
    The domain is [-1,1]^4, so the volume factor is 2^4 = 16.
    """

    def __init__(self):
        pass

    def integrate_ns_terms(self, u, p, w, grad_w, lap_w, nu=5e-5):
        """Compute all individual NS weak-form terms separately.

        Args:
            u: (B, 3, H, H, H, Ht) velocity
            p: (B, 1, H, H, H, Ht) pressure
            w: (K, H, H, H, Ht) test functions
            grad_w: [∂w/∂x, ∂w/∂y, ∂w/∂z, ∂w/∂t] each (K, H, H, H, Ht)
            lap_w: (K, H, H, H, Ht) ∇²w
            nu: viscosity
        Returns:
            r_time:  (B, K, 3)
            r_conv:  (B, K, 3)
            r_press: (B, K, 3)
            r_visc:  (B, K, 3)
            r_cont:  (B, K)
        """
        B = u.shape[0]
        K = w.shape[0]
        device = u.device
        dtype = u.dtype

        # All derivative-of-w terms have shape (K, H, H, H, Ht)
        # We unsqueeze to (1, K, ...) for broadcasting with (B, 1, ...)
        # Product: (B, 1, ...) * (1, K, ...) → (B, K, ...)
        # Mean over spatial/time dims (-4,-3,-2,-1) → (B, K)

        # Time derivative: ∫ w · ∂_t u → -∫ u · ∂_t w
        dw_dt = grad_w[3]
        r_time = torch.zeros(B, K, 3, device=device, dtype=dtype)
        for c in range(3):
            u_c = u[:, c].unsqueeze(1)
            term = -u_c * dw_dt.unsqueeze(0)
            r_time[:, :, c] = term.mean(dim=(-4, -3, -2, -1))

        # Convection via IBP + incompressibility: -∫ u_c u_j ∂w/∂x_j
        r_conv = torch.zeros(B, K, 3, device=device, dtype=dtype)
        for c in range(3):
            for j in range(3):
                dw_dxj = grad_w[j]
                u_c_u_j = (u[:, c] * u[:, j]).unsqueeze(1)
                term = -u_c_u_j * dw_dxj.unsqueeze(0)
                r_conv[:, :, c] = r_conv[:, :, c] + term.mean(dim=(-4, -3, -2, -1))

        # Pressure gradient: ∫ w · ∇p → -∫ p · ∇w
        r_press = torch.zeros(B, K, 3, device=device, dtype=dtype)
        for c in range(3):
            dw_dxc = grad_w[c]
            term = -p * dw_dxc.unsqueeze(0)
            r_press[:, :, c] = term.squeeze(1).mean(dim=(-4, -3, -2, -1))

        # Viscous: -ν∫ w · ∇²u → -ν∫ u · ∇²w
        r_visc = torch.zeros(B, K, 3, device=device, dtype=dtype)
        for c in range(3):
            u_c = u[:, c].unsqueeze(1)
            term = -nu * u_c * lap_w.unsqueeze(0)
            r_visc[:, :, c] = term.mean(dim=(-4, -3, -2, -1))

        # Continuity: ∫ w · (∇·u) → -∫ u · ∇w
        r_cont = torch.zeros(B, K, device=device, dtype=dtype)
        for c in range(3):
            dw_dxc = grad_w[c]
            u_c = u[:, c].unsqueeze(1)
            term = -u_c * dw_dxc.unsqueeze(0)
            r_cont = r_cont + term.mean(dim=(-4, -3, -2, -1))

        return r_time, r_conv, r_press, r_visc, r_cont

    def integrate_ns(self, u, p, w, grad_w, lap_w, nu=5e-5):
        """Compute NS residuals (combined).

        Returns:
            r_ns: (B, K, 3) NS residual per test function per component
            r_cont: (B, K) continuity residual
        """
        r_time, r_conv, r_press, r_visc, r_cont = self.integrate_ns_terms(
            u, p, w, grad_w, lap_w, nu,
        )
        r_ns = r_time + r_conv + r_press + r_visc
        return r_ns, r_cont

    # ---------------------------------------------------------------
    # Pressure-Poisson & Energy weak-form integration
    # ---------------------------------------------------------------

    def integrate_pp(self, u, p, w, lap_w, grad_w):
        """Pressure-Poisson weak form.

        Weak form of: nabla^2 p + nabla.((u.nabla)u) = 0

        IBP identities:
          int w * nabla^2 p = int p * nabla^2 w
          int w * nabla.((u.nabla)u) = -int (u.nabla)u . nabla w

        Args:
            u: (B, 3, H, H, H, Nt) velocity
            p: (B, 1, H, H, H, Nt) pressure
            w: (K, H, H, H, Nt) test functions
            lap_w: (K, H, H, H, Nt) spatial laplacian of w
            grad_w: [dw/dx, dw/dy, dw/dz, dw/dt] each (K, H, H, H, Nt)
        Returns:
            r_pp: (B, K) PP residual per test function
        """
        B = u.shape[0]
        K = w.shape[0]
        ndim = u.shape[1]
        device = u.device
        dtype = u.dtype

        p_s = p.squeeze(1).unsqueeze(1)
        r_pp = (p_s * lap_w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

        conv_contrib = torch.zeros(B, K, device=device, dtype=dtype)
        for c in range(ndim):
            for j in range(ndim):
                u_c = u[:, c].unsqueeze(1)
                u_j = u[:, j].unsqueeze(1)
                du_dxj = torch.gradient(u[:, c], dim=j+1, spacing=1.0)[0].unsqueeze(1)
                dw_dxj = grad_w[j].unsqueeze(0)
                conv_contrib = conv_contrib + (u_j * du_dxj * dw_dxj).mean(dim=(-4, -3, -2, -1))

        r_pp = r_pp - conv_contrib
        return r_pp

    def integrate_energy(self, u, p, w, lap_w, grad_w, nu=0.005):
        """Energy equation weak form.

        Energy: E = 0.5|u|^2
        PDE: dE/dt + nabla.(uE + up) = nu * nabla^2 E - nu * nabla u : nabla u

        IBP identities:
          int w * dE/dt = -int E * dw/dt
          int w * nabla.(uE) = -int (uE) . nabla w
          int w * nabla.(up) = -int (up) . nabla w
          nu * int w * nabla^2 E = nu * int E * nabla^2 w
          -nu * int w * (nabla u : nabla u) — no IBP, kept as is

        Returns:
            r_energy: (B, K) energy residual per test function
        """
        B = u.shape[0]
        K = w.shape[0]
        ndim = u.shape[1]
        device = u.device
        dtype = u.dtype

        E = 0.5 * (u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2)
        E_s = E.unsqueeze(1)

        dw_dt = grad_w[3].unsqueeze(0)
        r_energy = -(E_s * dw_dt).mean(dim=(-4, -3, -2, -1))

        for c in range(ndim):
            u_c = u[:, c].unsqueeze(1)
            dw_dxc = grad_w[c].unsqueeze(0)
            r_energy = r_energy - (u_c * E_s * dw_dxc).mean(dim=(-4, -3, -2, -1))

        p_s = p.squeeze(1).unsqueeze(1)
        for c in range(ndim):
            u_c = u[:, c].unsqueeze(1)
            dw_dxc = grad_w[c].unsqueeze(0)
            r_energy = r_energy - (u_c * p_s * dw_dxc).mean(dim=(-4, -3, -2, -1))

        r_energy = r_energy + nu * (E_s * lap_w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

        dissipation = torch.zeros_like(E)
        for i in range(ndim):
            for j in range(ndim):
                du_i_dx_j = torch.gradient(u[:, i], dim=j+1, spacing=1.0)[0]
                dissipation = dissipation + du_i_dx_j**2
        r_energy = r_energy - nu * (dissipation.unsqueeze(1) * w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

        return r_energy

    # ---------------------------------------------------------------
    # Custom residual computation for SPIDER-discovered equations
    # ---------------------------------------------------------------

    def compute_custom_residual(self, u, p, term_defs, coefficients,
                                 w, grad_w, lap_w, library_type='vector'):
        """Compute residual for a custom SPIDER-discovered equation.

        Each term's weak-form integral is computed via IBP, then combined
        with the discovered coefficients to form the PDE residual.

        Args:
            u: (B, 3, H, H, H, Ht) velocity
            p: (B, 1, H, H, H, Ht) pressure
            term_defs: list of dicts with keys 'type', 'fields'
            coefficients: (n_terms,) tensor of discovered coefficients
            w: (K, H, H, H, Ht) test functions
            grad_w: list of 4 gradient tensors (K, H, H, H, Ht)
            lap_w: (K, H, H, H, Ht) Laplacian
            library_type: 'vector' or 'scalar'
        Returns:
            residual: (B, K) per batch and test function
        """
        if library_type == 'vector':
            return self._custom_vector_residual(
                u, p, term_defs, coefficients, w, grad_w, lap_w)
        else:
            return self._custom_scalar_residual(
                u, p, term_defs, coefficients, w, grad_w, lap_w)

    def _custom_vector_residual(self, u, p, term_defs, coefficients,
                                 w, grad_w, lap_w):
        """Custom residual for vector (momentum) equation."""
        B = u.shape[0]
        ndim = u.shape[1]
        K = w.shape[0]
        device = u.device
        dtype = u.dtype
        residual = torch.zeros(B, K, device=device, dtype=dtype)

        for coeff, term in zip(coefficients, term_defs):
            ttype = term['type']

            if ttype == 'field':
                # u: Σ_c ∫ w · u_c
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)
                    total = total + (u_c * w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'time_deriv':
                # ∂_t u: Σ_c -∫ u_c · ∂_t w
                dw_dt = grad_w[3]
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)
                    total = total + (-u_c * dw_dt.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'gradient':
                # ∇p: -∫ p · Σ_c ∂w/∂x_c
                p_rep = p.squeeze(1).unsqueeze(1)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    dw_dxc = grad_w[c].unsqueeze(0)
                    total = total + (-p_rep * dw_dxc).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'field_product':
                # pu: Σ_c ∫ w · p · u_c
                p_rep = p.squeeze(1).unsqueeze(1)  # (B, 1, H, H, H, Ht)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)  # (B, 1, H, H, H, Ht)
                    prod = p_rep * u_c * w.unsqueeze(0)  # (B, K, H, H, H, Ht)
                    total = total + prod.mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'convection' or ttype == 'convection_alt':
                # (u·∇)u: -Σ_c Σ_j ∫ u_c · u_j · ∂w/∂x_j
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    for j in range(ndim):
                        u_c_u_j = (u[:, c] * u[:, j]).unsqueeze(1)
                        dw_dxj = grad_w[j].unsqueeze(0)
                        total = total + (-u_c_u_j * dw_dxj).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'laplacian':
                # ∇²u: Σ_c ∫ u_c · ∇²w
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)
                    total = total + (u_c * lap_w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'field_cube':
                # u²u: Σ_c ∫ w · |u|² · u_c
                u_sq = (u ** 2).sum(dim=1, keepdim=True)  # (B, 1, H, H, H, Ht)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)  # (B, 1, H, H, H, Ht)
                    prod = w.unsqueeze(0) * u_sq * u_c  # (B, K, H, H, H, Ht)
                    total = total + prod.mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'field_sq_field':
                # p²u: Σ_c ∫ w · p² · u_c
                p_sq = p ** 2  # (B, 1, H, H, H, Ht)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)  # (B, 1, H, H, H, Ht)
                    prod = w.unsqueeze(0) * p_sq * u_c  # (B, K, H, H, H, Ht)
                    total = total + prod.mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'field_gradient':
                # p∇p: -½ Σ_c ∫ p² · ∂w/∂x_c
                p_sq = (p ** 2).squeeze(1).unsqueeze(1)  # (B, 1, H, H, H, Ht)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    dw_dxc = grad_w[c].unsqueeze(0)
                    total = total + (-0.5 * p_sq * dw_dxc).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'field_div':
                # u(∇·u): Σ_c ∫ w · u_c · (∇·u)
                # Compute ∇·u with numerical gradients
                div_u = torch.zeros_like(u[:, 0])
                for j in range(ndim):
                    div_u = div_u + torch.gradient(u[:, j], dim=(-4, -3, -2))[0]
                div_u = div_u.unsqueeze(1)  # (B, 1, H, H, H, Ht)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)  # (B, 1, H, H, H, Ht)
                    prod = w.unsqueeze(0) * u_c * div_u  # (B, K, H, H, H, Ht)
                    total = total + prod.mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'grad_div':
                # ∇(∇·u): -Σ_c ∫ (∇·u) · ∂w/∂x_c
                div_u = torch.zeros_like(u[:, 0])
                for j in range(ndim):
                    div_u = div_u + torch.gradient(u[:, j], dim=(-4, -3, -2))[0]
                div_u = div_u.unsqueeze(1)  # (B, 1, H, H, H, Ht)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    dw_dxc = grad_w[c].unsqueeze(0)  # (1, K, H, H, H, Ht)
                    total = total + (-div_u * dw_dxc).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype in ('field_time_deriv', 'field_time_deriv_scalar',
                            'time_deriv2', 'gradient_time'):
                # Time-derivative terms — zero when data has no time dimension
                integral = torch.zeros(B, K, device=device, dtype=dtype)

            else:
                integral = torch.zeros(B, K, device=device, dtype=dtype)

            residual = residual + coeff * integral

        return residual

    def _custom_scalar_residual(self, u, p, term_defs, coefficients,
                                 w, grad_w, lap_w):
        """Custom residual for scalar (continuity) equation."""
        B = u.shape[0]
        ndim = u.shape[1]
        K = w.shape[0]
        device = u.device
        dtype = u.dtype
        residual = torch.zeros(B, K, device=device, dtype=dtype)

        for coeff, term in zip(coefficients, term_defs):
            ttype = term['type']

            if ttype == 'constant':
                # ∫ w
                integral = w.unsqueeze(0).mean(dim=(-4, -3, -2, -1)).expand(B, -1)

            elif ttype == 'scalar_field':
                # ∫ w · p
                p_rep = p.squeeze(1).unsqueeze(1)  # (B, 1, H, H, H, Ht)
                integral = (p_rep * w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

            elif ttype == 'divergence':
                # ∇·u: -Σ_c ∫ u_c · ∂w/∂x_c
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for c in range(ndim):
                    u_c = u[:, c].unsqueeze(1)
                    dw_dxc = grad_w[c].unsqueeze(0)
                    total = total + (-u_c * dw_dxc).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'scalar_time_deriv':
                integral = torch.zeros(B, K, device=device, dtype=dtype)

            elif ttype == 'scalar_sq':
                # ∫ w · p²
                p_sq = (p ** 2).squeeze(1).unsqueeze(1)
                integral = (p_sq * w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

            elif ttype == 'vector_magsq':
                # ∫ w · |u|²
                u_sq = (u ** 2).sum(dim=1, keepdim=True)  # (B, 1, H, H, H, Ht)
                integral = (u_sq * w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

            elif ttype == 'scalar_cube':
                # ∫ w · p³
                p_cube = (p ** 3).squeeze(1).unsqueeze(1)
                integral = (p_cube * w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

            elif ttype == 'vector_magsq_field':
                # ∫ w · |u|² · p
                u_sq = (u ** 2).sum(dim=1, keepdim=True)
                p_rep = p.squeeze(1).unsqueeze(1)
                integral = (u_sq * p_rep * w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

            elif ttype == 'scalar_laplacian':
                # ∇²p: ∫ p · ∇²w
                p_rep = p.squeeze(1).unsqueeze(1)
                integral = (p_rep * lap_w.unsqueeze(0)).mean(dim=(-4, -3, -2, -1))

            elif ttype == 'scalar_convection':
                # u·∇p: -Σ_j ∫ p · u_j · ∂w/∂x_j
                p_rep = p.squeeze(1).unsqueeze(1)
                total = torch.zeros(B, K, device=device, dtype=dtype)
                for j in range(ndim):
                    u_j = u[:, j].unsqueeze(1)
                    dw_dxj = grad_w[j].unsqueeze(0)
                    total = total + (-p_rep * u_j * dw_dxj).mean(dim=(-4, -3, -2, -1))
                integral = total

            elif ttype == 'scalar_field_div':
                # p(∇·u): ∫ w · p · (∇·u)
                div_u = torch.zeros_like(u[:, 0])
                for j in range(ndim):
                    div_u = div_u + torch.gradient(u[:, j], dim=(-4, -3, -2))[0]
                p_rep = p.squeeze(1)  # (B, H, H, H, Ht)
                product = w.unsqueeze(0) * p_rep.unsqueeze(1) * div_u.unsqueeze(1)
                integral = product.mean(dim=(-4, -3, -2, -1))

            elif ttype in ('scalar_field_time_deriv', 'scalar_time_deriv2',
                            'vector_dot_time_deriv'):
                integral = torch.zeros(B, K, device=device, dtype=dtype)

            else:
                integral = torch.zeros(B, K, device=device, dtype=dtype)

            residual = residual + coeff * integral

        return residual
