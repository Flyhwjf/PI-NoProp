"""Physics-informed loss L_phys^(t) using weak-form PDE residuals.

Implements eq.4.9:
    L_phys^(t) = (1/|K|) * Σ_k (||r_NS^(k)||² + ||r_cont^(k)||²)
"""
import torch
import torch.nn as nn
from .test_functions import TestFunctionGenerator
from .weak_form import WeakFormIntegrator


class PhysicsLoss(nn.Module):
    """Weak-form physics loss for PI-NoProp.

    Evaluates Navier-Stokes and continuity residuals in weak form
    using the reconstructed fields from the decoder.
    """

    def __init__(self, config, spider_vector_equation=None, spider_scalar_equation=None):
        """Initialize physics loss.

        Args:
            config: PINoPropConfig with physics settings
            spider_vector_equation: optional (coefficients, term_defs) tuple from
                SPIDER.discover() — used for custom momentum equation residual
            spider_scalar_equation: optional (coefficients, term_defs) tuple from
                SPIDER.discover() — used for custom continuity equation residual
        """
        super().__init__()
        self.config = config
        self.viscosity = config.physics.viscosity
        self.domain_size = config.physics.domain_size
        self.n_time = config.physics.n_time

        self.test_fn_gen = TestFunctionGenerator(
            beta=config.physics.beta,
            spatial_size=config.physics.domain_size,
            n_time=config.physics.n_time,
        )
        self.integrator = WeakFormIntegrator()

        # SPIDER-discovered equations for custom PDE residuals
        self.spider_vector_equation = spider_vector_equation
        self.spider_scalar_equation = spider_scalar_equation
        self.use_pp = getattr(config.physics, 'use_pressure_poisson', False)
        self.use_energy = getattr(config.physics, 'use_energy', False)

        # Precompute test functions and their gradients on init
        self.register_buffer('test_functions', None)
        self.register_buffer('grad_w_x', None)
        self.register_buffer('grad_w_y', None)
        self.register_buffer('grad_w_z', None)
        self.register_buffer('grad_w_t', None)
        self.register_buffer('lap_w', None)

        # Not precomputed yet
        self._ready = False

    def _ensure_test_functions(self, device):
        """Generate or retrieve cached test functions."""
        if not self._ready or self.test_functions is None or self.test_functions.device != device:
            w = self.test_fn_gen.generate(
                self.config.physics.n_test_functions, device=device,
            )
            self.test_functions = w

            grad_w, lap_w = self.test_fn_gen.compute_gradients(
                w, domain_size=self.domain_size)
            self.grad_w_x = grad_w[0]
            self.grad_w_y = grad_w[1]
            self.grad_w_z = grad_w[2]
            self.grad_w_t = grad_w[3]
            self.lap_w = lap_w
            self._ready = True

        return (self.test_functions,
                [self.grad_w_x, self.grad_w_y, self.grad_w_z, self.grad_w_t],
                self.lap_w)

    def forward(self, fields):
        """Compute physics loss from reconstructed fields.

        Args:
            fields: (batch_size, 4, H, H, H) or (batch_size, 4, H, H, H, Ht)
                    [u_x, u_y, u_z, p]
        Returns:
            loss: scalar physics loss
            residuals: dict with loss_ns, loss_cont, r_ns_norm, r_cont_norm
        """
        device = fields.device

        # Get or generate test functions
        w, grad_w, lap_w = self._ensure_test_functions(device)
        H = self.domain_size

        # Handle spatial-only input (no time dim) by adding fake time dim
        if fields.dim() == 5:
            # No time dimension → evaluate physics on this single snapshot
            # For the weak form we create a dummy time dimension
            fields = fields.unsqueeze(-1)  # (B, 4, H, H, H, 1)
            # The test functions have Ht=4 time steps, so we need to repeat
            if fields.shape[-1] == 1:
                fields = fields.expand(-1, -1, -1, -1, -1, self.n_time)

        # We need to match spatial dimensions
        # If fields are 32×32×32 but test functions are H×H×H, we need to subsample
        # or interpolate. For simplicity, use center crop if needed.
        if fields.shape[2] != H:
            # Center crop the fields to match test function size
            offset = (fields.shape[2] - H) // 2
            fields = fields[:, :, offset:offset+H, offset:offset+H, offset:offset+H, :]

        u = fields[:, :3]   # (B, 3, H, H, H, Ht)
        p = fields[:, 3:4]  # (B, 1, H, H, H, Ht)

        # Compute weak residuals — use SPIDER-discovered equations if provided
        has_custom_vector = self.spider_vector_equation is not None
        has_custom_scalar = self.spider_scalar_equation is not None

        if has_custom_vector or has_custom_scalar:
            if has_custom_vector:
                coeffs, tdefs = self.spider_vector_equation
                coeffs = coeffs.to(device) if hasattr(coeffs, 'to') else coeffs
                r_ns = self.integrator.compute_custom_residual(
                    u, p, tdefs, coeffs, w, grad_w, lap_w,
                    library_type='vector')
                loss_ns = torch.mean(r_ns ** 2)
            else:
                r_ns_fb, _ = self.integrator.integrate_ns(
                    u, p, w, grad_w, lap_w, nu=self.viscosity)
                loss_ns = torch.mean(r_ns_fb ** 2)
                r_ns = r_ns_fb

            if has_custom_scalar:
                coeffs, tdefs = self.spider_scalar_equation
                coeffs = coeffs.to(device) if hasattr(coeffs, 'to') else coeffs
                r_cont = self.integrator.compute_custom_residual(
                    u, p, tdefs, coeffs, w, grad_w, lap_w,
                    library_type='scalar')
                loss_cont = torch.mean(r_cont ** 2)
            else:
                _, r_cont_fb = self.integrator.integrate_ns(
                    u, p, w, grad_w, lap_w, nu=self.viscosity)
                loss_cont = torch.mean(r_cont_fb ** 2)
                r_cont = r_cont_fb
        else:
            # Standard path: hardcoded NS equations
            r_ns, r_cont = self.integrator.integrate_ns(
                u, p, w, grad_w, lap_w, nu=self.viscosity)
            loss_ns = torch.mean(r_ns ** 2)
            loss_cont = torch.mean(r_cont ** 2)

        total_loss = loss_ns
        if getattr(self.config.physics, 'use_continuity', True):
            total_loss = total_loss + loss_cont

        # Pressure-Poisson loss (if enabled)
        loss_pp = torch.tensor(0.0, device=device)
        if self.use_pp:
            r_pp = self.integrator.integrate_pp(u, p, w, lap_w, grad_w)
            loss_pp = torch.mean(r_pp ** 2)
            total_loss = total_loss + 0.5 * loss_pp

        # Energy equation loss (if enabled)
        loss_energy = torch.tensor(0.0, device=device)
        if self.use_energy:
            r_energy = self.integrator.integrate_energy(u, p, w, lap_w, grad_w, nu=self.viscosity)
            loss_energy = torch.mean(r_energy ** 2)
            total_loss = total_loss + 0.2 * loss_energy

        # Scale correction: weak form uses mean() which divides by N,
        # but integral = mean * 16 (domain volume [−1,1]⁴).
        # loss = mean(r²), so multiply by 16² = 256 to get correct integral magnitude.
        total_loss = total_loss * 256.0

        residuals = {
            'loss_ns': loss_ns.item(),
            'loss_cont': loss_cont.item(),
            'loss_pp': loss_pp.item(),
            'loss_energy': loss_energy.item(),
            'r_ns_norm': torch.norm(r_ns).item(),
            'r_cont_norm': torch.norm(r_cont).item(),
        }

        return total_loss, residuals

    def compute_eta_ns(self, fields):
        """Compute relative weak NS residual η_NS (eq.4.11).

        η_NS = ||R_NS||_weak / (||∂_t u|| + ||(u·∇)u|| + ||∇p|| + ν||∇²u||)

        Uses the standard NS decomposition regardless of SPIDER equations.
        Returns 0.0 if the denominators are zero.
        """
        device = fields.device
        w, grad_w, lap_w = self._ensure_test_functions(device)
        H = self.domain_size

        if fields.dim() == 5:
            fields = fields.unsqueeze(-1)
            if fields.shape[-1] == 1:
                fields = fields.expand(-1, -1, -1, -1, -1, self.n_time)

        if fields.shape[2] != H:
            offset = (fields.shape[2] - H) // 2
            fields = fields[:, :, offset:offset+H, offset:offset+H, offset:offset+H, :]

        u = fields[:, :3]
        p = fields[:, 3:4]

        r_time, r_conv, r_press, r_visc, r_cont = self.integrator.integrate_ns_terms(
            u, p, w, grad_w, lap_w, nu=self.viscosity,
        )

        r_ns = r_time + r_conv + r_press + r_visc
        r_ns_norm = torch.norm(r_ns).item()
        denom = (torch.norm(r_time).item() + torch.norm(r_conv).item()
                 + torch.norm(r_press).item() + torch.norm(r_visc).item())

        if denom > 0:
            return r_ns_norm / denom
        return 0.0
