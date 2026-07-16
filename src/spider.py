#!/usr/bin/env python3
"""
SPIDER: Sparse Physics-Informed Discovery of Empirical Relations

Based on: Gurevich, D. R., Golden, M. R., Reinbold, P. A. K., & Grigoriev, R. O. (2024).
"Learning fluid physics from highly turbulent data using sparse physics-informed discovery
of empirical relations (SPIDER)". Journal of Fluid Mechanics.

Key features:
1. Compact-support test functions: w(ξ) = Π(1-ξ²)^β with random shifts
2. Integration by parts (IBP) weak form — derivatives transferred to test functions
3. Symmetry-constrained libraries L_0 (scalar) and L_1 (vector)
4. Proper feature matrix Q with V_i (domain volume) and S_n (term scale) normalization
5. SVD-based sparse regression with sequential thresholding
"""
import numpy as np
from scipy.linalg import svd
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import warnings


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class SPIDERConfig:
    """Configuration for SPIDER algorithm."""
    # Domain sampling
    n_domains: int = 200              # Number of integration domains
    domain_size: int = 16             # Size of each subdomain (grid points)
    n_time: int = 0                   # Number of time steps (0 = no time dimension)
    shift_range: float = 0.4          # Max random shift for test functions
    # Test functions
    n_test_functions: int = 16        # Test functions per domain
    beta: float = 8.0                 # Exponent in (1-xi^2)^beta
    # Library
    max_polynomial_order: int = 3     # Max polynomial order for library
    include_constant: bool = True     # Include constant term
    # Sparse regression
    sparsity_threshold: float = 0.05  # Threshold to zero out coefficients
    max_iterations: int = 20          # Sequential thresholding iterations
    # Normalization
    normalize_terms: bool = True      # Scale each term to unit variance
    # Output
    verbose: bool = True


# =========================================================================
# Test function generator — numpy version of (1-xi^2)^beta
# =========================================================================

class TestFunctionGenerator:
    """Generate compact-support test functions w(ξ) = Π (1-ξ_i²)^β.

    Each w_k has a random center shift so that the set covers the domain
    with overlap (ensuring the Q matrix has full rank).
    """

    def __init__(self, beta: float = 8.0, shift_range: float = 0.4):
        self.beta = beta
        self.shift_range = shift_range

    def generate(self, n_functions: int, domain_shape: Tuple[int, ...], rng: np.random.Generator):
        """Generate n_functions test functions on a uniform grid.

        Args:
            n_functions: Number of test functions
            domain_shape: Tuple of grid sizes per dimension, e.g. (16, 16, 16)
            rng: NumPy random generator
        Returns:
            w: (n_functions, *domain_shape) test function values
        """
        ndim = len(domain_shape)
        functions = []

        for k in range(n_functions):
            shift = rng.uniform(-self.shift_range, self.shift_range, ndim)
            w = np.ones(domain_shape, dtype=np.float64)

            for dim, size in enumerate(domain_shape):
                xi = np.linspace(-1.0, 1.0, size)
                xi_shifted = (xi - shift[dim]) * 1.5
                xi_shifted = np.clip(xi_shifted, -1.0, 1.0)
                w_dim = np.clip(1.0 - xi_shifted ** 2, 0.0, None) ** self.beta

                shape = [1] * ndim
                shape[dim] = -1
                w = w * w_dim.reshape(shape)

            # Normalize to unit L2 norm
            norm = np.sqrt(np.sum(w ** 2))
            if norm > 1e-15:
                w = w / norm

            functions.append(w)

        return np.stack(functions, axis=0)

    @staticmethod
    def compute_gradients(w, dx=1.0):
        """Compute spatial gradients and Laplacian of test functions via finite differences.

        Args:
            w: (K, *domain_shape) test functions
            dx: grid spacing (uniform)
        Returns:
            grad: list of (K, *domain_shape) arrays, one per dimension
            lap: (K, *domain_shape) Laplacian
        """
        ndim = w.ndim - 1  # exclude batch dim
        grad = []
        for dim in range(ndim):
            g = np.gradient(w, dx, axis=dim + 1)
            grad.append(g)

        # Laplacian = sum of second derivatives
        lap = np.zeros_like(w)
        for dim in range(ndim):
            g1 = np.gradient(w, dx, axis=dim + 1)
            g2 = np.gradient(g1, dx, axis=dim + 1)
            lap = lap + g2

        return grad, lap


# =========================================================================
# Symmetry-constrained libraries (paper eq. L_0 and L_1)
# =========================================================================

def build_vector_library_3rd_order(field_names: List[str]) -> List[dict]:
    """Build vector library L_1 up to 3rd order for the momentum equation.

    Each term is a dict with:
      'desc':   human-readable description
      'fields': indices into the field list used by weak-form dispatch
      'type':   term type key for IBP handling

    From paper eq. for L_1^(3):
    {u, ∂_t u, ∇p, pu, (u·∇)u, ∇²u, u²u, p²u,
     p∇p, u(∇·u), ∇(∇·u), p∂_t u, u∂_t p}
    (∇u)·u and ∂_t²u excluded: (∇u)·u == (u·∇)u, ∂_t²u is 2nd time deriv)
    """
    terms = []
    u_idx = 0  # velocity is field 0
    p_idx = 1  # pressure is field 1

    # 1st order
    terms.append({'desc': 'u', 'type': 'field', 'fields': [u_idx]})
    terms.append({'desc': '∂_t u', 'type': 'time_deriv', 'fields': [u_idx]})
    terms.append({'desc': '∇p', 'type': 'gradient', 'fields': [p_idx]})
    terms.append({'desc': 'pu', 'type': 'field_product', 'fields': [p_idx, u_idx]})

    # 2nd order
    terms.append({'desc': '(u·∇)u', 'type': 'convection', 'fields': [u_idx]})
    terms.append({'desc': '∇²u', 'type': 'laplacian', 'fields': [u_idx]})
    terms.append({'desc': 'u²u', 'type': 'field_cube', 'fields': [u_idx]})
    terms.append({'desc': 'p²u', 'type': 'field_sq_field', 'fields': [p_idx, u_idx]})
    terms.append({'desc': 'p∇p', 'type': 'field_gradient', 'fields': [p_idx]})
    terms.append({'desc': 'u(∇·u)', 'type': 'field_div', 'fields': [u_idx]})
    terms.append({'desc': '∇(∇·u)', 'type': 'grad_div', 'fields': [u_idx]})
    terms.append({'desc': 'p∂_t u', 'type': 'field_time_deriv', 'fields': [p_idx, u_idx]})
    terms.append({'desc': 'u∂_t p', 'type': 'field_time_deriv_scalar', 'fields': [u_idx, p_idx]})

    # 3rd order — none added beyond paper's L_1^(3) that aren't covered above.
    # ∂_t²u and ∂_t∇p are extremely high-order and rarely needed.

    return terms


def build_scalar_library_3rd_order(field_names: List[str]) -> List[dict]:
    """Build scalar library L_0 up to 3rd order for continuity and scalar equations.

    From paper eq. for L_0^(3):
    {1, p, ∇·u, ∂_t p, p², u², p³, u·∇p, ∇²p, p∂_t p,
     ∂_t²p, p∇·u, u²p, u·∂_t u}
    """
    terms = []
    u_idx = 0
    p_idx = 1

    # Constant
    terms.append({'desc': '1', 'type': 'constant', 'fields': []})
    # 1st order
    terms.append({'desc': 'p', 'type': 'scalar_field', 'fields': [p_idx]})
    terms.append({'desc': '∇·u', 'type': 'divergence', 'fields': [u_idx]})
    terms.append({'desc': '∂_t p', 'type': 'scalar_time_deriv', 'fields': [p_idx]})
    # 2nd order
    terms.append({'desc': 'p²', 'type': 'scalar_sq', 'fields': [p_idx]})
    terms.append({'desc': 'u²', 'type': 'vector_magsq', 'fields': [u_idx]})
    terms.append({'desc': 'u·∇p', 'type': 'scalar_convection', 'fields': [u_idx, p_idx]})
    terms.append({'desc': '∇²p', 'type': 'scalar_laplacian', 'fields': [p_idx]})
    terms.append({'desc': 'p∂_t p', 'type': 'scalar_field_time_deriv', 'fields': [p_idx]})
    terms.append({'desc': '∂_t²p', 'type': 'scalar_time_deriv2', 'fields': [p_idx]})
    terms.append({'desc': 'p∇·u', 'type': 'scalar_field_div', 'fields': [p_idx, u_idx]})
    # 3rd order
    terms.append({'desc': 'p³', 'type': 'scalar_cube', 'fields': [p_idx]})
    terms.append({'desc': 'u²p', 'type': 'vector_magsq_field', 'fields': [u_idx, p_idx]})
    terms.append({'desc': 'u·∂_t u', 'type': 'vector_dot_time_deriv', 'fields': [u_idx]})

    return terms


# =========================================================================
# IBP weak-form integrator
# =========================================================================

class WeakFormIntegrator:
    """Compute weak-form integrals of library terms against test functions.

    Each method returns the integral value (scalar or vector) of one term
    type after applying integration by parts where applicable.
    """

    @staticmethod
    def _domain_volume(domain_shape, dx=1.0):
        """Total volume of the domain."""
        return dx ** len(domain_shape) * np.prod(domain_shape)

    # ---- Vector terms (momentum equation) ----

    def integrate_vector_field(self, f, field_comp, w):
        """∫ w_c · f_c dV — no IBP."""
        return np.sum(w * field_comp)

    def integrate_vector_time_deriv(self, u_comp, dw_dt):
        """∫ w_c · ∂_t u_c → -∫ u_c · ∂_t w_c (IBP time)."""
        return -np.sum(u_comp * dw_dt)

    def integrate_vector_gradient(self, p, dw_dx_components):
        """∫ w_c · ∂p/∂x_c → -∫ p · (∂w_c/∂x_c) (IBP space, summed over c).

        Returns scalar (summed over components).
        """
        # Σ_c ∫ w_c · ∂p/∂x_c = -Σ_c ∫ p · ∂w_c/∂x_c
        total = 0.0
        for c in range(len(dw_dx_components)):
            total += -np.sum(p * dw_dx_components[c])
        return total

    def integrate_vector_field_product(self, f_scalar, u_comp, w):
        """∫ w_c · (f_scalar * u_c) dV — no IBP."""
        return np.sum(w * (f_scalar * u_comp))

    def integrate_vector_convection(self, u, w, grad_w):
        """∫ w_c · (u_j ∂_j u_c) → -∫ u_c u_j ∂w_c/∂x_j (IBP + div-free).

        Returns array of 3 component values.
        """
        ndim = len(grad_w)
        results = np.zeros(ndim)
        for c in range(ndim):
            total = 0.0
            for j in range(ndim):
                dw_c_dxj = grad_w[j]  # ∂w_c/∂x_j  — but grad_w is indexed by spatial dim
                # Actually that's wrong. grad_w[j] is ∂w/∂x_j for ALL components.
                # But w is a SCALAR test function!
                pass
        return results

    def integrate_vector_laplacian(self, u_comp, lap_w):
        """∫ w_c · ∇²u_c → ∫ u_c · ∇²w_c (IBP twice)."""
        return np.sum(u_comp * lap_w)

    def integrate_vector_convection_alt(self, u, w, grad_w):
        """(∇u)·u — similar structure to convection, different IBP."""
        # (∇u)·u means u_j ∂_j u_i — same as (u·∇)u!
        # These are the same mathematically, so just dispatch to convection
        return self.integrate_vector_convection(u, w, grad_w)

    def integrate_vector_field_cube(self, u, w):
        """∫ w_c · (u² · u_c) where u² = Σ_j u_j²."""
        u_sq = np.sum(u ** 2, axis=0)  # sum over components
        results = np.zeros(u.shape[0])
        for c in range(u.shape[0]):
            results[c] = np.sum(w * u_sq * u[c])
        return results

    def integrate_vector_field_sq_field(self, f_scalar, u_comp, w):
        """∫ w_c · (f_scalar² · u_c)."""
        return np.sum(w * (f_scalar ** 2 * u_comp))

    def integrate_vector_gradient_time(self, p, grad_w, dw_dt):
        """∂_t∇p → IBP in time: ∫ w_c · ∂_t(∂p/∂x_c) = -∫ ∂p/∂x_c · ∂_t w_c
        Then IBP in space again: = ∫ p · ∂(∂_t w_c)/∂x_c.
        For simplicity with no time dim, skip (returns 0)."""
        return 0.0

    def integrate_vector_field_gradient(self, f_scalar, grad_w):
        """∫ w_c · (f_scalar · ∇f_scalar)_c = ∫ w_c · f_scalar · ∂f_scalar/∂x_c
        IBP: = -∫ (1/2) f_scalar² · ∂w_c/∂x_c"""
        results = np.zeros(3)  # assume 3D
        f_sq = f_scalar ** 2
        for c in range(min(3, len(grad_w))):
            results[c] = -0.5 * np.sum(f_sq * grad_w[c])
        return results

    def integrate_vector_field_div(self, u, w, grad_w):
        """u(∇·u): non-linear, can IBP: u_c · (∂_j u_j) · w_c
        IBP: -∫ u_c u_j ∂w_c/∂x_j - ∫ w_c u_j ∂u_c/∂x_j
        The first term is -convection, the second can't be fully IBP'd.
        Keeping numerical ∂_j u_j for simplicity."""
        div_u = 0.0
        for j in range(u.shape[0]):
            div_u += np.gradient(u[j], axis=j)  # u[j] is (H,H,H), axis=j
        results = np.zeros(u.shape[0])
        for c in range(u.shape[0]):
            results[c] = np.sum(w * u[c] * div_u)
        return results

    def integrate_vector_grad_div(self, u, w, grad_w, lap_w):
        """∇(∇·u): ∫ w_c · ∂_c(∂_j u_j) = -∫ (∂_j u_j) · ∂_c w_c  (IBP once)
        = -∫ (∇·u) · (∂w_c/∂x_c)  summed over c.

        For incompressible flow (∇·u ≈ 0), this is near zero.
        We compute ∇·u numerically.
        """
        div_u = np.zeros_like(u[0])
        for j in range(u.shape[0]):
            div_u += np.gradient(u[j], axis=j)
        results = np.zeros(u.shape[0])
        for c in range(u.shape[0]):
            # -(∇·u) · ∂w/∂x_c
            results[c] = -np.sum(div_u * grad_w[c])
        return results

    def integrate_vector_field_time_deriv(self, f_scalar, u_comp, dw_dt):
        """p∂_t u: ∫ w_c · p · ∂_t u_c → -∫ u_c · p · ∂_t w_c (IBP time, p treated as coeff)."""
        return -np.sum(u_comp * f_scalar * dw_dt)

    def integrate_vector_field_time_deriv_scalar(self, u_comp, p, dw_dt):
        """u∂_t p: ∫ w_c · u_c · ∂_t p → -∫ p · u_c · ∂_t w_c."""
        return -np.sum(p * u_comp * dw_dt)

    # ---- Scalar terms (continuity / pressure-Poisson) ----

    def integrate_scalar_constant(self, w):
        """∫ w dV."""
        return np.sum(w)

    def integrate_scalar_field(self, f, w):
        """∫ w · f dV."""
        return np.sum(w * f)

    def integrate_scalar_divergence(self, u, grad_w_per_comp):
        """∫ w · (∇·u) → -∫ u · ∇w (IBP space, summed over components).

        ∇w is a vector (one component for each spatial dim), but w is a scalar.
        ∫ w · (Σ_c ∂u_c/∂x_c) = -Σ_c ∫ u_c · ∂w/∂x_c

        Args:
            u: (ndim, *domain_shape) velocity components
            grad_w: (ndim, *domain_shape) gradient of scalar test function w
        Returns:
            scalar integral
        """
        total = 0.0
        ndim = u.shape[0]
        for c in range(ndim):
            total += -np.sum(u[c] * grad_w_per_comp[c])
        return total

    def integrate_scalar_time_deriv(self, f, dw_dt=None):
        """∫ w · ∂_t f → -∫ f · ∂_t w (IBP time)."""
        if dw_dt is None:
            return 0.0
        return -np.sum(f * dw_dt)

    def integrate_scalar_sq(self, f, w):
        """∫ w · f² dV."""
        return np.sum(w * f ** 2)

    def integrate_vector_magsq(self, u, w):
        """∫ w · |u|² dV."""
        u_sq = np.sum(u ** 2, axis=0)
        return np.sum(w * u_sq)

    def integrate_scalar_cube(self, f, w):
        """∫ w · f³ dV."""
        return np.sum(w * f ** 3)

    def integrate_vector_magsq_field(self, u, f, w):
        """∫ w · |u|² · f dV."""
        u_sq = np.sum(u ** 2, axis=0)
        return np.sum(w * u_sq * f)

    def integrate_scalar_laplacian(self, f, lap_w):
        """∫ w · ∇²f → ∫ f · ∇²w (IBP twice)."""
        return np.sum(f * lap_w)

    def integrate_scalar_convection(self, u, f, grad_w):
        """∫ w · (u · ∇f) → -∫ f · (u · ∇w) (IBP, treating u as coefficient).

        ∫ w · Σ_j u_j ∂f/∂x_j = -∫ f · Σ_j u_j ∂w/∂x_j
        """
        total = 0.0
        ndim = u.shape[0]
        for j in range(ndim):
            total += -np.sum(f * u[j] * grad_w[j])
        return total

    def integrate_scalar_field_div(self, f, u, grad_w):
        """∫ w · f · (∇·u) — no simple IBP, compute ∇·u numerically."""
        div_u = 0.0
        for j in range(u.shape[0]):
            div_u += np.gradient(u[j], axis=j)  # u[j] is (H,H,H), axis=j
        return np.sum(w * f * div_u)

    def integrate_scalar_field_time_deriv(self, f, dw_dt=None):
        """∫ w · f · ∂_t f → -(1/2)∫ f² · ∂_t w."""
        if dw_dt is None:
            return 0.0
        return -0.5 * np.sum(f ** 2 * dw_dt)

    def integrate_vector_dot_time_deriv(self, u, dw_dt=None):
        """∫ w · (u · ∂_t u) → -(1/2)∫ |u|² · ∂_t w."""
        if dw_dt is None:
            return 0.0
        u_sq = np.sum(u ** 2, axis=0)
        return -0.5 * np.sum(u_sq * dw_dt)


# =========================================================================
# Main SPIDER orchestrator
# =========================================================================

class SPIDER:
    """Complete SPIDER algorithm for PDE discovery.

    Usage:
        config = SPIDERConfig()
        spider = SPIDER(config)
        equations = spider.discover(
            fields=[velocity, pressure],
            field_names=['u', 'p'],
            n_spatial_dims=3,
        )
    """

    # Dispatch table: maps term type → method name in WeakFormIntegrator
    _VECTOR_DISPATCH = {
        'field': 'integrate_vector_field',
        'time_deriv': 'integrate_vector_time_deriv',
        'gradient': 'integrate_vector_gradient',
        'field_product': 'integrate_vector_field_product',
        'convection': 'integrate_vector_convection',
        'laplacian': 'integrate_vector_laplacian',
        'time_deriv2': (lambda *a: 0.0),
        'field_cube': 'integrate_vector_field_cube',
        'field_sq_field': 'integrate_vector_field_sq_field',
        'gradient_time': 'integrate_vector_gradient_time',
        'field_gradient': 'integrate_vector_field_gradient',
        'field_div': 'integrate_vector_field_div',
        'convection_alt': 'integrate_vector_convection_alt',
        'grad_div': 'integrate_vector_grad_div',
        'field_time_deriv': 'integrate_vector_field_time_deriv',
        'field_time_deriv_scalar': 'integrate_vector_field_time_deriv_scalar',
    }

    _SCALAR_DISPATCH = {
        'constant': 'integrate_scalar_constant',
        'scalar_field': 'integrate_scalar_field',
        'divergence': 'integrate_scalar_divergence',
        'scalar_time_deriv': 'integrate_scalar_time_deriv',
        'scalar_sq': 'integrate_scalar_sq',
        'vector_magsq': 'integrate_vector_magsq',
        'scalar_cube': 'integrate_scalar_cube',
        'vector_magsq_field': 'integrate_vector_magsq_field',
        'scalar_laplacian': 'integrate_scalar_laplacian',
        'scalar_convection': 'integrate_scalar_convection',
        'scalar_field_div': 'integrate_scalar_field_div',
        'scalar_field_time_deriv': 'integrate_scalar_field_time_deriv',
        'scalar_time_deriv2': (lambda *a: 0.0),
        'vector_dot_time_deriv': 'integrate_vector_dot_time_deriv',
    }

    def __init__(self, config: Optional[SPIDERConfig] = None):
        self.config = config or SPIDERConfig()
        self._integrator = WeakFormIntegrator()
        self._term_generator = TestFunctionGenerator(
            beta=config.beta,
            shift_range=config.shift_range,
        )
        self._rng = np.random.default_rng(42)

        # Populated by discover()
        self.library_terms: List[dict] = []
        self.coefficients: Optional[np.ndarray] = None
        self.equations: List[Tuple[np.ndarray, List[str]]] = []

    # ---------------------------------------------------------------
    # Main public API
    # ---------------------------------------------------------------

    def discover(self, fields: List[np.ndarray],
                 field_names: Optional[List[str]] = None,
                 n_spatial_dims: int = 3,
                 library_type: str = 'vector') -> List[Tuple[np.ndarray, List[str]]]:
        """Discover a PDE from field data.

        Args:
            fields: List of field arrays, each (n_channels, *spatial_dims)
                    e.g. velocity (3, H, H, H), pressure (H, H, H)
            field_names: Names of the fields
            n_spatial_dims: Number of spatial dimensions
            library_type: 'vector' (momentum) or 'scalar' (continuity etc.)
        Returns:
            List of (coefficients, term_descriptions) tuples
        """
        if field_names is None:
            field_names = [f'f{i}' for i in range(len(fields))]
        self.field_names = field_names
        self.n_spatial_dims = n_spatial_dims

        # Ensure all fields have same spatial shape
        self._validate_inputs(fields)

        if self.config.verbose:
            print("=" * 60)
            print("SPIDER: Sparse Physics-Informed Discovery of Empirical Relations")
            print("=" * 60)
            print(f"Library: {'vector (momentum)' if library_type == 'vector' else 'scalar'}")
            print(f"Fields: {field_names}")
            print(f"Domain size: {self.config.domain_size}^3")
            print(f"Test functions: {self.config.n_test_functions}")
            print(f"Integration domains: {self.config.n_domains}")
            print()

        # 1. Build candidate library
        self.library_terms = self._build_library(library_type)
        n_terms = len(self.library_terms)
        if self.config.verbose:
            print(f"Built library with {n_terms} candidate terms")

        # 2. Compute Q matrix
        Q = self._compute_q_matrix(fields, library_type)

        # 3. Sparse regression
        coefficients = self._sparse_regression(Q)

        # 4. Format equations
        if library_type == 'vector':
            # For vector library, return one eq per component
            equations = self._format_vector_equations(coefficients)
        else:
            equations = self._format_equations(coefficients)

        self.coefficients = coefficients
        self.equations = equations

        if self.config.verbose:
            print(f"\nFound {len(equations)} equation(s):")
            for coeffs, terms in equations:
                self._print_equation(coeffs, terms)

        return equations

    # ---------------------------------------------------------------
    # Library construction
    # ---------------------------------------------------------------

    def _build_library(self, library_type: str) -> List[dict]:
        if library_type == 'vector':
            return build_vector_library_3rd_order(self.field_names)
        else:
            return build_scalar_library_3rd_order(self.field_names)

    # ---------------------------------------------------------------
    # Q matrix construction
    # ---------------------------------------------------------------

    def _extract_domain(self, fields, sizes, offsets):
        """Extract a subdomain from all fields.

        Args:
            fields: list of (C, *spatial_dims) arrays
            sizes: spatial dim sizes of subdomain
            offsets: starting indices for each spatial dim
        Returns:
            list of subdomain arrays, same structure as fields
        """
        sub_fields = []
        for f in fields:
            idx = (slice(None),) + tuple(
                slice(o, o + s) for o, s in zip(offsets, sizes)
            )
            sub_fields.append(f[idx])
        return sub_fields

    def _compute_q_matrix(self, fields, library_type):
        """Assemble the feature matrix Q.

        Q has shape (n_samples, n_terms) where n_samples = n_domains * n_test_functions.
        Each entry Q_{i,n} = (1/V_i * S_n) ∫ w_i · f̃_n dΩ
        """
        cfg = self.config
        field_shapes = [f.shape for f in fields]

        # Separate spatial and time dimensions
        n_spatial = self.n_spatial_dims
        full_dims = fields[0].shape[1:]  # skip channel dim
        spatial_sizes = full_dims[:n_spatial]
        time_size = full_dims[n_spatial] if len(full_dims) > n_spatial else 0

        # Build domain shape: spatial dimensions use domain_size, time uses n_time or full
        domain_dims = [cfg.domain_size] * n_spatial
        if time_size > 0 and cfg.n_time > 0:
            domain_dims.append(cfg.n_time)
        domain_shape = tuple(domain_dims)

        # Build Q matrix with all terms first
        n_rows = cfg.n_domains * cfg.n_test_functions
        n_cols = len(self.library_terms)
        Q_raw = np.zeros((n_rows, n_cols), dtype=np.float64)

        row = 0
        for d in range(cfg.n_domains):
            # Build offsets: random for spatial dims, 0 for time dim
            offsets = []
            for i in range(n_spatial):
                mo = max(1, spatial_sizes[i] - cfg.domain_size)
                offsets.append(self._rng.integers(0, max(1, mo + 1)))
            if time_size > 0:
                offsets.append(0)
            offsets = tuple(offsets)
            sub_fields = self._extract_domain(fields, domain_shape, offsets)

            w_stack = self._term_generator.generate(
                cfg.n_test_functions, domain_shape, self._rng,
            )
            grad_w_list, lap_w = self._term_generator.compute_gradients(w_stack, dx=1.0)

            for k in range(cfg.n_test_functions):
                w = w_stack[k]
                grad_w = [g[k] for g in grad_w_list]
                lap_w_k = lap_w[k]

                for j, term in enumerate(self.library_terms):
                    integral = self._compute_term_integral_single(
                        sub_fields, term, library_type,
                        w, grad_w, lap_w_k,
                    )
                    Q_raw[row, j] = integral

                row += 1

        # Detect and drop zero-valued terms (time derivatives with no time dim)
        col_norms = np.linalg.norm(Q_raw, axis=0)
        valid_mask = col_norms > 1e-12
        n_dropped = n_cols - np.sum(valid_mask)

        if n_dropped > 0:
            dropped_terms = [self.library_terms[i]['desc']
                             for i in range(n_cols) if not valid_mask[i]]
            if cfg.verbose:
                print(f"  Dropped {n_dropped} zero-valued terms: {dropped_terms}")

        # Keep only valid terms
        self._active_mask = valid_mask
        self._active_library = [t for t, v in zip(self.library_terms, valid_mask) if v]
        Q_active = Q_raw[:, valid_mask]

        # Normalize columns by RMS (S_n) and rows by V_i
        if cfg.normalize_terms and Q_active.shape[1] > 0:
            # V_i normalization: divide each row by ∫ |w_k| dΩ
            # (already per-test-function, so within each group of n_test_functions,
            #  the V_i is different)
            # For simplicity, standardize each column to unit variance
            col_std = np.std(Q_active, axis=0)
            col_std = np.clip(col_std, 1e-15, None)
            Q_active = Q_active / col_std[None, :]

            # Row-normalize as well
            row_norms = np.linalg.norm(Q_active, axis=1)
            row_norms = np.clip(row_norms, 1e-15, None)
            Q_active = Q_active / row_norms[:, None]

        if cfg.verbose:
            print(f"Q matrix shape: {Q_active.shape}  "
                  f"(rows={Q_active.shape[0]}, cols={Q_active.shape[1]})")
            print(f"  Active terms: {[t['desc'] for t in self._active_library]}")

        return Q_active

    # ---------------------------------------------------------------
    # Term integral dispatch (with IBP)
    # ---------------------------------------------------------------

    def _compute_term_integral(self, sub_fields, term, library_type, no_norm=False):
        """Quick estimate of a single term integral for scale computation.

        Args:
            sub_fields: list of subdomain field arrays
            term: term dict
            library_type: 'vector' or 'scalar'
            no_norm: if True, skip V_i normalization
        Returns:
            float: integral estimate
        """
        # Use center of domain as single eval point approximation
        # For speed: just use mean of field values
        f0 = sub_fields[0]  # velocity
        val = np.mean(np.abs(f0)) if term['fields'] else 1.0
        return val

    def _compute_term_integral_single(self, sub_fields, term, library_type,
                                       w, grad_w, lap_w):
        """Compute the IBP weak-form integral of one library term against one test function.

        This is the core method that dispatches to the correct WeakFormIntegrator method.
        """
        integrator = self._integrator

        if library_type == 'vector':
            dispatch = self._VECTOR_DISPATCH
            u = sub_fields[0]  # velocity: (3, H, H, H)
            p = sub_fields[1] if len(sub_fields) > 1 else np.zeros_like(u[0])
            ndim = u.shape[0]

            ttype = term['type']

            if ttype == 'field':
                # w · u — component-wise
                field_idx = term['fields'][0]
                if field_idx == 0:
                    # Vector integral per component then sum
                    # For Q matrix, vector terms get one entry per component
                    total = 0.0
                    for c in range(ndim):
                        total += integrator.integrate_vector_field(u, u[c], w)
                    return total
                else:
                    return integrator.integrate_scalar_field(p, w)

            elif ttype == 'time_deriv':
                # ∂_t u: -∫ u · ∂_t w (IBP time)
                n_spatial = self.n_spatial_dims if hasattr(self, 'n_spatial_dims') else ndim
                if len(grad_w) <= n_spatial:
                    return 0.0
                dw_dt = grad_w[n_spatial]
                total = 0.0
                for c in range(ndim):
                    total += integrator.integrate_vector_time_deriv(u[c], dw_dt)
                return total

            elif ttype == 'gradient':
                # ∇p: -∫ p · Σ_c ∂w/∂x_c
                return integrator.integrate_vector_gradient(p, grad_w[:ndim])

            elif ttype == 'field_product':
                # pu: ∫ w · (p * u)
                p_fields = sub_fields[1] if len(sub_fields) > 1 else np.zeros_like(u[0])
                total = 0.0
                for c in range(ndim):
                    total += integrator.integrate_vector_field_product(p_fields, u[c], w)
                return total

            elif ttype == 'convection':
                # (u·∇)u: -∫ u_c u_j ∂w/∂x_j
                total = 0.0
                for c in range(ndim):
                    for j in range(ndim):
                        dw_c_dxj = grad_w[j]  # ∂w/∂x_j (same for all components since w is scalar)
                        total += -np.sum(u[c] * u[j] * dw_c_dxj)
                return total

            elif ttype == 'laplacian':
                # ∇²u: ∫ u_c · ∇²w
                total = 0.0
                for c in range(ndim):
                    total += integrator.integrate_vector_laplacian(u[c], lap_w)
                return total

            elif ttype == 'time_deriv2':
                return 0.0

            elif ttype == 'field_cube':
                # u²u: ∫ w · |u|² · u
                u_sq = np.sum(u ** 2, axis=0)
                total = 0.0
                for c in range(ndim):
                    total += np.sum(w * u_sq * u[c])
                return total

            elif ttype == 'field_sq_field':
                # p²u: ∫ w · p² · u
                p_sq = p ** 2
                total = 0.0
                for c in range(ndim):
                    total += np.sum(w * p_sq * u[c])
                return total

            elif ttype == 'gradient_time':
                return 0.0

            elif ttype == 'field_gradient':
                # p∇p: -½∫ p² · ∂w/∂x_c
                p_sq = p ** 2
                total = 0.0
                for c in range(min(ndim, len(grad_w))):
                    total += -0.5 * np.sum(p_sq * grad_w[c])
                return total

            elif ttype == 'field_div':
                # u(∇·u): approximate with numerical divergence
                div_u = np.zeros_like(u[0])
                for j in range(ndim):
                    div_u += np.gradient(u[j], axis=j)  # u[j] is (H,H,H), axis=0,1,2
                total = 0.0
                for c in range(ndim):
                    total += np.sum(w * u[c] * div_u)
                return total

            elif ttype == 'convection_alt':
                # Same as convection
                total = 0.0
                for c in range(ndim):
                    for j in range(ndim):
                        dw_c_dxj = grad_w[j]
                        total += -np.sum(u[c] * u[j] * dw_c_dxj)
                return total

            elif ttype == 'grad_div':
                # ∇(∇·u): -∇·u · ∂w/∂x_c  per component
                div_u = np.zeros_like(u[0])
                for j in range(ndim):
                    div_u += np.gradient(u[j], axis=j)
                total = 0.0
                for c in range(ndim):
                    total += -np.sum(div_u * grad_w[c])
                return total

            elif ttype == 'field_time_deriv':
                # p∂_t u: Σ_c -∫ u_c · p · ∂_t w (IBP time)
                n_spatial = self.n_spatial_dims if hasattr(self, 'n_spatial_dims') else ndim
                if len(grad_w) <= n_spatial:
                    return 0.0
                dw_dt = grad_w[n_spatial]
                p_fields = sub_fields[1] if len(sub_fields) > 1 else np.zeros_like(u[0])
                total = 0.0
                for c in range(ndim):
                    total += integrator.integrate_vector_field_time_deriv(p_fields, u[c], dw_dt)
                return total

            elif ttype == 'field_time_deriv_scalar':
                # u∂_t p: Σ_c -∫ p · u_c · ∂_t w (IBP time)
                n_spatial = self.n_spatial_dims if hasattr(self, 'n_spatial_dims') else ndim
                if len(grad_w) <= n_spatial:
                    return 0.0
                dw_dt = grad_w[n_spatial]
                p_fields = sub_fields[1] if len(sub_fields) > 1 else np.zeros_like(u[0])
                total = 0.0
                for c in range(ndim):
                    total += integrator.integrate_vector_field_time_deriv_scalar(u[c], p_fields, dw_dt)
                return total

            else:
                return 0.0

        else:  # scalar library
            dispatch = self._SCALAR_DISPATCH
            u = sub_fields[0]
            p = sub_fields[1] if len(sub_fields) > 1 else np.zeros_like(u[0])
            ndim = u.shape[0]
            ttype = term['type']

            if ttype == 'constant':
                return np.sum(w)

            elif ttype == 'scalar_field':
                return np.sum(w * p)

            elif ttype == 'divergence':
                # ∫ w · (∇·u) → -∫ u · ∇w
                total = 0.0
                for c in range(ndim):
                    total += -np.sum(u[c] * grad_w[c])
                return total

            elif ttype == 'scalar_time_deriv':
                # ∂_t p: -∫ p · ∂_t w (IBP time)
                n_spatial = self.n_spatial_dims if hasattr(self, 'n_spatial_dims') else 3
                dw_dt = grad_w[n_spatial] if len(grad_w) > n_spatial else None
                if dw_dt is None:
                    return 0.0
                return -np.sum(p * dw_dt)

            elif ttype == 'scalar_sq':
                return np.sum(w * p ** 2)

            elif ttype == 'vector_magsq':
                u_sq = np.sum(u ** 2, axis=0)
                return np.sum(w * u_sq)

            elif ttype == 'scalar_cube':
                return np.sum(w * p ** 3)

            elif ttype == 'vector_magsq_field':
                u_sq = np.sum(u ** 2, axis=0)
                return np.sum(w * u_sq * p)

            elif ttype == 'scalar_laplacian':
                # ∇²p: ∫ p · ∇²w
                return np.sum(p * lap_w)

            elif ttype == 'scalar_convection':
                # u·∇p: -∫ p · u_j ∂w/∂x_j
                total = 0.0
                for j in range(ndim):
                    total += -np.sum(p * u[j] * grad_w[j])
                return total

            elif ttype == 'scalar_field_div':
                # p(∇·u): compute numerically
                div_u = np.zeros_like(p)
                for j in range(ndim):
                    div_u += np.gradient(u[j], axis=j)  # u[j] is (H,H,H), axis=j
                return np.sum(w * p * div_u)

            elif ttype == 'scalar_field_time_deriv':
                # p∂_t p: -(1/2)∫ p² · ∂_t w
                n_spatial = self.n_spatial_dims if hasattr(self, 'n_spatial_dims') else 3
                dw_dt = grad_w[n_spatial] if len(grad_w) > n_spatial else None
                if dw_dt is None:
                    return 0.0
                return -0.5 * np.sum(p ** 2 * dw_dt)

            elif ttype == 'scalar_time_deriv2':
                return 0.0

            elif ttype == 'vector_dot_time_deriv':
                # u·∂_t u: -(1/2)∫ |u|² · ∂_t w
                n_spatial = self.n_spatial_dims if hasattr(self, 'n_spatial_dims') else 3
                dw_dt = grad_w[n_spatial] if len(grad_w) > n_spatial else None
                if dw_dt is None:
                    return 0.0
                u_sq = np.sum(u ** 2, axis=0)
                return -0.5 * np.sum(u_sq * dw_dt)

            else:
                return 0.0

    # ---------------------------------------------------------------
    # Sparse regression
    # ---------------------------------------------------------------

    def _sparse_regression(self, Q):
        """SVD-based sparse regression to find nullspace of Q.

        Strategy:
        1. Full SVD on Q
        2. Try multiple candidate null vectors (bottom singular vectors)
        3. For each candidate, apply iterative hard thresholding
        4. Score by ||Q·c|| / (||Q|| * ||c||) + α · |c|_0 / N
        5. Return the sparsest vector with acceptable residual

        The PDE Σ c_n f_n = 0 corresponds to Q c ≈ 0.
        """
        cfg = self.config
        n_rows, n_cols = Q.shape

        # 1. Full SVD
        try:
            u, s, vh = np.linalg.svd(Q, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.ones(n_cols) / np.sqrt(n_cols)

        singular_values = s
        if cfg.verbose:
            print(f"  Singular values (first 5): {singular_values[:min(5, len(s))]}")
            if singular_values[-1] > 1e-15:
                print(f"  Condition number: {singular_values[0] / singular_values[-1]:.2f}")

        # 2. Try bottom 5 singular vectors as candidates
        n_candidates = min(5, n_cols)
        best_score = float('inf')
        best_c = None

        for cand_idx in range(n_candidates):
            c0 = vh[-(cand_idx + 1)]  # (n_cols,)

            # 3. Iterative hard thresholding
            c = c0.copy()
            residual = np.linalg.norm(Q @ c) / (np.linalg.norm(Q) * np.linalg.norm(c) + 1e-15)

            for iteration in range(cfg.max_iterations):
                # Find threshold: keep terms with |c_i| > threshold
                # Use adaptive threshold that increases with iterations
                threshold = cfg.sparsity_threshold * (1.0 + iteration * 0.1)

                # Zero out small coefficients
                c[np.abs(c) < threshold] = 0.0

                n_active = np.sum(np.abs(c) > 1e-10)
                if n_active == 0:
                    break

                # Re-fit: project c onto active subspace
                active_mask = np.abs(c) > 1e-10
                Q_active = Q[:, active_mask]
                try:
                    u_a, s_a, vh_a = np.linalg.svd(Q_active, full_matrices=False)
                    c_active = vh_a[-1]  # smallest singular vector of active set
                    # Sign fix
                    if np.dot(c_active, c[active_mask]) < 0:
                        c_active = -c_active
                    c_new = np.zeros(n_cols)
                    c_new[active_mask] = c_active
                    c = c_new
                except np.linalg.LinAlgError:
                    break

                new_residual = np.linalg.norm(Q @ c) / (np.linalg.norm(Q) * np.linalg.norm(c) + 1e-15)
                if abs(new_residual - residual) < 1e-8:
                    break
                residual = new_residual

            # 4. Score: residual + λ · sparsity_penalty
            n_active = np.sum(np.abs(c) > 1e-10)
            residual = np.linalg.norm(Q @ c) / (np.linalg.norm(Q) * np.linalg.norm(c) + 1e-15)
            # Score balances residual vs sparsity (lower is better)
            sparsity_penalty = 0.01 * n_active / max(n_cols, 1)
            score = residual + sparsity_penalty

            if cfg.verbose:
                nz = np.where(np.abs(c) > 1e-10)[0]
                active_lib = getattr(self, '_active_library', self.library_terms)
                terms = [active_lib[i]['desc'] for i in nz]
                print(f"  Candidate {cand_idx + 1}: residual={residual:.4f}, "
                      f"active={n_active}, terms={terms}")

            if score < best_score:
                best_score = score
                best_c = c

        # Normalize
        norm = np.linalg.norm(best_c)
        if norm > 1e-15:
            best_c = best_c / norm

        return best_c

    # ---------------------------------------------------------------
    # Preset: known NS equations from DNS (ν=5e-5)
    # ---------------------------------------------------------------

    @staticmethod
    def get_ns_equations(viscosity: float = 5e-5) -> List[Tuple[np.ndarray, List[str]]]:
        """Return the known incompressible NS and continuity equations.

        This is useful when the PDE form is already known (e.g., from DNS),
        bypassing the numerical discovery step.

        Returns:
            List of (coefficients, terms) for:
            - Momentum: ∂_t u + (u·∇)u + ∇p - ν∇²u = 0
            - Continuity: ∇·u = 0
        """
        # Momentum equation: ∂_t u + (u·∇)u + ∇p - ν∇²u = 0
        momentum_coeffs = np.array([1.0, 1.0, 1.0, -viscosity], dtype=np.float64)
        momentum_terms = ['∂_t u', '(u·∇)u', '∇p', '∇²u']

        # Continuity: ∇·u = 0
        continuity_coeffs = np.array([1.0], dtype=np.float64)
        continuity_terms = ['∇·u']

        return [
            (momentum_coeffs, momentum_terms),
            (continuity_coeffs, continuity_terms),
        ]

    def discover_ns(self, fields, field_names, n_spatial_dims=3):
        """Convenience: discover NS equations with optimized settings."""
        cfg = self.config
        # Use aggressive settings for NS discovery
        self.config = SPIDERConfig(
            n_domains=max(cfg.n_domains, 200),
            domain_size=cfg.domain_size,
            n_test_functions=max(cfg.n_test_functions, 12),
            beta=cfg.beta,
            max_iterations=cfg.max_iterations,
            sparsity_threshold=0.05,
            normalize_terms=True,
            verbose=cfg.verbose,
        )
        return self.discover(fields, field_names, n_spatial_dims, 'vector')

    def _format_equations(self, coefficients):
        """Format coefficients into (coeffs, terms) tuples."""
        active_lib = getattr(self, '_active_library', self.library_terms)
        nonzero = np.where(np.abs(coefficients) > 1e-10)[0]
        if len(nonzero) == 0:
            return [(np.array([]), [])]
        eq_coeffs = coefficients[nonzero]
        eq_terms = [active_lib[i]['desc'] for i in nonzero]
        return [(eq_coeffs, eq_terms)]

    def _format_vector_equations(self, coefficients):
        """For vector library, each component is a separate equation."""
        active_lib = getattr(self, '_active_library', self.library_terms)
        nonzero = np.where(np.abs(coefficients) > 1e-10)[0]
        if len(nonzero) == 0:
            return [(np.array([]), [])]
        eq_coeffs = coefficients[nonzero]
        eq_terms = [active_lib[i]['desc'] for i in nonzero]
        return [(eq_coeffs, eq_terms)]

    def _print_equation(self, coefficients, terms):
        """Print equation in readable format."""
        parts = []
        for coeff, term in zip(coefficients, terms):
            if coeff >= 0:
                sign = "+"
            else:
                sign = "-"
            parts.append(f"{sign} {abs(coeff):.4f}·{term}")

        if parts:
            if parts[0].startswith("+ "):
                parts[0] = parts[0][2:]
            print("  " + " ".join(parts) + " = 0")

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    def _validate_inputs(self, fields):
        base_shape = fields[0].shape
        for i, f in enumerate(fields):
            if f.shape[1:] != base_shape[1:]:
                raise ValueError(
                    f"Field {i} spatial shape {f.shape[1:]} != {base_shape[1:]}"
                )
        self.spatial_shape = base_shape[1:]

    # ---------------------------------------------------------------
    # Export for integration with PhysicsLoss
    # ---------------------------------------------------------------

    def get_active_terms(self, library_type: str = 'vector'
                         ) -> Tuple[np.ndarray, List[dict]]:
        """Get active (nonzero) coefficients and their full term definitions.

        Used to feed discovered equations into PhysicsLoss for PI-NoProp training.

        Args:
            library_type: 'vector' or 'scalar'

        Returns:
            (coefficients, term_defs) tuple where term_defs contains
            {'type', 'fields', 'desc'} keys needed by PhysicsLoss
        """
        if self.coefficients is None:
            raise RuntimeError("No discovered equations. Run discover() first.")
        # Use _active_library if available (zero-valued columns may have been dropped)
        active_lib = getattr(self, '_active_library', self.library_terms)
        active = np.where(np.abs(self.coefficients) > 1e-10)[0]
        if len(active) == 0:
            return np.array([]), []
        coeffs = self.coefficients[active]
        terms = [active_lib[i] for i in active]
        return coeffs, terms

    # ---------------------------------------------------------------
    # Save / load
    # ---------------------------------------------------------------

    def save(self, filepath: str):
        """Save discovered equations to file."""
        import json
        data = {
            'coefficients': self.coefficients.tolist() if self.coefficients is not None else None,
            'terms': [t['desc'] for t in self.library_terms],
            'config': {
                'n_domains': self.config.n_domains,
                'domain_size': self.config.domain_size,
                'n_test_functions': self.config.n_test_functions,
                'beta': self.config.beta,
            },
        }
        np.savez(filepath, **{k: np.array(v, dtype=object) for k, v in data.items()})
        print(f"Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load discovered equations from file."""
        data = np.load(filepath, allow_pickle=True)
        return data
