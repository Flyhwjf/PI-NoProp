"""Test function generation for weak-form PDE evaluation.

Implements eq.4.8:
    w(x,t) = Π_{ξ∈{x,y,z,t}} (1 - ξ̃²)^β
where ξ̃ maps each dimension to [-1, 1].
"""
import torch
import numpy as np


class TestFunctionGenerator:
    """Generate tensor-product test functions for weak-form integration.

    Each test function w_k has compact support over the full domain
    and vanishes at boundaries (ensuring no boundary terms after IBP).
    """

    def __init__(self, beta=8.0, spatial_size=16, n_time=4):
        self.beta = beta
        self.spatial_size = spatial_size
        self.n_time = n_time

    def generate(self, n_functions, device='cpu'):
        """Generate n_functions test functions on a grid.

        Each w(ξ) = Π (1 - ξ_i²)^β where ξ_i ∈ [-1, 1].

        Returns:
            w: (n_functions, spatial_size, spatial_size, spatial_size, n_time)
        """
        H = self.spatial_size
        Ht = self.n_time
        functions = []

        for k in range(n_functions):
            # Random center shift (fraction of domain)
            shift = np.random.uniform(-0.4, 0.4, 4)

            w = torch.ones(H, H, H, Ht)
            for dim in range(4):
                size = H if dim < 3 else Ht
                # Create coordinate grid on [-1, 1]
                xi = torch.linspace(-1, 1, size)
                # Apply random shift and scale
                xi_shifted = (xi - shift[dim]) * 1.5
                xi_shifted = torch.clamp(xi_shifted, -1.0, 1.0)
                w_dim = torch.clamp(1.0 - xi_shifted ** 2, min=0.0) ** self.beta

                shape = [1, 1, 1, 1]
                shape[dim] = -1
                w = w * w_dim.reshape(shape)

            functions.append(w)

        w_stack = torch.stack(functions, dim=0)
        # NO L2 normalization — keep the original magnitude
        # (SPIDER paper uses unnormalized test functions w ~ O(1))

        return w_stack.to(device)

    def compute_gradients(self, w, dx=1.0, dy=1.0, dz=1.0, dt=1.0, domain_size=32):
        """Precompute spatial and temporal gradients of test functions.
        
        Args:
            w: (K, H, H, H, Ht) test functions
            dx, dy, dz: spatial grid spacing (default: 2/domain_size)
            dt: temporal spacing (default: 2/n_time)
            domain_size: number of grid points per spatial dimension
            
        Returns:
            grad_w: list of 4 tensors each (K, H, H, H, Ht)
            lap_w: (K, H, H, H, Ht)
        """
        K, H, _, _, Ht = w.shape
        # Use correct physical spacing
        if dx == 1.0 and dy == 1.0 and dz == 1.0:
            dx = dy = dz = 2.0 / domain_size
            dt = 2.0 / Ht
        grad_w = []
        for dim in range(4):
            spacing = [dx, dy, dz, dt][dim]
            g = torch.zeros_like(w)
            g[:, :, :, :] = torch.gradient(w, spacing=spacing, dim=dim + 1)[0]
            grad_w.append(g)

        lap_w = torch.zeros_like(w)
        for dim in range(3):
            spacing = [dx, dy, dz][dim]
            g1 = torch.gradient(w, spacing=spacing, dim=dim + 1)[0]
            g2 = torch.gradient(g1, spacing=spacing, dim=dim + 1)[0]
            lap_w = lap_w + g2

        return grad_w, lap_w
