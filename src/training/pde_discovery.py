"""Bridge: run SPIDER on training data and convert discovered equations
into the (coefficients, term_defs) format accepted by PhysicsLoss.

Loads full 4D (3 spatial + time) fields directly from npz files,
bypassing the dataloader's t=0 truncation so SPIDER can discover
time-derivative terms.

Usage:
    from src.training.pde_discovery import discover_pdes
    vec_eq, scal_eq = discover_pdes(cfg, region='centre')
    physics_loss = PhysicsLoss(config,
        spider_vector_equation=vec_eq,
        spider_scalar_equation=scal_eq,
    )
"""
import numpy as np
import torch
from pathlib import Path

from ..spider import SPIDER, SPIDERConfig
from ..physics.loss import PhysicsLoss


def _load_fields_direct(data_dir, region='centre', n_files=16):
    """Load full 4D velocity+pressure directly from npz files.

    Each npz contains velocity(3, 32, 32, 32, 32) and pressure(32, 32, 32, 32).
    Returns fields stacked across n_files as (N, ..., ...).

    Returns:
        velocity: (N, 3, 32, 32, 32, 32) float32
        pressure: (N, 32, 32, 32, 32) float32
    """
    import glob
    files = sorted(glob.glob(str(Path(data_dir) / region / 'sub_*.npz')))[:n_files]
    if not files:
        raise RuntimeError(f"No npz files found in {data_dir}/{region}")
    vel_list, prs_list = [], []
    for f in files:
        d = np.load(f)
        vel_list.append(d['velocity'])     # (3, 32, 32, 32, 32)
        prs_list.append(d['pressure'])     # (32, 32, 32, 32)
    velocity = np.stack(vel_list, axis=0).astype(np.float32)   # (N, 3, 32, 32, 32, 32)
    pressure = np.stack(prs_list, axis=0)[:, np.newaxis, ...]  # (N, 1, 32, 32, 32, 32)
    return velocity, pressure


def discover_pdes(cfg, device='cpu', region='centre', n_files=16,
                   verbose=True):
    """Run SPIDER equation discovery on full 4D training data.
    
    Loads velocity+pressure directly from npz files (preserves time dim),
    then runs SPIDER with n_time > 0 so time-derivative terms are found.

    Args:
        cfg: PINoPropConfig
        device: torch device (for output tensors)
        region: 'centre' or 'edge'
        n_files: number of npz files to load (more = better statistics)
        verbose: print progress

    Returns:
        vector_eq: (coefficients_tensor, term_defs_list) or None
        scalar_eq: (coefficients_tensor, term_defs_list) or None
    """
    if verbose:
        print("=" * 60)
        print("SPIDER PDE Discovery for PI-NoProp Training (4D)")
        print("=" * 60)

    # 1. Load full 4D fields
    if verbose:
        print(f"Loading {n_files} files from {cfg.data.data_dir}/{region} ...")
    velocity, pressure = _load_fields_direct(cfg.data.data_dir, region, n_files)
    # velocity: (N, 3, 32, 32, 32, 32), pressure: (N, 32, 32, 32, 32)
    # Take first batch for SPIDER: (3, 32, 32, 32, 32) and (32, 32, 32, 32)
    vel_4d = velocity[0]        # (3, 32, 32, 32, 32) — 3 channels + 4 dims
    prs_4d = pressure[0]        # (32, 32, 32, 32)
    H = vel_4d.shape[1]
    Ht = vel_4d.shape[-1]
    if verbose:
        print(f"  velocity shape: {vel_4d.shape}  (3 spatial dims + {Ht} time steps)")
        print(f"  pressure shape: {prs_4d.shape}")

    # 2. SPIDER config — include time dimension
    sub_size = min(cfg.physics.domain_size, max(4, H // 2))
    n_time_spider = min(cfg.physics.n_time, Ht)
    spider_cfg = SPIDERConfig(
        n_domains=cfg.physics.n_test_functions * 12,
        domain_size=sub_size,
        n_time=n_time_spider,       # ← KEY: enable time dimension
        n_test_functions=cfg.physics.n_test_functions,
        beta=cfg.physics.beta,
        max_iterations=30,
        sparsity_threshold=0.05,
        normalize_terms=True,
        verbose=verbose,
    )
    spider = SPIDER(spider_cfg)

    if verbose:
        print("\nDiscovering vector (momentum) equation ...")
    spider.discover(
        fields=[vel_4d, prs_4d],
        field_names=['u', 'p'],
        n_spatial_dims=3,
        library_type='vector',
    )
    vec_coeffs, vec_terms = spider.get_active_terms(library_type='vector')

    if len(vec_coeffs) == 0:
        if verbose:
            print("WARNING: No vector equation terms discovered. "
                  "Falling back to known NS equation.")
        vec_eq = None
    else:
        vec_eq = (torch.tensor(vec_coeffs, dtype=torch.float32), vec_terms)
        if verbose:
            print(f"  Vector equation: {len(vec_coeffs)} active terms")
            for c, t in zip(vec_coeffs, vec_terms):
                print(f"    {c:+.6f} × {t['desc']}")

    # 3. Run SPIDER — scalar (continuity) equation
    if verbose:
        print("\nDiscovering scalar (continuity) equation ...")
    spider.discover(
        fields=[vel_4d, prs_4d],
        field_names=['u', 'p'],
        n_spatial_dims=3,
        library_type='scalar',
    )
    scal_coeffs, scal_terms = spider.get_active_terms(library_type='scalar')

    if len(scal_coeffs) == 0:
        if verbose:
            print("WARNING: No scalar equation terms discovered. "
                  "Falling back to known continuity equation.")
        scal_eq = None
    else:
        scal_eq = (torch.tensor(scal_coeffs, dtype=torch.float32), scal_terms)
        if verbose:
            print(f"  Scalar equation: {len(scal_coeffs)} active terms")
            for c, t in zip(scal_coeffs, scal_terms):
                print(f"    {c:+.6f} × {t['desc']}")

    if verbose:
        print("\nPDE discovery complete.")

    return vec_eq, scal_eq


def build_physics_loss_with_spider(config, dataloader, device='cpu',
                                   max_batches=None, verbose=True):
    """Convenience: discover PDEs and build PhysicsLoss with the results.

    If discovery fails for an equation type, PhysicsLoss falls back to the
    built-in NS / continuity equations for that type.

    Args:
        config: PINoPropConfig
        dataloader: training DataLoader
        device: torch device
        max_batches: limit batches for discovery
        verbose: print progress

    Returns:
        physics_loss: PhysicsLoss module (on device)
        vec_eq: discovered vector equation tuple (or None)
        scal_eq: discovered scalar equation tuple (or None)
    """
    vec_eq, scal_eq = discover_pdes(
        dataloader, config, device=device,
        max_batches=max_batches, verbose=verbose,
    )

    physics_loss = PhysicsLoss(
        config,
        spider_vector_equation=vec_eq,
        spider_scalar_equation=scal_eq,
    ).to(device)

    return physics_loss, vec_eq, scal_eq