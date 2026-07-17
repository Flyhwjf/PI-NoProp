"""Global configuration dataclasses for Physics-Informed NoProp."""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_dir: str = 'data/generated'
    regions: List[str] = field(default_factory=lambda: ['centre', 'edge'])
    subdomain_size: int = 32
    n_subdomains: int = 256
    n_classes: int = 10
    n_timesteps: int = 32
    n_channels: int = 4
    noise_level: float = 0.0
    batch_size: int = 16
    num_workers: int = 0
    val_split: float = 0.125          # 32/256
    n_train: int = 192
    n_val: int = 32
    n_test: int = 32
    cache_dir: str = 'data/cache'
    use_cache: bool = True
    cache_in_memory: bool = True
    split_seed: int = 42
    pin_memory: bool = True
    cache_on_device: bool = True
    trajectory_disjoint: bool = False
    ns_term_means: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ns_term_covariance: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class DiffusionConfig:
    """NoProp diffusion schedule configuration."""
    T: int = 10
    s: float = 0.008
    schedule: str = 'cosine'
    eta: float = 0.1


@dataclass
class NoPropConfig:
    """NoProp block architecture configuration."""
    embedding_dim: int = 128
    condition_dim: int = 128
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    activation: str = 'relu'
    normalize_condition: bool = False


@dataclass
class DecoderConfig:
    """Decoder network configuration (32^3 output)."""
    latent_dim: int = 128
    output_channels: int = 4
    base_channels: int = 32
    use_conv: bool = True
    n_upsample_layers: int = 3          # 4→8→16→32
    use_compact_physics_decoder: bool = True
    use_temporal_decoder: bool = False


@dataclass
class PhysicsConfig:
    """Weak-form physics loss configuration."""
    n_test_functions: int = 8
    beta: float = 8.0
    domain_size: int = 16                # integration domain size (<= subdomain_size)
    n_time: int = 4
    lambda_weight: float = 0.01
    viscosity: float = 0.005              # HIT nu
    use_continuity: bool = True
    use_pressure_poisson: bool = True
    use_energy: bool = False
    spatial_only: bool = True
    pressure_poisson_weight: float = 0.25
    physics_grid_size: int = 16
    use_full_ns: bool = False
    discovered_artifact: str = 'outputs/spider/hit_full_ns_v2.json'
    dns_grid_size: int = 64
    box_length: float = 6.283185307179586
    snapshot_dt: float = 0.002
    divergence_weight: float = 0.25
    energy_weight: float = 0.25
    # Coefficients of [convection, pressure gradient, velocity Laplacian]
    # after fixing the time-derivative coefficient to one.  ``None`` is the
    # no-equation ablation and disables the physics condition branch.
    condition_coefficients: Optional[List[float]] = None


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    lr: float = 1e-3
    pretrain_lr: float = 1e-3
    weight_decay: float = 1e-4
    n_pretrain_epochs: int = 50
    n_epochs: int = 50
    log_interval: int = 10
    save_dir: str = 'outputs/models'
    rec_weight: float = 1.0
    local_rec_weight: float = 0.0
    local_steps_per_block: int = 500
    classifier_epochs: int = 30
    use_amp: bool = True
    amp_dtype: str = 'float16'
    use_tf32: bool = True
    freeze_encoder: bool = True
    freeze_decoder: bool = True
    cache_conditions: bool = True
    grad_clip_norm: float = 1.0
    fused_optimizer: bool = True
    benchmark_batches: int = 20
    fast_batch_size: int = 64


@dataclass
class PINoPropConfig:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    noprop: NoPropConfig = field(default_factory=NoPropConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = 'cuda'
    seed: int = 42
