"""Explicit, isolated experiment protocols shared by the auxiliary scripts.

Run names mirror run_experiment.py; checkpoint configuration, never current
defaults, determines how an existing model and its dataset are reconstructed.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


CONTEXT_ENCODERS = ('single', 'dual_scale', 'residual_dual_scale',
                    'residual_warmstart')


def lambda_token(value):
    return f'{value:g}'.replace('.', 'p')


@dataclass(frozen=True)
class ExperimentProtocol:
    input_size: int = 32
    target_size: int = 16
    context_encoder: str = 'residual_warmstart'
    lambda_phys: float | None = None
    cache_override: str | None = None
    device: str | None = None

    def __post_init__(self):
        if self.input_size not in (16, 32) or self.target_size not in (16, 32):
            raise ValueError('input and target sizes must be 16 or 32')
        if self.target_size > self.input_size:
            raise ValueError('--target-size cannot exceed --input-size')
        if self.context_encoder not in CONTEXT_ENCODERS:
            raise ValueError(f'unknown context encoder: {self.context_encoder}')
        if self.input_size == self.target_size:
            object.__setattr__(self, 'context_encoder', 'single')
        if self.lambda_phys is None:
            object.__setattr__(self, 'lambda_phys', self.default_lambda)
        if not math.isfinite(self.lambda_phys) or self.lambda_phys < 0:
            raise ValueError('--lambda-phys must be finite and nonnegative')

    @property
    def default_lambda(self):
        return 0.01 if (self.input_size, self.target_size) == (16, 16) else 0.1

    @property
    def prefix(self):
        if (self.input_size, self.target_size) == (16, 16):
            return 'full_ns_v4'
        return (f'full_ns_v5_input{self.input_size}_target{self.target_size}_'
                f'{self.context_encoder}')

    @property
    def cache_dir(self):
        return self.cache_override or (
            'data/cache_hit_ns' if (self.input_size, self.target_size) == (16, 16)
            else f'data/cache_hit_ns_input{self.input_size}_target{self.target_size}')

    @property
    def runtime_device(self):
        return self.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def model_revision(self):
        return ('trainable-physics-condition-fusion-prototype-readout'
                if self.prefix == 'full_ns_v4' else
                'cached-residual-warmstart-physics-condition-prototype-readout')

    def run_id(self, source, region, seed, *, weight=None, relation='ns', tag=''):
        weight = (0.0 if source == 'none' else self.lambda_phys) if weight is None else weight
        suffix = '' if relation == 'ns' else f'_{relation}'
        suffix += f'_{tag}' if tag else ''
        return (f'{self.prefix}_{source}_{region}_lambda{lambda_token(weight)}'
                f'_seed{seed}{suffix}')

    def baseline_run_id(self, method, region, seed):
        suffix = ('' if self.lambda_phys == self.default_lambda else
                  f'_lambda{lambda_token(self.lambda_phys)}')
        return f'{self.prefix}_{method}_{region}_seed{seed}{suffix}'

    def shared_path(self, root, source, region, seed, architecture='conv'):
        suffix = '' if architecture == 'conv' else f'_{architecture}'
        return (Path(root)/'outputs/models'/
                f'shared_{self.prefix}_{source}_{region}_seed{seed}{suffix}.pt')

    def aggregate_path(self, root, name):
        suffix = ('' if self.lambda_phys == self.default_lambda else
                  f'_lambda{lambda_token(self.lambda_phys)}')
        return Path(root)/'outputs/aggregate'/f'{self.prefix}{suffix}_{name}.json'

    def metadata(self):
        return {'prefix': self.prefix, 'input_spatial_size': self.input_size,
                'target_spatial_size': self.target_size,
                'spatial_context_mode': self.context_encoder,
                'lambda_weight': self.lambda_phys, 'cache_dir': self.cache_dir}

    def cli_args(self, *, weight=None):
        arguments = ['--input-size', str(self.input_size),
                     '--target-size', str(self.target_size),
                     '--context-encoder', self.context_encoder,
                     '--lambda-phys', str(self.lambda_phys if weight is None else weight),
                     '--cache-dir', self.cache_dir]
        if self.device is not None:
            arguments += ['--device', self.device]
        return arguments

    def apply_config(self, config):
        config.device = self.runtime_device
        config.data.data_dir = 'data/generated_hit_ns'
        config.data.cache_dir = self.cache_dir
        config.data.subdomain_size = self.input_size
        config.data.target_subdomain_size = self.target_size
        config.data.n_subdomains = 64
        config.data.n_classes = 5
        config.data.batch_size = 32
        config.data.trajectory_disjoint = True
        config.noprop.spatial_context_mode = self.context_encoder
        config.noprop.normalize_condition = True
        config.decoder.use_temporal_decoder = True
        config.decoder.architecture = 'conv'
        config.decoder.use_conv = True
        config.physics.physics_grid_size = self.target_size
        config.physics.lambda_weight = self.lambda_phys
        return config


def add_protocol_arguments(parser):
    parser.add_argument('--input-size', type=int, choices=(16, 32), default=32,
                        help='velocity-pressure context cube size (default: 32)')
    parser.add_argument('--target-size', type=int, choices=(16, 32), default=16,
                        help='centred label/physics cube size (default: 16)')
    parser.add_argument('--context-encoder', choices=CONTEXT_ENCODERS,
                        default='residual_warmstart')
    parser.add_argument('--lambda-phys', type=float, default=None,
                        help='default: 0.1 for v5, 0.01 for input16/target16 v4')
    parser.add_argument('--cache-dir', default=None,
                        help='size-specific cache override; evaluation must match checkpoint')
    parser.add_argument('--device', default=None,
                        help='runtime device override; otherwise CUDA if available, else CPU')


def protocol_from_args(args, parser=None):
    try:
        return ExperimentProtocol(args.input_size, args.target_size,
                                  args.context_encoder, args.lambda_phys,
                                  args.cache_dir, args.device)
    except ValueError as error:
        if parser is not None:
            parser.error(str(error))
        raise


def validate_metadata(metadata, protocol, *, legacy_ok=False):
    """Reject incompatible aggregates or baseline metrics before reusing them."""
    if legacy_ok and protocol.prefix == 'full_ns_v4' and 'prefix' not in metadata:
        return
    expected = protocol.metadata()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'cached protocol mismatch for {key}: '
                             f'{metadata.get(key)!r} != {value!r}')


def protect_legacy_output(path, protocol):
    if protocol.prefix == 'full_ns_v4' and Path(path).exists():
        raise ValueError(f'refusing to overwrite legacy v4 output: {path}')


def validate_checkpoint(config, protocol, *, source, region, seed, weight=None,
                        relation='ns', architecture='conv'):
    target = getattr(config.data, 'target_subdomain_size', None) or config.data.subdomain_size
    expected_weight = (0.0 if source == 'none' else protocol.lambda_phys) if weight is None else weight
    actual = {
        'input_size': config.data.subdomain_size, 'target_size': target,
        'context_encoder': getattr(config.noprop, 'spatial_context_mode', 'single'),
        'physics_grid_size': config.physics.physics_grid_size,
        'lambda_phys': config.physics.lambda_weight, 'seed': config.seed,
        'regions': config.data.regions,
        'architecture': getattr(config.decoder, 'architecture', 'conv'),
        'pressure_poisson': config.physics.use_pressure_poisson,
        'energy': config.physics.use_energy,
        'trajectory_disjoint': config.data.trajectory_disjoint,
        'samples_per_trajectory': config.data.n_subdomains,
        'n_classes': config.data.n_classes, 'n_time': config.physics.n_time,
        'normalize_condition': config.noprop.normalize_condition,
    }
    expected = {
        'input_size': protocol.input_size, 'target_size': protocol.target_size,
        'context_encoder': protocol.context_encoder,
        'physics_grid_size': protocol.target_size,
        'lambda_phys': expected_weight, 'seed': seed, 'regions': [region],
        'architecture': architecture, 'pressure_poisson': relation in ('ns_pp', 'full'),
        'energy': relation == 'full',
        'trajectory_disjoint': True, 'samples_per_trajectory': 64,
        'n_classes': 5, 'n_time': 9, 'normalize_condition': True,
    }
    if actual != expected:
        differences = {key: (actual[key], value) for key, value in expected.items()
                       if actual[key] != value}
        raise ValueError(f'checkpoint protocol mismatch (actual, expected): {differences}')
    enabled = config.physics.condition_coefficients is not None
    if enabled != (source != 'none'):
        raise ValueError('checkpoint physics condition does not match requested source')
    if protocol.cache_override is not None:
        if Path(config.data.cache_dir).resolve() != Path(protocol.cache_dir).resolve():
            raise ValueError('checkpoint cache_dir differs from explicit --cache-dir')


def load_run(run, protocol, *, source, region, seed, weight=None, relation='ns',
             architecture='conv'):
    """Load on CPU and validate even when only reusing previously computed metrics."""
    run = Path(run)
    checkpoint = torch.load(run/'checkpoint.pt', map_location='cpu', weights_only=False)
    config = copy.deepcopy(checkpoint['config'])
    validate_checkpoint(config, protocol, source=source, region=region, seed=seed,
                        weight=weight, relation=relation, architecture=architecture)
    record = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    for key, expected in (('physics_source', source), ('region', region), ('seed', seed),
                          ('relation_set', relation), ('decoder_architecture', architecture)):
        if record.get(key, expected if protocol.prefix == 'full_ns_v4' else None) != expected:
            raise ValueError(f'{run}: metrics/checkpoint mismatch for {key}')
    for key, expected in (('input_spatial_size', protocol.input_size),
                          ('target_spatial_size', protocol.target_size),
                          ('spatial_context_mode', protocol.context_encoder)):
        if record.get(key, expected if protocol.prefix == 'full_ns_v4' else None) != expected:
            raise ValueError(f'{run}: metrics/checkpoint mismatch for {key}')
    # Only deployment changes the saved config.  Preserve data/cache paths,
    # physics settings, widths, latent dimensions and prototype configuration.
    config.device = protocol.runtime_device
    return checkpoint, config, record


def make_decoder(config):
    from src.decoder import LinearTemporalPhysicsDecoder, TemporalPhysicsDecoder
    if getattr(config.decoder, 'architecture', 'conv') == 'linear':
        return LinearTemporalPhysicsDecoder(
            config.decoder.latent_dim, config.physics.n_time,
            config.decoder.output_channels, config.physics.physics_grid_size)
    return TemporalPhysicsDecoder(
        config.decoder.latent_dim, config.decoder.base_channels,
        config.physics.n_time, config.decoder.output_channels)


def assert_matched_condition(reference, candidate):
    """Decoder-only changes must preserve *all* frozen condition/prototype state."""
    prefixes = ('encoder.', 'physics_encoder.', 'condition_fusion.', 'label_embed.')
    buffers = {'physics_coefficients', 'physics_mean', 'physics_std'}
    def select(checkpoint):
        return {key: value for key, value in checkpoint['model_state_dict'].items()
                if key.startswith(prefixes) or key in buffers}
    left, right = select(reference), select(candidate)
    complete = (all(any(key.startswith(prefix) for key in left) for prefix in prefixes)
                and buffers.issubset(left))
    if not complete or left.keys() != right.keys():
        raise ValueError('decoder comparison lacks matched full condition/prototypes')
    changed = [key for key in left if not torch.equal(left[key], right[key])]
    if changed:
        raise ValueError('decoder ablation changed frozen condition/prototypes: '
                         f'{changed[:8]}; run_experiment.py must reuse the current '
                         'protocol conv condition checkpoint, without context adaptation')
