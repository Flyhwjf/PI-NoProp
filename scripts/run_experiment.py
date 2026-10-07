"""Strictly local PI-NoProp with a SPIDER-discovered full NS equation."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PINoPropConfig
from src.data.hit_dataset import create_hit_dataloaders
from src.decoder import LinearTemporalPhysicsDecoder, TemporalPhysicsDecoder
from src.noprop.model import NoPropModel
from src.physics.temporal_ns_loss import TemporalNSPhysicsLoss
from src.training.local_trainer import LocalNoPropTrainer, configure_torch
from src.training.pretrain import (align_label_embeddings_to_encoder,
                                   adapt_residual_context,
                                   initialize_residual_context_from_local,
                                   load_residual_context_warmstart,
                                   load_shared_components,
                                   load_shared_condition_components,
                                   pretrain_decoder,
                                   pretrain_encoder, save_shared_components)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_local_blocks(model, seed):
    """Initialize local blocks independently of shared-stage RNG usage.

    Encoder/decoder pretraining and checkpoint loading consume different
    amounts of random state.  Resetting only the independently trained NoProp
    blocks here makes their initial weights identical whether shared
    components were rebuilt or loaded, and across matched input encoders.
    """
    devices = ([torch.cuda.current_device()]
               if torch.cuda.is_available() else [])
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for module in model.blocks.modules():
            reset = getattr(module, 'reset_parameters', None)
            if reset is not None:
                reset()


def analytic_artifact(viscosity):
    return {
        'schema_version': 2,
        'method': 'analytic full NS ablation',
        'validation': {'passed': True, 'failure_reasons': []},
        'equation': {
            'terms': ['time_derivative', 'convection',
                      'pressure_gradient', 'velocity_laplacian'],
            'coefficients': [1.0, 1.0, 1.0, -viscosity],
            'normalization': 'time-derivative coefficient fixed to one',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', choices=['low_enstrophy', 'high_enstrophy'],
                        default='low_enstrophy')
    parser.add_argument('--physics-source', choices=['none', 'analytic', 'discovered'],
                        default='discovered')
    parser.add_argument('--equation-path', default='outputs/spider/full_ns_equation.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps-per-block', type=int, default=100)
    parser.add_argument('--condition-epochs', '--classifier-epochs',
                        dest='condition_epochs', type=int, default=30,
                        help='epochs for the temporary condition-pretraining head '
                             '(the final readout is not trained)')
    parser.add_argument('--pretrain-epochs', type=int, default=40)
    parser.add_argument('--lambda-phys', type=float, default=None,
                        help='defaults to 0.1 for 32^3 context and 0.01 for '
                             'the legacy 16^3 protocol')
    parser.add_argument('--relation-set', choices=['ns', 'ns_pp', 'full'],
                        default='ns')
    parser.add_argument('--decoder-architecture', choices=['conv', 'linear'],
                        default='conv')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--input-size', type=int, choices=[16, 32], default=32,
                        help='spatial size of the velocity-pressure context')
    parser.add_argument('--target-size', type=int, choices=[16, 32], default=16,
                        help='centred label/physics target size')
    parser.add_argument('--cache-dir', default=None,
                        help='override the size-specific learning cache')
    parser.add_argument('--context-encoder',
                        choices=['single', 'dual_scale', 'residual_dual_scale',
                                 'residual_warmstart'],
                        default='residual_warmstart',
                        help='spatial encoder used when input exceeds target size')
    parser.add_argument('--context-adapt-epochs', type=int, default=20,
                        help='validation-selected context-only warm-start epochs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rebuild-shared', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    config = PINoPropConfig()
    config.device = args.device
    config.seed = args.seed
    if args.target_size > args.input_size:
        parser.error('--target-size cannot exceed --input-size')
    config.data.data_dir = 'data/generated_hit_ns'
    config.data.cache_dir = (args.cache_dir or (
        'data/cache_hit_ns' if (args.input_size, args.target_size) == (16, 16)
        else f'data/cache_hit_ns_input{args.input_size}_target{args.target_size}'))
    config.data.regions = [args.region]
    config.data.subdomain_size = args.input_size
    config.data.target_subdomain_size = args.target_size
    config.data.n_subdomains = 64  # samples generated per independent trajectory
    config.data.n_classes = 5
    config.data.batch_size = args.batch_size
    config.data.trajectory_disjoint = True
    config.noprop.normalize_condition = True
    config.noprop.spatial_context_mode = (
        args.context_encoder if args.input_size > args.target_size else 'single')
    config.decoder.use_temporal_decoder = True
    config.decoder.architecture = args.decoder_architecture
    config.decoder.use_conv = args.decoder_architecture == 'conv'
    config.physics.n_time = 9
    config.physics.beta = 4.0
    config.physics.n_test_functions = 8
    config.physics.physics_grid_size = args.target_size
    config.physics.use_continuity = True
    config.physics.use_pressure_poisson = args.relation_set in ('ns_pp', 'full')
    config.physics.use_energy = args.relation_set == 'full'
    config.physics.use_full_ns = True
    default_lambda = (0.01 if (args.input_size, args.target_size) == (16, 16)
                      else 0.1)
    requested_lambda = (default_lambda if args.lambda_phys is None
                        else args.lambda_phys)
    config.physics.lambda_weight = (0.0 if args.physics_source == 'none'
                                    else requested_lambda)
    config.physics.discovered_artifact = args.equation_path
    config.training.local_steps_per_block = args.steps_per_block
    # Keep the legacy config field for checkpoint compatibility; these epochs
    # belong to condition pretraining, not to the final prototype readout.
    config.training.classifier_epochs = args.condition_epochs
    config.training.n_pretrain_epochs = args.pretrain_epochs
    if args.smoke:
        config.training.local_steps_per_block = 1
        config.training.classifier_epochs = 1
        config.training.n_pretrain_epochs = 1
        args.context_adapt_epochs = 1

    seed_everything(config.seed)
    configure_torch(config)
    loaders = create_hit_dataloaders(config)[args.region]
    discovered = json.loads(Path(args.equation_path).read_text(encoding='utf-8'))
    if not discovered.get('validation', {}).get('passed'):
        raise ValueError('the full-NS SPIDER artifact has not passed validation')
    training_artifact = (discovered if args.physics_source == 'discovered'
                         else analytic_artifact(config.physics.viscosity))

    if args.physics_source == 'none':
        config.physics.condition_coefficients = None
    else:
        config.physics.condition_coefficients = [
            float(value) for value in training_artifact['equation']['coefficients'][1:4]]

    model = NoPropModel(config).to(config.device)
    if args.decoder_architecture == 'linear':
        decoder = LinearTemporalPhysicsDecoder(
            config.decoder.latent_dim, config.physics.n_time,
            config.decoder.output_channels,
            config.physics.physics_grid_size).to(config.device)
    else:
        decoder = TemporalPhysicsDecoder(
            config.decoder.latent_dim, config.decoder.base_channels,
            config.physics.n_time, config.decoder.output_channels).to(config.device)
    physics = TemporalNSPhysicsLoss(config, training_artifact).to(config.device)
    protocol_prefix = ('full_ns_v4'
                       if (args.input_size, args.target_size) == (16, 16)
                       else (f'full_ns_v5_input{args.input_size}_target'
                             f'{args.target_size}_{config.noprop.spatial_context_mode}'))
    shared_suffix = '_smoke' if args.smoke else ''
    architecture_suffix = ('' if args.decoder_architecture == 'conv'
                           else f'_{args.decoder_architecture}')
    shared_path = (Path('outputs/models') /
                   f'shared_{protocol_prefix}_{args.physics_source}_{args.region}_seed'
                   f'{args.seed}{architecture_suffix}{shared_suffix}.pt')
    condition_pretrain_seconds = decoder_pretrain_seconds = 0.0
    context_adaptation = None
    if shared_path.exists() and not args.rebuild_shared:
        load_shared_components(model, decoder, shared_path, config.device)
        print(f'Loaded shared components: {shared_path}')
    else:
        matched_condition_path = (
            Path('outputs/models') /
            f'shared_{protocol_prefix}_{args.physics_source}_{args.region}_seed'
            f'{args.seed}{shared_suffix}.pt')
        if (args.decoder_architecture != 'conv'
                and matched_condition_path.exists()):
            # Decoder ablations reuse the full selected context pathway and
            # prototypes from the main run, not a second adaptation run.
            load_shared_condition_components(
                model, matched_condition_path, config.device)
            print(f'Loaded matched condition components: {matched_condition_path}')
            context_enabled = getattr(model.encoder, 'context_enabled', None)
            if config.noprop.spatial_context_mode == 'residual_warmstart':
                # The main convolutional decoder was pretrained on centre-only
                # conditions before context adaptation. Match that latent
                # supervision while retaining the selected 32^3 condition state.
                model.encoder.context_enabled = False
            stage_started = time.perf_counter()
            try:
                pretrain_decoder(model, decoder, loaders['train'], config)
            finally:
                if context_enabled is not None:
                    model.encoder.context_enabled = context_enabled
            decoder_pretrain_seconds = time.perf_counter()-stage_started
            save_shared_components(model, decoder, shared_path)
            print(f'Saved shared components: {shared_path}')
        elif config.noprop.spatial_context_mode == 'residual_warmstart':
            reference_shared_path = (
                Path('outputs/models') /
                f'shared_full_ns_v4_{args.physics_source}_{args.region}_seed'
                f'{args.seed}.pt')
            stage_started = time.perf_counter()
            decoder_loaded = False
            if reference_shared_path.exists():
                decoder_loaded = args.decoder_architecture == 'conv'
                load_residual_context_warmstart(
                    model, decoder, reference_shared_path, config.device,
                    load_decoder=decoder_loaded)
            else:
                # A fresh checkout may not contain the large v4 checkpoints.
                # Train the centre pathway on the exact target cube first,
                # then copy it into the zero-gated residual encoder.
                model.encoder.context_enabled = False
                pretrain_encoder(model, loaders['train'], config,
                                 epochs=config.training.classifier_epochs,
                                 val_loader=loaders['val'])
                align_label_embeddings_to_encoder(model, loaders['train'], config)
                pretrain_decoder(model, decoder, loaders['train'], config)
                decoder_loaded = True
                initialize_residual_context_from_local(model)
            seed_everything(args.seed + 10_000)
            context_adaptation = adapt_residual_context(
                model, loaders['train'], loaders['val'], config,
                epochs=args.context_adapt_epochs)
            condition_pretrain_seconds = time.perf_counter()-stage_started
            if not decoder_loaded:
                decoder_started = time.perf_counter()
                pretrain_decoder(model, decoder, loaders['train'], config)
                decoder_pretrain_seconds = time.perf_counter()-decoder_started
            save_shared_components(model, decoder, shared_path)
            print(f'Saved shared components: {shared_path}')
        else:
            reference_shared_path = (Path('outputs/models') /
                                     f'shared_{protocol_prefix}_{args.physics_source}_'
                                     f'{args.region}_seed{args.seed}{shared_suffix}.pt')
            if args.decoder_architecture != 'conv' and reference_shared_path.exists():
                load_shared_condition_components(
                    model, reference_shared_path, config.device)
                print(f'Loaded matched condition components: {reference_shared_path}')
            else:
                stage_started = time.perf_counter()
                pretrain_encoder(model, loaders['train'], config,
                                 epochs=config.training.classifier_epochs,
                                 val_loader=loaders['val'])
                align_label_embeddings_to_encoder(model, loaders['train'], config)
                condition_pretrain_seconds = time.perf_counter()-stage_started
            stage_started = time.perf_counter()
            pretrain_decoder(model, decoder, loaders['train'], config)
            decoder_pretrain_seconds = time.perf_counter()-stage_started
            save_shared_components(model, decoder, shared_path)
            print(f'Saved shared components: {shared_path}')

    block_initialization_seed = args.seed + 20_000
    local_training_seed = args.seed + 30_000
    reset_local_blocks(model, block_initialization_seed)
    trainer = LocalNoPropTrainer(model, decoder, physics, config)
    condition_cache_stats = trainer.prepare_condition_cache(loaders['train'])
    # The local schedule, shuffled batches, diffusion noise, and sampled block
    # indices must not depend on whether shared components came from cache.
    seed_everything(local_training_seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    local_history = trainer.train_blocks(loaders['train'])
    block_seconds = time.perf_counter()-started

    # The final latent is classified by cosine similarity to the frozen label
    # prototypes; no separately trained output head is needed.
    classifier_history = []

    train_readout = trainer.evaluate(loaders['train'])
    validation_readout = trainer.evaluate(loaders['val'])
    training_test = trainer.evaluate(loaders['test'], include_physics=True)
    trainer.physics_loss = TemporalNSPhysicsLoss(config, discovered).to(config.device)
    common_test = trainer.evaluate(loaders['test'], include_physics=True)
    result = {
        'schema_version': 5 if protocol_prefix != 'full_ns_v4' else 4,
        'method': ('PI-NoProp-full-NS-v4' if protocol_prefix == 'full_ns_v4'
                   else f'PI-NoProp-{protocol_prefix}'),
        'model_revision': (
            'trainable-physics-condition-fusion-prototype-readout'
            if protocol_prefix == 'full_ns_v4' else
            'cached-residual-warmstart-physics-condition-prototype-readout-'
            f'input{args.input_size}-target{args.target_size}'),
        'readout': 'cosine similarity to frozen label embeddings',
        'input_spatial_size': args.input_size,
        'target_spatial_size': args.target_size,
        'spatial_context_mode': config.noprop.spatial_context_mode,
        'cache_dir': config.data.cache_dir,
        'physics_source': args.physics_source,
        'relation_set': args.relation_set,
        'decoder_architecture': args.decoder_architecture,
        'decoder_parameters': sum(parameter.numel()
                                  for parameter in decoder.parameters()),
        'equation_path': args.equation_path,
        'equation': training_artifact['equation'] if args.physics_source != 'none' else None,
        'region': args.region,
        'seed': args.seed,
        'trajectory_disjoint': True,
        'test': common_test,
        'train_readout': train_readout,
        'validation_readout': validation_readout,
        'training_equation_test_metrics': training_test,
        'block_train_seconds': block_seconds,
        'condition_pretrain_seconds': condition_pretrain_seconds,
        'decoder_pretrain_seconds': decoder_pretrain_seconds,
        'condition_cache_seconds': condition_cache_stats['seconds'],
        'condition_cache_samples': condition_cache_stats['samples'],
        'context_adaptation': context_adaptation,
        'context_gate_norm': (
            float(model.encoder.context_gate.detach().norm())
            if hasattr(model.encoder, 'context_gate') else None),
        'block_initialization_seed': block_initialization_seed,
        'local_training_seed': local_training_seed,
        'peak_memory_mb': (torch.cuda.max_memory_allocated()/1024**2
                           if torch.cuda.is_available() else 0.0),
        'block_updates': trainer.block_updates.tolist(),
    }
    suffix = '_smoke' if args.smoke else ''
    relation_suffix = '' if args.relation_set == 'ns' else f'_{args.relation_set}'
    tag_suffix = f'_{args.run_tag}' if args.run_tag else ''
    run_id = (f'{protocol_prefix}_{args.physics_source}_{args.region}_lambda'
              f'{config.physics.lambda_weight:g}_seed{args.seed}'
              f'{relation_suffix}{tag_suffix}{suffix}').replace('.', 'p')
    run_dir = Path('outputs/runs')/run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(run_dir/'checkpoint.pt', extra={'result': result})
    (run_dir/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (run_dir/'config.json').write_text(json.dumps(asdict(config), indent=2),
                                      encoding='utf-8')
    np.savez_compressed(run_dir/'history.npz',
                        local=np.asarray(local_history, dtype=object),
                        classifier=np.asarray(classifier_history, dtype=object))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
