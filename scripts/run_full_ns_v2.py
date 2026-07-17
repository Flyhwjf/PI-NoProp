"""Strictly local PI-NoProp with a SPIDER-discovered full NS equation."""
from __future__ import annotations

import argparse
import copy
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
from src.data.hit_ns_v2 import create_hit_ns_v2_dataloaders
from src.decoder import TemporalPhysicsDecoder
from src.noprop.model import NoPropModel
from src.physics.temporal_ns_loss import TemporalNSPhysicsLoss
from src.training.local_trainer import LocalNoPropTrainer, configure_torch
from src.training.pretrain import (align_label_embeddings_to_encoder,
                                   load_shared_components, pretrain_decoder,
                                   pretrain_encoder, save_shared_components)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    parser.add_argument('--equation-path', default='outputs/spider/hit_full_ns_v2.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps-per-block', type=int, default=100)
    parser.add_argument('--classifier-epochs', type=int, default=30)
    parser.add_argument('--pretrain-epochs', type=int, default=40)
    parser.add_argument('--lambda-phys', type=float, default=0.01)
    parser.add_argument('--relation-set', choices=['ns', 'ns_pp', 'full'],
                        default='ns')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rebuild-shared', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    config = PINoPropConfig()
    config.device = args.device
    config.seed = args.seed
    config.data.data_dir = 'data/generated_hit_ns_v2'
    config.data.cache_dir = 'data/cache_hit_ns_v2'
    config.data.regions = [args.region]
    config.data.subdomain_size = 16
    config.data.n_subdomains = 64  # samples generated per independent trajectory
    config.data.n_classes = 5
    config.data.batch_size = args.batch_size
    config.data.trajectory_disjoint = True
    config.noprop.normalize_condition = True
    config.decoder.use_temporal_decoder = True
    config.physics.n_time = 9
    config.physics.beta = 4.0
    config.physics.n_test_functions = 8
    config.physics.physics_grid_size = 16
    config.physics.use_continuity = True
    config.physics.use_pressure_poisson = args.relation_set in ('ns_pp', 'full')
    config.physics.use_energy = args.relation_set == 'full'
    config.physics.use_full_ns = True
    config.physics.lambda_weight = (0.0 if args.physics_source == 'none'
                                    else args.lambda_phys)
    config.physics.discovered_artifact = args.equation_path
    config.training.local_steps_per_block = args.steps_per_block
    config.training.classifier_epochs = args.classifier_epochs
    config.training.n_pretrain_epochs = args.pretrain_epochs
    if args.smoke:
        config.training.local_steps_per_block = 1
        config.training.classifier_epochs = 1
        config.training.n_pretrain_epochs = 1

    seed_everything(config.seed)
    configure_torch(config)
    loaders = create_hit_ns_v2_dataloaders(config)[args.region]
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
    decoder = TemporalPhysicsDecoder(
        config.decoder.latent_dim, config.decoder.base_channels,
        config.physics.n_time, config.decoder.output_channels).to(config.device)
    physics = TemporalNSPhysicsLoss(config, training_artifact).to(config.device)
    shared_suffix = '_smoke' if args.smoke else ''
    shared_path = (Path('outputs/models') /
                   f'shared_full_ns_v3_{args.physics_source}_{args.region}_seed'
                   f'{args.seed}{shared_suffix}.pt')
    if shared_path.exists() and not args.rebuild_shared:
        load_shared_components(model, decoder, shared_path, config.device)
        print(f'Loaded shared components: {shared_path}')
    else:
        pretrain_encoder(model, loaders['train'], config,
                         epochs=config.training.classifier_epochs,
                         val_loader=loaders['val'])
        align_label_embeddings_to_encoder(model, loaders['train'], config)
        pretrain_decoder(model, decoder, loaders['train'], config)
        save_shared_components(model, decoder, shared_path)
        print(f'Saved shared components: {shared_path}')

    trainer = LocalNoPropTrainer(model, decoder, physics, config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    local_history = trainer.train_blocks(loaders['train'])
    block_seconds = time.perf_counter()-started

    classifier_history, best_state, best_accuracy = [], None, -1.0
    for epoch in range(config.training.classifier_epochs):
        train_metrics = trainer.train_classifier_epoch(loaders['train'])
        val_metrics = trainer.evaluate(loaders['val'])
        classifier_history.append({'epoch': epoch, 'train': train_metrics,
                                   'val': val_metrics})
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            best_state = copy.deepcopy(model.classifier.state_dict())
    if best_state is not None:
        model.classifier.load_state_dict(best_state)

    training_test = trainer.evaluate(loaders['test'], include_physics=True)
    trainer.physics_loss = TemporalNSPhysicsLoss(config, discovered).to(config.device)
    common_test = trainer.evaluate(loaders['test'], include_physics=True)
    result = {
        'schema_version': 3,
        'method': 'PI-NoProp-full-NS-v3',
        'model_revision': 'physics-conditioned-conv-consistent-schedule',
        'physics_source': args.physics_source,
        'relation_set': args.relation_set,
        'equation_path': args.equation_path,
        'equation': training_artifact['equation'] if args.physics_source != 'none' else None,
        'region': args.region,
        'seed': args.seed,
        'trajectory_disjoint': True,
        'test': common_test,
        'training_equation_test_metrics': training_test,
        'block_train_seconds': block_seconds,
        'peak_memory_mb': (torch.cuda.max_memory_allocated()/1024**2
                           if torch.cuda.is_available() else 0.0),
        'block_updates': trainer.block_updates.tolist(),
    }
    suffix = '_smoke' if args.smoke else ''
    relation_suffix = '' if args.relation_set == 'ns' else f'_{args.relation_set}'
    tag_suffix = f'_{args.run_tag}' if args.run_tag else ''
    run_id = (f'full_ns_v3_{args.physics_source}_{args.region}_lambda'
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
