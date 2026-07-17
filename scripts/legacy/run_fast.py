"""Optimized, strictly local PI-NoProp training entry point."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PINoPropConfig
from src.data.dataset import create_cached_dataloaders, cache_loaders_on_device
from src.decoder import FieldDecoder, PhysicsFieldDecoder
from src.noprop.model import NoPropModel
from src.physics.spatial_loss import SpatialPhysicsLoss
from src.physics.discovered_equation import load_validated_equation
from src.training.local_trainer import LocalNoPropTrainer, configure_torch
from src.training.pretrain import (load_shared_components, pretrain_decoder,
                                   pretrain_encoder,
                                   align_label_embeddings_to_encoder,
                                   save_shared_components)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', choices=['centre', 'edge'], default='centre')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps-per-block', type=int, default=None)
    parser.add_argument('--classifier-epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lambda-phys', type=float, default=None)
    parser.add_argument('--physics-source', choices=['analytic', 'discovered', 'none'],
                        default='analytic')
    parser.add_argument('--equation-path',
                        default='outputs/spider/hit_spatial_equation.json')
    parser.add_argument('--local-rec-weight', type=float, default=None)
    parser.add_argument('--no-continuity', action='store_true')
    parser.add_argument('--no-pressure-poisson', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rebuild-shared', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='Use tiny step counts for correctness checks')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = PINoPropConfig()
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.noprop.normalize_condition = True
    cfg.data.regions = [args.region]
    cfg.data.batch_size = args.batch_size or cfg.training.fast_batch_size
    if args.steps_per_block is not None:
        cfg.training.local_steps_per_block = args.steps_per_block
    if args.classifier_epochs is not None:
        cfg.training.classifier_epochs = args.classifier_epochs
    if args.lambda_phys is not None:
        cfg.physics.lambda_weight = args.lambda_phys
    if args.local_rec_weight is not None:
        cfg.training.local_rec_weight = args.local_rec_weight
    if args.no_continuity:
        cfg.physics.use_continuity = False
    if args.no_pressure_poisson:
        cfg.physics.use_pressure_poisson = False
    if args.physics_source == 'none':
        cfg.physics.use_continuity = False
        cfg.physics.use_pressure_poisson = False
    if args.smoke:
        cfg.training.local_steps_per_block = 1
        cfg.training.classifier_epochs = 1
        cfg.training.n_pretrain_epochs = 1

    seed_everything(cfg.seed)
    configure_torch(cfg)
    all_loaders = create_cached_dataloaders(cfg)
    if cfg.data.cache_on_device and str(cfg.device).startswith('cuda'):
        all_loaders = cache_loaders_on_device(all_loaders, cfg.device)
    loaders = all_loaders[args.region]
    model = NoPropModel(cfg).to(cfg.device)
    decoder_class = (PhysicsFieldDecoder
                     if cfg.decoder.use_compact_physics_decoder else FieldDecoder)
    decoder = decoder_class(cfg.decoder.latent_dim, cfg.decoder.base_channels,
                            cfg.decoder.output_channels).to(cfg.device)
    discovered_artifact = (load_validated_equation(args.equation_path)
                           if args.physics_source == 'discovered' else None)
    physics = SpatialPhysicsLoss(cfg, discovered_artifact=discovered_artifact).to(cfg.device)

    decoder_kind = ('compact16' if cfg.decoder.use_compact_physics_decoder
                    else 'full32')
    shared_suffix = '_smoke' if args.smoke else ''
    shared_path = (Path('outputs/models') /
                   f'shared_fast_{args.region}_{decoder_kind}{shared_suffix}.pt')
    if shared_path.exists() and not args.rebuild_shared:
        load_shared_components(model, decoder, shared_path, cfg.device)
        print(f'Loaded shared components: {shared_path}')
    else:
        pretrain_encoder(model, loaders['train'], cfg, val_loader=loaders['val'])
        align_label_embeddings_to_encoder(model, loaders['train'], cfg)
        pretrain_decoder(model, decoder, loaders['train'], cfg)
        save_shared_components(model, decoder, shared_path)
        print(f'Saved shared components: {shared_path}')

    trainer = LocalNoPropTrainer(model, decoder, physics, cfg)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = trainer.train_blocks(loaders['train'])
    block_seconds = time.perf_counter() - start

    classifier_history = []
    best_classifier = None
    best_val_accuracy = -1.0
    for epoch in range(cfg.training.classifier_epochs):
        train_metrics = trainer.train_classifier_epoch(loaders['train'])
        val_metrics = trainer.evaluate(loaders['val'])
        classifier_history.append({'epoch': epoch, 'train': train_metrics,
                                   'val': val_metrics})
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_classifier = copy.deepcopy(model.classifier.state_dict())
        if epoch == cfg.training.classifier_epochs - 1 or (epoch + 1) % 10 == 0:
            print(f'classifier {epoch+1}/{cfg.training.classifier_epochs}: '
                  f'val_acc={val_metrics["accuracy"]:.1f}%')

    if best_classifier is not None:
        model.classifier.load_state_dict(best_classifier)

    test_metrics = trainer.evaluate(loaders['test'], include_physics=True)
    test_metrics['eta_pp_training_source'] = test_metrics['eta_pp']
    analytic_evaluator = SpatialPhysicsLoss(cfg).to(cfg.device)
    training_evaluator = trainer.physics_loss
    trainer.physics_loss = analytic_evaluator
    analytic_metrics = trainer.evaluate(loaders['test'], include_physics=True)
    test_metrics['eta_pp_analytic'] = analytic_metrics['eta_pp']
    equation_file = Path(args.equation_path)
    if equation_file.exists():
        evaluation_artifact = (discovered_artifact
                               or load_validated_equation(equation_file))
        trainer.physics_loss = SpatialPhysicsLoss(
            cfg, discovered_artifact=evaluation_artifact).to(cfg.device)
        discovered_metrics = trainer.evaluate(loaders['test'], include_physics=True)
        test_metrics['eta_pp_discovered'] = discovered_metrics['eta_pp']
        test_metrics['eta_pp'] = discovered_metrics['eta_pp']
    trainer.physics_loss = training_evaluator
    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
               if torch.cuda.is_available() else 0.0)
    constraints = ('divpp' if (cfg.physics.use_continuity and
                               cfg.physics.use_pressure_poisson)
                   else 'div' if cfg.physics.use_continuity
                   else 'pp' if cfg.physics.use_pressure_poisson else 'none')
    smoke_suffix = '_smoke' if args.smoke else ''
    lambda_token = f'lambda{cfg.physics.lambda_weight:g}'.replace('.', 'p')
    source = ('spider' if args.physics_source == 'discovered'
              else 'analytic' if args.physics_source == 'analytic' else 'none')
    run_id = (f'local_fast_{source}_{constraints}_{args.region}_{lambda_token}_seed{cfg.seed}_'
              f'steps{cfg.training.local_steps_per_block}{smoke_suffix}')
    run_dir = Path('outputs/runs') / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(run_dir / 'checkpoint.pt', extra={'test': test_metrics})
    (run_dir / 'config.json').write_text(
        json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    result = {
        'run_id': run_id,
        'method': 'pi_noprop_local_fast',
        'physics_source': source,
        'equation_path': args.equation_path if discovered_artifact else None,
        'equation': discovered_artifact['equation'] if discovered_artifact else None,
        'region': args.region,
        'seed': cfg.seed,
        'test': test_metrics,
        'block_train_seconds': block_seconds,
        'peak_memory_mb': peak_mb,
        'block_updates': trainer.block_updates.tolist(),
        'n_parameters_total': sum(p.numel() for p in model.parameters())
                              + sum(p.numel() for p in decoder.parameters()),
        'n_parameters_trainable_local': sum(
            p.numel() for block in model.blocks for p in block.parameters()),
    }
    (run_dir / 'metrics.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(run_dir / 'history.npz',
                        local=np.asarray(history, dtype=object),
                        classifier=np.asarray(classifier_history, dtype=object))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
