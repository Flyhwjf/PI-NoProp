"""Noise robustness of clean-trained vanilla and discovered PI-NoProp."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hit_dataset import create_hit_dataloaders
from src.noprop.model import NoPropModel
from src.physics.temporal_ns_loss import TemporalNSPhysicsLoss
from scripts.experiment_protocol import (ExperimentProtocol, add_protocol_arguments,
                                        load_run, make_decoder, protocol_from_args,
                                        validate_metadata)

REGIONS = ('low_enstrophy', 'high_enstrophy')
METHODS = ('none', 'discovered')
SEEDS = (42, 123, 456)
LEVELS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
REPETITIONS = 5


def ns_terms_from_standardized(fields, means, stds, dx, target_size=None):
    if target_size is not None:
        if any(size < target_size or (size-target_size) % 2 for size in fields.shape[-3:]):
            raise ValueError('physics target must be centred inside the input context')
        starts = [(size-target_size)//2 for size in fields.shape[-3:]]
        fields = fields[..., starts[0]:starts[0]+target_size,
                        starts[1]:starts[1]+target_size,
                        starts[2]:starts[2]+target_size]
    output_dtype = fields.dtype
    # Match the float64, second-order cache construction on clean inputs.
    physical = (fields*stds+means).double()
    velocity, pressure = physical[:, :3], physical[:, 3]
    gradients = torch.stack([
        torch.gradient(velocity, spacing=dx, dim=axis+2, edge_order=2)[0]
        for axis in range(3)], dim=2)
    convection = torch.einsum('bjxyz,bijxyz->bixyz', velocity, gradients)
    pressure_gradient = torch.stack([
        torch.gradient(pressure, spacing=dx, dim=axis+1, edge_order=2)[0]
        for axis in range(3)], dim=1)
    laplacian = sum(
        torch.gradient(torch.gradient(
            velocity, spacing=dx, dim=axis+2, edge_order=2)[0],
            spacing=dx, dim=axis+2, edge_order=2)[0]
        for axis in range(3))
    energy = 0.5*velocity.square().sum(1).mean((1, 2, 3)).clamp_min(1e-12)
    return torch.stack([
        (velocity*term).sum(1).mean((1, 2, 3))/energy
        for term in (convection, pressure_gradient, laplacian)], dim=1).to(output_dtype)


@torch.no_grad()
def evaluate(model, decoder, physics, loader, level, repetition, means, stds, dx,
             model_seed, target_size=None):
    device = means.device
    generator = torch.Generator(device=device).manual_seed(91_000+repetition)
    # Match the main evaluation and hold NoProp's initial latent fixed while
    # varying only the observation-noise realization.
    inference_seed = int(model_seed)+10_000
    torch.manual_seed(inference_seed)
    if device.type == 'cuda': torch.cuda.manual_seed_all(inference_seed)
    model.eval(); decoder.eval(); correct = total = 0; ns = batches = 0
    for batch in loader:
        fields = batch['field'].to(device)
        noisy = fields+level*torch.randn(fields.shape, generator=generator,
                                         device=device, dtype=fields.dtype)
        terms = ns_terms_from_standardized(noisy, means, stds, dx, target_size)
        labels = batch['label'].to(device)
        logits, latents = model(noisy, ns_terms=terms, return_all_latents=True)
        correct += int((logits.argmax(-1) == labels).sum()); total += len(labels)
        ns += physics.evaluate_metrics(decoder(latents[-1]))['eta_ns']; batches += 1
    return 100*correct/max(total, 1), ns/max(batches, 1)


def run_dir(method, region, seed, protocol=None):
    protocol = protocol or ExperimentProtocol()
    return ROOT/'outputs/runs'/protocol.run_id(method, region, seed)


def build_parser():
    parser = argparse.ArgumentParser()
    add_protocol_arguments(parser)
    parser.add_argument('--overwrite', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    protocol = protocol_from_args(args, parser)
    output = protocol.aggregate_path(ROOT, 'noise')
    previous = None
    if output.exists() and not args.overwrite:
        previous = json.loads(output.read_text())
        validate_metadata(previous.get('protocol', {}), protocol)
    result = {'schema_version': 5, 'protocol': {**protocol.metadata(),
        'levels_in_channel_standard_deviations': list(LEVELS),
        'noise_definition': 'additive standard Gaussian noise in discovery-standardised channel units',
        'seeds': list(SEEDS), 'repetitions_per_seed': REPETITIONS,
        'clean_trained': True, 'trajectory_disjoint_test': True,
        'inference_latent': 'fixed per trained-model seed across noise levels and repetitions',
        'varied_randomness': 'observation noise only',
        'model_revision': protocol.model_revision,
        'readout': 'cosine similarity to frozen label embeddings'}, 'results': {}}
    for region in REGIONS:
        result['results'][region] = {}
        for method in METHODS:
            retained = {}
            if previous is not None:
                retained = previous.get('results', {}).get(region, {}).get(method, {})
            result['results'][region][method] = {
                str(level): retained[str(level)] for level in LEVELS
                if str(level) in retained}
            missing = [level for level in LEVELS if str(level) not in retained]
            print('evaluate predictive noise levels' if missing else
                  'validate cached predictive noise levels', missing, region, method, flush=True)
            seed_records = {level: {'accuracy': [], 'eta_ns': []} for level in missing}
            for seed in SEEDS:
                checkpoint, config, _ = load_run(
                    run_dir(method, region, seed, protocol), protocol,
                    source=method, region=region, seed=seed)
                if not missing:
                    continue
                loaders = create_hit_dataloaders(config)[region]
                model = NoPropModel(config).to(config.device)
                decoder = make_decoder(config).to(config.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                artifact_path = Path(config.physics.discovered_artifact)
                if not artifact_path.is_absolute():
                    artifact_path = ROOT/artifact_path
                artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
                physics = TemporalNSPhysicsLoss(config, artifact).to(config.device)
                with np.load(Path(config.data.cache_dir)/'stats.npz') as stats:
                    means = torch.tensor(stats['means'], device=config.device).view(1,4,1,1,1)
                    stds = torch.tensor(stats['stds'], device=config.device).view(1,4,1,1,1)
                dx = config.physics.box_length/config.physics.dns_grid_size
                for level in missing:
                    repeated = [evaluate(model, decoder, physics, loaders['test'],
                                         level, repetition, means, stds, dx, seed,
                                         config.physics.physics_grid_size)
                                for repetition in range(REPETITIONS)]
                    seed_records[level]['accuracy'].append(
                        float(np.mean([value[0] for value in repeated])))
                    seed_records[level]['eta_ns'].append(
                        float(np.mean([value[1] for value in repeated])))
            for level in missing:
                record = seed_records[level]
                result['results'][region][method][str(level)] = {
                    key: {'values': values, 'mean': float(np.mean(values)),
                          'std': float(np.std(values, ddof=1))}
                    for key, values in record.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
