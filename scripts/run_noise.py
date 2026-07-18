"""Noise robustness of clean-trained vanilla and discovered PI-NoProp."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hit_dataset import create_hit_dataloaders
from src.decoder import TemporalPhysicsDecoder
from src.noprop.model import NoPropModel
from src.physics.temporal_ns_loss import TemporalNSPhysicsLoss

REGIONS = ('low_enstrophy', 'high_enstrophy')
METHODS = ('none', 'discovered')
SEEDS = (42, 123, 456)
LEVELS = (0.0, 0.1, 0.5, 1.0)
REPETITIONS = 5


def ns_terms_from_standardized(fields, means, stds, dx):
    physical = fields*stds+means
    velocity, pressure = physical[:, :3], physical[:, 3]
    gradients = torch.stack([
        torch.gradient(velocity, spacing=dx, dim=axis+2)[0]
        for axis in range(3)], dim=2)
    convection = torch.einsum('bjxyz,bijxyz->bixyz', velocity, gradients)
    pressure_gradient = torch.stack([
        torch.gradient(pressure, spacing=dx, dim=axis+1)[0]
        for axis in range(3)], dim=1)
    laplacian = sum(
        torch.gradient(torch.gradient(velocity, spacing=dx, dim=axis+2)[0],
                       spacing=dx, dim=axis+2)[0]
        for axis in range(3))
    energy = 0.5*velocity.square().sum(1).mean((1, 2, 3)).clamp_min(1e-12)
    return torch.stack([
        (velocity*term).sum(1).mean((1, 2, 3))/energy
        for term in (convection, pressure_gradient, laplacian)], dim=1)


@torch.no_grad()
def evaluate(model, decoder, physics, loader, level, repetition, means, stds, dx):
    device = means.device
    generator = torch.Generator(device=device).manual_seed(91_000+repetition)
    torch.manual_seed(71_000+repetition)
    if device.type == 'cuda': torch.cuda.manual_seed_all(71_000+repetition)
    model.eval(); decoder.eval(); correct = total = 0; ns = batches = 0
    for batch in loader:
        fields = batch['field'].to(device)
        noisy = fields+level*torch.randn(fields.shape, generator=generator,
                                         device=device, dtype=fields.dtype)
        terms = ns_terms_from_standardized(noisy, means, stds, dx)
        labels = batch['label'].to(device)
        logits, latents = model(noisy, ns_terms=terms, return_all_latents=True)
        correct += int((logits.argmax(-1) == labels).sum()); total += len(labels)
        ns += physics.evaluate_metrics(decoder(latents[-1]))['eta_ns']; batches += 1
    return 100*correct/max(total, 1), ns/max(batches, 1)


def main():
    artifact = json.loads((ROOT/'outputs/spider/full_ns_equation.json').read_text())
    result = {'schema_version': 2, 'protocol': {
        'levels_in_channel_standard_deviations': list(LEVELS),
        'seeds': list(SEEDS), 'repetitions_per_seed': REPETITIONS,
        'clean_trained': True, 'trajectory_disjoint_test': True,
        'model_revision': 'trainable-physics-condition-fusion'}, 'results': {}}
    for region in REGIONS:
        result['results'][region] = {}
        for method in METHODS:
            seed_records = {level: {'accuracy': [], 'eta_ns': []} for level in LEVELS}
            for seed in SEEDS:
                weight = '0' if method == 'none' else '0p01'
                run = (ROOT/'outputs/runs'/
                       f'full_ns_v4_{method}_{region}_lambda{weight}_seed{seed}')
                checkpoint = torch.load(run/'checkpoint.pt', map_location='cuda',
                                        weights_only=False)
                config = checkpoint['config']; config.data.regions = [region]
                config.data.data_dir = 'data/generated_hit_ns'
                config.data.cache_dir = 'data/cache_hit_ns'
                loaders = create_hit_dataloaders(config)[region]
                model = NoPropModel(config).to(config.device)
                decoder = TemporalPhysicsDecoder(128, 32, 9, 4).to(config.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                decoder.load_state_dict(checkpoint['decoder_state_dict'])
                physics = TemporalNSPhysicsLoss(config, artifact).to(config.device)
                stats = np.load(ROOT/'data/cache_hit_ns/stats.npz')
                means = torch.tensor(stats['means'], device=config.device).view(1,4,1,1,1)
                stds = torch.tensor(stats['stds'], device=config.device).view(1,4,1,1,1)
                dx = config.physics.box_length/config.physics.dns_grid_size
                for level in LEVELS:
                    repeated = [evaluate(model, decoder, physics, loaders['test'],
                                         level, repetition, means, stds, dx)
                                for repetition in range(REPETITIONS)]
                    seed_records[level]['accuracy'].append(
                        float(np.mean([value[0] for value in repeated])))
                    seed_records[level]['eta_ns'].append(
                        float(np.mean([value[1] for value in repeated])))
            result['results'][region][method] = {}
            for level in LEVELS:
                record = seed_records[level]
                result['results'][region][method][str(level)] = {
                    key: {'values': values, 'mean': float(np.mean(values)),
                          'std': float(np.std(values, ddof=1))}
                    for key, values in record.items()}
    output = ROOT/'outputs/aggregate/full_ns_noise.json'
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
