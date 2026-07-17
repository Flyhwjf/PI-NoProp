"""Microbenchmark the legacy and optimized PI-NoProp hot paths."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import PINoPropConfig
from src.data.dataset import create_cached_dataloaders
from src.decoder import FieldDecoder, PhysicsFieldDecoder
from src.noprop.model import NoPropModel
from src.physics.loss import PhysicsLoss
from src.physics.spatial_loss import SpatialPhysicsLoss
from src.training.local_trainer import LocalNoPropTrainer, configure_torch


def timed(function, repetitions=10):
    for _ in range(3):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repetitions):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repetitions


def main():
    cfg = PINoPropConfig()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.data.regions = ['centre']
    cfg.data.batch_size = 2
    configure_torch(cfg)
    batch = next(iter(create_cached_dataloaders(cfg)['centre']['train']))
    fields = batch['field'].to(cfg.device)
    model = NoPropModel(cfg).to(cfg.device)
    decoder = FieldDecoder().to(cfg.device)
    compact_decoder = PhysicsFieldDecoder().to(cfg.device)
    legacy = PhysicsLoss(cfg).to(cfg.device)
    spatial = SpatialPhysicsLoss(cfg).to(cfg.device)
    trainer = LocalNoPropTrainer(model, decoder, spatial, cfg)

    model.eval()
    with torch.no_grad():
        _, latents = model(fields, return_all_latents=True)
        decoded = decoder(latents[-1])
    report = {
        'legacy_physics_ms': timed(lambda: legacy(decoded), 5),
        'spatial_physics_ms': timed(lambda: spatial(decoded), 10),
        'legacy_decode_physics_T_ms': timed(
            lambda: sum(legacy(decoder(z))[0] for z in latents[1:]), 2),
        'optimized_local_step_ms': timed(lambda: trainer.train_local_step(batch, 0), 10),
        'compact_decoder_ms': timed(lambda: compact_decoder(latents[-1]), 20),
        'full_decoder_ms': timed(lambda: decoder(latents[-1]), 20),
    }
    report['physics_speedup'] = (report['legacy_physics_ms']
                                 / report['spatial_physics_ms'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
