import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.config import PINoPropConfig
from src.decoder import TemporalPhysicsDecoder
from src.noprop.model import NoPropModel
from src.physics.temporal_ns_loss import TemporalNSPhysicsLoss
from src.training.local_trainer import LocalNoPropTrainer


class TestTemporalNSLoss(unittest.TestCase):
    def _loss(self, root):
        np.savez(Path(root)/'stats.npz', means=np.zeros(4, np.float32),
                 stds=np.ones(4, np.float32))
        config = PINoPropConfig()
        config.data.cache_dir = root
        config.physics.physics_grid_size = 8
        config.physics.n_time = 7
        config.physics.n_test_functions = 3
        config.physics.beta = 4
        config.physics.dns_grid_size = 24
        config.physics.snapshot_dt = 1e-3
        artifact = {
            'validation': {'passed': True},
            'equation': {
                'terms': ['time_derivative', 'convection',
                          'pressure_gradient', 'velocity_laplacian'],
                'coefficients': [1.0, 1.0, 1.0, -0.005],
            },
        }
        return config, TemporalNSPhysicsLoss(config, artifact)

    def test_temporal_decoder_shape(self):
        decoder = TemporalPhysicsDecoder(latent_dim=12, base_channels=8, n_time=7)
        output = decoder(torch.randn(2, 12))
        self.assertEqual(output.shape, (2, 7, 4, 16, 16, 16))

    def test_full_ns_loss_is_differentiable(self):
        with tempfile.TemporaryDirectory() as root:
            _, loss_module = self._loss(root)
            fields = torch.randn(2, 7, 4, 8, 8, 8, requires_grad=True)
            loss, metrics = loss_module(fields)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(fields.grad.abs().sum().item(), 0)
            self.assertIn('eta_ns', metrics)

    def test_pressure_poisson_and_energy_relations_are_differentiable(self):
        with tempfile.TemporaryDirectory() as root:
            _, loss_module = self._loss(root)
            loss_module.use_pressure_poisson = True
            loss_module.use_energy = True
            fields = torch.randn(2, 7, 4, 8, 8, 8, requires_grad=True)
            loss, metrics = loss_module(fields)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(fields.grad.abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(metrics['eta_pp']))
            self.assertTrue(torch.isfinite(metrics['eta_energy']))

    def test_unvalidated_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            np.savez(Path(root)/'stats.npz', means=np.zeros(4, np.float32),
                     stds=np.ones(4, np.float32))
            config = PINoPropConfig()
            config.data.cache_dir = root
            with self.assertRaises(ValueError):
                TemporalNSPhysicsLoss(config, {
                    'validation': {'passed': False},
                    'equation': {'terms': [], 'coefficients': []},
                })

    def test_full_ns_local_step_updates_only_selected_block(self):
        with tempfile.TemporaryDirectory() as root:
            config, loss_module = self._loss(root)
            config.device = 'cpu'
            config.training.use_amp = False
            config.physics.lambda_weight = 0.01
            config.physics.use_full_ns = True
            config.physics.use_pressure_poisson = False
            config.noprop.embedding_dim = 12
            config.noprop.condition_dim = 12
            config.noprop.hidden_dim = 24
            config.decoder.latent_dim = 12
            model = NoPropModel(config)
            decoder = TemporalPhysicsDecoder(
                latent_dim=12, base_channels=8, n_time=7)
            trainer = LocalNoPropTrainer(model, decoder, loss_module, config)
            for module in (model.physics_encoder, model.condition_fusion):
                self.assertTrue(all(not parameter.requires_grad
                                    for parameter in module.parameters()))
            batch = {
                'field': torch.randn(2, 4, 16, 16, 16),
                'sequence': torch.randn(2, 7, 4, 16, 16, 16),
                'label': torch.tensor([0, 1]),
            }
            metrics = trainer.train_local_step(batch, 2)
            self.assertIn('eta_ns', metrics)
            for index, block in enumerate(model.blocks):
                gradients = [parameter.grad for parameter in block.parameters()]
                self.assertEqual(any(value is not None for value in gradients), index == 2)

    def test_cache_has_trajectory_disjoint_splits(self):
        root = Path('data/cache_hit_ns')
        if not (root/'trajectory_ids.npy').exists():
            self.skipTest('formal cache has not been built')
        trajectory_ids = np.load(root/'trajectory_ids.npy')
        splits = np.load(root/'splits.npy')
        groups = [set(trajectory_ids[splits == code].tolist()) for code in range(3)]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])


if __name__ == '__main__':
    unittest.main()
