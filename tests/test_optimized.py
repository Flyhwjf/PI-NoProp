import sys
import unittest
import json
import tempfile
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import PINoPropConfig
from src.decoder import FieldDecoder
from src.noprop.model import NoPropModel
from src.physics.spatial_loss import SpatialPhysicsLoss
from src.physics.discovered_equation import load_validated_equation
from src.training.local_trainer import LocalNoPropTrainer


class OptimizedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = PINoPropConfig()
        self.cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg.data.batch_size = 2
        self.cfg.training.use_amp = torch.cuda.is_available()
        self.model = NoPropModel(self.cfg).to(self.cfg.device)
        self.decoder = FieldDecoder().to(self.cfg.device)
        self.physics = SpatialPhysicsLoss(self.cfg).to(self.cfg.device)

    def test_spatial_loss_is_differentiable(self):
        fields = torch.randn(2, 4, 32, 32, 32, device=self.cfg.device,
                             requires_grad=True)
        loss, metrics = self.physics(fields)
        loss.backward()
        self.assertIsNotNone(fields.grad)
        self.assertTrue(torch.isfinite(fields.grad).all())
        self.assertIn('eta_pp', metrics)

    def test_constant_velocity_has_zero_weak_divergence(self):
        fields = torch.zeros(1, 4, 16, 16, 16, device=self.cfg.device)
        fields[:, 0] = 1.0
        r_div, _ = self.physics.residuals(fields)
        self.assertLess(float(r_div.abs().max()), 1e-4)

    def test_only_selected_block_receives_gradient(self):
        trainer = LocalNoPropTrainer(self.model, self.decoder, self.physics, self.cfg)
        batch = {
            'field': torch.randn(2, 4, 32, 32, 32),
            'label': torch.tensor([0, 1]),
        }
        trainer.train_local_step(batch, 3)
        for index, block in enumerate(self.model.blocks):
            gradients = [parameter.grad for parameter in block.parameters()]
            if index == 3:
                self.assertTrue(any(gradient is not None for gradient in gradients))
            else:
                self.assertTrue(all(gradient is None for gradient in gradients))
        self.assertTrue(all(parameter.grad is None
                            for parameter in self.model.encoder.parameters()))
        self.assertTrue(all(parameter.grad is None
                            for parameter in self.decoder.parameters()))

    def test_classifier_does_not_backpropagate_into_blocks(self):
        trainer = LocalNoPropTrainer(self.model, self.decoder, self.physics, self.cfg)
        batch = {
            'field': torch.randn(2, 4, 32, 32, 32),
            'label': torch.tensor([0, 1]),
        }
        trainer.train_classifier_epoch([batch])
        self.assertTrue(all(parameter.grad is None
                            for block in self.model.blocks
                            for parameter in block.parameters()))
        self.assertTrue(any(parameter.grad is not None
                            for parameter in self.model.classifier.parameters()))

    def test_persisted_splits_are_disjoint(self):
        split_path = Path('data/cache/centre_split.npz')
        if not split_path.exists():
            self.skipTest('first-frame cache has not been prepared')
        split = np.load(split_path)
        train, val, test = map(lambda key: set(split[key].tolist()),
                               ('train', 'val', 'test'))
        self.assertFalse(train & val)
        self.assertFalse(train & test)
        self.assertFalse(val & test)
        self.assertEqual((len(train), len(val), len(test)), (192, 32, 32))

    def test_unvalidated_spider_artifact_is_rejected(self):
        artifact = {
            'schema_version': 1,
            'equation': {'terms': ['pressure_laplacian', 'convection_divergence'],
                         'coefficients': [1.0, 0.3]},
            'validation': {'passed': False, 'failure_reasons': ['test failure']},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'equation.json'
            path.write_text(json.dumps(artifact), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_validated_equation(path)

    def test_discovered_coefficients_reach_spatial_loss(self):
        artifact = {
            'equation': {
                'terms': ['pressure_laplacian', 'convection_divergence'],
                'coefficients': [1.0, 0.293],
            }
        }
        loss = SpatialPhysicsLoss(self.cfg, artifact).to(self.cfg.device)
        self.assertAlmostEqual(float(loss.pp_coefficients[1]), 0.293, places=5)
        self.assertEqual(loss.physics_source, 'spider_discovered')


if __name__ == '__main__':
    unittest.main()
