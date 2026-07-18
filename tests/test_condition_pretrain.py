import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import PINoPropConfig
from src.noprop.model import NoPropModel
from src.training.pretrain import (load_shared_components, pretrain_encoder,
                                   save_shared_components)


class TestConditionPretraining(unittest.TestCase):
    @staticmethod
    def _config():
        config = PINoPropConfig()
        config.device = 'cpu'
        config.data.n_classes = 2
        config.data.batch_size = 4
        config.noprop.condition_dim = 8
        config.noprop.embedding_dim = 8
        config.noprop.hidden_dim = 16
        config.decoder.latent_dim = 8
        config.physics.condition_coefficients = [1.0, 0.5, -0.1]
        config.training.use_amp = False
        config.training.fused_optimizer = False
        return config

    @staticmethod
    def _loader():
        generator = torch.Generator().manual_seed(9)
        rates = torch.linspace(-1.0, 1.0, 12)
        samples = []
        for rate in rates:
            samples.append({
                'field': torch.randn(4, 8, 8, 8, generator=generator),
                'ns_terms': torch.tensor([rate, 0.2*rate, -0.1*rate]),
                'label': torch.tensor(int(rate > 0), dtype=torch.long),
            })
        return DataLoader(samples, batch_size=4, shuffle=False)

    def test_physics_encoder_and_fusion_are_optimized(self):
        torch.manual_seed(3)
        config = self._config()
        model = NoPropModel(config)
        before_physics = [parameter.detach().clone()
                          for parameter in model.physics_encoder.parameters()]
        before_fusion = [parameter.detach().clone()
                         for parameter in model.condition_fusion.parameters()]

        pretrain_encoder(model, self._loader(), config, epochs=1)

        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(
                                before_physics, model.physics_encoder.parameters())))
        self.assertTrue(any(not torch.equal(old, new)
                            for old, new in zip(
                                before_fusion, model.condition_fusion.parameters())))

    def test_shared_checkpoint_round_trips_condition_modules(self):
        torch.manual_seed(5)
        config = self._config()
        model = NoPropModel(config)
        decoder = nn.Linear(8, 4)
        pretrain_encoder(model, self._loader(), config, epochs=1)

        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'shared.pt'
            save_shared_components(model, decoder, path)
            restored = NoPropModel(config)
            restored_decoder = nn.Linear(8, 4)
            load_shared_components(restored, restored_decoder, path, 'cpu')

        for name in ('encoder', 'physics_encoder', 'condition_fusion',
                     'label_embed'):
            expected = getattr(model, name).state_dict()
            actual = getattr(restored, name).state_dict()
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])
        for expected, actual in zip(decoder.parameters(),
                                    restored_decoder.parameters()):
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
