import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import PINoPropConfig
from src.noprop.model import NoPropModel
from src.training.pretrain import (load_shared_components, pretrain_encoder,
                                   save_shared_components,
                                   build_condition_cache,
                                   lookup_condition_cache,
                                   load_shared_condition_components)


class _IndexedConditionDataset(Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(17)
        self.indices = torch.arange(6).numpy()
        self.fields = torch.randn(6, 4, 8, 8, 8, generator=generator)
        rates = torch.linspace(-1.0, 1.0, 6)
        self.terms = torch.stack([rates, 0.2*rates, -0.1*rates], dim=1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {
            'field': self.fields[index],
            'ns_terms': self.terms[index],
            'label': torch.tensor(index % 2, dtype=torch.long),
            'idx': int(self.indices[index]),
        }


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

    def test_cached_conditions_equal_direct_encoder_output(self):
        config = self._config()
        config.data.cache_on_device = False
        model = NoPropModel(config)
        loader = DataLoader(_IndexedConditionDataset(), batch_size=3,
                            shuffle=True)
        cache = build_condition_cache(model, loader, config)
        self.assertIsNotNone(cache)
        batch = next(iter(loader))
        cached = lookup_condition_cache(cache, batch, torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            direct = model.encode_condition(batch['field'], batch['ns_terms'])
        torch.testing.assert_close(cached, direct)

    def test_decoder_ablation_reuses_selected_context_and_prototypes(self):
        config = self._config()
        config.data.subdomain_size = 32
        config.data.target_subdomain_size = 16
        config.noprop.spatial_context_mode = 'residual_warmstart'
        model = NoPropModel(config)
        with torch.no_grad():
            model.encoder.context_gate.fill_(0.2)
        model.eval()
        fields = torch.randn(2, 4, 32, 32, 32)
        terms = torch.randn(2, 3)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'shared.pt'
            save_shared_components(model, nn.Linear(8, 4), path)
            restored = NoPropModel(config)
            # The ablation decoder has a different shape and is not loaded.
            load_shared_condition_components(restored, path, 'cpu')
            restored.eval()
            with torch.no_grad():
                expected = model.encode_condition(fields, terms)
                actual = restored.encode_condition(fields, terms)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(restored._label_prototypes(),
                                       model._label_prototypes())
            self.assertTrue(restored.encoder.context_enabled)

    def test_dual_scale_encoder_initially_preserves_centre_feature(self):
        config = self._config()
        config.data.subdomain_size = 32
        config.data.target_subdomain_size = 16
        config.noprop.spatial_context_mode = 'dual_scale'
        model = NoPropModel(config)
        fields = torch.randn(2, 4, 32, 32, 32)
        model.encoder.eval()
        with torch.no_grad():
            encoded = model.encoder(fields)
            centre = model.encoder.backbone(fields[:, :, 8:24, 8:24, 8:24])
        torch.testing.assert_close(encoded, centre)

    def test_residual_context_encoder_initially_preserves_centre_feature(self):
        config = self._config()
        config.data.subdomain_size = 32
        config.data.target_subdomain_size = 16
        config.noprop.spatial_context_mode = 'residual_dual_scale'
        model = NoPropModel(config)
        fields = torch.randn(2, 4, 32, 32, 32)
        model.encoder.eval()
        with torch.no_grad():
            encoded = model.encoder(fields)
            centre = model.encoder.local_backbone(
                fields[:, :, 8:24, 8:24, 8:24])
        torch.testing.assert_close(encoded, centre)


if __name__ == '__main__':
    unittest.main()
