import unittest

import torch

from src.noprop.classifier import PrototypeReadout


class TestPrototypeReadout(unittest.TestCase):
    def test_readout_is_parameter_free_and_uses_prototype_geometry(self):
        readout = PrototypeReadout(latent_dim=3, n_classes=3)
        self.assertEqual(sum(parameter.numel() for parameter in readout.parameters()), 0)

        prototypes = torch.eye(3)
        latents = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        logits = readout(latents, prototypes)
        self.assertEqual(logits.argmax(dim=-1).tolist(), [0, 1, 2])
        self.assertEqual(tuple(logits.shape), (3, 3))

    def test_incompatible_prototype_shape_is_rejected(self):
        readout = PrototypeReadout(latent_dim=4, n_classes=2)
        with self.assertRaises(ValueError):
            readout(torch.randn(1, 4), torch.randn(3, 4))


if __name__ == '__main__':
    unittest.main()
