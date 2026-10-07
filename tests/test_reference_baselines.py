import unittest

import torch

from src.baselines.noprop_reference import (ReferenceNoProp3D, ContinuousNoProp3D,
                                          LocalFieldEncoder)


class TestReferenceBaselines(unittest.TestCase):
    def test_context_encoder_preserves_legacy_output(self):
        encoder = LocalFieldEncoder()
        fields = torch.randn(2, 4, 16, 16, 16)
        torch.testing.assert_close(encoder(fields), encoder.net(fields))

    def test_reference_and_ct_accept_full_32_context(self):
        fields = torch.randn(2, 4, 32, 32, 32)
        labels = torch.tensor([1, 3])
        model = ReferenceNoProp3D(n_classes=5, n_blocks=2)
        model.local_loss(fields, labels, 1).backward()
        self.assertTrue(all(parameter.grad is None
                            for parameter in model.steps[0].parameters()))
        self.assertEqual(tuple(model(fields).shape), (2, 5))
        continuous = ContinuousNoProp3D(n_classes=5)
        loss = continuous.flow_matching_loss(fields, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(tuple(continuous.integrate(
            fields, steps=2, adjoint=False).shape), (2, 5))

    def test_reference_noprop_loss_is_strictly_local(self):
        model = ReferenceNoProp3D(n_classes=5, n_blocks=3)
        fields = torch.randn(2, 4, 16, 16, 16)
        labels = torch.tensor([1, 3])
        model.local_loss(fields, labels, 1).backward()

        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.steps[1].parameters()))
        for index in (0, 2):
            self.assertTrue(all(parameter.grad is None
                                for parameter in model.steps[index].parameters()))

    def test_reference_noprop_inference_shape(self):
        model = ReferenceNoProp3D(n_classes=5, n_blocks=2)
        output = model(torch.randn(2, 4, 16, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 5))

    def test_continuous_noprop_flow_and_adjoint_shapes(self):
        model = ContinuousNoProp3D(n_classes=5)
        fields = torch.randn(2, 4, 16, 16, 16)
        labels = torch.tensor([0, 4])
        loss = model.flow_matching_loss(fields, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

        model.zero_grad(set_to_none=True)
        output = model.integrate(fields, steps=2, adjoint=True)
        self.assertEqual(tuple(output.shape), (2, 5))
        output.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
