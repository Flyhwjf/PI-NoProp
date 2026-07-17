import unittest

import torch

from src.noprop.diffusion import NoiseSchedule


class TestNoPropSchedule(unittest.TestCase):
    def test_perfect_denoiser_preserves_train_inference_marginals(self):
        schedule = NoiseSchedule(T=10)
        target = torch.randn(7, 16)
        noise = torch.randn_like(target)
        value = noise.clone()
        for index in range(schedule.T):
            a, b, c = schedule.get_coeffs(index)
            self.assertEqual(float(c), 0.0)
            value = a*target+b*value
            signal = schedule.signal[index+1]
            expected = signal.sqrt()*target+(1-signal).sqrt()*noise
            torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(value, target, atol=2e-6, rtol=2e-6)

    def test_first_block_is_trained_from_noise(self):
        schedule = NoiseSchedule(T=10)
        self.assertEqual(float(schedule.get_input_signal(0)), 0.0)
        self.assertEqual(float(schedule.signal[-1]), 1.0)
        self.assertTrue(torch.all(schedule.signal[1:] >= schedule.signal[:-1]))


if __name__ == '__main__':
    unittest.main()
