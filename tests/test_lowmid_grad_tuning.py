import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn

from attack import LazyAggregationAttacker, _LOWMID_GRAD_FFT_BANDS
from gradient_analysis import FFT_BANDS

try:
    import main
except ImportError as exc:
    main = None
    MAIN_IMPORT_ERROR = exc
else:
    MAIN_IMPORT_ERROR = None


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 16 * 16, 4)

    def forward(self, x, return_attn=False):
        return self.linear(x.flatten(1))


def make_attacker(**kwargs):
    return LazyAggregationAttacker(
        TinyModel(),
        epsilon=0.1,
        steps=2,
        ti_sigma=0,
        layers=(-1,),
        device=torch.device("cpu"),
        **kwargs,
    )


def band_energy(attacker, x, bands):
    return sum(attacker._fft_project_grad(x, band).square().sum().item() for band in bands)


def lowmid_ratio(attacker, x):
    lowmid = band_energy(attacker, x, range(6))
    high = band_energy(attacker, x, range(6, 8))
    return lowmid / (lowmid + high)


class LowMidGradientTuningTests(unittest.TestCase):
    def test_private_fft_bands_match_gradient_analysis(self):
        self.assertEqual(_LOWMID_GRAD_FFT_BANDS, FFT_BANDS)

    def test_disabled_tuning_returns_original_gradient_object(self):
        attacker = make_attacker(lowmid_grad_tuning=False)
        grad = torch.randn(2, 3, 16, 16)
        self.assertIs(attacker._tune_lowmid_gradient(grad), grad)

    def test_enabled_tuning_preserves_shape_and_is_finite(self):
        attacker = make_attacker(lowmid_grad_tuning=True)
        grad = torch.randn(2, 3, 16, 16)
        tuned = attacker._tune_lowmid_gradient(grad)
        self.assertEqual(tuned.shape, grad.shape)
        self.assertTrue(torch.isfinite(tuned).all())

    def test_zero_rotation_strength_reconstructs_original_gradient(self):
        attacker = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.0)
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        tuned = attacker._tune_lowmid_gradient(grad)
        self.assertTrue(torch.allclose(tuned, grad, atol=1e-10, rtol=1e-10))

    def test_rotation_increases_lowmid_energy_ratio_without_dropping_high(self):
        attacker = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.5)
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        tuned = attacker._tune_lowmid_gradient(grad)
        self.assertGreater(lowmid_ratio(attacker, tuned), lowmid_ratio(attacker, grad))
        self.assertGreater(band_energy(attacker, tuned, range(6, 8)), 1e-12)

    def test_stronger_rotation_increases_lowmid_ratio_more(self):
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        weak = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.25)._tune_lowmid_gradient(grad)
        strong_attacker = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.75)
        strong = strong_attacker._tune_lowmid_gradient(grad)
        self.assertGreater(lowmid_ratio(strong_attacker, strong), lowmid_ratio(strong_attacker, weak))

    def test_preserve_norm_matches_per_sample_original_norm(self):
        attacker = make_attacker(
            lowmid_grad_tuning=True,
            lowmid_grad_rotation_strength=0.5,
            lowmid_grad_preserve_norm=True,
        )
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        tuned = attacker._tune_lowmid_gradient(grad)
        self.assertTrue(torch.allclose(tuned.flatten(1).norm(dim=1), grad.flatten(1).norm(dim=1), atol=1e-10))

    def test_degenerate_gradients_are_stable(self):
        attacker = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.5)
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        pure_lowmid = sum((attacker._fft_project_grad(grad, band) for band in range(6)), torch.zeros_like(grad))
        pure_high = sum((attacker._fft_project_grad(grad, band) for band in range(6, 8)), torch.zeros_like(grad))
        for sample in (pure_lowmid, pure_high, torch.zeros_like(grad)):
            tuned = attacker._tune_lowmid_gradient(sample)
            self.assertTrue(torch.isfinite(tuned).all())
            self.assertTrue(torch.allclose(tuned, sample, atol=1e-10, rtol=1e-10))

    def test_attack_batch_smoke_with_tuning(self):
        attacker = make_attacker(lowmid_grad_tuning=True, lowmid_grad_rotation_strength=0.5)
        images = torch.randn(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        adv = attacker.attack_batch(images, labels)
        self.assertEqual(adv.shape, images.shape)

    def test_constructor_validates_lowmid_options(self):
        with self.assertRaises(ValueError):
            make_attacker(lowmid_grad_rotation_strength=-0.1)
        with self.assertRaises(ValueError):
            make_attacker(lowmid_grad_rotation_strength=1.0)
        with self.assertRaises(ValueError):
            make_attacker(lowmid_grad_preserve_norm=1)


@unittest.skipIf(main is None, f"main import failed: {MAIN_IMPORT_ERROR}")
class LowMidGradientTuningCLITests(unittest.TestCase):
    def test_parse_args_parses_lowmid_flags(self):
        argv = [
            "main.py",
            "--lowmid-grad-tuning",
            "--lowmid-grad-rotation-strength",
            "0.25",
            "--no-lowmid-grad-preserve-norm",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = main.parse_args()
        self.assertTrue(args.lowmid_grad_tuning)
        self.assertEqual(args.lowmid_grad_rotation_strength, 0.25)
        self.assertFalse(args.lowmid_grad_preserve_norm)

    def test_create_attacker_forwards_lowmid_options(self):
        attacker = main.create_attacker(
            model=TinyModel(),
            epsilon=0.1,
            step_size=None,
            steps=2,
            layers=(-1,),
            ti_sigma=0,
            lowmid_grad_tuning=True,
            lowmid_grad_rotation_strength=0.25,
            lowmid_grad_preserve_norm=False,
        )
        self.assertTrue(attacker.lowmid_grad_tuning)
        self.assertEqual(attacker.lowmid_grad_rotation_strength, 0.25)
        self.assertFalse(attacker.lowmid_grad_preserve_norm)


if __name__ == "__main__":
    unittest.main()
