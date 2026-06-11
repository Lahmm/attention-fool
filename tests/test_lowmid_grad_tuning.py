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

    def test_dim_resonance_augmentation_is_finite_lowmid_biased(self):
        torch.manual_seed(9)
        attacker = make_attacker(guide_aug_strength=0.5, dim_resize_range=(0.5, 0.5))
        pixels = torch.rand(2, 3, 16, 16)
        augmented = attacker._augment_full_image(pixels, "dim_resonance")
        delta = augmented - pixels
        self.assertEqual(augmented.shape, pixels.shape)
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertGreater(lowmid_ratio(attacker, delta), 0.75)

    def test_no_step_projection_allows_delta_to_exceed_epsilon(self):
        images = torch.zeros(1, 3, 16, 16)
        labels = torch.tensor([1])
        projected = LazyAggregationAttacker(
            TinyModel(), epsilon=0.1, step_size=0.1, steps=2, ti_sigma=0, layers=(-1,),
            project_each_step=True, device=torch.device("cpu")
        )
        unprojected = LazyAggregationAttacker(
            TinyModel(), epsilon=0.1, step_size=0.1, steps=2, ti_sigma=0, layers=(-1,),
            project_each_step=False, device=torch.device("cpu")
        )
        for attacker in (projected, unprojected):
            attacker._attack_grad = lambda pixels, _labels, _guide: torch.ones_like(pixels)
        projected_adv = projected._denormalize(projected.attack_batch(images, labels))
        unprojected_adv = unprojected._denormalize(unprojected.attack_batch(images, labels))
        clean = projected._denormalize(images)
        self.assertLessEqual((projected_adv - clean).abs().max().item(), 0.10001)
        self.assertGreater((unprojected_adv - clean).abs().max().item(), 0.15)


    def test_lowmid_dss_sign_filter_threshold_is_monotonic(self):
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        term_grads = (grad, -grad, grad)
        loose = make_attacker(
            lowmid_dss_filter=True,
            lowmid_dss_consistency="sign",
            lowmid_dss_agreement_threshold=0.34,
        )._apply_lowmid_dss_filter(grad, term_grads)
        strict = make_attacker(
            lowmid_dss_filter=True,
            lowmid_dss_consistency="sign",
            lowmid_dss_agreement_threshold=1.0,
        )._apply_lowmid_dss_filter(grad, term_grads)
        self.assertGreaterEqual(band_energy(make_attacker(), loose, range(6)), band_energy(make_attacker(), strict, range(6)))

    def test_lowmid_dss_filter_preserves_high_component(self):
        attacker = make_attacker(lowmid_dss_filter=True, lowmid_dss_consistency="sign")
        grad = torch.randn(2, 3, 16, 16, dtype=torch.float64)
        term_grads = (grad, -grad, grad)
        filtered = attacker._apply_lowmid_dss_filter(grad, term_grads)
        raw_high = sum((attacker._fft_project_grad(grad, band) for band in range(6, 8)), torch.zeros_like(grad))
        filtered_high = sum((attacker._fft_project_grad(filtered, band) for band in range(6, 8)), torch.zeros_like(filtered))
        self.assertTrue(torch.allclose(filtered_high, raw_high, atol=1e-10, rtol=1e-10))

    def test_lowmid_dss_cos_filter_is_finite_and_shape_preserving(self):
        attacker = make_attacker(lowmid_dss_filter=True, lowmid_dss_consistency="cos")
        grad = torch.randn(2, 3, 16, 16)
        term_grads = (grad, grad + 0.01 * torch.randn_like(grad), -grad)
        filtered = attacker._apply_lowmid_dss_filter(grad, term_grads)
        self.assertEqual(filtered.shape, grad.shape)
        self.assertTrue(torch.isfinite(filtered).all())

    def test_lowmid_dss_disabled_returns_original_gradient_object(self):
        attacker = make_attacker(lowmid_dss_filter=False)
        grad = torch.randn(2, 3, 16, 16)
        self.assertIs(attacker._apply_lowmid_dss_filter(grad, (grad,)), grad)

    def test_constructor_validates_lowmid_options(self):
        with self.assertRaises(ValueError):
            make_attacker(lowmid_grad_rotation_strength=-0.1)
        with self.assertRaises(ValueError):
            make_attacker(lowmid_grad_rotation_strength=1.0)
        with self.assertRaises(ValueError):
            make_attacker(lowmid_grad_preserve_norm=1)
        with self.assertRaises(ValueError):
            make_attacker(lowmid_dss_consistency="bad")
        with self.assertRaises(ValueError):
            make_attacker(lowmid_dss_agreement_threshold=1.5)


@unittest.skipIf(main is None, f"main import failed: {MAIN_IMPORT_ERROR}")
class LowMidGradientTuningCLITests(unittest.TestCase):
    def test_parse_args_parses_lowmid_flags(self):
        argv = [
            "main.py",
            "--lowmid-grad-tuning",
            "--lowmid-grad-rotation-strength",
            "0.25",
            "--no-lowmid-grad-preserve-norm",
            "--lowmid-dss-filter",
            "--lowmid-dss-consistency",
            "cos",
            "--lowmid-dss-agreement-threshold",
            "0.8",
            "--no-step-projection",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = main.parse_args()
        self.assertTrue(args.lowmid_grad_tuning)
        self.assertEqual(args.lowmid_grad_rotation_strength, 0.25)
        self.assertFalse(args.lowmid_grad_preserve_norm)
        self.assertTrue(args.lowmid_dss_filter)
        self.assertEqual(args.lowmid_dss_consistency, "cos")
        self.assertEqual(args.lowmid_dss_agreement_threshold, 0.8)
        self.assertFalse(args.project_each_step)

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
            lowmid_dss_filter=True,
            lowmid_dss_consistency="cos",
            lowmid_dss_agreement_threshold=0.8,
            project_each_step=False,
        )
        self.assertTrue(attacker.lowmid_grad_tuning)
        self.assertEqual(attacker.lowmid_grad_rotation_strength, 0.25)
        self.assertFalse(attacker.lowmid_grad_preserve_norm)
        self.assertTrue(attacker.lowmid_dss_filter)
        self.assertEqual(attacker.lowmid_dss_consistency, "cos")
        self.assertEqual(attacker.lowmid_dss_agreement_threshold, 0.8)
        self.assertFalse(attacker.project_each_step)


if __name__ == "__main__":
    unittest.main()
