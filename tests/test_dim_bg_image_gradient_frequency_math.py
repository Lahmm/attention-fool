import unittest

import torch

from gradient_analysis import FFT_BANDS
from dim_bg_image_gradient_frequency_math import (
    ablate_image_band,
    band_energy_ratios,
    cross_band_patch_correlation,
    factorial_interaction,
    fft_band_count,
    region_band_energy,
    report_rule,
    same_fft_band_edges,
)


class DimBgImageGradientFrequencyMathTests(unittest.TestCase):
    def test_image_and_gradient_frequency_use_same_fft_bands(self):
        self.assertEqual(same_fft_band_edges(), tuple(FFT_BANDS))
        self.assertEqual(fft_band_count(), 8)

    def test_image_and_gradient_energy_ratios_sum_to_one(self):
        x = torch.randn(3, 3, 32, 32, dtype=torch.float64)
        g = torch.randn(3, 3, 32, 32, dtype=torch.float64)
        self.assertTrue(torch.allclose(band_energy_ratios(x).sum(1), torch.ones(3, dtype=torch.float64), atol=1e-8))
        self.assertTrue(torch.allclose(band_energy_ratios(g).sum(1), torch.ones(3, dtype=torch.float64), atol=1e-8))

    def test_region_band_energy_shape(self):
        grad = torch.randn(4, 3, 16, 16)
        guide = torch.rand(4, 1, 16, 16)
        values = region_band_energy(grad, guide)
        self.assertEqual(tuple(values.shape), (4, 2))

    def test_factorial_interaction_formula(self):
        plain = torch.tensor([1.0, 2.0])
        dim = torch.tensor([3.0, 5.0])
        bg = torch.tensor([7.0, 11.0])
        dim_bg = torch.tensor([13.0, 17.0])
        self.assertTrue(torch.equal(factorial_interaction(plain, dim, bg, dim_bg), torch.tensor([4.0, 3.0])))

    def test_cross_band_correlation_shape(self):
        image = torch.rand(2, 8, 16)
        grad = torch.rand(2, 8, 16)
        corr = cross_band_patch_correlation(image, grad)
        self.assertEqual(tuple(corr.shape), (8, 8))

    def test_band_ablation_preserves_shape_and_clamps(self):
        x = torch.rand(2, 3, 32, 32)
        y = ablate_image_band(x, 2, eta=0.25)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 1.0)

    def test_report_rule(self):
        self.assertEqual(report_rule(True, True, True), "supported")
        self.assertEqual(report_rule(True, False, True), "association_only")
        self.assertEqual(report_rule(False, True, True), "inconclusive")


if __name__ == "__main__":
    unittest.main()
