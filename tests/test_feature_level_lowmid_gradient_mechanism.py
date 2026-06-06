import unittest

import numpy as np
import torch

from feature_level_lowmid_gradient_mechanism import (
    FEATURE_REGIONS,
    classify_report_rule,
    dim_cross_band_leakage,
    feature_region_masks,
    feature_stability_scores,
    patch_mask_to_pixel,
    region_band_energy_from_patch_masks,
)


class FeatureLevelLowmidGradientMechanismTests(unittest.TestCase):
    def test_feature_stability_shape(self):
        base = torch.randn(2, 197, 32)
        probes = [base + 0.01 * torch.randn_like(base), base + 0.02 * torch.randn_like(base)]
        scores = feature_stability_scores(base, probes)
        self.assertEqual(tuple(scores.shape), (2, 196))

    def test_feature_region_masks_shape(self):
        torch.manual_seed(3)
        tokens = torch.randn(2, 197, 32)
        attn = [torch.randn(2, 4, 197, 197)]
        pixels = torch.rand(2, 3, 224, 224)
        guide = torch.rand(2, 1, 224, 224)
        masks, scores = feature_region_masks(tokens, attn, pixels, guide, patch_size=16)
        self.assertEqual(tuple(masks.shape), (2, len(FEATURE_REGIONS), 196))
        self.assertEqual(tuple(scores.shape), (2, len(FEATURE_REGIONS), 196))

    def test_patch_mask_to_pixel_shape(self):
        mask = torch.ones(2, 196)
        pixel = patch_mask_to_pixel(mask, 224, 224)
        self.assertEqual(tuple(pixel.shape), (2, 1, 224, 224))

    def test_region_band_energy_shape(self):
        grad = torch.randn(2, 3, 32, 32)
        masks = torch.ones(2, len(FEATURE_REGIONS), 16)
        values = region_band_energy_from_patch_masks(grad, masks, patch_size=8)
        self.assertEqual(tuple(values.shape), (2, len(FEATURE_REGIONS), 8))

    def test_dim_cross_band_leakage_shape(self):
        result = dim_cross_band_leakage(img_size=32, samples=2, modes=("resize_pad",), device=torch.device("cpu"))
        self.assertEqual(set(result), {"resize_pad"})
        self.assertEqual(result["resize_pad"].shape, (8, 8))
        self.assertTrue(np.isfinite(result["resize_pad"]).all())

    def test_report_rule(self):
        self.assertEqual(classify_report_rule(0.1, 0.2, 6), "supported")
        self.assertEqual(classify_report_rule(0.1, -0.2, 2), "spectrum_only")
        self.assertEqual(classify_report_rule(-0.1, 0.2, 2), "association_only")
        self.assertEqual(classify_report_rule(-0.1, -0.2, 0), "inconclusive")


if __name__ == "__main__":
    unittest.main()
