import math
import unittest

import torch

from experiments.patch_score_layer_semantic_maturity_experiment import (
    align_phase_scores,
    class_geometry,
    linear_cka,
    phase_shift,
    score_distribution_metrics,
)


class LayerSemanticMaturityTests(unittest.TestCase):
    def test_linear_cka_supports_different_feature_widths(self):
        torch.manual_seed(3)
        latent = torch.randn(16, 3)
        left = latent @ torch.randn(3, 5)
        right = latent @ torch.randn(3, 7)
        self.assertGreater(linear_cka(left, right), 0.7)
        self.assertAlmostEqual(linear_cka(left, left), 1.0, places=5)

    def test_class_geometry_reports_separation_and_split_knn(self):
        features = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]]
        )
        labels = torch.tensor([0, 0, 1, 1])
        metrics = class_geometry(features, labels)
        self.assertGreater(metrics["class_cosine_margin"], 1.5)
        self.assertEqual(metrics["split_1nn_evaluated"], 2)
        self.assertEqual(metrics["split_1nn_accuracy"], 1.0)

    def test_class_geometry_handles_unique_labels(self):
        metrics = class_geometry(torch.eye(3), torch.arange(3))
        self.assertTrue(math.isnan(metrics["within_class_cosine"]))
        self.assertTrue(math.isnan(metrics["split_1nn_accuracy"]))
        self.assertEqual(metrics["split_1nn_evaluated"], 0)

    def test_score_distribution_is_normalized(self):
        entropy, mass, std = score_distribution_metrics(
            torch.zeros(2, 20), temperature=0.1, top_ratio=0.15
        )
        self.assertTrue(torch.allclose(entropy, torch.ones_like(entropy)))
        self.assertTrue(torch.allclose(mass, torch.full_like(mass, 0.15)))
        self.assertTrue(torch.equal(std, torch.zeros_like(std)))

    def test_phase_alignment_retains_shapes(self):
        scores = torch.arange(16, dtype=torch.float32).reshape(1, 16)
        shifted_pixels = phase_shift(torch.zeros(1, 3, 32, 32), 4, 3)
        aligned = align_phase_scores(scores, (4, 4), (32, 32), 4, 3)
        self.assertEqual(tuple(shifted_pixels.shape), (1, 3, 32, 32))
        self.assertEqual(tuple(aligned.shape), (1, 16))


if __name__ == "__main__":
    unittest.main()
