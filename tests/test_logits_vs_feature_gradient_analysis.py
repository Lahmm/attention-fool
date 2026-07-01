import sys
import unittest
from pathlib import Path

import torch
import numpy as np

# experiments/ scripts use bare imports (e.g. "from causal_analysis import ...")
# and rely on the script directory being on sys.path.  Replicate that here.
_EXPERIMENTS_DIR = str(Path(__file__).resolve().parent.parent / "experiments")
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

from attack import LMDSSAttacker
from logits_vs_feature_gradient_analysis import (
    BAND_GROUPS,
    FFT_BAND_COUNT,
    band_direction_derivative,
    band_energy_ratios,
    compute_source_gradient,
    estimate_gradient_noise,
    grad_l2_norm,
    grad_sign_consistency,
    group_energy_ratio,
)


class TinyTokenModel(torch.nn.Module):
    """Minimal ViT-like model with two blocks, supporting return_tokens."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 4)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, 3, padding=1),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 3, 3, padding=1),
            ),
        ])
        self.attn_logits = []
        self.block_tokens = []

    def _as_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Convert feature map to token sequence: [B, 196, 3]"""
        b, c, h, w = x.shape
        patches = torch.nn.functional.avg_pool2d(x, kernel_size=4, stride=4)
        patches = patches.flatten(2).transpose(1, 2)  # [B, N, C]
        cls = patches.mean(dim=1, keepdim=True)
        return torch.cat((cls, patches), dim=1)  # [B, N+1, C]

    def forward(self, x, return_attn=False, return_tokens=False):
        del return_attn
        feat0 = self.blocks[0](x)
        tokens0 = self._as_tokens(feat0)
        feat1 = self.blocks[1](x + feat0)  # residual
        tokens1 = self._as_tokens(feat1)
        block_tokens = [tokens0, tokens1]

        # Global average pool of patch tokens, then classify
        pooled = tokens1[:, 1:, :].mean(dim=1)  # [B, C]
        logits = self.head(pooled)
        return (logits, block_tokens) if return_tokens else logits


def make_attacker(**kwargs):
    return LMDSSAttacker(
        TinyTokenModel(),
        epsilon=0.1,
        steps=2,
        ti_sigma=0,
        device=torch.device("cpu"),
        **kwargs,
    )


class LogitsVsFeatureGradientTests(unittest.TestCase):
    def test_compute_logits_gradient_shape(self):
        torch.manual_seed(3)
        attacker = make_attacker(attack_loss="logits")
        pixels = torch.rand(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        grad = compute_source_gradient(attacker, pixels, labels, None, attack_loss="logits")
        self.assertEqual(grad.shape, pixels.shape)
        self.assertFalse(torch.allclose(grad, torch.zeros_like(grad)))

    def test_compute_feature_gradient_shape(self):
        torch.manual_seed(3)
        attacker = make_attacker(attack_loss="feature", feature_layer=1)
        clean = torch.rand(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        # Pass perturbed pixels for gradient computation, clean pixels for feature target
        perturbed = clean + 0.01 * torch.randn_like(clean)
        grad = compute_source_gradient(attacker, perturbed, labels, None,
                                       attack_loss="feature", feature_layer=1,
                                       clean_pixels=clean)
        self.assertEqual(grad.shape, clean.shape)
        self.assertFalse(torch.allclose(grad, torch.zeros_like(grad)))

    def test_logits_and_feature_gradients_differ(self):
        """The core hypothesis: logits and feature gradients should differ structurally."""
        torch.manual_seed(3)
        attacker = make_attacker(attack_loss="feature", feature_layer=1)
        clean = torch.rand(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        perturbed = clean + 0.01 * torch.randn_like(clean)
        grad_logits = compute_source_gradient(attacker, perturbed, labels, None, attack_loss="logits")
        grad_feature = compute_source_gradient(attacker, perturbed, labels, None,
                                                attack_loss="feature", feature_layer=1,
                                                clean_pixels=clean)
        # They should not be identical
        self.assertFalse(torch.allclose(grad_logits, grad_feature))

    def test_band_energy_ratios_shape_and_sum(self):
        x = torch.randn(4, 3, 32, 32, dtype=torch.float64)
        ratios = band_energy_ratios(x)
        self.assertEqual(ratios.shape, (4, FFT_BAND_COUNT))
        self.assertTrue(torch.allclose(ratios.sum(1), torch.ones(4, dtype=torch.float64), atol=1e-8))

    def test_group_energy_ratio(self):
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        low = group_energy_ratio(x, BAND_GROUPS["low"])
        high = group_energy_ratio(x, BAND_GROUPS["high"])
        self.assertEqual(low.shape, (2,))
        self.assertTrue((low >= 0).all())
        self.assertTrue((high >= 0).all())

    def test_grad_l2_norm_shape(self):
        x = torch.randn(4, 3, 16, 16)
        norms = grad_l2_norm(x)
        self.assertEqual(norms.shape, (4,))
        self.assertTrue((norms > 0).all())

    def test_grad_sign_consistency(self):
        grads = [torch.randn(4, 3, 8, 8) for _ in range(3)]
        score = grad_sign_consistency(grads)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # Identical grads should have score = 1.0
        self.assertAlmostEqual(grad_sign_consistency([grads[0], grads[0]]), 1.0)

    def test_band_direction_derivative_shape(self):
        source = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        dd = band_direction_derivative(source, target)
        self.assertEqual(dd.shape, (2, FFT_BAND_COUNT))

    def test_feature_gradient_has_different_frequency_spectrum(self):
        """Feature attack gradient should have measurably different frequency distribution."""
        torch.manual_seed(7)
        attacker = make_attacker(attack_loss="feature", feature_layer=1)
        clean = torch.rand(2, 3, 32, 32)
        labels = torch.tensor([1, 2])
        perturbed = clean + 0.01 * torch.randn_like(clean)
        grad_logits = compute_source_gradient(attacker, perturbed, labels, None, attack_loss="logits")
        grad_feature = compute_source_gradient(attacker, perturbed, labels, None,
                                                attack_loss="feature", feature_layer=1,
                                                clean_pixels=clean)

        logits_ratios = band_energy_ratios(grad_logits)
        feature_ratios = band_energy_ratios(grad_feature)

        # The ratios should differ — this just checks the measurement works
        diff = (logits_ratios - feature_ratios).abs().max().item()
        self.assertGreater(diff, 1e-6,
                           "Expected logits and feature gradients to have different frequency spectra")

    def test_gradient_noise_estimation_runs(self):
        torch.manual_seed(3)
        attacker = make_attacker(attack_loss="logits")
        # Enable augmentations so gradients vary across seeds
        attacker.guide_aug = True
        attacker.guide_aug_methods = ("dropout",)
        attacker.guide_aug_copies = 1
        attacker.guide_aug_strength = 0.1
        pixels = torch.rand(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        noise = estimate_gradient_noise(
            attacker, pixels, labels,
            attack_loss="logits", feature_layer=10, num_samples=3,
        )
        self.assertIn("gradient_cv", noise)
        self.assertIn("sign_consistency_across_seeds", noise)
        self.assertGreaterEqual(noise["sign_consistency_across_seeds"], 0.0)
        self.assertLessEqual(noise["sign_consistency_across_seeds"], 1.0)


class ReportLogicTests(unittest.TestCase):
    def test_build_conclusion_includes_key_sections(self):
        from experiments.logits_vs_feature_gradient_analysis import build_conclusion_zh
        # Minimal report structure for testing
        report = {
            "conclusions": {
                "frequency_spectrum": {
                    "logits_low_mid_energy_ratio": 0.45,
                    "feature_low_mid_energy_ratio": 0.58,
                    "logits_high_energy_ratio": 0.22,
                    "feature_high_energy_ratio": 0.14,
                    "low_mid_shift": 0.13,
                    "interpretation": "feature attack concentrates MORE gradient energy in low/mid frequencies",
                },
                "transfer_direction": {
                    "avg_logits_direction_derivative": 0.0012,
                    "avg_feature_direction_derivative": 0.0034,
                    "direction_advantage": 0.0022,
                    "feature_wins_over_logits": 6,
                    "total_target_models": 8,
                },
                "gradient_noise": {
                    "logits_cv": 0.85,
                    "feature_cv": 0.62,
                    "logits_sign_consistency": 0.72,
                    "feature_sign_consistency": 0.83,
                },
                "gradient_norms": {},
                "cross_model_coherence": {
                    "logits": {"positive_model_fraction": 0.5, "model_signs": {}},
                    "feature": {"positive_model_fraction": 0.875, "model_signs": {}},
                },
                "per_model_advantage": {},
            },
            "trace_steps": [1, 10, 20, 40],
            "target_models": ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
        }
        text = build_conclusion_zh(report)
        self.assertIn("频率谱分布", text)
        self.assertIn("迁移方向对齐", text)
        self.assertIn("梯度噪声", text)
        self.assertIn("跨模型梯度符号一致性", text)
        self.assertIn("核心机制总结", text)
        self.assertIn("机制 A", text)
        self.assertIn("机制 B", text)
        self.assertIn("机制 C", text)


if __name__ == "__main__":
    unittest.main()
