import unittest

import torch
import torch.nn as nn

from attack import LMDSSAttacker
from experiments.dim_gradient_mechanism import (
    VARIANTS,
    build_report,
    classify_band,
    compute_source_gradient_variants,
    fft_energy_ratio_sum,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3 * 16 * 16, 5)

    def forward(self, x, return_attn=False):
        return self.linear(x.flatten(1))


def _attacker():
    return LMDSSAttacker(
        TinyModel(),
        steps=1,
        ti_sigma=0,
        input_diversity=True,
        dim_resize_range=(0.55, 0.9),
        dim_mode="full-random",
        guide_aug=False,
        device=torch.device("cpu"),
    )


class DimGradientMechanismTests(unittest.TestCase):
    def test_dim_random_average_gradient_shape(self):
        torch.manual_seed(3)
        attacker = _attacker()
        pixels = torch.rand(2, 3, 16, 16)
        labels = torch.tensor([1, 2])
        gradients, samples = compute_source_gradient_variants(attacker, pixels, labels, dim_samples=3)
        self.assertEqual(set(gradients), set(VARIANTS))
        self.assertEqual(len(samples), 3)
        for grad in gradients.values():
            self.assertEqual(tuple(grad.shape), tuple(pixels.shape))

    def test_fixed_dim_reuses_transform(self):
        attacker = _attacker()
        attacker.dim_mode = "full-fixed"
        images = torch.randn(1, 3, 16, 16)
        torch.manual_seed(11)
        first = attacker._input_diversity(images)
        second = attacker._input_diversity(images)
        self.assertTrue(torch.equal(first, second))

    def test_forward_only_forward_transform_backward_identity(self):
        attacker = _attacker()
        attacker.dim_mode = "full-fixed"
        images = torch.randn(1, 3, 16, 16, requires_grad=True)
        torch.manual_seed(5)
        full = attacker._input_diversity(images)
        full.sum().backward()
        full_grad = images.grad.clone()
        images.grad.zero_()
        attacker.dim_mode = "forward-only"
        torch.manual_seed(5)
        forward = attacker._input_diversity(images)
        forward.sum().backward()
        self.assertTrue(torch.allclose(forward, full))
        self.assertTrue(torch.equal(images.grad, torch.ones_like(images)))
        self.assertFalse(torch.equal(full_grad, torch.ones_like(images)))

    def test_fft_band_energy_ratios_sum_to_one(self):
        x = torch.randn(4, 3, 32, 32, dtype=torch.float64)
        self.assertTrue(torch.allclose(fft_energy_ratio_sum(x), torch.ones(4, dtype=torch.float64), atol=1e-8))

    def test_report_rule_classifies_and_falls_back_to_inconclusive(self):
        models = [f"m{i}" for i in range(8)]

        def payload(seed, improved=True):
            metrics = {"1": {}}
            for band in range(8):
                metrics["1"][str(band)] = {
                    "source_energy_ratio": {variant: 0.1 for variant in VARIANTS},
                    "target_cosine": {variant: {model: 0.0 for model in models} for variant in VARIANTS},
                    "target_direction_derivative": {variant: {model: 0.0 for model in models} for variant in VARIANTS},
                    "transform_coherence": {"dim_random_average": 0.8},
                }
            metrics["1"]["0"]["source_energy_ratio"]["dim_random_average"] = 0.2 if improved else 0.1
            for model in models[:6 if improved else 2]:
                metrics["1"]["0"]["target_direction_derivative"]["dim_random_average"][model] = 1.0
            return {
                "protocol": "dim_gradient_mechanism_v1",
                "seed": seed,
                "trace_steps": [1],
                "target_models": models,
                "metrics": metrics,
            }

        report = build_report([payload(0), payload(1)])
        classes = classify_band(report, 1, 0)
        self.assertIn("enhanced", classes)
        self.assertIn("transfer_improved", classes)
        inconclusive = build_report([payload(0, improved=False), payload(1, improved=False)])
        self.assertEqual(classify_band(inconclusive, 1, 0), ["inconclusive"])


if __name__ == "__main__":
    unittest.main()
