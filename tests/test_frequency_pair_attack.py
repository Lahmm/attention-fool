import unittest

import torch

from attack import PatchScoreAttacker
from experiments.frequency_pair_attack import FrequencyPairAttacker


class DummyModel:
    model_mean = (0.5, 0.5, 0.5)
    model_std = (0.5, 0.5, 0.5)

    def eval(self):
        return self


def make_attacker(**kwargs):
    return FrequencyPairAttacker(
        DummyModel(),
        steps=1,
        input_diversity_groups=1,
        input_diversity_views_per_group=2,
        gaussian_alpha=0.0,
        device=torch.device("cpu"),
        **kwargs,
    )


class FrequencyPairAttackTests(unittest.TestCase):
    def test_production_default_is_unchanged(self):
        production = PatchScoreAttacker(DummyModel(), device=torch.device("cpu"))
        self.assertEqual(production.attack_method, "original_score_postdrop_phase_pair")
        self.assertFalse(hasattr(production, "frequency_sigma"))

    def test_fourier_components_are_complementary(self):
        attacker = make_attacker(frequency_sigma=2.0)
        pixels = torch.rand(2, 3, 16, 16)
        low, high = attacker._frequency_components(pixels)
        self.assertTrue(torch.allclose(low + high, pixels, atol=1e-6, rtol=1e-6))

        constant = torch.full((1, 3, 16, 16), 0.4)
        _, constant_high = attacker._frequency_components(constant)
        self.assertLess(float(constant_high.abs().max()), 1e-6)

    def test_frequency_views_are_valid_and_distinct(self):
        attacker = make_attacker()
        pixels = torch.zeros(1, 3, 16, 16)
        pixels[:, :, 4:12, 4:12] = 1.0
        low = attacker._frequency_view(pixels, attacker.frequency_low_residual_scale)
        high = attacker._frequency_view(pixels, attacker.frequency_high_residual_scale)
        self.assertGreaterEqual(float(low.min()), 0.0)
        self.assertLessEqual(float(low.max()), 1.0)
        self.assertGreaterEqual(float(high.min()), 0.0)
        self.assertLessEqual(float(high.max()), 1.0)
        self.assertFalse(torch.equal(low, high))

    def test_pair_reuses_one_mask_and_emits_two_views(self):
        attacker = make_attacker()
        pixels = torch.rand(1, 3, 8, 8, requires_grad=True)
        drop_mask = torch.tensor([[True, False, False, False]])
        attacker._compute_mainline_drop_mask = lambda _pixels: (drop_mask, (2, 2))

        views = list(attacker._iter_original_score_postdrop_phase_pair(pixels))

        self.assertEqual(len(views), 2)
        self.assertEqual(attacker._actual_forward_view_count, 2)
        self.assertTrue(torch.equal(views[0][1], views[1][1]))
        self.assertTrue(torch.equal(views[0][0][:, :, :4, :4], torch.zeros(1, 3, 4, 4)))
        self.assertTrue(torch.equal(views[1][0][:, :, :4, :4], torch.zeros(1, 3, 4, 4)))
        self.assertFalse(torch.equal(views[0][0], views[1][0]))

    def test_invalid_frequency_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            make_attacker(frequency_sigma=0.0)
        with self.assertRaises(ValueError):
            make_attacker(frequency_low_residual_scale=1.1)
        with self.assertRaises(ValueError):
            make_attacker(frequency_high_residual_scale=0.9)


if __name__ == "__main__":
    unittest.main()
