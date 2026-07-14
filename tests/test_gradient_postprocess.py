import sys
import unittest

import torch

from gradient_observer import GradientObserver
from attack import GRADIENT_POSTPROCESS_MODES, PatchScoreAttacker
from main import parse_args


class DummyModel:
    def eval(self):
        return self


def make_attacker(**kwargs):
    return PatchScoreAttacker(model=DummyModel(), device=torch.device("cpu"), **kwargs)


class GradientPostprocessTests(unittest.TestCase):
    def test_mean_matches_original_stack_mean(self):
        gradients = torch.arange(20 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(
            20, 2, 3, 2, 2
        )
        expected = torch.stack([gradients[index] for index in range(20)]).mean(dim=0)

        actual = PatchScoreAttacker._aggregate_gradients(gradients, "mean")

        self.assertTrue(torch.equal(actual, expected))

    def test_view_l2_mean_gives_views_equal_weight(self):
        gradients = torch.tensor(
            [
                [[[[3.0]]]],
                [[[[4.0]]]],
            ]
        )

        actual = PatchScoreAttacker._aggregate_gradients(gradients, "view_l2_mean")

        self.assertTrue(torch.allclose(actual, torch.tensor([[[[1.0]]]])))

    def test_sign_consensus_uses_coordinate_majority(self):
        gradients = torch.tensor(
            [
                [[[[2.0, -1.0, 1.0]]]],
                [[[[1.0, -3.0, -2.0]]]],
                [[[[5.0, 4.0, -3.0]]]],
            ]
        )

        actual = PatchScoreAttacker._aggregate_gradients(gradients, "sign_consensus")

        self.assertTrue(torch.equal(actual, torch.tensor([[[[1.0, -1.0, -1.0]]]])))

    def test_transport_endpoints_match_mean_and_sign_consensus(self):
        gradients = torch.tensor(
            [
                [[[[10.0, -1.0]]]],
                [[[[1.0, 3.0]]]],
                [[[[1.0, 3.0]]]],
            ]
        )
        mean = PatchScoreAttacker._aggregate_gradients(gradients, "mean")
        sign = PatchScoreAttacker._aggregate_gradients(gradients, "sign_consensus")

        at_zero = PatchScoreAttacker._aggregate_gradients(
            gradients, "sign_consensus_transport", gradient_consensus_lambda=0.0
        )
        at_one = PatchScoreAttacker._aggregate_gradients(
            gradients, "sign_consensus_transport", gradient_consensus_lambda=1.0
        )

        self.assertTrue(torch.equal(at_zero, mean))
        self.assertTrue(torch.equal(at_one, sign))

    def test_attack_grad_keeps_all_twenty_view_gradients(self):
        attacker = make_attacker()
        pixels = torch.zeros(2, 3, 2, 2, requires_grad=True)
        losses = [(pixels * float(index + 1)).sum() for index in range(20)]
        attacker._iter_attack_losses = lambda _pixels, _labels: iter(losses)

        actual = attacker._attack_grad(pixels, torch.zeros(2, dtype=torch.long))

        self.assertEqual(attacker.input_diversity_groups * attacker.input_diversity_views_per_group, 20)
        self.assertTrue(torch.equal(actual, torch.full_like(pixels, 10.5)))

    def test_attack_batch_keeps_raw_gradient_scale_and_projects(self):
        attacker = make_attacker(steps=1, epsilon=0.1, step_size=0.2, use_momentum=False)
        attacker._compute_generic_patch_scores = lambda _pixels: None

        def raw_gradient(pixels, _labels, **_kwargs):
            attacker._actual_forward_view_count = 20
            return torch.full_like(pixels, 2.0)

        attacker._attack_grad = raw_gradient
        observer = GradientObserver()
        images = torch.zeros(1, 3, 2, 2)
        labels = torch.zeros(1, dtype=torch.long)

        adversarial = attacker.attack_batch(images, labels, observer=observer)

        self.assertAlmostEqual(observer._records[0]["per_sample"][0]["norm_abs_mean"], 2.0)
        clean_pixels = attacker._denormalize(images)
        adversarial_pixels = attacker._denormalize(adversarial)
        self.assertLessEqual(float((adversarial_pixels - clean_pixels).abs().max()), 0.1 + 1e-6)

    def test_invalid_modes_and_lambda_are_rejected(self):
        gradients = torch.ones(2, 1, 1, 1, 1)
        with self.assertRaises(ValueError):
            PatchScoreAttacker._aggregate_gradients(gradients, "invalid")
        with self.assertRaises(ValueError):
            PatchScoreAttacker._aggregate_gradients(gradients, "mean", -0.01)
        with self.assertRaises(ValueError):
            PatchScoreAttacker._aggregate_gradients(gradients, "mean", 1.01)
        with self.assertRaises(ValueError):
            make_attacker(input_diversity_groups=11, input_diversity_views_per_group=2)

    def test_cli_defaults_preserve_old_mainline(self):
        old_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.gradient_postprocess, "mean")
        self.assertEqual(args.gradient_consensus_lambda, 0.2)
        self.assertIsNone(args.seed)
        self.assertTrue(
            {"mean", "view_l2_mean", "sign_consensus", "sign_consensus_transport"}.issubset(
                GRADIENT_POSTPROCESS_MODES
            )
        )
