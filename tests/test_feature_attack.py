import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

import main
from attack import LMDSSAttacker


class TinyTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 4)

    def forward(self, x, return_attn=False, return_tokens=False):
        del return_attn
        patches = F.avg_pool2d(x, kernel_size=4, stride=4).flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat((cls, patches), dim=1)
        block_tokens = [tokens, torch.tanh(tokens)]
        logits = self.head(block_tokens[-1][:, 0])
        return (logits, block_tokens) if return_tokens else logits


class TinyMapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 4)

    def forward(self, x, return_attn=False, return_tokens=False):
        del return_attn
        feature0 = x
        feature1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        logits = self.head(feature1.mean(dim=(2, 3)))
        return (logits, [feature0, feature1]) if return_tokens else logits

    @staticmethod
    def prepare_feature_tokens(features):
        return features.flatten(2).transpose(1, 2)


class TinyStageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 4)

    def forward(self, x, return_attn=False, return_tokens=False, return_stage_tokens=False):
        del return_attn
        patches = F.avg_pool2d(x, kernel_size=4, stride=4).flatten(2).transpose(1, 2)
        cls = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat((cls, patches), dim=1)
        block_tokens = [tokens, torch.tanh(tokens)]
        stage_tokens = [x, F.avg_pool2d(x, kernel_size=2, stride=2)]
        logits = self.head(block_tokens[-1][:, 0])
        outputs = [logits]
        if return_tokens:
            outputs.append(block_tokens)
        if return_stage_tokens:
            outputs.append(stage_tokens)
        return tuple(outputs) if len(outputs) > 1 else logits

    @staticmethod
    def prepare_feature_tokens(features):
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 3 and features.size(1) > 1:
            return features[:, 1:, :]
        return features


def make_attacker(**kwargs):
    return LMDSSAttacker(
        TinyTokenModel(),
        epsilon=0.1,
        steps=2,
        ti_sigma=0,
        device=torch.device("cpu"),
        **kwargs,
    )


class FeatureAttackTests(unittest.TestCase):
    def test_feature_attack_smoke_and_budget(self):
        torch.manual_seed(3)
        attacker = make_attacker(attack_loss="feature", feature_layer=1)
        images = torch.rand(2, 3, 16, 16) * 2.0 - 1.0
        labels = torch.tensor([1, 2])
        adversarial = attacker.attack_batch(images, labels)
        clean_pixels = attacker._denormalize(images)
        adversarial_pixels = attacker._denormalize(adversarial)
        self.assertEqual(adversarial.shape, images.shape)
        self.assertLessEqual((adversarial_pixels - clean_pixels).abs().max().item(), 0.1 + 1e-6)

    def test_feature_attack_supports_negative_layer(self):
        attacker = make_attacker(attack_loss="feature", feature_layer=-1)
        images = torch.rand(1, 3, 16, 16) * 2.0 - 1.0
        adversarial = attacker.attack_batch(images, torch.tensor([0]))
        self.assertEqual(adversarial.shape, images.shape)


    def test_feature_trajectory_dropout_generates_nine_feature_terms(self):
        torch.manual_seed(23)
        attacker = make_attacker(
            attack_loss="feature",
            feature_layer=1,
            guide_aug=True,
            guide_aug_methods=("feature_trajectory_dropout",),
            guide_aug_copies=9,
            guide_aug_strength=0.2,
        )
        pixels = torch.rand(2, 3, 16, 16, requires_grad=True)
        labels = torch.tensor([1, 2])
        with torch.no_grad():
            target = attacker._extract_layer_patch_features(pixels).detach()
        gradient, terms = attacker._attack_grad_terms(pixels, labels, target)
        self.assertEqual(len(terms), 9)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.flatten(1).norm(dim=1).min().detach()), 0.0)


    def test_dim_consensus_trajectory_generates_requested_feature_terms(self):
        torch.manual_seed(41)
        attacker = make_attacker(
            attack_loss="feature",
            feature_layer=1,
            guide_aug=True,
            guide_aug_methods=("dim_consensus_trajectory",),
            guide_aug_copies=5,
            guide_aug_strength=0.2,
            input_diversity=True,
        )
        pixels = torch.rand(2, 3, 16, 16, requires_grad=True)
        labels = torch.tensor([1, 2])
        with torch.no_grad():
            target = attacker._extract_layer_patch_features(pixels).detach()
        gradient, terms = attacker._attack_grad_terms(pixels, labels, target)
        self.assertEqual(len(terms), 5)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.flatten(1).norm(dim=1).min().detach()), 0.0)


    def test_dim_consensus_evidence_trajectory_generates_requested_feature_terms(self):
        torch.manual_seed(43)
        attacker = make_attacker(
            attack_loss="feature",
            feature_layer=1,
            guide_aug=True,
            guide_aug_methods=("dim_consensus_evidence_trajectory",),
            guide_aug_copies=5,
            guide_aug_strength=0.2,
            input_diversity=True,
        )
        pixels = torch.rand(2, 3, 16, 16, requires_grad=True)
        labels = torch.tensor([1, 2])
        with torch.no_grad():
            target = attacker._extract_layer_patch_features(pixels).detach()
        gradient, terms = attacker._attack_grad_terms(pixels, labels, target)
        self.assertEqual(len(terms), 5)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.flatten(1).norm(dim=1).min().detach()), 0.0)


    def test_spatial_sign_reinforcement_boosts_locally_consistent_update(self):
        attacker = make_attacker(
            spatial_sign_reinforcement=True,
            spatial_sign_reinforcement_sigma=0.5,
            spatial_sign_reinforcement_strength=0.5,
        )
        update = torch.ones(1, 1, 8, 8)
        tuned = attacker._apply_spatial_sign_reinforcement(update)
        self.assertEqual(tuned.shape, update.shape)
        self.assertTrue(torch.all(tuned > update))
        self.assertAlmostEqual(float(tuned.mean()), 1.5, places=4)

    def test_spatial_sign_reinforcement_is_weaker_on_conflicting_local_signs(self):
        attacker = make_attacker(
            spatial_sign_reinforcement=True,
            spatial_sign_reinforcement_sigma=1.0,
            spatial_sign_reinforcement_strength=0.5,
        )
        uniform = torch.ones(1, 1, 8, 8)
        checker = torch.ones(1, 1, 8, 8)
        checker[:, :, ::2, 1::2] = -1.0
        checker[:, :, 1::2, ::2] = -1.0
        tuned_uniform = attacker._apply_spatial_sign_reinforcement(uniform)
        tuned_checker = attacker._apply_spatial_sign_reinforcement(checker)
        uniform_gain = (tuned_uniform - uniform).abs().mean()
        checker_gain = (tuned_checker - checker).abs().mean()
        self.assertLess(float(checker_gain), float(uniform_gain))

    def test_spatial_sign_reinforcement_can_be_disabled(self):
        attacker = make_attacker(spatial_sign_reinforcement=False)
        update = torch.randn(1, 1, 4, 4)
        tuned = attacker._apply_spatial_sign_reinforcement(update)
        self.assertTrue(torch.equal(tuned, update))

    def test_spatial_sign_reinforcement_validates_parameters(self):
        with self.assertRaises(ValueError):
            make_attacker(spatial_sign_reinforcement_sigma=0)
        with self.assertRaises(ValueError):
            make_attacker(spatial_sign_reinforcement_strength=-0.1)


    def test_grad_momentum_agreement_reinforces_only_currently_supported_update_signs(self):
        attacker = make_attacker(
            grad_momentum_agreement=True,
            grad_momentum_agreement_strength=0.5,
        )
        update = torch.tensor([[[[2.0, -2.0, 2.0, -2.0, 0.0]]]])
        grad = torch.tensor([[[[1.0, -1.0, -1.0, 1.0, 1.0]]]])
        tuned = attacker._apply_grad_momentum_agreement(update, grad)
        expected = torch.tensor([[[[2.5, -2.5, 2.0, -2.0, 0.0]]]])
        self.assertTrue(torch.equal(tuned, expected))

    def test_grad_momentum_agreement_spatial_smoothing_weights_local_agreement(self):
        attacker = make_attacker(
            grad_momentum_agreement=True,
            grad_momentum_agreement_strength=0.5,
            grad_momentum_agreement_sigma=1.0,
        )
        update = torch.ones(1, 1, 8, 8)
        uniform_grad = torch.ones_like(update)
        isolated_grad = -torch.ones_like(update)
        isolated_grad[:, :, 4, 4] = 1.0

        tuned_uniform = attacker._apply_grad_momentum_agreement(update, uniform_grad)
        tuned_isolated = attacker._apply_grad_momentum_agreement(update, isolated_grad)

        self.assertAlmostEqual(float(tuned_uniform.mean()), 1.5, places=4)
        self.assertLess(float(tuned_isolated[:, :, 4, 4]), float(tuned_uniform[:, :, 4, 4]))
        self.assertGreater(float(tuned_isolated[:, :, 4, 5]), 1.0)

    def test_grad_momentum_agreement_suppresses_conflicting_update_signs(self):
        attacker = make_attacker(
            grad_momentum_agreement=True,
            grad_momentum_agreement_strength=0.5,
            grad_momentum_conflict_suppression_strength=0.25,
        )
        update = torch.tensor([[[[2.0, -2.0, 2.0, -2.0, 0.0]]]])
        grad = torch.tensor([[[[1.0, -1.0, -1.0, 1.0, 1.0]]]])
        tuned = attacker._apply_grad_momentum_agreement(update, grad)
        expected = torch.tensor([[[[2.5, -2.5, 1.75, -1.75, 0.0]]]])
        self.assertTrue(torch.equal(tuned, expected))

    def test_grad_momentum_conflict_suppression_can_run_without_reinforcement(self):
        attacker = make_attacker(
            grad_momentum_agreement=True,
            grad_momentum_agreement_strength=0.0,
            grad_momentum_conflict_suppression_strength=0.25,
        )
        update = torch.tensor([[[[2.0, -2.0, 2.0, -2.0, 0.0]]]])
        grad = torch.tensor([[[[1.0, -1.0, -1.0, 1.0, 1.0]]]])
        tuned = attacker._apply_grad_momentum_agreement(update, grad)
        expected = torch.tensor([[[[2.0, -2.0, 1.75, -1.75, 0.0]]]])
        self.assertTrue(torch.equal(tuned, expected))

    def test_grad_momentum_agreement_can_be_disabled(self):
        attacker = make_attacker(grad_momentum_agreement=False)
        update = torch.randn(1, 1, 2, 2)
        grad = torch.randn(1, 1, 2, 2)
        tuned = attacker._apply_grad_momentum_agreement(update, grad)
        self.assertTrue(torch.equal(tuned, update))

    def test_grad_momentum_agreement_validates_parameters(self):
        with self.assertRaises(ValueError):
            make_attacker(grad_momentum_agreement_strength=-0.1)
        with self.assertRaises(ValueError):
            make_attacker(grad_momentum_agreement_sigma=-0.1)
        with self.assertRaises(ValueError):
            make_attacker(grad_momentum_conflict_suppression_strength=-0.1)


    def test_feature_attack_supports_4d_feature_maps(self):
        attacker = LMDSSAttacker(
            TinyMapModel(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            attack_loss="feature",
            feature_layer=-2,
            device=torch.device("cpu"),
        )
        images = torch.rand(2, 3, 16, 16) * 2.0 - 1.0
        adversarial = attacker.attack_batch(images, torch.tensor([1, 2]))
        self.assertEqual(adversarial.shape, images.shape)

    def test_stage_feature_scope_uses_stage_outputs(self):
        attacker = LMDSSAttacker(
            TinyStageModel(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            attack_loss="feature",
            feature_layer=-1,
            feature_scope="stage",
            device=torch.device("cpu"),
        )
        images = torch.rand(2, 3, 16, 16) * 2.0 - 1.0
        adversarial = attacker.attack_batch(images, torch.tensor([1, 2]))
        self.assertEqual(adversarial.shape, images.shape)

    def test_feature_layer_range_is_validated_on_forward(self):
        attacker = make_attacker(attack_loss="feature", feature_layer=2)
        images = torch.rand(1, 3, 16, 16) * 2.0 - 1.0
        with self.assertRaisesRegex(ValueError, "out of range"):
            attacker.attack_batch(images, torch.tensor([0]))

    def test_stage_feature_layer_range_is_validated_on_forward(self):
        attacker = LMDSSAttacker(
            TinyStageModel(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            attack_loss="feature",
            feature_layer=2,
            feature_scope="stage",
            device=torch.device("cpu"),
        )
        images = torch.rand(1, 3, 16, 16) * 2.0 - 1.0
        with self.assertRaisesRegex(ValueError, "out of range"):
            attacker.attack_batch(images, torch.tensor([0]))

    def test_constructor_rejects_unknown_loss(self):
        with self.assertRaises(ValueError):
            make_attacker(attack_loss="unknown")

    def test_constructor_rejects_unknown_feature_scope(self):
        with self.assertRaises(ValueError):
            make_attacker(attack_loss="feature", feature_scope="unknown")

    def test_cli_exposes_feature_attack_options(self):
        argv = ["main.py", "--attack-loss", "feature", "--feature-layer", "-2", "--feature-scope", "stage"]
        with mock.patch.object(sys, "argv", argv):
            args = main.parse_args()
        self.assertEqual(args.attack_loss, "feature")
        self.assertEqual(args.feature_layer, -2)
        self.assertEqual(args.feature_scope, "stage")
        self.assertEqual(args.whitebox_model, main.DEFAULT_MODEL_NAME)

    def test_create_attacker_forwards_feature_options(self):
        attacker = main.create_attacker(
            model=TinyTokenModel(),
            epsilon=0.1,
            step_size=None,
            steps=2,
            ti_sigma=0,
            attack_loss="feature",
            feature_layer=1,
            feature_scope="stage",
        )
        self.assertEqual(attacker.attack_loss, "feature")
        self.assertEqual(attacker.feature_layer, 1)
        self.assertEqual(attacker.feature_scope, "stage")


if __name__ == "__main__":
    unittest.main()
