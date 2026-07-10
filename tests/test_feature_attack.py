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


class TinyPatchEmbedTokenModel(TinyTokenModel):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, 2, kernel_size=4, stride=4, bias=False)
        with torch.no_grad():
            self.patch_embed.proj.weight.zero_()
            self.patch_embed.proj.weight[0, 0].fill_(1.0)
            self.patch_embed.proj.weight[1, 1].fill_(1.0)


class TinyPatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 3, kernel_size=4, stride=4, bias=False)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyTokenMixBlock(nn.Module):
    calls = 0

    def forward(self, x):
        type(self).calls += 1
        cls = x[:, :1] + x[:, 1:].mean(dim=1, keepdim=True)
        return torch.cat((cls, x[:, 1:]), dim=1)


class TinyTimmViTBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 3))
        self.pos_embed = nn.Parameter(torch.zeros(1, 10, 3))
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList([TinyTokenMixBlock(), TinyTokenMixBlock(), TinyTokenMixBlock()])
        self.norm = nn.Identity()
        self.head = nn.Linear(3, 4)

    def _pos_embed(self, x):
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat((cls, x), dim=1) + self.pos_embed[:, : x.size(1) + 1]

    def forward_head(self, x, pre_logits=False):
        out = x[:, 0]
        return out if pre_logits else self.head(out)


class TinyTimmViTWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyTimmViTBackbone()

    def forward(self, x, return_attn=False, return_tokens=False):
        del return_attn
        TinyTokenMixBlock.calls = 0
        tokens = self.model.patch_embed(x)
        tokens = self.model._pos_embed(tokens)
        tokens = self.model.patch_drop(tokens)
        tokens = self.model.norm_pre(tokens)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        logits = self.model.forward_head(tokens)
        return (logits, [tokens]) if return_tokens else logits


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

    def test_patch_dropout_score_mode_selects_high_or_low_score_patches(self):
        pixels = torch.ones(1, 3, 12, 12)
        scores = torch.arange(9, dtype=torch.float32).view(1, 9)

        high_attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.0,
            patch_dropout_ratio=1.0,
            patch_dropout_score_mode="high",
        )
        high_attacker._patch_scores = scores
        high_pixels = high_attacker._patch_dropout_pixels(pixels)

        low_attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.0,
            patch_dropout_ratio=1.0,
            patch_dropout_score_mode="low",
        )
        low_attacker._patch_scores = scores
        low_pixels = low_attacker._patch_dropout_pixels(pixels)

        high_zero_mask = high_pixels[:, :1].eq(0).float()
        low_zero_mask = low_pixels[:, :1].eq(0).float()
        expected_high = torch.tensor(
            [[[[0, 0, 0], [0, 0, 1], [1, 1, 1]]]], dtype=torch.float32
        )
        expected_low = torch.tensor(
            [[[[1, 1, 1], [1, 0, 0], [0, 0, 0]]]], dtype=torch.float32
        )
        expected_high = F.interpolate(expected_high, size=(12, 12), mode="nearest")
        expected_low = F.interpolate(expected_low, size=(12, 12), mode="nearest")

        torch.testing.assert_close(high_zero_mask, expected_high)
        torch.testing.assert_close(low_zero_mask, expected_low)

    def test_patch_dropout_context_high_score_blend_fills_selected_patches(self):
        pixels = torch.zeros(1, 3, 12, 12)
        scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        for index in range(9):
            row, col = divmod(index, 3)
            value = 0.2 if index < 4 else 0.8
            pixels[:, :, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = value

        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.0,
            patch_dropout_ratio=1.0,
            patch_dropout_score_mode="low",
            patch_dropout_fill_mode="context_high_score_blend",
        )
        attacker._patch_scores = scores
        filled = attacker._patch_dropout_pixels(pixels)

        low_region = filled[:, :, :4, :4]
        high_region = filled[:, :, 8:12, 8:12]
        self.assertGreater(float(low_region.mean()), 0.2)
        self.assertLess(float(low_region.mean()), 0.8)
        torch.testing.assert_close(high_region, pixels[:, :, 8:12, 8:12])

    def test_patch_dropout_random_high_score_inpaint_uses_random_high_score_donor(self):
        pixels = torch.zeros(1, 3, 12, 12)
        scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        for index in range(9):
            row, col = divmod(index, 3)
            value = 0.2 if index < 4 else 0.8
            pixels[:, :, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = value

        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.0,
            patch_dropout_ratio=1.0,
            patch_dropout_score_mode="low",
            patch_dropout_fill_mode="random_high_score_inpaint",
        )
        attacker._patch_scores = scores
        filled = attacker._patch_dropout_pixels(pixels)

        low_region = filled[:, :, :4, :4]
        high_region = filled[:, :, 8:12, 8:12]
        torch.testing.assert_close(low_region, torch.full_like(low_region, 0.8))
        torch.testing.assert_close(high_region, pixels[:, :, 8:12, 8:12])

    def test_patch_dropout_nearest_high_score_inpaint_uses_nearest_donor(self):
        pixels = torch.zeros(1, 3, 12, 12)
        scores = torch.tensor([[0.0, 1.0, 1.0, 1.0, 9.0, 2.0, 2.0, 2.0, 8.0]])
        for index in range(9):
            row, col = divmod(index, 3)
            pixels[:, :, row * 4:(row + 1) * 4, col * 4:(col + 1) * 4] = float(index) / 10.0

        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.0,
            patch_dropout_ratio=1.0,
            patch_dropout_score_mode="low",
            patch_dropout_fill_mode="nearest_high_score_inpaint",
        )
        attacker._patch_scores = scores
        filled = attacker._patch_dropout_pixels(pixels)

        # Patch 0 is nearer to high-score patch 4 than high-score patch 8.
        torch.testing.assert_close(filled[:, :, :4, :4], pixels[:, :, 4:8, 4:8])
        torch.testing.assert_close(filled[:, :, 8:12, 8:12], pixels[:, :, 8:12, 8:12])

    def test_patch_dropout_score_mode_validates_parameters(self):
        with self.assertRaises(ValueError):
            make_attacker(
                guide_aug=True,
                guide_aug_methods=("patch_dropout",),
                patch_dropout_score_mode="middle",
            )

    def test_patch_dropout_fill_mode_validates_parameters(self):
        with self.assertRaises(ValueError):
            make_attacker(
                guide_aug=True,
                guide_aug_methods=("patch_dropout",),
                patch_dropout_fill_mode="unknown",
            )

    def test_patch_dropout_noise_mode_validates_parameters(self):
        with self.assertRaises(ValueError):
            make_attacker(
                guide_aug=True,
                guide_aug_methods=("patch_dropout",),
                patch_dropout_noise_mode="unknown",
            )

    def test_patch_dropout_patch_embed_rowspace_noise_is_patch_structured(self):
        torch.manual_seed(17)
        attacker = LMDSSAttacker(
            TinyPatchEmbedTokenModel(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.05,
            patch_dropout_ratio=0.0,
            patch_dropout_noise_mode="patch_embed_rowspace",
        )
        attacker._patch_scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        pixels = torch.full((1, 3, 12, 12), 0.5)

        noised = attacker._patch_dropout_pixels(pixels)
        delta = noised - pixels

        self.assertGreater(float(delta.abs().max()), 0.0)
        first_patch = delta[:, :, :4, :4]
        torch.testing.assert_close(first_patch, first_patch[:, :, :1, :1].expand_as(first_patch))

    def test_patch_dropout_antithetic_gaussian_pairs_noise(self):
        torch.manual_seed(19)
        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.01,
            patch_dropout_ratio=0.0,
            patch_dropout_noise_mode="antithetic_gaussian",
        )
        attacker._patch_scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        pixels = torch.full((1, 3, 12, 12), 0.5)

        first = attacker._patch_dropout_pixels(pixels)
        second = attacker._patch_dropout_pixels(pixels)

        torch.testing.assert_close(first + second, pixels * 2.0, atol=1e-6, rtol=1e-6)

    def test_patch_dropout_rademacher_cubature_has_fixed_pixel_radius(self):
        torch.manual_seed(29)
        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.0,
            patch_dropout_noise_mode="rademacher_cubature",
        )
        attacker._patch_scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        pixels = torch.full((1, 3, 12, 12), 0.5)

        noised = attacker._patch_dropout_pixels(pixels)
        magnitudes = (noised - pixels).abs()

        torch.testing.assert_close(magnitudes, torch.full_like(magnitudes, 0.1))

    def test_patch_dropout_patch_cov_gaussian_keeps_pixel_and_patch_variation(self):
        torch.manual_seed(31)
        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.01,
            patch_dropout_ratio=0.0,
            patch_dropout_noise_mode="patch_cov_gaussian",
        )
        attacker._patch_scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        pixels = torch.full((1, 3, 12, 12), 0.5)

        noised = attacker._patch_dropout_pixels(pixels)
        delta = noised - pixels
        first_patch = delta[:, :, :4, :4]
        patch_means = delta.view(1, 3, 3, 4, 3, 4).mean(dim=(3, 5))

        self.assertEqual(noised.shape, pixels.shape)
        self.assertGreater(float(delta.abs().max()), 0.0)
        self.assertGreater(float(first_patch.std()), 0.0)
        self.assertGreater(float(patch_means.std()), 0.0)

    def test_patch_dropout_score_weighted_gaussian_scales_high_score_patches(self):
        torch.manual_seed(37)
        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.01,
            patch_dropout_ratio=0.0,
            patch_dropout_noise_mode="score_weighted_gaussian",
        )
        attacker._patch_scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        pixels = torch.full((1, 3, 12, 12), 0.5)

        total_low = 0.0
        total_high = 0.0
        for _ in range(256):
            delta = attacker._patch_dropout_pixels(pixels) - pixels
            patch_energy = delta.square().view(1, 3, 3, 4, 3, 4).mean(dim=(1, 3, 5))
            total_low += float(patch_energy[0, 0, 0])
            total_high += float(patch_energy[0, 2, 2])

        self.assertGreater(total_high, total_low)

    def test_patch_dropout_inverse_score_weighted_gaussian_scales_low_score_patches(self):
        torch.manual_seed(39)
        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.01,
            patch_dropout_ratio=0.0,
            patch_dropout_noise_mode="inverse_score_weighted_gaussian",
        )
        attacker._patch_scores = torch.arange(9, dtype=torch.float32).view(1, 9)
        pixels = torch.full((1, 3, 12, 12), 0.5)

        total_low = 0.0
        total_high = 0.0
        for _ in range(256):
            delta = attacker._patch_dropout_pixels(pixels) - pixels
            patch_energy = delta.square().view(1, 3, 3, 4, 3, 4).mean(dim=(1, 3, 5))
            total_low += float(patch_energy[0, 0, 0])
            total_high += float(patch_energy[0, 2, 2])

        self.assertGreater(total_low, total_high)

    def test_patch_dropout_opponent_channel_gaussian_has_negative_channel_covariance(self):
        torch.manual_seed(43)
        attacker = make_attacker(
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            guide_aug_strength=0.01,
            patch_dropout_ratio=0.0,
            patch_dropout_noise_mode="opponent_channel_gaussian",
        )
        pixels = torch.full((4096, 3, 1, 1), 0.5)
        noise = attacker._patch_dropout_noise(pixels, grid_size=1) / attacker.guide_aug_strength
        samples = noise.flatten(2).squeeze(-1)
        covariance = samples.transpose(0, 1).matmul(samples) / samples.size(0)

        torch.testing.assert_close(covariance.diag(), torch.ones(3), atol=0.08, rtol=0.08)
        self.assertLess(float(covariance[0, 1]), -0.15)
        self.assertLess(float(covariance[0, 2]), -0.15)
        self.assertLess(float(covariance[1, 2]), -0.15)

    def test_token_patch_dropout_loss_backpropagates_to_pixels(self):
        torch.manual_seed(47)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            guide_aug_copies=2,
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])

        loss = attacker._attack_loss_for_token_patch_dropout(pixels, labels)
        grad = torch.autograd.grad(loss, pixels)[0]

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_token_patch_dropout_opponent_noise_projects_through_patch_embed(self):
        torch.manual_seed(53)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            guide_aug_strength=0.1,
            patch_dropout_noise_mode="opponent_channel_gaussian",
        )
        patch_tokens = torch.ones(2, 9, 3)

        noise = attacker._token_patch_dropout_noise(patch_tokens, attacker.model.model)

        self.assertEqual(noise.shape, patch_tokens.shape)
        self.assertTrue(torch.isfinite(noise).all())
        self.assertGreater(float(noise.abs().sum()), 0.0)
        torch.testing.assert_close(
            noise.square().mean(dim=(1, 2)).sqrt(),
            torch.full((2,), 0.1),
            atol=0.03,
            rtol=0.3,
        )

    def test_token_patch_dropout_layer_runs_prefix_and_suffix_blocks(self):
        torch.manual_seed(59)
        TinyTokenMixBlock.calls = 0
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
            token_patch_dropout_layer=2,
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])

        loss = attacker._attack_loss_for_token_patch_dropout(pixels, labels)
        grad = torch.autograd.grad(loss, pixels)[0]

        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(TinyTokenMixBlock.calls, 3)

    def test_token_patch_dropout_context_fill_preserves_undropped_noise(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            patch_dropout_fill_mode="context_high_score_blend",
        )
        patch_tokens = torch.tensor(
            [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [9.0, 9.0]]]
        )
        noisy_patches = patch_tokens + 10.0
        scores = torch.tensor([[0.0, 0.0, 0.0, 3.0]])
        drop_mask = torch.tensor([[True, False, False, False]])

        filled = attacker._token_patch_dropout_fill(patch_tokens, noisy_patches, scores, drop_mask)

        torch.testing.assert_close(filled[0, 0], torch.tensor([5.0, 5.0]))
        torch.testing.assert_close(filled[0, 1:], noisy_patches[0, 1:])

    def test_token_patch_dropout_cls_noise_backpropagates_to_pixels(self):
        torch.manual_seed(71)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
            token_cls_noise=True,
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])

        loss = attacker._attack_loss_for_token_patch_dropout(pixels, labels)
        grad = torch.autograd.grad(loss, pixels)[0]

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_token_patch_dropout_score_cls_noise_backpropagates_to_pixels(self):
        torch.manual_seed(79)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
            token_cls_noise=True,
            token_score_cls_noise=True,
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])

        loss = attacker._attack_loss_for_token_patch_dropout(pixels, labels)
        grad = torch.autograd.grad(loss, pixels)[0]

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_token_patch_dropout_score_only_cls_noise_backpropagates_to_pixels(self):
        torch.manual_seed(83)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
            token_score_cls_noise=True,
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])

        loss = attacker._attack_loss_for_token_patch_dropout(pixels, labels)
        grad = torch.autograd.grad(loss, pixels)[0]

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_token_patch_dropout_score_patch_noise_backpropagates_to_pixels(self):
        torch.manual_seed(97)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
            patch_dropout_noise_mode="opponent_channel_gaussian",
            token_score_cls_noise=True,
            token_score_patch_noise=True,
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])

        loss = attacker._attack_loss_for_token_patch_dropout(pixels, labels)
        grad = torch.autograd.grad(loss, pixels)[0]

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_patch_dropout_cls_jitter_loss_backpropagates_to_pixels(self):
        torch.manual_seed(73)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("patch_dropout_cls_jitter",),
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])
        attacker._compute_patch_scores(pixels.detach())
        forward_pixels = attacker._patch_dropout_pixels(pixels)

        loss = attacker._attack_loss_for_patch_dropout_cls_jitter(forward_pixels, labels)
        grad = torch.autograd.grad(loss, pixels)[0]

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_patch_token_dropout_mix_generates_requested_terms(self):
        torch.manual_seed(89)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            guide_aug=True,
            guide_aug_methods=("patch_token_dropout_mix",),
            guide_aug_copies=4,
            guide_aug_strength=0.1,
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
            patch_dropout_noise_mode="opponent_channel_gaussian",
            token_score_cls_noise=True,
        )
        pixels = torch.rand(2, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([1, 2])
        attacker._compute_patch_scores(pixels.detach())

        losses = tuple(attacker._iter_attack_losses(pixels, labels))
        grad = torch.autograd.grad(sum(losses), pixels)[0]

        self.assertEqual(len(losses), 4)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_patch_dropout_extreme_sampling_selects_lowest_low_scores(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            patch_dropout_ratio=0.5,
            patch_dropout_score_mode="low",
            patch_dropout_sampling_mode="extreme",
        )
        scores = torch.tensor([[0.10, 0.20, 0.30, 0.80, 0.90]])
        candidate_mask = scores < scores.median(dim=1, keepdim=True).values

        drop_mask = attacker._sample_patch_dropout_mask(scores, candidate_mask)

        self.assertEqual(drop_mask.tolist(), [[True, False, False, False, False]])

    def test_patch_dropout_score_weighted_sampling_keeps_candidate_subset(self):
        torch.manual_seed(61)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            patch_dropout_ratio=1.0,
            patch_dropout_score_mode="low",
            patch_dropout_sampling_mode="score_weighted",
        )
        scores = torch.tensor([[0.10, 0.20, 0.30, 0.80, 0.90]])
        candidate_mask = scores < scores.median(dim=1, keepdim=True).values

        drop_mask = attacker._sample_patch_dropout_mask(scores, candidate_mask)

        self.assertTrue(torch.equal(drop_mask, candidate_mask))

    def test_patch_score_candidate_mask_defaults_to_median_split(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            patch_dropout_score_mode="low",
        )
        scores = torch.tensor([[0.10, 0.20, 0.30, 0.80, 0.90]])

        candidate_mask = attacker._patch_score_candidate_mask(scores)

        self.assertEqual(candidate_mask.tolist(), [[True, True, False, False, False]])

    def test_patch_score_quantile_jitter_validates_parameters(self):
        with self.assertRaises(ValueError):
            LMDSSAttacker(
                TinyTimmViTWrapper(),
                epsilon=0.1,
                steps=2,
                ti_sigma=0,
                device=torch.device("cpu"),
                patch_dropout_score_quantile_jitter=0.5,
            )

    def test_patch_score_noise_validates_parameters(self):
        with self.assertRaises(ValueError):
            LMDSSAttacker(
                TinyTimmViTWrapper(),
                epsilon=0.1,
                steps=2,
                ti_sigma=0,
                device=torch.device("cpu"),
                patch_dropout_score_noise=-0.1,
            )

    def test_patch_dropout_bernoulli_sampling_keeps_candidate_subset(self):
        torch.manual_seed(67)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(),
            epsilon=0.1,
            steps=2,
            ti_sigma=0,
            device=torch.device("cpu"),
            patch_dropout_ratio=0.0,
            patch_dropout_score_mode="low",
            patch_dropout_sampling_mode="bernoulli",
        )
        scores = torch.tensor([[0.10, 0.20, 0.30, 0.80, 0.90]])
        candidate_mask = scores < scores.median(dim=1, keepdim=True).values
        attacker.patch_dropout_ratio = 1e-12

        drop_mask = attacker._sample_patch_dropout_mask(scores, candidate_mask)

        self.assertEqual(int(drop_mask.sum()), 1)
        self.assertFalse(bool((drop_mask & ~candidate_mask).any()))


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


    def test_view_consistent_agreement_reinforces_by_view_support(self):
        attacker = make_attacker(
            view_consistent_agreement=True,
            view_consistent_agreement_strength=0.6,
        )
        update = torch.tensor([[[[1.0, 1.0, 1.0, -1.0]]]])
        term_grads = (
            torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]]),
            torch.tensor([[[[1.0, -1.0, -1.0, -1.0]]]]),
            torch.tensor([[[[-1.0, -1.0, -1.0, 1.0]]]]),
        )
        tuned = attacker._apply_view_consistent_agreement(update, term_grads)
        expected = torch.tensor([[[[1.4, 1.2, 1.0, -1.4]]]])
        self.assertTrue(torch.allclose(tuned, expected))

    def test_view_consistent_agreement_threshold_filters_weak_support(self):
        attacker = make_attacker(
            view_consistent_agreement=True,
            view_consistent_agreement_strength=0.6,
            view_consistent_agreement_threshold=0.5,
        )
        update = torch.tensor([[[[1.0, 1.0, 1.0, -1.0]]]])
        term_grads = (
            torch.tensor([[[[1.0, 1.0, -1.0, -1.0]]]]),
            torch.tensor([[[[1.0, -1.0, -1.0, -1.0]]]]),
            torch.tensor([[[[-1.0, -1.0, -1.0, 1.0]]]]),
        )
        tuned = attacker._apply_view_consistent_agreement(update, term_grads)
        expected = torch.tensor([[[[1.4, 1.0, 1.0, -1.4]]]])
        self.assertTrue(torch.allclose(tuned, expected))

    def test_view_consistent_agreement_can_be_disabled_or_missing_terms(self):
        update = torch.randn(1, 1, 2, 2)
        disabled = make_attacker(view_consistent_agreement=False)
        enabled = make_attacker(view_consistent_agreement=True)
        self.assertTrue(torch.equal(disabled._apply_view_consistent_agreement(update, (update,)), update))
        self.assertTrue(torch.equal(enabled._apply_view_consistent_agreement(update, None), update))

    def test_view_consistent_agreement_validates_parameters(self):
        with self.assertRaises(ValueError):
            make_attacker(view_consistent_agreement_strength=-0.1)
        with self.assertRaises(ValueError):
            make_attacker(view_consistent_agreement_threshold=-0.1)
        with self.assertRaises(ValueError):
            make_attacker(view_consistent_agreement_threshold=1.1)


    def test_cross_step_sign_vote_uses_recent_majority_signs(self):
        attacker = make_attacker(
            cross_step_sign_vote=True,
            cross_step_sign_vote_window=3,
            cross_step_sign_vote_strength=0.6,
        )
        history = []
        positive = torch.ones(1, 1, 1, 1)
        negative = -torch.ones(1, 1, 1, 1)

        first = attacker._apply_cross_step_sign_vote(positive, history)
        second = attacker._apply_cross_step_sign_vote(negative, history)
        third = attacker._apply_cross_step_sign_vote(positive, history)
        fourth = attacker._apply_cross_step_sign_vote(negative, history)

        self.assertTrue(torch.allclose(first, torch.tensor([[[[1.6]]]])))
        self.assertTrue(torch.allclose(second, torch.tensor([[[[-1.0]]]])))
        self.assertTrue(torch.allclose(third, torch.tensor([[[[1.2]]]])))
        self.assertTrue(torch.allclose(fourth, torch.tensor([[[[-1.2]]]])))
        self.assertEqual(len(history), 3)

    def test_cross_step_sign_vote_can_be_disabled(self):
        attacker = make_attacker(cross_step_sign_vote=False)
        history = []
        update = torch.randn(1, 1, 2, 2)
        tuned = attacker._apply_cross_step_sign_vote(update, history)
        self.assertTrue(torch.equal(tuned, update))
        self.assertEqual(history, [])

    def test_cross_step_sign_vote_validates_parameters(self):
        with self.assertRaises(ValueError):
            make_attacker(cross_step_sign_vote_window=0)
        with self.assertRaises(ValueError):
            make_attacker(cross_step_sign_vote_strength=-0.1)


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

    def test_cli_exposes_patch_dropout_score_mode(self):
        argv = [
            "main.py",
            "--patch-dropout-score-mode",
            "low",
            "--patch-dropout-sampling-mode",
            "extreme",
            "--patch-dropout-score-quantile-jitter",
            "0.1",
            "--patch-dropout-score-noise",
            "0.2",
            "--patch-dropout-fill-mode",
            "context_high_score_blend",
            "--patch-dropout-noise-mode",
            "inverse_score_weighted_gaussian",
            "--token-patch-dropout-layer",
            "2",
            "--token-cls-noise",
            "--token-score-cls-noise",
            "--token-score-patch-noise",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = main.parse_args()
        self.assertEqual(args.patch_dropout_score_mode, "low")
        self.assertEqual(args.patch_dropout_sampling_mode, "extreme")
        self.assertEqual(args.patch_dropout_score_quantile_jitter, 0.1)
        self.assertEqual(args.patch_dropout_score_noise, 0.2)
        self.assertEqual(args.patch_dropout_fill_mode, "context_high_score_blend")
        self.assertEqual(args.patch_dropout_noise_mode, "inverse_score_weighted_gaussian")
        self.assertEqual(args.token_patch_dropout_layer, 2)
        self.assertTrue(args.token_cls_noise)
        self.assertTrue(args.token_score_cls_noise)
        self.assertTrue(args.token_score_patch_noise)

    def test_cli_exposes_token_patch_dropout_method(self):
        argv = ["main.py", "--guide-aug-method", "token_patch_dropout"]
        with mock.patch.object(sys, "argv", argv):
            args = main.parse_args()
        self.assertEqual(args.guide_aug_method, ("token_patch_dropout",))

    def test_cli_exposes_patch_dropout_cls_jitter_method(self):
        argv = ["main.py", "--guide-aug-method", "patch_dropout_cls_jitter"]
        with mock.patch.object(sys, "argv", argv):
            args = main.parse_args()
        self.assertEqual(args.guide_aug_method, ("patch_dropout_cls_jitter",))

    def test_cli_exposes_patch_token_dropout_mix_method(self):
        argv = ["main.py", "--guide-aug-method", "patch_token_dropout_mix"]
        with mock.patch.object(sys, "argv", argv):
            args = main.parse_args()
        self.assertEqual(args.guide_aug_method, ("patch_token_dropout_mix",))

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

    def test_create_attacker_forwards_patch_dropout_score_mode(self):
        attacker = main.create_attacker(
            model=TinyTokenModel(),
            epsilon=0.1,
            step_size=None,
            steps=2,
            ti_sigma=0,
            guide_aug=True,
            guide_aug_methods=("patch_dropout",),
            patch_dropout_score_mode="low",
            patch_dropout_sampling_mode="score_weighted",
            patch_dropout_score_quantile_jitter=0.1,
            patch_dropout_score_noise=0.2,
            patch_dropout_fill_mode="context_high_score_blend",
            patch_dropout_noise_mode="patch_embed_rowspace",
            token_patch_dropout_layer=2,
            token_cls_noise=True,
            token_score_cls_noise=True,
            token_score_patch_noise=True,
        )
        self.assertEqual(attacker.patch_dropout_score_mode, "low")
        self.assertEqual(attacker.patch_dropout_sampling_mode, "score_weighted")
        self.assertEqual(attacker.patch_dropout_score_quantile_jitter, 0.1)
        self.assertEqual(attacker.patch_dropout_score_noise, 0.2)
        self.assertEqual(attacker.patch_dropout_fill_mode, "context_high_score_blend")
        self.assertEqual(attacker.patch_dropout_noise_mode, "patch_embed_rowspace")
        self.assertEqual(attacker.token_patch_dropout_layer, 2)
        self.assertTrue(attacker.token_cls_noise)
        self.assertTrue(attacker.token_score_cls_noise)
        self.assertTrue(attacker.token_score_patch_noise)

    # --- input diversity (方案一) tests ---

    def test_input_diversity_budget_validation(self):
        base_kwargs = dict(
            epsilon=0.1, steps=2, ti_sigma=0, device=torch.device("cpu"),
            guide_aug=True, guide_aug_methods=("token_patch_dropout",),
        )
        LMDSSAttacker(TinyTimmViTWrapper(), **base_kwargs,
                      input_diversity_groups=20, input_diversity_views_per_group=1)
        LMDSSAttacker(TinyTimmViTWrapper(), **base_kwargs,
                      input_diversity_groups=10, input_diversity_views_per_group=2)
        LMDSSAttacker(TinyTimmViTWrapper(), **base_kwargs,
                      input_diversity_groups=5, input_diversity_views_per_group=4)
        with self.assertRaises(ValueError):
            LMDSSAttacker(TinyTimmViTWrapper(), **base_kwargs,
                          input_diversity_groups=11, input_diversity_views_per_group=2)

    def test_input_diversity_zero_shift_is_identity(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"))
        pixels = torch.rand(2, 3, 12, 12)
        shifted = attacker._apply_phase_shift(pixels, 0, 0)
        self.assertTrue(torch.equal(shifted, pixels))

    def test_input_diversity_phase_shift_moves_content(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"))
        pixels = torch.zeros(1, 3, 12, 12)
        pixels[:, :, 0, 0] = 1.0
        shifted = attacker._apply_phase_shift(pixels, 4, 4)
        self.assertEqual(shifted.shape, (1, 3, 12, 12))
        self.assertGreater(float(shifted[:, :, 4, 4].max()), 0.9)

    def test_input_diversity_10x2_produces_20_views(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=10, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(2, 2),
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 20)

    def test_input_diversity_group_has_two_different_views(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=10, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(4, 4),
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        grads = [torch.autograd.grad(loss, pixels, retain_graph=True)[0].clone()
                 for loss in losses]
        v0, v1 = grads[0], grads[1]
        self.assertFalse(torch.allclose(v0, v1, atol=1e-6),
                         msg="View 0 and View 1 gradients are identical despite phase shift")

    def test_input_diversity_independent_randomness_per_view(self):
        torch.manual_seed(42)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            patch_dropout_ratio=0.5, input_diversity_groups=10,
            input_diversity_views_per_group=2, input_diversity_phase_shift=(0, 0),
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        loss_vals = [float(loss.detach()) for loss in losses]
        self.assertGreater(len(set(round(v, 6) for v in loss_vals)), 1,
                           msg="All 20 views produced identical loss despite independent randomness")

    def test_input_diversity_nested_mean_equals_flat_mean(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=10, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(0, 0),
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        grads = tuple(torch.autograd.grad(loss, pixels, retain_graph=True)[0] for loss in losses)
        flat_mean = torch.stack(grads, dim=0).mean(dim=0)
        group_means = [(grads[g * 2] + grads[g * 2 + 1]) / 2.0 for g in range(10)]
        nested_mean = torch.stack(group_means, dim=0).mean(dim=0)
        self.assertTrue(torch.allclose(flat_mean, nested_mean, atol=1e-6, rtol=1e-6))

    def test_cli_exposes_input_diversity_args(self):
        from main import parse_args
        import sys
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "main.py", "--input-diversity-groups", "10",
                "--input-diversity-views-per-group", "2",
                "--input-diversity-phase-shift", "4,4",
                "--input-diversity-phase-shift-set", "4,4;8,8",
                "--input-diversity-pair-aggregation", "mean",
            ]
            args = parse_args()
            self.assertEqual(args.input_diversity_groups, 10)
            self.assertEqual(args.input_diversity_views_per_group, 2)
            self.assertEqual(args.input_diversity_phase_shift, (4, 4))
            self.assertEqual(args.input_diversity_phase_shift_set, ((4, 4), (8, 8)))
            self.assertEqual(args.input_diversity_pair_aggregation, "mean")
        finally:
            sys.argv = old_argv

    # --- 方案二: Cross-Patch Counterfactual Transport ---

    def test_cross_patch_permutation_rotate180(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 cross_patch_transport_mode="rotate180")
        perm = attacker._build_cross_patch_permutation(9)  # 3x3 grid
        self.assertEqual(perm[0].item(), 8)
        self.assertEqual(perm[8].item(), 0)
        self.assertEqual(perm[4].item(), 4)

    def test_cross_patch_permutation_mirror_x(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 cross_patch_transport_mode="mirror_x")
        perm = attacker._build_cross_patch_permutation(9)
        self.assertEqual(perm[0].item(), 2)
        self.assertEqual(perm[4].item(), 4)
        self.assertEqual(perm[2].item(), 0)

    def test_cross_patch_transport_alpha_zero_is_identity(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 cross_patch_transport_mode="rotate180",
                                 cross_patch_transport_alpha=0.0)
        patch_tokens = torch.rand(1, 9, 3)
        drop_mask = torch.zeros(1, 9, dtype=torch.bool)
        result = attacker._apply_cross_patch_transport(patch_tokens, drop_mask)
        self.assertTrue(torch.allclose(result, patch_tokens, atol=1e-6))

    def test_cross_patch_transport_modifies_kept_not_dropped(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 cross_patch_transport_mode="rotate180",
                                 cross_patch_transport_alpha=0.2)
        patch_tokens = torch.rand(1, 9, 3)
        drop_mask = torch.tensor([[True, True, True, True, True, True, True, True, False]])
        result = attacker._apply_cross_patch_transport(patch_tokens, drop_mask)
        self.assertTrue(torch.equal(result[0, :8], patch_tokens[0, :8]))
        self.assertFalse(torch.allclose(result[0, 8:], patch_tokens[0, 8:], atol=1e-6))

    def test_cross_patch_transport_in_pipeline(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=1, input_diversity_views_per_group=1,
            cross_patch_transport_mode="rotate180",
            cross_patch_transport_alpha=0.2,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 1)
        self.assertTrue(torch.isfinite(losses[0]))

    # --- 方案四: Kept-Token Orthogonal Residual ---

    def test_pair_swap_rotation_matrix_is_orthogonal(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 kept_token_rotation_mode="pair_swap")
        R = attacker._build_kept_token_rotation_matrix(6)
        self.assertTrue(torch.allclose(R @ R.T, torch.eye(6), atol=1e-6))

    def test_hadamard_rotation_matrix_is_orthogonal(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 kept_token_rotation_mode="hadamard_block")
        R = attacker._build_kept_token_rotation_matrix(4)
        self.assertTrue(torch.allclose(R @ R.T, torch.eye(4), atol=1e-6))

    def test_kept_token_rotation_alpha_zero_is_identity(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 kept_token_rotation_mode="pair_swap",
                                 kept_token_rotation_alpha=0.0)
        patch_tokens = torch.rand(1, 9, 6)
        drop_mask = torch.zeros(1, 9, dtype=torch.bool)
        result = attacker._apply_kept_token_rotation(patch_tokens, drop_mask)
        self.assertTrue(torch.allclose(result, patch_tokens, atol=1e-6))

    def test_kept_token_rotation_modifies_kept_not_dropped(self):
        attacker = LMDSSAttacker(TinyTimmViTWrapper(), epsilon=0.1, steps=2,
                                 ti_sigma=0, device=torch.device("cpu"),
                                 kept_token_rotation_mode="pair_swap",
                                 kept_token_rotation_alpha=0.5)
        patch_tokens = torch.rand(1, 9, 6)
        # Keep 3 patches so centering doesn't zero out the residual
        drop_mask = torch.tensor([[True, True, True, True, True, True, False, False, False]])
        result = attacker._apply_kept_token_rotation(patch_tokens, drop_mask)
        # Dropped patches unchanged
        self.assertTrue(torch.equal(result[0, :6], patch_tokens[0, :6]))
        # Kept patches modified
        self.assertFalse(torch.allclose(result[0, 6:], patch_tokens[0, 6:], atol=1e-6))

    def test_kept_token_rotation_in_pipeline(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=1, input_diversity_views_per_group=1,
            kept_token_rotation_mode="pair_swap",
            kept_token_rotation_alpha=0.1,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 1)
        self.assertTrue(torch.isfinite(losses[0]))

    # --- 方案三: Pair-Difference Gradient ---

    def test_pair_difference_lambda_zero_equals_plain_mean(self):
        torch.manual_seed(42)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=2, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(0, 0),
            input_diversity_pair_aggregation="difference_mix",
            input_diversity_lambda_difference=0.0,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        grad, term_grads = attacker._attack_grad_terms(pixels, labels)
        flat_mean = torch.stack(term_grads, dim=0).mean(dim=0)
        self.assertTrue(torch.allclose(grad, flat_mean, atol=1e-6, rtol=1e-6))

    def test_original_score_postdrop_phase_pair_uses_shared_pixel_mask_and_20_view_budget(self):
        torch.manual_seed(47)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("original_score_postdrop_phase_pair",),
            input_diversity_groups=2, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(2, 2), feature_layer=3,
            patch_dropout_ratio=0.5, patch_dropout_score_mode="high",
            token_score_cls_noise=False, patch_dropout_noise_mode="gaussian",
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 4)
        self.assertEqual(attacker._actual_forward_view_count, 4)
        gradients = [
            torch.autograd.grad(loss, pixels, retain_graph=index < len(losses) - 1)[0]
            for index, loss in enumerate(losses)
        ]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertFalse(torch.allclose(gradients[0], gradients[1]))

    def test_patch_drop_mask_to_image_expands_square_patch_mask(self):
        drop_mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)
        image_mask = LMDSSAttacker._patch_drop_mask_to_image(drop_mask, 8, 8)
        self.assertEqual(tuple(image_mask.shape), (1, 1, 8, 8))
        self.assertEqual(float(image_mask.sum()), 16.0)

    def test_pair_difference_with_lambda_modifies_gradient(self):
        torch.manual_seed(42)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=2, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(0, 0),
            input_diversity_pair_aggregation="difference_mix",
            input_diversity_lambda_difference=0.2,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        grad, term_grads = attacker._attack_grad_terms(pixels, labels)
        flat_mean = torch.stack(term_grads, dim=0).mean(dim=0)
        self.assertFalse(torch.allclose(grad, flat_mean, atol=1e-6))

    def test_pair_difference_views_per_group_1_skips(self):
        torch.manual_seed(42)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=4, input_diversity_views_per_group=1,
            input_diversity_pair_aggregation="difference_mix",
            input_diversity_lambda_difference=0.2,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        grad, term_grads = attacker._attack_grad_terms(pixels, labels)
        flat_mean = torch.stack(term_grads, dim=0).mean(dim=0)
        self.assertTrue(torch.allclose(grad, flat_mean, atol=1e-6, rtol=1e-6))

    # --- Combination tests ---

    def test_transport_plus_rotation_combined(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            cross_patch_transport_mode="rotate180",
            cross_patch_transport_alpha=0.1,
            kept_token_rotation_mode="pair_swap",
            kept_token_rotation_alpha=0.1,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 20)
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))

    def test_phase_shift_plus_transport_combined(self):
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=5, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(2, 2),
            cross_patch_transport_mode="mirror_x",
            cross_patch_transport_alpha=0.1,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 10)
        for loss in losses:
            self.assertTrue(torch.isfinite(loss))

    def test_phase_shift_plus_pair_difference_combined(self):
        torch.manual_seed(42)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=3, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(2, 2),
            input_diversity_pair_aggregation="difference_mix",
            input_diversity_lambda_difference=0.1,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        grad, term_grads = attacker._attack_grad_terms(pixels, labels)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(len(term_grads), 6)

    def test_all_three_combined(self):
        torch.manual_seed(42)
        attacker = LMDSSAttacker(
            TinyTimmViTWrapper(), epsilon=0.1, steps=2, ti_sigma=0,
            device=torch.device("cpu"), guide_aug=True,
            guide_aug_methods=("token_patch_dropout",),
            input_diversity_groups=3, input_diversity_views_per_group=2,
            input_diversity_phase_shift=(2, 2),
            input_diversity_pair_aggregation="difference_mix",
            input_diversity_lambda_difference=0.1,
            cross_patch_transport_mode="rotate180",
            cross_patch_transport_alpha=0.1,
        )
        pixels = torch.rand(1, 3, 12, 12, requires_grad=True)
        labels = torch.tensor([0])
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 6)
        grads = [torch.autograd.grad(loss, pixels, retain_graph=True)[0].clone()
                 for loss in losses]
        for g in grads:
            self.assertTrue(torch.isfinite(g).all())


if __name__ == "__main__":
    unittest.main()
