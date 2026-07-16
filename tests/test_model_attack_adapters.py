import gc
import os
import unittest
from types import SimpleNamespace

import torch

from attack import PatchScoreAttacker
from nets import build_whitebox_model
from nets.base import AttackFeatureState


class DummyModel:
    model_mean = (0.5, 0.5, 0.5)
    model_std = (0.5, 0.5, 0.5)

    def eval(self):
        return self


class MainlineBudgetAndNoiseTests(unittest.TestCase):
    def make_attacker(self, **kwargs):
        return PatchScoreAttacker(
            DummyModel(),
            steps=1,
            input_diversity_groups=1,
            input_diversity_views_per_group=2,
            device=torch.device("cpu"),
            **kwargs,
        )

    def test_native_patch_budgets_track_fifteen_percent(self):
        attacker = self.make_attacker(patch_dropout_ratio=0.3)
        expected = {196: ((14, 14), 29), 64: ((8, 8), 10), 49: ((7, 7), 7)}
        for count, (grid_size, expected_dropped) in expected.items():
            scores = torch.linspace(-1.0, 1.0, count).unsqueeze(0)
            candidates = attacker._patch_score_candidate_mask(scores)
            mask = attacker._sample_patch_dropout_mask(scores, candidates)
            self.assertEqual(int(mask.sum()), expected_dropped)
            self.assertLessEqual(abs(float(mask.float().mean()) - 0.15), 1.0 / count)
            image_mask = attacker._patch_drop_mask_to_image(mask, grid_size, 224, 224)
            self.assertAlmostEqual(float(image_mask.mean()), expected_dropped / count, places=6)

    def test_strict_opponent_noise_has_target_rms(self):
        torch.manual_seed(11)
        attacker = self.make_attacker(guide_aug_strength=0.2)
        local_tokens = torch.randn(2, 4, 5)
        state = AttackFeatureState(
            local_tokens=local_tokens,
            grid_size=(2, 2),
            context=None,
            rgb_projection_weight=torch.randn(5, 3, 1, 1),
            projection_kernel=(1, 1),
            projection_stride=(1, 1),
            projection_padding=(0, 0),
        )

        noise = attacker._strict_opponent_feature_noise(state)

        source_rms = local_tokens.square().mean(dim=(1, 2)).sqrt()
        noise_rms = noise.square().mean(dim=(1, 2)).sqrt()
        self.assertTrue(torch.allclose(noise_rms, 0.2 * source_rms, rtol=1e-5, atol=1e-6))
        self.assertEqual(attacker.mainline_metadata()["feature_noise_type"],
                         "opponent_channel_rgb_projection")

    def test_post_dropout_feature_noise_strength_is_independent(self):
        torch.manual_seed(11)
        attacker = self.make_attacker(
            guide_aug_strength=0.2,
            post_dropout_feature_noise_strength=0.05,
        )
        local_tokens = torch.randn(2, 4, 5)
        state = AttackFeatureState(
            local_tokens=local_tokens,
            grid_size=(2, 2),
            context=None,
            rgb_projection_weight=torch.randn(5, 3, 1, 1),
            projection_kernel=(1, 1),
            projection_stride=(1, 1),
            projection_padding=(0, 0),
        )
        noise = attacker._strict_opponent_feature_noise(state)
        source_rms = local_tokens.square().mean(dim=(1, 2)).sqrt()
        noise_rms = noise.square().mean(dim=(1, 2)).sqrt()
        self.assertTrue(torch.allclose(noise_rms, 0.05 * source_rms, rtol=1e-5, atol=1e-6))
        self.assertAlmostEqual(
            attacker.mainline_metadata()["post_dropout_feature_noise_strength"],
            0.05,
        )

    def test_strict_projection_matches_original_vit_noise_formula(self):
        attacker = self.make_attacker(guide_aug_strength=0.2)
        local_tokens = torch.randn(1, 4, 5)
        weight = torch.randn(5, 3, 2, 2)
        state = AttackFeatureState(
            local_tokens=local_tokens,
            grid_size=(2, 2),
            context=None,
            rgb_projection_weight=weight,
            projection_kernel=(2, 2),
            projection_stride=(2, 2),
            projection_padding=(0, 0),
        )
        base_model = SimpleNamespace(
            patch_embed=SimpleNamespace(proj=SimpleNamespace(weight=weight))
        )

        torch.manual_seed(29)
        strict = attacker._strict_opponent_feature_noise(state)
        torch.manual_seed(29)
        original = attacker._token_patch_dropout_noise(local_tokens, base_model)

        self.assertTrue(torch.equal(strict, original))

    def test_projection_mask_uses_rgb_receptive_fields(self):
        attacker = self.make_attacker()
        state = AttackFeatureState(
            local_tokens=torch.zeros(1, 4, 2),
            grid_size=(2, 2),
            context=None,
            rgb_projection_weight=torch.ones(2, 3, 2, 2),
            projection_kernel=(2, 2),
            projection_stride=(2, 2),
            projection_padding=(0, 0),
        )
        image_mask = torch.zeros(1, 1, 4, 4)
        image_mask[:, :, :2, :2] = 1.0

        projected = attacker._image_mask_to_projection_drop_mask(image_mask, state)

        self.assertTrue(torch.equal(projected, torch.tensor([[True, False, False, False]])))

    def test_invalid_rgb_projection_never_falls_back_to_gaussian(self):
        attacker = self.make_attacker()
        state = AttackFeatureState(
            local_tokens=torch.zeros(1, 1, 2),
            grid_size=(1, 1),
            context=None,
            rgb_projection_weight=torch.ones(2, 4, 1, 1),
            projection_kernel=(1, 1),
            projection_stride=(1, 1),
            projection_padding=(0, 0),
        )
        with self.assertRaisesRegex(ValueError, "RGB Conv2d"):
            attacker._strict_opponent_feature_noise(state)

    def test_both_retained_noises_are_rms_matched(self):
        local_tokens = torch.randn(2, 4, 5)
        state = AttackFeatureState(
            local_tokens=local_tokens,
            grid_size=(2, 2),
            context=None,
            rgb_projection_weight=torch.randn(5, 3, 1, 1),
            projection_kernel=(1, 1),
            projection_stride=(1, 1),
            projection_padding=(0, 0),
        )
        image_mask = torch.zeros(2, 1, 2, 2)
        expected_rms = 0.2 * local_tokens.square().mean(dim=(1, 2)).sqrt()

        for noise_type in ("gaussian", "opponent_projected"):
            with self.subTest(noise_type=noise_type):
                attacker = self.make_attacker(
                    post_dropout_feature_noise_type=noise_type,
                )
                noise = attacker._build_post_dropout_feature_noise(
                    local_tokens,
                    state,
                    image_mask,
                )
                actual_rms = noise.square().mean(dim=(1, 2)).sqrt()
                self.assertTrue(
                    torch.allclose(actual_rms, expected_rms, rtol=1e-5, atol=1e-6)
                )

    def test_both_retained_noises_are_kept_only(self):
        local_tokens = torch.randn(1, 4, 5)
        state = AttackFeatureState(
            local_tokens=local_tokens,
            grid_size=(2, 2),
            context=None,
            rgb_projection_weight=torch.randn(5, 3, 1, 1),
            projection_kernel=(1, 1),
            projection_stride=(1, 1),
            projection_padding=(0, 0),
        )
        image_mask = torch.zeros(1, 1, 2, 2)
        image_mask[:, :, 0, 0] = 1.0

        for noise_type in ("gaussian", "opponent_projected"):
            with self.subTest(noise_type=noise_type):
                attacker = self.make_attacker(
                    post_dropout_feature_noise_type=noise_type,
                )
                noise = attacker._build_post_dropout_feature_noise(
                    local_tokens,
                    state,
                    image_mask,
                )
                self.assertTrue(torch.equal(noise[:, 0], torch.zeros_like(noise[:, 0])))
                self.assertGreater(float(noise[:, 1:].abs().sum()), 0.0)

    def test_patch_score_layer_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "patch_score_layer"):
            self.make_attacker(patch_score_layer="middle")


@unittest.skipUnless(os.environ.get("RUN_MODEL_SMOKE") == "1", "set RUN_MODEL_SMOKE=1")
class RealModelMainlineSmokeTests(unittest.TestCase):
    EXPECTED = {
        "vit_base_patch16_224": ((14, 14), 29, "blocks[11]"),
        "cait_s24_224": ((14, 14), 29, "blocks[23]+blocks_token_only[1]"),
        "pit_b_224": ((8, 8), 10, "transformers[2].blocks[3]"),
        "visformer_small": ((7, 7), 7, "stage3[3]+gap"),
    }

    def test_full_two_view_one_step_mainline(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_name, (grid_size, drop_count, score_source) in self.EXPECTED.items():
            for noise_type in ("gaussian", "opponent_projected"):
                with self.subTest(model=model_name, noise=noise_type):
                    torch.manual_seed(7)
                    model = build_whitebox_model(
                        num_classes=1000,
                        model_name=model_name,
                        pretrained=False,
                        device=device,
                    )
                    attacker = PatchScoreAttacker(
                        model,
                        epsilon=1.0 / 255.0,
                        steps=1,
                        input_diversity_groups=1,
                        input_diversity_views_per_group=2,
                        input_diversity_phase_shift_set=((4, 4),),
                        post_dropout_feature_noise_type=noise_type,
                        device=device,
                    )
                    images = torch.zeros(1, 3, 224, 224)
                    normalized = attacker._normalize(attacker._denormalize(images.to(device)))
                    with torch.no_grad():
                        direct_logits = model(normalized)
                        state = model.prepare_attack_feature_state(normalized)
                        resumed_logits = model.forward_from_attack_feature_state(
                            state,
                            state.local_tokens,
                        )
                    self.assertTrue(torch.equal(direct_logits, resumed_logits))
                    adversarial = attacker.attack_batch(
                        images,
                        torch.zeros(1, dtype=torch.long),
                    )
                    metadata = attacker.mainline_metadata()

                    self.assertEqual(metadata["score_grid"], list(grid_size))
                    self.assertEqual(metadata["actual_patch_drop_count"], drop_count)
                    self.assertEqual(metadata["score_source"], score_source)
                    expected_noise = (
                        "feature_iid_gaussian"
                        if noise_type == "gaussian"
                        else "opponent_channel_rgb_projection"
                    )
                    self.assertEqual(metadata["feature_noise_type"], expected_noise)
                    self.assertTrue(torch.isfinite(adversarial).all())
                    clean_pixels = attacker._denormalize(images.to(device))
                    adv_pixels = attacker._denormalize(adversarial)
                    self.assertLessEqual(
                        float((adv_pixels - clean_pixels).abs().max()),
                        1.0 / 255.0 + 1e-6,
                    )

                    del (
                        attacker,
                        model,
                        adversarial,
                        clean_pixels,
                        adv_pixels,
                        normalized,
                        direct_logits,
                        state,
                        resumed_logits,
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def test_retained_control_attacks_run_on_vit(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_whitebox_model(
            num_classes=1000,
            model_name="vit_base_patch16_224",
            pretrained=False,
            device=device,
        )
        images = torch.zeros(1, 3, 224, 224)
        labels = torch.zeros(1, dtype=torch.long)
        configs = (
            {
                "attack_method": "patch_dropout",
                "guide_aug_copies": 1,
                "feature_layer": -1,
            },
            {
                "attack_method": "token_patch_dropout",
                "input_diversity_groups": 1,
                "input_diversity_views_per_group": 1,
            },
            {
                "attack_method": "none",
                "input_diversity": True,
                "nesterov": True,
                "ti_sigma": 1.0,
            },
        )
        for config in configs:
            with self.subTest(attack_method=config["attack_method"]):
                attacker = PatchScoreAttacker(
                    model,
                    epsilon=1.0 / 255.0,
                    steps=1,
                    gaussian_alpha=0.0,
                    device=device,
                    **config,
                )
                adversarial = attacker.attack_batch(images, labels)
                self.assertTrue(torch.isfinite(adversarial).all())
        del attacker, adversarial, model, images, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    unittest.main()
