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

    def test_generalized_mainline_rejects_non_opponent_noise(self):
        with self.assertRaisesRegex(ValueError, "strictly requires opponent-channel"):
            self.make_attacker(patch_dropout_noise_mode="gaussian")


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
            with self.subTest(model=model_name):
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
                adversarial = attacker.attack_batch(images, torch.zeros(1, dtype=torch.long))
                metadata = attacker.mainline_metadata()

                self.assertEqual(metadata["score_grid"], list(grid_size))
                self.assertEqual(metadata["actual_patch_drop_count"], drop_count)
                self.assertEqual(metadata["score_source"], score_source)
                self.assertEqual(metadata["feature_noise_type"],
                                 "opponent_channel_rgb_projection")
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


if __name__ == "__main__":
    unittest.main()
