import json
import os
import types
import unittest

import torch
from torch import nn

from gradient_replay import GradientReplay
from nets.base import AttackFeatureState, conv2d_attack_metadata
from vit_progressive_patch_score_attack import (
    MODEL_NAME,
    ProgressiveMaskSchedule,
    ViTProgressivePatchScoreAttacker,
    _parse_float_list,
    _parse_int_list,
)


class TinyPatchEmbed(nn.Module):
    def __init__(self, dimension: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(3, dimension, kernel_size=2, stride=2, bias=False)
        self.grid_size = (2, 2)

    def forward(self, x):
        local = self.proj(x).flatten(2).transpose(1, 2)
        cls = torch.zeros(
            x.size(0), 1, local.size(-1), device=x.device, dtype=x.dtype
        )
        return torch.cat((cls, local), dim=1)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def forward(self, tokens):
        self.last_input = tokens.detach().clone()
        # Make CLS and local tokens interact so that a preceding drop affects
        # the later checkpoint trajectory.
        return tokens + 0.1 * tokens.mean(dim=1, keepdim=True)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatchEmbed()
        self._pos_embed = nn.Identity()
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(12))
        self.norm = nn.Identity()
        self.head = nn.Linear(4, 2, bias=False)

    def forward_head(self, tokens):
        return self.head(tokens[:, 0])


class TinyViTWrapper(nn.Module):
    model_name = MODEL_NAME
    model_mean = (0.0, 0.0, 0.0)
    model_std = (1.0, 1.0, 1.0)

    def __init__(self):
        super().__init__()
        self.model = TinyViT()

    def patch_score_layer_candidates(self):
        return ("block3", "block6", "block9", "block12")

    def prepare_attack_feature_state(self, x):
        tokens = self.model.patch_embed(x)
        return AttackFeatureState(
            local_tokens=tokens[:, 1:],
            grid_size=(2, 2),
            context={"prefix_tokens": tokens[:, :1]},
            **conv2d_attack_metadata(self.model.patch_embed.proj),
        )

    def eval(self):
        super().eval()
        return self


class ProgressiveViTTests(unittest.TestCase):
    def make_attacker(self, **overrides):
        arguments = {
            "checkpoints": (3, 6, 9),
            "drop_ratios": (0.25, 0.25, 0.25),
            "score_cls_noise_strength": 0.0,
            "opponent_noise_strength": 0.0,
            "steps": 1,
            "input_diversity_groups": 1,
            "input_diversity_views_per_group": 2,
            "input_diversity_phase_shift_set": ((0, 0),),
            "use_momentum": False,
            "gaussian_alpha": 0.0,
            "device": torch.device("cpu"),
        }
        arguments.update(overrides)
        return ViTProgressivePatchScoreAttacker(TinyViTWrapper(), **arguments)

    def test_list_parsers(self):
        self.assertEqual(_parse_int_list("3, 6,9"), (3, 6, 9))
        self.assertEqual(_parse_float_list(".05, 0.1, .2"), (0.05, 0.1, 0.2))

    def test_independent_ratios_do_not_hit_parent_single_budget_limit(self):
        attacker = self.make_attacker(drop_ratios=(0.4, 0.4, 0.4))
        self.assertEqual(attacker.progressive_drop_ratios, (0.4, 0.4, 0.4))
        self.assertEqual(attacker.patch_dropout_ratio, 0.0)

    def test_metadata_is_json_serializable(self):
        attacker = self.make_attacker()
        encoded = json.dumps(attacker.mainline_metadata())
        self.assertIn('"model_mean": [0.0, 0.0, 0.0]', encoded)

    def test_phase_pair_invariants_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError, "requires.*attack_method"):
            self.make_attacker(attack_method="none")
        with self.assertRaisesRegex(ValueError, "two views"):
            self.make_attacker(input_diversity_views_per_group=1)
        with self.assertRaisesRegex(ValueError, "progressive patch_selector"):
            self.make_attacker(patch_selector="no_drop")

    def test_sampled_tokens_belong_to_current_high_half(self):
        attacker = self.make_attacker()
        scores = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [8.0, 3.0, 5.0, 1.0]]
        )
        mask = attacker._sample_high_mask(scores, 0.25, checkpoint=3)
        high_half = torch.zeros_like(mask)
        high_half.scatter_(1, torch.topk(scores, 2, dim=1).indices, True)
        self.assertFalse(bool(torch.any(mask & ~high_half)))
        self.assertTrue(torch.equal(mask.sum(dim=1), torch.ones(2, dtype=torch.long)))

    def test_sampled_tokens_belong_to_current_low_half(self):
        attacker = self.make_attacker(
            patch_selector="low",
            score_cls_noise_strength=0.2,
        )
        scores = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [8.0, 3.0, 5.0, 1.0]]
        )
        mask = attacker._sample_low_mask(scores, 0.25, checkpoint=3)
        low_half = torch.zeros_like(mask)
        low_half.scatter_(1, torch.topk(scores, 2, dim=1, largest=False).indices, True)
        self.assertFalse(bool(torch.any(mask & ~low_half)))
        self.assertTrue(torch.equal(mask.sum(dim=1), torch.ones(2, dtype=torch.long)))
        metadata = attacker.mainline_metadata()
        self.assertEqual(metadata["patch_selector"], "low")
        self.assertTrue(metadata["score_cls_noise_active"])

    def test_high_is_an_explicit_alias_for_patch_score_selection(self):
        attacker = self.make_attacker(patch_selector="high")
        schedule = attacker._build_mask_schedule(torch.rand(1, 3, 4, 4))
        self.assertEqual(schedule.counts, (1, 1, 1))
        self.assertEqual(attacker.mainline_metadata()["patch_selector"], "high")

    def test_zero_score_cls_noise_is_recorded_as_inactive(self):
        attacker = self.make_attacker(
            patch_selector="high",
            score_cls_noise_strength=0.0,
        )
        metadata = attacker.mainline_metadata()
        self.assertFalse(metadata["score_cls_noise_active"])
        self.assertEqual(metadata["score_cls_noise_strength"], 0.0)
        self.assertEqual(metadata["score_reference"], "current_cls_without_noise")

    def test_random_selector_skips_scores_and_uses_full_patch_budget(self):
        attacker = self.make_attacker(patch_selector="random")

        def score_must_not_run(*_args, **_kwargs):
            raise AssertionError("random progressive selection must not compute patch scores")

        attacker._score_at_checkpoint = score_must_not_run
        schedule = attacker._build_mask_schedule(torch.rand(2, 3, 4, 4))
        self.assertEqual(schedule.counts, (1, 1, 1))
        schedule.validate(batch_size=2, token_count=4)
        metadata = attacker.mainline_metadata()
        self.assertEqual(metadata["attack_method"], "vit_progressive_random")
        self.assertEqual(metadata["patch_selector"], "random")
        self.assertEqual(metadata["score_reference"], "none_uniform_all_local_tokens")
        self.assertFalse(metadata["score_cls_noise_active"])

    def test_schedule_is_sequential_and_can_repeat_positions(self):
        attacker = self.make_attacker()
        pixels = torch.rand(1, 3, 4, 4)
        checkpoint_inputs = {}

        def record_scores(_self, tokens, *, checkpoint):
            checkpoint_inputs[checkpoint] = tokens.detach().clone()
            return torch.arange(tokens.size(1) - 1, device=tokens.device)[None].expand(
                tokens.size(0), -1
            )

        def repeat_first_position(_self, scores, _ratio, *, checkpoint):
            del checkpoint
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[:, 0] = True
            return mask

        attacker._score_at_checkpoint = types.MethodType(record_scores, attacker)
        attacker._sample_high_mask = types.MethodType(repeat_first_position, attacker)
        schedule = attacker._build_mask_schedule(pixels)

        self.assertTrue(all(bool(mask[0, 0]) for mask in schedule.masks))
        base = attacker._vit
        with torch.no_grad():
            embedded = attacker._embed_vit_tokens(base, pixels)
            at_three = attacker._run_vit_blocks(base.blocks, embedded, 0, 3)
            self.assertTrue(torch.allclose(checkpoint_inputs[3], at_three))

            after_three = attacker._apply_local_mask(at_three, schedule.masks[0])
            at_six = attacker._run_vit_blocks(base.blocks, after_three, 3, 6)
            uninterrupted_six = attacker._run_vit_blocks(base.blocks, embedded, 0, 6)
            self.assertTrue(torch.allclose(checkpoint_inputs[6], at_six))
            self.assertFalse(torch.allclose(checkpoint_inputs[6], uninterrupted_six))

            after_six = attacker._apply_local_mask(at_six, schedule.masks[1])
            at_nine = attacker._run_vit_blocks(base.blocks, after_six, 6, 9)
            without_middle_drop = attacker._run_vit_blocks(base.blocks, at_six, 6, 9)
            self.assertTrue(torch.allclose(checkpoint_inputs[9], at_nine))
            self.assertFalse(torch.allclose(checkpoint_inputs[9], without_middle_drop))

    def test_phase_schedule_preserves_counts(self):
        attacker = self.make_attacker()
        schedule = attacker._build_mask_schedule(torch.rand(1, 3, 4, 4))
        shifted = attacker._phase_mask_schedule(
            schedule, [(1, 1)], height=4, width=4
        )
        self.assertEqual(shifted.counts, schedule.counts)
        for original, phase in zip(schedule.masks, shifted.masks):
            self.assertEqual(int(original.sum()), int(phase.sum()))

    def test_opponent_noise_is_local_and_union_kept_only(self):
        attacker = self.make_attacker(opponent_noise_strength=1.0)
        pixels = torch.rand(1, 3, 4, 4)
        labels = torch.zeros(1, dtype=torch.long)
        first = torch.tensor([[True, False, False, False]])
        middle = torch.tensor([[False, True, False, False]])
        late = torch.tensor([[True, False, False, False]])
        schedule = ProgressiveMaskSchedule(
            checkpoints=(3, 6, 9),
            masks=(first, middle, late),
            counts=(1, 1, 1),
            grid_size=(2, 2),
        )

        def unit_noise(_self, state):
            return torch.ones_like(state.local_tokens)

        attacker._strict_opponent_feature_noise = types.MethodType(unit_noise, attacker)
        state = attacker.model.prepare_attack_feature_state(attacker._normalize(pixels))
        attacker._forward_with_schedule(pixels, labels, schedule)
        block_input = attacker._vit.blocks[0].last_input

        self.assertTrue(torch.equal(block_input[:, :1], state.context["prefix_tokens"]))
        expected_local = state.local_tokens.clone()
        expected_local[:, 2:] += 1.0
        self.assertTrue(torch.allclose(block_input[:, 1:], expected_local))

    def test_projected_opponent_noise_is_feature_rms_matched(self):
        attacker = self.make_attacker(opponent_noise_strength=0.25)
        pixels = torch.rand(1, 3, 4, 4)
        with torch.no_grad():
            attacker._vit.patch_embed.proj.weight.fill_(1.0)
        state = attacker.model.prepare_attack_feature_state(attacker._normalize(pixels))

        def deterministic_noise(_self, tensor, _event):
            return torch.arange(
                1,
                tensor.numel() + 1,
                device=tensor.device,
                dtype=tensor.dtype,
            ).view_as(tensor)

        attacker._randn_like = types.MethodType(deterministic_noise, attacker)
        feature_noise = attacker._strict_opponent_feature_noise(state)
        token_rms = state.local_tokens.square().mean(dim=(1, 2)).sqrt()
        noise_rms = feature_noise.square().mean(dim=(1, 2)).sqrt()
        self.assertTrue(torch.allclose(noise_rms, 0.25 * token_rms, rtol=1e-5))

    def test_default_scale_has_twenty_views_and_three_hundred_selections(self):
        attacker = self.make_attacker(steps=10, input_diversity_groups=10)
        pixels = torch.rand(1, 3, 4, 4, requires_grad=True)
        labels = torch.zeros(1, dtype=torch.long)
        losses = list(attacker._iter_attack_losses(pixels, labels))
        metadata = attacker.mainline_metadata()

        self.assertEqual(len(losses), 20)
        self.assertEqual(attacker._actual_forward_view_count, 20)
        self.assertEqual(attacker._progressive_schedule_count, 10)
        self.assertEqual(attacker._progressive_checkpoint_selection_count, 30)
        self.assertEqual(metadata["mask_schedule_count_per_image"], 100)
        self.assertEqual(metadata["checkpoint_mask_selection_count_per_image"], 300)

    def test_replay_reproduces_checkpoint_masks(self):
        attacker = self.make_attacker(score_cls_noise_strength=0.2)
        pixels = torch.rand(1, 3, 4, 4)

        def replayed_schedule():
            replay = GradientReplay(1234)
            replay.begin_batch(["sample.png"])
            replay.set_context(step=2, group=4, view=-1)
            attacker._gradient_replay = replay
            try:
                return attacker._build_mask_schedule(pixels)
            finally:
                attacker._gradient_replay = None

        first = replayed_schedule()
        second = replayed_schedule()
        for first_mask, second_mask in zip(first.masks, second.masks):
            self.assertTrue(torch.equal(first_mask, second_mask))

    def test_replay_reproduces_random_checkpoint_masks(self):
        attacker = self.make_attacker(patch_selector="random")
        pixels = torch.rand(1, 3, 4, 4)

        def replayed_schedule():
            replay = GradientReplay(4321)
            replay.begin_batch(["sample.png"])
            replay.set_context(step=3, group=5, view=-1)
            attacker._gradient_replay = replay
            try:
                return attacker._build_mask_schedule(pixels)
            finally:
                attacker._gradient_replay = None

        first = replayed_schedule()
        second = replayed_schedule()
        for first_mask, second_mask in zip(first.masks, second.masks):
            self.assertTrue(torch.equal(first_mask, second_mask))

    def test_two_views_backpropagate_and_attack_respects_epsilon(self):
        attacker = self.make_attacker()
        pixels = torch.rand(1, 3, 4, 4, requires_grad=True)
        labels = torch.zeros(1, dtype=torch.long)
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 2)
        gradients = [torch.autograd.grad(loss, pixels)[0] for loss in losses]
        self.assertEqual(attacker._actual_forward_view_count, 2)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

        clean = torch.rand(1, 3, 4, 4)
        adversarial = attacker.attack_batch(clean, labels)
        clean_pixels = attacker._denormalize(clean)
        adversarial_pixels = attacker._denormalize(adversarial)
        self.assertLessEqual(
            float((adversarial_pixels - clean_pixels).abs().max()),
            attacker.epsilon + 1e-6,
        )

    @unittest.skipUnless(
        os.environ.get("RUN_REAL_VIT_PROGRESSIVE_SMOKE") == "1",
        "set RUN_REAL_VIT_PROGRESSIVE_SMOKE=1 to run the real timm ViT smoke test",
    )
    def test_real_vit_forward_smoke(self):
        from nets import build_whitebox_model

        model = build_whitebox_model(
            num_classes=2,
            model_name=MODEL_NAME,
            pretrained=False,
            device=torch.device("cpu"),
        )
        for patch_selector in ("patch_score", "high", "low", "random"):
            with self.subTest(patch_selector=patch_selector):
                attacker = ViTProgressivePatchScoreAttacker(
                    model,
                    patch_selector=patch_selector,
                    steps=1,
                    input_diversity_groups=1,
                    input_diversity_views_per_group=2,
                    use_momentum=False,
                    gaussian_alpha=0.0,
                    device=torch.device("cpu"),
                )
                pixels = torch.rand(1, 3, 224, 224, requires_grad=True)
                labels = torch.zeros(1, dtype=torch.long)
                schedule = attacker._build_mask_schedule(pixels)
                loss = attacker._forward_with_schedule(pixels, labels, schedule)
                gradient = torch.autograd.grad(loss, pixels)[0]
                self.assertEqual(schedule.counts, (10, 10, 10))
                self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
