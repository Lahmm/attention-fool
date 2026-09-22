import json
import os
import types
import unittest

import torch
from torch import nn

from gradient_replay import GradientReplay
from nets.base import (
    ProgressiveAttackState,
    ProgressiveInputState,
    conv2d_progressive_metadata,
)
from nets.vit import DEFAULT_MODEL_NAME
from progressive_attack import (
    ProgressiveMaskSchedule,
    ProgressiveMaskSelection,
    ProgressiveRouteDisruptionAttacker,
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
    model_name = DEFAULT_MODEL_NAME
    model_mean = (0.0, 0.0, 0.0)
    model_std = (1.0, 1.0, 1.0)

    def __init__(self):
        super().__init__()
        self.model = TinyViT()

    def prepare_progressive_input(self, x):
        tokens = self.model.patch_embed(x)
        return ProgressiveInputState(
            local_tokens=tokens[:, 1:],
            grid_size=(2, 2),
            context={"prefix_tokens": tokens[:, :1]},
            **conv2d_progressive_metadata(self.model.patch_embed.proj),
        )

    def progressive_checkpoint_candidates(self):
        return tuple(f"block{index}" for index in range(1, 12))

    def default_progressive_checkpoints(self):
        return ("block3", "block10")

    def default_progressive_drop_ratios(self):
        return (0.051020408163, 0.051020408163)

    def begin_progressive_forward(self, x):
        initial = self.prepare_progressive_input(x)
        return ProgressiveAttackState(
            initial.local_tokens,
            initial.grid_size,
            {"prefix_tokens": initial.context["prefix_tokens"], "block_index": 0},
        )

    def replace_progressive_local_tokens(self, state, local_tokens):
        return ProgressiveAttackState(local_tokens, state.grid_size, dict(state.context))

    def advance_progressive_state(self, state, checkpoint_id):
        target = int(checkpoint_id.removeprefix("block"))
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        for block in self.model.blocks[state.context["block_index"] : target]:
            tokens = block(tokens)
        return ProgressiveAttackState(
            tokens[:, 1:],
            state.grid_size,
            {"prefix_tokens": tokens[:, :1], "block_index": target},
        )

    def apply_progressive_mask(self, state, mask):
        local = torch.where(mask.unsqueeze(-1), torch.zeros_like(state.local_tokens), state.local_tokens)
        return ProgressiveAttackState(local, state.grid_size, dict(state.context))

    def finish_progressive_forward(self, state):
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        for block in self.model.blocks[state.context["block_index"] :]:
            tokens = block(tokens)
        return self.model.forward_head(self.model.norm(tokens))

    def eval(self):
        super().eval()
        return self


class ProgressiveViTTests(unittest.TestCase):
    def make_attacker(self, **overrides):
        arguments = {
            "checkpoints": (3, 10),
            "drop_ratios": (0.25, 0.25),
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
        return ProgressiveRouteDisruptionAttacker(TinyViTWrapper(), **arguments)

    def test_vit_defaults_are_block3_and_block10(self):
        attacker = self.make_attacker(checkpoints=None, drop_ratios=None)
        self.assertEqual(attacker.progressive_checkpoints, ("block3", "block10"))
        self.assertEqual(
            attacker.progressive_drop_ratios,
            (0.051020408163, 0.051020408163),
        )

    def test_checkpoint_drop_ratios_are_independent(self):
        attacker = self.make_attacker(drop_ratios=(0.4, 0.4))
        self.assertEqual(attacker.progressive_drop_ratios, (0.4, 0.4))

    def test_checkpoint_count_is_dynamic_and_matches_ratios(self):
        attacker = self.make_attacker(
            checkpoints=(2, 4, 6, 9),
            drop_ratios=(0.25, 0.25, 0.25, 0.25),
        )
        pixels = torch.rand(1, 3, 4, 4)
        schedule = attacker._build_mask_schedule(pixels)
        self.assertEqual(schedule.checkpoints, ("block2", "block4", "block6", "block9"))
        self.assertEqual(schedule.counts, (1, 1, 1, 1))
        shifted = attacker._phase_mask_schedule(schedule, [(0, 0)], height=4, width=4)
        self.assertEqual(shifted.counts, schedule.counts)
        metadata = attacker.mainline_metadata()
        self.assertEqual(metadata["checkpoint_mask_selection_count_per_image"], 4)

    def test_checkpoint_and_ratio_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.make_attacker(
                checkpoints=(2, 4, 6, 9),
                drop_ratios=(0.25, 0.25, 0.25),
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.make_attacker(checkpoints=(), drop_ratios=())

    def test_dynamic_checkpoints_remain_unique_and_ordered(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.make_attacker(
                checkpoints=(2, 6, 4, 9),
                drop_ratios=(0.25, 0.25, 0.25, 0.25),
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.make_attacker(
                checkpoints=(2, 4, 4, 9),
                drop_ratios=(0.25, 0.25, 0.25, 0.25),
            )

    def test_metadata_is_json_serializable(self):
        attacker = self.make_attacker()
        encoded = json.dumps(attacker.mainline_metadata())
        self.assertIn('"model_mean": [0.0, 0.0, 0.0]', encoded)

    def test_phase_pair_requires_two_views(self):
        with self.assertRaisesRegex(ValueError, "two views"):
            self.make_attacker(input_diversity_views_per_group=1)

    def test_random_drop_uses_all_local_tokens_and_exact_budget(self):
        attacker = self.make_attacker()
        schedule = attacker._build_mask_schedule(torch.rand(2, 3, 4, 4))
        self.assertEqual(schedule.counts, (1, 1))
        schedule.validate(batch_size=2, token_count=4)
        metadata = attacker.mainline_metadata()
        self.assertEqual(metadata["attack_method"], "progressive_route_disruption")
        self.assertEqual(metadata["drop_map_policy"], "uniform_random_all_local_tokens")

    def test_schedule_is_sequential_and_can_repeat_positions(self):
        attacker = self.make_attacker()
        pixels = torch.rand(1, 3, 4, 4)

        def repeat_first_position(
            _self, *, batch_size, token_count, ratio, checkpoint, device
        ):
            del ratio, checkpoint
            mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
            mask[:, 0] = True
            return mask

        attacker._sample_random_mask = types.MethodType(repeat_first_position, attacker)
        schedule = attacker._build_mask_schedule(pixels)

        self.assertTrue(all(bool(mask[0, 0]) for mask in schedule.masks))
        with torch.no_grad():
            state = attacker.model.begin_progressive_forward(attacker._normalize(pixels))
            at_three = attacker.model.advance_progressive_state(state, "block3")
            after_three = attacker.model.apply_progressive_mask(at_three, schedule.masks[0])
            at_ten = attacker.model.advance_progressive_state(after_three, "block10")
            uninterrupted_ten = attacker.model.advance_progressive_state(at_three, "block10")
            self.assertFalse(torch.allclose(at_ten.local_tokens, uninterrupted_ten.local_tokens))

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
        late = torch.tensor([[False, True, False, False]])
        schedule = ProgressiveMaskSchedule((
            ProgressiveMaskSelection("block3", first, 1, (2, 2)),
            ProgressiveMaskSelection("block10", late, 1, (2, 2)),
        ))

        def unit_noise(_self, state):
            return torch.ones_like(state.local_tokens)

        attacker._strict_opponent_feature_noise = types.MethodType(unit_noise, attacker)
        state = attacker.model.prepare_progressive_input(attacker._normalize(pixels))
        attacker._forward_with_schedule(pixels, labels, schedule)
        block_input = attacker.model.model.blocks[0].last_input

        self.assertTrue(torch.equal(block_input[:, :1], state.context["prefix_tokens"]))
        expected_local = state.local_tokens.clone()
        expected_local[:, 2:] += 1.0
        self.assertTrue(torch.allclose(block_input[:, 1:], expected_local))

    def test_projected_opponent_noise_is_feature_rms_matched(self):
        attacker = self.make_attacker(opponent_noise_strength=0.25)
        pixels = torch.rand(1, 3, 4, 4)
        with torch.no_grad():
            attacker.model.model.patch_embed.proj.weight.fill_(1.0)
        state = attacker.model.prepare_progressive_input(attacker._normalize(pixels))

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

    def test_default_scale_has_twenty_views_and_two_hundred_selections(self):
        attacker = self.make_attacker(steps=10, input_diversity_groups=10)
        pixels = torch.rand(1, 3, 4, 4, requires_grad=True)
        labels = torch.zeros(1, dtype=torch.long)
        losses = list(attacker._iter_attack_losses(pixels, labels))
        metadata = attacker.mainline_metadata()

        self.assertEqual(len(losses), 20)
        self.assertEqual(attacker._actual_forward_view_count, 20)
        self.assertEqual(attacker._progressive_schedule_count, 10)
        self.assertEqual(attacker._progressive_checkpoint_selection_count, 20)
        self.assertEqual(metadata["mask_schedule_count_per_image"], 100)
        self.assertEqual(metadata["checkpoint_mask_selection_count_per_image"], 200)

    def test_replay_reproduces_checkpoint_masks(self):
        attacker = self.make_attacker()
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
            model_name=DEFAULT_MODEL_NAME,
            pretrained=False,
            device=torch.device("cpu"),
        )
        attacker = ProgressiveRouteDisruptionAttacker(
            model,
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
        self.assertEqual(schedule.counts, (10, 10))
        self.assertTrue(torch.isfinite(gradient).all())


if __name__ == "__main__":
    unittest.main()
