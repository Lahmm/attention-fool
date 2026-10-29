import unittest
from unittest.mock import patch

import torch
from torch import nn

from nets.vit import ViTAdapter
from progressive_attack import ProgressiveMaskSchedule, ProgressiveMaskSelection, ProgressiveRouteDisruptionAttacker
from utils import SEED


class TinyPatchEmbed(nn.Module):
    grid_size = (14, 14)

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def forward(self, x):
        self.last_input = x.detach().clone()
        return x + 0.1 * x.mean(dim=1, keepdim=True)


class TinyViT(nn.Module):
    def __init__(self, prefixes):
        super().__init__()
        self.num_prefix_tokens = prefixes
        self.pretrained_cfg = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
        self.patch_embed = TinyPatchEmbed()
        self.patch_drop = self.norm_pre = self.norm = nn.Identity()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(12))
        self.head = nn.Linear(4, 2, bias=False)

    def _pos_embed(self, local):
        return torch.cat((local.new_zeros(local.size(0), self.num_prefix_tokens, 4), local), dim=1)

    def forward_head(self, tokens):
        return self.head(tokens[:, :self.num_prefix_tokens].mean(dim=1))


def make_model(prefixes=1):
    torch.manual_seed(SEED)
    model_name = 'vit_base_patch16_224' if prefixes == 1 else 'deit_base_distilled_patch16_224'
    with patch('timm.create_model', return_value=TinyViT(prefixes)):
        model = ViTAdapter(model_name, pretrained=False, device=torch.device('cpu'))
    model.dynamic_checkpoint = False
    return model


def make_attacker(**overrides):
    options = dict(steps=2, input_diversity_groups=1, device=torch.device('cpu'))
    options.update(overrides)
    return ProgressiveRouteDisruptionAttacker(make_model(), **options)


class ProgressiveAttackTests(unittest.TestCase):
    def setUp(self):
        self.pixels = torch.linspace(0.1, 0.9, 3 * 28 * 28).view(1, 3, 28, 28)
        self.labels = torch.tensor([1])

    def test_schedule_uses_current_vit_budget_and_fresh_masks(self):
        attacker = make_attacker()
        first = attacker._build_mask_schedule(self.pixels)
        second = attacker._build_mask_schedule(self.pixels)
        self.assertEqual(first.checkpoints, ('block3', 'block10'))
        self.assertEqual(first.counts, (10, 10))
        self.assertEqual(first.grid_sizes, ((14, 14), (14, 14)))
        first.validate(batch_size=1)
        self.assertFalse(torch.equal(first.masks[0], second.masks[0]))
        self.assertFalse(torch.equal(first.masks[0], first.masks[1]))

    def test_checkpoint_masks_can_repeat_and_change_later_trajectory(self):
        attacker = make_attacker()
        mask = torch.zeros(1, 196, dtype=torch.bool)
        mask[:, :10] = True
        with patch.object(attacker, '_sample_random_mask', return_value=mask):
            schedule = attacker._build_mask_schedule(self.pixels)
        self.assertTrue(torch.equal(*schedule.masks))
        initial = attacker.model.begin_progressive_forward(self.pixels)
        early = attacker.model.advance_progressive_state(initial, 'block3')
        dropped = attacker.model.apply_progressive_mask(early, mask)
        self.assertEqual(int(torch.count_nonzero(dropped.local_tokens[:, :10])), 0)
        interrupted = attacker.model.advance_progressive_state(dropped, 'block10')
        uninterrupted = attacker.model.advance_progressive_state(early, 'block10')
        self.assertFalse(torch.allclose(interrupted.local_tokens, uninterrupted.local_tokens))
        self.assertGreater(int(torch.count_nonzero(interrupted.local_tokens[:, :10])), 0)

    def test_phase_preserves_each_native_grid_budget(self):
        attacker = make_attacker()
        first = torch.zeros(1, 196, dtype=torch.bool)
        first[:, 20:30] = True
        late = torch.zeros(1, 64, dtype=torch.bool)
        late[:, 20:26] = True
        schedule = ProgressiveMaskSchedule((
            ProgressiveMaskSelection('early', first, 10, (14, 14)),
            ProgressiveMaskSelection('late', late, 6, (8, 8)),
        ))
        shifted = attacker._phase_mask_schedule(schedule, [(8, 8)], height=224, width=224)
        self.assertEqual(shifted.counts, (10, 6))
        self.assertEqual(shifted.grid_sizes, ((14, 14), (8, 8)))
        shifted.validate(batch_size=1)
        self.assertFalse(torch.equal(shifted.masks[0], first))

    def test_reflect_translation_moves_interior_by_requested_pixels(self):
        original = torch.arange(28 * 28).view(1, 1, 28, 28).float()
        shifted = make_attacker()._apply_phase_shift(original, 8, 8)
        torch.testing.assert_close(shifted[..., 8:, 8:], original[..., :-8, :-8])
        self.assertEqual(float(shifted[0, 0, 0, 0]), float(original[0, 0, 8, 8]))

    def test_noise_is_rms_matched_per_image_with_stopped_scale(self):
        attacker = make_attacker()
        state = attacker.model.begin_progressive_forward(self.pixels.repeat(2, 1, 1, 1).requires_grad_(True))
        noise = attacker._strict_opponent_feature_noise(state)
        self.assertFalse(noise.requires_grad)
        actual = noise.square().mean((1, 2)).sqrt()
        expected = 0.2 * state.local_tokens.detach().square().mean((1, 2)).sqrt()
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.equal(noise[0], noise[1]))

    def test_noise_reaches_all_locals_once_and_preserves_both_prefixes(self):
        model = make_model(prefixes=2)
        attacker = ProgressiveRouteDisruptionAttacker(model, steps=1, input_diversity_groups=1, device=torch.device('cpu'))
        schedule = attacker._build_mask_schedule(self.pixels)
        self.assertEqual(schedule.checkpoints, ('block2', 'block9'))
        initial = model.begin_progressive_forward(attacker._normalize(self.pixels))
        with patch.object(attacker, '_strict_opponent_feature_noise', return_value=torch.ones_like(initial.local_tokens)) as noise, \
             patch.object(model, 'begin_progressive_forward', wraps=model.begin_progressive_forward) as begin:
            attacker._forward_with_schedule(self.pixels, self.labels, schedule)
        self.assertEqual(noise.call_count, 1)
        self.assertEqual(begin.call_count, 1)
        first = model.model.blocks[0].last_input
        torch.testing.assert_close(first[:, :2], initial.context['prefix_tokens'])
        torch.testing.assert_close(first[:, 2:], initial.local_tokens + 1)
        for mask, index in zip(schedule.masks, (2, 9)):
            after_drop = model.model.blocks[index].last_input[:, 2:]
            self.assertEqual(int(torch.count_nonzero(after_drop[mask])), 0)

    def test_original_only_keeps_fresh_schedules_without_phase(self):
        attacker = make_attacker(input_diversity_groups=2, input_diversity_views_per_group=1,
                                 input_diversity_phase_shift_set=())
        with patch.object(attacker, '_pick_input_diversity_phases', side_effect=AssertionError('phase called')), \
             patch.object(attacker, '_build_mask_schedule', wraps=attacker._build_mask_schedule) as schedules:
            attacker.attack_batch(self.pixels * 2 - 1, self.labels)
        self.assertEqual(schedules.call_count, 4)
        self.assertEqual(attacker._actual_forward_view_count, 2)

    def test_mean_pixel_gradient_is_used_directly(self):
        attacker = make_attacker(input_diversity_groups=2)
        def initialize():
            torch.manual_seed(SEED)
            attacker._actual_forward_view_count = 0
        initialize()
        pixels = self.pixels.clone().requires_grad_(True)
        expected = torch.stack([torch.autograd.grad(loss, pixels)[0]
                                for loss in attacker._iter_attack_losses(pixels, self.labels)]).mean(0)
        initialize()
        actual = attacker._attack_gradients(pixels, self.labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(actual.norm()), 0)

    def test_update_uses_momentum_and_projects_relative_to_clean_pixels(self):
        attacker = make_attacker(steps=2)
        gradients = [torch.full_like(self.pixels, 2), torch.full_like(self.pixels, -1)]
        def gradient(*args):
            attacker._actual_forward_view_count = 2
            return gradients.pop(0)
        images = self.pixels * 2 - 1
        with patch.object(attacker, '_attack_gradients', side_effect=gradient):
            adversarial = attacker.attack_batch(images, self.labels)
        pixels = attacker._denormalize(adversarial)
        torch.testing.assert_close(pixels, self.pixels + attacker.epsilon)
        self.assertLessEqual(float((pixels - self.pixels).abs().max()), attacker.epsilon + 1e-6)
        self.assertGreaterEqual(float(pixels.min()), 0)
        self.assertLessEqual(float(pixels.max()), 1)

    def test_seeded_attack_is_repeatable_for_fixed_batch_and_flow(self):
        images = (self.pixels * 2 - 1).repeat(2, 1, 1, 1)
        labels = self.labels.repeat(2)
        attacker = make_attacker()
        torch.manual_seed(SEED)
        first = attacker.attack_batch(images, labels)
        torch.manual_seed(SEED)
        repeated = attacker.attack_batch(images, labels)
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)

    def test_analysis_checkpoint_override_controls_traversal_depth(self):
        model = make_model()
        model.checkpoints = ('block1', 'block4')
        attacker = ProgressiveRouteDisruptionAttacker(
            model, steps=1, input_diversity_groups=1,
            drop_ratios=(10 / 196, 10 / 196), device=torch.device('cpu'),
        )
        schedule = attacker._build_mask_schedule(self.pixels)
        self.assertEqual(schedule.checkpoints, ('block1', 'block4'))
        self.assertEqual(schedule.counts, (10, 10))

    def test_analysis_drop_budgets_record_actual_integer_counts(self):
        settings = (
            (('block3', 'block10'), (5 / 196, 5 / 196), (5, 5)),
            (('block3', 'block10'), (20 / 196, 20 / 196), (20, 20)),
            (('block3', 'block6', 'block10'), (7 / 196, 6 / 196, 7 / 196), (7, 6, 7)),
        )
        for checkpoints, ratios, counts in settings:
            with self.subTest(checkpoints=checkpoints, counts=counts):
                model = make_model()
                model.checkpoints = checkpoints
                attacker = ProgressiveRouteDisruptionAttacker(
                    model, steps=1, input_diversity_groups=1,
                    drop_ratios=ratios, device=torch.device('cpu'),
                )
                schedule = attacker._build_mask_schedule(self.pixels)
                self.assertEqual(schedule.checkpoints, checkpoints)
                self.assertEqual(schedule.counts, counts)
                self.assertEqual(sum(schedule.counts), sum(counts))

    def test_zero_phase_shift_preserves_pixels_and_masks(self):
        attacker = make_attacker(input_diversity_phase_shift_set=((0, 0),))
        schedule = attacker._build_mask_schedule(self.pixels)
        shifted = attacker._phase_mask_schedule(
            schedule, [(0, 0)], height=self.pixels.size(-2), width=self.pixels.size(-1)
        )
        for original, transformed in zip(schedule.masks, shifted.masks):
            self.assertTrue(torch.equal(original, transformed))
        shifted_pixels = attacker._apply_samplewise_phase_shifts(self.pixels, [(0, 0)])
        self.assertTrue(torch.equal(shifted_pixels, self.pixels))

    def test_zero_shift_pair_keeps_two_independent_noisy_views_per_group(self):
        attacker = make_attacker(
            input_diversity_groups=2, input_diversity_phase_shift_set=((0, 0),)
        )
        pixels = self.pixels.clone().requires_grad_(True)
        with patch.object(attacker, '_build_mask_schedule',
                          wraps=attacker._build_mask_schedule) as schedules, \
             patch.object(attacker, '_strict_opponent_feature_noise',
                          wraps=attacker._strict_opponent_feature_noise) as noise:
            gradient = attacker._attack_gradients(pixels, self.labels)
        self.assertEqual(schedules.call_count, 2)
        self.assertEqual(noise.call_count, 4)
        self.assertEqual(attacker._actual_forward_view_count, 4)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_isotropic_rgb_noise_changes_directions_but_keeps_rms(self):
        opponent = make_attacker()
        isotropic = make_attacker(noise_structure='isotropic_rgb')
        self.assertEqual(opponent.noise_direction_variances, (0.5, 1.25, 1.25))
        self.assertEqual(isotropic.noise_direction_variances, (1.0, 1.0, 1.0))
        state = opponent.model.begin_progressive_forward(self.pixels.repeat(2, 1, 1, 1))
        torch.manual_seed(SEED)
        opponent_noise = opponent._strict_opponent_feature_noise(state)
        torch.manual_seed(SEED)
        isotropic_noise = isotropic._strict_opponent_feature_noise(state)
        self.assertFalse(torch.equal(opponent_noise, isotropic_noise))
        expected_rms = 0.2 * state.local_tokens.detach().square().mean((1, 2)).sqrt()
        for noise in (opponent_noise, isotropic_noise):
            actual_rms = noise.square().mean((1, 2)).sqrt()
            torch.testing.assert_close(actual_rms, expected_rms)
        params = isotropic.parameters()
        self.assertEqual(params['noise_structure'], 'isotropic_rgb')
        self.assertEqual(params['noise_direction_variances'], [1.0, 1.0, 1.0])

    def test_disabling_progressive_drop_skips_schedules_and_keeps_optimization(self):
        attacker = make_attacker(
            progressive_drop=False, opponent_noise_strength=0.0,
            input_diversity_groups=1, input_diversity_views_per_group=1,
        )
        pixels = self.pixels.clone().requires_grad_(True)
        with patch.object(attacker, '_build_mask_schedule',
                          side_effect=AssertionError('schedule must not be built')):
            gradient = attacker._attack_gradients(pixels, self.labels)
        self.assertEqual(attacker._actual_forward_view_count, 1)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)
        params = attacker.parameters()
        self.assertFalse(params['progressive_drop_enabled'])
        self.assertFalse(params['opponent_noise_enabled'])
        self.assertEqual(params['drop_counts'], [])

    def test_invalid_budgets_are_rejected(self):
        for settings in (dict(drop_ratios=(0.1,)), dict(input_diversity_views_per_group=3),
                         dict(input_diversity_groups=11), dict(input_diversity_phase_shift_set=())):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_attacker(**settings)


if __name__ == '__main__':
    unittest.main()
