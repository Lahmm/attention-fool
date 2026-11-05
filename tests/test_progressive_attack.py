import unittest
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from nets.base import SampleCheckpoint
from nets.vit import ViTAdapter
from progressive_attack import ProgressiveMaskSelection, ProgressiveRouteDisruptionAttacker
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
    return model


def make_checkpoint(checkpoint, layer, variance, grid):
    return SampleCheckpoint(checkpoint, layer, variance, 2., 1., .5, 1., .5, grid)


def make_attacker(**overrides):
    options = dict(steps=2, input_diversity_groups=1, device=torch.device('cpu'))
    options.update(overrides)
    attacker = ProgressiveRouteDisruptionAttacker(make_model(), **options)
    attacker._prepare_dynamic_mask_layout((make_checkpoint('block10', 'block11', .1, (14, 14)),), 1)
    return attacker


def reflect_shift_reference(tensor, dx, dy):
    if dx == 0 and dy == 0:
        return tensor
    padded = F.pad(tensor, (max(0, dx), max(0, -dx), max(0, dy), max(0, -dy)), mode="reflect")
    return padded[..., max(0, -dy):max(0, -dy) + tensor.size(-2),
                  max(0, -dx):max(0, -dx) + tensor.size(-1)]


def validate_schedule(schedule, batch_size):
    assert schedule
    coverage = torch.zeros(batch_size, dtype=torch.long)
    for item in schedule:
        assert item.mask.shape == (batch_size, item.grid_size[0] * item.grid_size[1])
        assert item.mask.dtype == torch.bool
        expected = torch.full((batch_size,), item.count, device=item.mask.device, dtype=torch.long)
        if item.active_samples is not None:
            expected.masked_fill_(~item.active_samples, 0)
        torch.testing.assert_close(item.mask.sum(1), expected, rtol=0, atol=0)
        coverage += (expected > 0).cpu()
    torch.testing.assert_close(coverage, torch.ones_like(coverage), rtol=0, atol=0)


class ProgressiveAttackTests(unittest.TestCase):
    def setUp(self):
        self.pixels = torch.linspace(0.1, 0.9, 3 * 28 * 28).view(1, 3, 28, 28)
        self.labels = torch.tensor([1])

    def test_schedule_uses_current_vit_budget_and_fresh_masks(self):
        attacker = make_attacker()
        with patch.object(attacker.model, 'begin_progressive_forward', side_effect=AssertionError('mask forward')), \
             patch.object(torch, 'equal', side_effect=AssertionError('mask synchronization')):
            first = attacker._build_mask_schedule(self.pixels)
            second = attacker._build_mask_schedule(self.pixels)
        self.assertEqual(tuple(item.checkpoint for item in first), ('block10',))
        self.assertEqual(tuple(item.count for item in first), (20,))
        self.assertEqual(tuple(item.grid_size for item in first), ((14, 14),))
        validate_schedule(first, batch_size=1)
        self.assertFalse(torch.equal(first[0].mask, second[0].mask))


    def test_independent_views_preserve_each_native_grid_budget(self):
        attacker = make_attacker(drop_ratio=0.15306122449)
        for grid, count in (((14, 14), 30), ((7, 7), 8), ((27, 27), 112)):
            with self.subTest(grid=grid), \
                 patch.object(attacker.model, 'checkpoint_grid_size', return_value=grid):
                choice = make_checkpoint('block10', 'block11', .1, grid)
                attacker._prepare_dynamic_mask_layout((choice,), 1)
                first = attacker._build_mask_schedule(self.pixels)
                second = attacker._build_mask_schedule(self.pixels)
                for schedule in (first, second):
                    validate_schedule(schedule, batch_size=1)
                    self.assertEqual((schedule[0].grid_size, schedule[0].count), (grid, count))
                self.assertFalse(torch.equal(first[0].mask, second[0].mask))

    def test_phase_sampler_assigns_half_the_mass_to_identity(self):
        attacker = make_attacker()
        with patch('torch.randint', return_value=torch.arange(6)) as sample:
            phases = attacker._pick_input_diversity_phases(6)
        sample.assert_called_once_with(6, (6,), device='cpu')
        self.assertEqual(phases, [(0, 0), (0, 0), (0, 0), (4, 4), (8, 8), (12, 12)])

    def test_each_view_independently_selects_phase_and_uses_its_own_native_mask(self):
        attacker = make_attacker(input_diversity_groups=2)
        phases = [[(0, 0)], [(8, 8)], [(4, 4)], [(0, 0)]]
        schedules = []
        build = attacker._build_mask_schedule
        def capture(values):
            schedule = build(values)
            schedules.append(schedule)
            return schedule
        with patch.object(attacker, '_pick_input_diversity_phases', side_effect=phases) as sample, \
             patch.object(attacker, '_build_mask_schedule', side_effect=capture), \
             patch.object(attacker, '_forward_with_schedule', return_value=torch.tensor(0.)) as forward:
            losses = list(attacker._iter_attack_losses(self.pixels, self.labels))
        self.assertEqual(len(losses), 4)
        self.assertEqual(sample.call_count, 4)
        for index, call in enumerate(forward.call_args_list):
            expected = reflect_shift_reference(self.pixels, *phases[index][0])
            torch.testing.assert_close(call.args[0], expected, rtol=0, atol=0)
            self.assertIs(call.args[2], schedules[index])
            validate_schedule(schedules[index], batch_size=1)
        self.assertFalse(torch.equal(schedules[0][0].mask, schedules[1][0].mask))
        params = attacker.parameters()
        self.assertEqual(params['input_diversity_sampling'], 'independent_identity_or_phase_views')
        self.assertEqual(params['input_diversity_phase_probability'], .5)
        self.assertEqual(params['mask_schedule_count_per_image'], 8)

    def test_reflect_translation_moves_interior_by_requested_pixels(self):
        original = torch.arange(28 * 28).view(1, 1, 28, 28).float()
        shifted = make_attacker()._apply_samplewise_phase_shifts(original, [(8, 8)])
        torch.testing.assert_close(shifted[..., 8:, 8:], original[..., :-8, :-8])
        self.assertEqual(float(shifted[0, 0, 0, 0]), float(original[0, 0, 8, 8]))

    def test_batched_reflection_matches_individual_shifts_and_gradients(self):
        attacker = make_attacker()
        phases = [(0, 0), (8, 12), (-8, 4), (12, -12)]
        pixels = self.pixels.repeat(4, 1, 1, 1).requires_grad_(True)
        expected = torch.cat([reflect_shift_reference(pixels[i:i+1], *phase)
                              for i, phase in enumerate(phases)])
        actual = attacker._apply_samplewise_phase_shifts(pixels, phases)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.linspace(-1, 1, pixels.numel()).reshape_as(pixels)
        actual_grad = torch.autograd.grad((actual * weights).sum(), pixels)[0]
        expected_grad = torch.autograd.grad((expected * weights).sum(), pixels)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

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
        attacker._prepare_dynamic_mask_layout((make_checkpoint('block2', 'block3', .1, (14, 14)),), 1)
        schedule = attacker._build_mask_schedule(self.pixels)
        self.assertEqual(tuple(item.checkpoint for item in schedule), ('block2',))
        initial = model.begin_progressive_forward(attacker._normalize(self.pixels))
        with patch.object(attacker, '_strict_opponent_feature_noise', return_value=torch.ones_like(initial.local_tokens)) as noise, \
             patch.object(model, 'begin_progressive_forward', wraps=model.begin_progressive_forward) as begin:
            attacker._forward_with_schedule(self.pixels, self.labels, schedule)
        self.assertEqual(noise.call_count, 1)
        self.assertEqual(begin.call_count, 1)
        first = model.model.blocks[0].last_input
        torch.testing.assert_close(first[:, :2], initial.context['prefix_tokens'])
        torch.testing.assert_close(first[:, 2:], initial.local_tokens + 1)
        for selection, index in zip(schedule, (2,)):
            after_drop = model.model.blocks[index].last_input[:, 2:]
            self.assertEqual(int(torch.count_nonzero(after_drop[selection.mask])), 0)

    def test_original_only_keeps_fresh_schedules_without_phase(self):
        attacker = make_attacker(input_diversity_groups=2, input_diversity_views_per_group=1,
                                 input_diversity_phase_shift_set=())
        with patch.object(attacker, '_pick_input_diversity_phases', side_effect=AssertionError('phase called')), \
             patch.object(attacker, '_build_mask_schedule', wraps=attacker._build_mask_schedule) as schedules, \
             patch.object(attacker, '_forward_with_schedule', wraps=attacker._forward_with_schedule) as forwards:
            attacker.attack_batch(self.pixels * 2 - 1, self.labels)
        self.assertEqual(schedules.call_count, 4)
        self.assertEqual(forwards.call_count, 4)

    def test_mean_pixel_gradient_is_used_directly(self):
        attacker = make_attacker(input_diversity_groups=2)
        def initialize():
            torch.manual_seed(SEED)
        initialize()
        pixels = self.pixels.clone().requires_grad_(True)
        expected = torch.stack([torch.autograd.grad(loss, pixels)[0]
                                for loss in attacker._iter_attack_losses(pixels, self.labels)]).mean(0)
        initialize()
        actual = attacker._attack_gradients(pixels, self.labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float(actual.norm()), 0)

    def test_gradient_aggregation_rejects_missing_or_extra_views(self):
        attacker = make_attacker()
        pixels = self.pixels.clone().requires_grad_(True)
        for count in (0, 1, 3):
            with self.subTest(count=count), \
                 patch.object(attacker, '_iter_attack_losses', return_value=(pixels.sum() for _ in range(count))), \
                 self.assertRaisesRegex(RuntimeError, 'view count mismatch'):
                attacker._attack_gradients(pixels, self.labels)

    def test_update_uses_momentum_and_projects_relative_to_clean_pixels(self):
        attacker = make_attacker(steps=2)
        gradients = [torch.full_like(self.pixels, 2), torch.full_like(self.pixels, -1)]
        def gradient(*args):
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


    def test_zero_phase_preserves_loss_and_pixel_gradient(self):
        attacker = make_attacker(input_diversity_phase_shift_set=((0, 0),))
        schedule = attacker._build_mask_schedule(self.pixels)
        pixels = self.pixels.clone().requires_grad_(True)
        shifted = attacker._apply_samplewise_phase_shifts(pixels, [(0, 0)])
        torch.testing.assert_close(shifted, pixels, rtol=0, atol=0)
        torch.manual_seed(SEED)
        original_loss = attacker._forward_with_schedule(pixels, self.labels, schedule)
        original_gradient = torch.autograd.grad(original_loss, pixels)[0]
        torch.manual_seed(SEED)
        shifted_loss = attacker._forward_with_schedule(shifted, self.labels, schedule)
        shifted_gradient = torch.autograd.grad(shifted_loss, pixels)[0]
        torch.testing.assert_close(shifted_loss, original_loss, rtol=0, atol=0)
        torch.testing.assert_close(shifted_gradient, original_gradient, rtol=0, atol=0)

    def test_zero_phase_views_have_independent_masks_and_noise(self):
        attacker = make_attacker(input_diversity_groups=2, input_diversity_phase_shift_set=((0, 0),))
        pixels = self.pixels.clone().requires_grad_(True)
        schedules, noises = [], []
        build, sample_noise = attacker._build_mask_schedule, attacker._strict_opponent_feature_noise
        def capture_schedule(values):
            value = build(values)
            schedules.append(value)
            return value
        def capture_noise(state):
            value = sample_noise(state)
            noises.append(value)
            return value
        with patch.object(attacker, '_build_mask_schedule', side_effect=capture_schedule), \
             patch.object(attacker, '_strict_opponent_feature_noise', side_effect=capture_noise):
            gradient = attacker._attack_gradients(pixels, self.labels)
        self.assertEqual(len(schedules), 4)
        self.assertEqual(len(noises), 4)
        self.assertFalse(torch.equal(schedules[0][0].mask, schedules[1][0].mask))
        self.assertFalse(torch.equal(noises[0], noises[1]))
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
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)
        params = attacker.parameters()
        self.assertFalse(params['progressive_drop_enabled'])
        self.assertFalse(params['opponent_noise_enabled'])
        self.assertEqual(params['sample_checkpoints'], [])

    def test_invalid_budgets_are_rejected(self):
        for settings in (dict(drop_ratio=0), dict(drop_ratio=.6), dict(drop_ratio=float('nan')), dict(input_diversity_views_per_group=3),
                         dict(input_diversity_groups=11), dict(input_diversity_phase_shift_set=())):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                make_attacker(**settings)


if __name__ == '__main__':
    unittest.main()
