import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nets import build_whitebox_model
from nets.base import select_contraction_peaks, contraction_statistics
from progressive_attack import ProgressiveMaskSelection, ProgressiveRouteDisruptionAttacker
from tests.test_progressive_attack import make_model, make_checkpoint, reflect_shift_reference, validate_schedule


class DynamicCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_contraction_selects_each_image_independently_and_breaks_ties(self):
        reference = torch.tensor([[1., 0.], [1., 0.]])
        broad = torch.tensor([[[1., 0.], [-1., 0.]]]).expand(2, -1, -1)
        narrow = torch.tensor([[[.6, .8], [-.6, .8]]]).expand(2, -1, -1)
        early_out = torch.stack((narrow[0], broad[1]))
        late_out = torch.stack((broad[0], narrow[1]))
        observations = [('block1', 'block0', (1, 2), early_out, reference, broad),
                        ('block2', 'block1', (1, 2), late_out, reference, broad)]
        choices = select_contraction_peaks(observations)
        self.assertEqual([c.checkpoint for c in choices], ['block0', 'block1'])
        for choice in choices:
            self.assertAlmostEqual(choice.direction_variance_ratio, 2.)
            self.assertAlmostEqual(choice.contraction_fraction, .64)
            self.assertAlmostEqual(choice.contraction_score, 1.28)
        for i in range(2):
            individual = [(layer, cp, grid, local[i:i+1], ref[i:i+1], before[i:i+1])
                          for layer, cp, grid, local, ref, before in observations]
            self.assertEqual(select_contraction_peaks(individual)[0], choices[i])
        tied = [observations[0], ('block2', 'block1', *observations[0][2:])]
        self.assertEqual([c.checkpoint for c in select_contraction_peaks(tied)], ['block0', 'block0'])

    def test_expansion_zero_dispersion_and_all_zero_ties(self):
        reference = torch.tensor([[1., 0.]])
        broad = torch.tensor([[[1., 0.], [-1., 0.]]])
        narrow = torch.tensor([[[.6, .8], [-.6, .8]]])
        flat = broad[:, :1].expand_as(broad)
        for before, after in [(narrow, broad), (flat, narrow), (broad, flat)]:
            result = contraction_statistics(after, reference, before)
            self.assertEqual(result['contraction_score'].item(), 0.)
            self.assertTrue(torch.isfinite(result['contraction_fraction']).all())
        observations = [('first', 'block0', (1, 2), broad, reference, narrow),
                        ('second', 'block1', (1, 2), broad, reference, narrow)]
        self.assertEqual(select_contraction_peaks(observations)[0].checkpoint, 'block0')

    def test_geometry_matches_covariance_definition_and_rotation(self):
        torch.manual_seed(42)
        before, after, ref = torch.randn(2, 13, 5), torch.randn(2, 13, 5), torch.randn(2, 5)
        u = F.normalize(after.double(), dim=-1)
        ui = F.normalize(before.double(), dim=-1)
        g = F.normalize(ref.double(), dim=-1)
        c = u-u.mean(1, keepdim=True)
        ci = ui-ui.mean(1, keepdim=True)
        covariance = c.transpose(1, 2) @ c / 13
        prior = ci.transpose(1, 2) @ ci / 13
        total = covariance.diagonal(dim1=1, dim2=2).sum(1)
        input_total = prior.diagonal(dim1=1, dim2=2).sum(1)
        variance = torch.einsum('bi,bij,bj->b', g, covariance, g)
        expected = 5*variance/total*(1-total/input_total).clamp_min(0)
        actual = contraction_statistics(after, ref, before)
        torch.testing.assert_close(actual['contraction_score'], expected)
        rotation = torch.linalg.qr(torch.randn(5, 5)).Q
        transformed = contraction_statistics(after @ rotation, ref @ rotation, before @ rotation)
        torch.testing.assert_close(actual['contraction_score'], transformed['contraction_score'], rtol=2e-5, atol=2e-6)


    def test_selection_rejects_nonfinite_and_zero_representations(self):
        reference = torch.tensor([[1., 0.]])
        good = torch.ones(1, 3, 2)
        for invalid, message in ((float('nan'), 'non-finite'), (float('inf'), 'non-finite'), (0., 'nonzero')):
            for invalid_input in (True, False):
                bad = torch.full_like(good, invalid)
                local, before = (good, bad) if invalid_input else (bad, good)
                with self.subTest(value=invalid, input=invalid_input), self.assertRaisesRegex(ValueError, message):
                    select_contraction_peaks([('block1', 'block0', (1, 3), local, reference, before)])


    def test_vit_and_distilled_deit_use_final_normalized_classification_cls(self):
        pixels = torch.linspace(.1, .9, 2*3*28*28).reshape(2, 3, 28, 28)
        for prefixes in (1, 2):
            model = make_model(prefixes)
            model.model.norm = torch.nn.LayerNorm(4)
            with torch.no_grad():
                state = model.begin_progressive_forward(pixels)
                tokens = torch.cat((state.context['prefix_tokens'], state.local_tokens), dim=1)
                snapshots = []
                for block in model.model.blocks:
                    before = tokens[:, prefixes:].clone()
                    tokens = block(tokens)
                    snapshots.append((before, tokens[:, prefixes:].clone()))
                reference = model.model.norm(tokens)[:, 0]
            with patch('nets.vit.select_contraction_peaks', wraps=select_contraction_peaks) as scoring:
                rng = torch.get_rng_state().clone()
                choices = model.select_contraction_checkpoints(pixels)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            observations = scoring.call_args.args[0]
            for obs, (before, after) in zip(observations, snapshots):
                torch.testing.assert_close(obs[3], after, rtol=0, atol=0)
                torch.testing.assert_close(obs[4], reference, rtol=0, atol=0)
                torch.testing.assert_close(obs[5], before, rtol=0, atol=0)
            expected = select_contraction_peaks([(f'block{i+1}', f'block{i}', state.grid_size, after, reference, before)
                                                for i, (before, after) in enumerate(snapshots)])
            self.assertEqual(choices, expected)
            self.assertFalse(model.training)


    def test_cait_uses_raw_final_cls_after_class_attention_without_final_layernorm(self):
        torch.manual_seed(42)
        adapter = build_whitebox_model('cait_s24_224', pretrained=False, device=torch.device('cpu'))
        pixels = torch.linspace(.1, .9, 2*3*224*224).reshape(2, 3, 224, 224)
        with torch.no_grad():
            local = adapter.begin_progressive_forward(pixels).local_tokens
            snapshots = []
            for block in adapter.model.blocks:
                before = local.clone()
                local = block(local)
                snapshots.append((before, local.clone()))
            cls = adapter.model.cls_token.expand(2, -1, -1)
            for block in adapter.model.blocks_token_only:
                cls = block(local, cls)
        with patch('nets.cait.select_contraction_peaks', wraps=select_contraction_peaks) as scoring, \
             patch.object(adapter.model.norm, 'forward', side_effect=AssertionError('final LayerNorm during scoring')):
            choices = adapter.select_contraction_checkpoints(pixels)
        observations = scoring.call_args.args[0]
        self.assertEqual(len(observations), 24)
        for i, (obs, (before, after)) in enumerate(zip(observations, snapshots)):
            self.assertEqual(obs[:2], (f'block{i+1}', f'block{i}'))
            torch.testing.assert_close(obs[3], after, rtol=0, atol=0)
            torch.testing.assert_close(obs[4], cls[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(obs[5], before, rtol=0, atol=0)
        expected = select_contraction_peaks([(f'block{i+1}', f'block{i}', adapter.model.patch_embed.grid_size,
                                             after, cls[:, 0], before) for i, (before, after) in enumerate(snapshots)])
        self.assertEqual(choices, expected)
        attacker = ProgressiveRouteDisruptionAttacker(adapter, device=torch.device('cpu'))
        params = attacker.parameters()
        self.assertEqual(params['patch_score_reference'], adapter.checkpoint_score_reference)
        self.assertEqual(params['checkpoint_selection_metric'], 'global_direction_variance_ratio_times_positive_fractional_contraction')
        self.assertNotIn('experiment_only', params)


    def test_mixed_batch_selects_once_and_keeps_boundaries_across_steps_and_phase_views(self):
        model = make_model()
        pixels = torch.linspace(.1, .9, 3 * 3 * 28 * 28).reshape(3, 3, 28, 28)
        labels = torch.tensor([0, 1, 0])
        choices = (make_checkpoint('block10', 'block11', .3, (14, 14)),
                   make_checkpoint('block0', 'block1', .4, (14, 14)),
                   make_checkpoint('block10', 'block11', .2, (14, 14)))
        attacker = ProgressiveRouteDisruptionAttacker(model, steps=2, input_diversity_groups=2,
                                                      device=torch.device('cpu'))
        schedules = []
        build = attacker._build_mask_schedule
        def capture(values):
            schedule = build(values)
            schedules.append(schedule)
            return schedule
        with patch.object(model, 'select_contraction_checkpoints', return_value=choices) as select, \
             patch.object(attacker, '_build_mask_schedule', side_effect=capture), \
             patch.object(model, 'begin_progressive_forward', wraps=model.begin_progressive_forward) as begin:
            output = attacker.attack_batch(attacker._normalize_output(pixels), labels)
        select.assert_called_once()
        torch.testing.assert_close(select.call_args.args[0], attacker._normalize(pixels))
        self.assertEqual(begin.call_count, 8)
        self.assertTrue(all(call.args[0].size(0) == 3 for call in begin.call_args_list))
        self.assertEqual(len(schedules), 4)
        self.assertEqual({tuple(selection.checkpoint for selection in item) for item in schedules}, {('block0', 'block10')})
        for schedule in schedules:
            self.assertEqual(tuple(item.count for item in schedule), (20, 20))
            validate_schedule(schedule, batch_size=3)
            self.assertEqual(schedule[0].mask.sum(dim=1).tolist(), [0, 20, 0])
            self.assertEqual(schedule[1].mask.sum(dim=1).tolist(), [20, 0, 20])
        for boundary in (0, 1):
            masks = [schedule[boundary].mask for schedule in schedules]
            self.assertFalse(torch.equal(masks[0], masks[1]))
        self.assertIsNone(attacker._dynamic_mask_layout)
        self.assertLessEqual(float((attacker._denormalize(output) - pixels).abs().max()), attacker.epsilon + 1e-6)
        params = attacker.parameters()
        self.assertEqual(params['checkpoint_histogram'], {'block10': 2, 'block0': 1})
        self.assertEqual([item['sample_index'] for item in params['sample_checkpoints']], [0, 1, 2])
        self.assertEqual(params['checkpoint_mask_selection_count_per_image'], 4)
        self.assertNotIn('checkpoints', params)

    def test_heterogeneous_batch_matches_individual_losses_gradients_noise_and_phase_masks(self):
        model = make_model()
        attacker = ProgressiveRouteDisruptionAttacker(model, device=torch.device('cpu'))
        pixels = torch.linspace(.1, .9, 3 * 3 * 28 * 28).reshape(3, 3, 28, 28).requires_grad_(True)
        labels = torch.tensor([0, 1, 0])
        choices = (make_checkpoint('block10', 'block11', .3, (14, 14)),
                   make_checkpoint('block0', 'block1', .4, (14, 14)),
                   make_checkpoint('block10', 'block11', .2, (14, 14)))
        attacker._prepare_dynamic_mask_layout(choices, 3)
        with patch.object(model, 'begin_progressive_forward', side_effect=AssertionError('mask forward')):
            schedule = attacker._build_mask_schedule(pixels)
        phases = [(4, 4), (-8, 8), (12, -12)]
        shifted = attacker._phase_mask_schedule(schedule, phases, height=28, width=28)
        validate_schedule(shifted, batch_size=3)
        for original, paired in zip(schedule, shifted):
            torch.testing.assert_close(paired.mask.sum(dim=1), original.mask.sum(dim=1))
        with patch.object(attacker, '_projected_gaussian_noise',
                          side_effect=lambda state: .01 * state.local_tokens.detach().sin()):
            for is_phase, combined in ((False, schedule), (True, shifted)):
                view = attacker._apply_samplewise_phase_shifts(pixels, phases) if is_phase else pixels
                full = attacker._forward_with_schedule(view, labels, combined)
                expected = torch.autograd.grad(full, pixels)[0]
                actual = torch.zeros_like(pixels)
                losses = []
                for index, choice in enumerate(choices):
                    selected = next(item for item in combined if item.checkpoint == choice.checkpoint)
                    individual = (ProgressiveMaskSelection(
                        selected.checkpoint, selected.mask[index:index+1], selected.count, selected.grid_size,
                    ),)
                    part = pixels[index:index+1].detach().requires_grad_(True)
                    part_view = reflect_shift_reference(part, *phases[index]) if is_phase else part
                    loss = attacker._forward_with_schedule(part_view, labels[index:index+1], individual) / 3
                    actual[index:index+1] = torch.autograd.grad(loss, part)[0]
                    losses.append(loss.detach())
                torch.testing.assert_close(sum(losses), full.detach())
                torch.testing.assert_close(actual, expected)

    def test_drop_off_skips_clean_selection_and_exception_clears_active_boundary(self):
        model = make_model()
        pixels = torch.linspace(.1, .9, 3 * 28 * 28).reshape(1, 3, 28, 28)
        attacker = ProgressiveRouteDisruptionAttacker(model, progressive_drop=False, drop_ratio=.1,
                                                      steps=1, input_diversity_groups=1, device=torch.device('cpu'))
        with patch.object(model, 'select_contraction_checkpoints', side_effect=AssertionError('selection called')):
            attacker.attack_batch(attacker._normalize_output(pixels), torch.tensor([0]))
        self.assertFalse(attacker.parameters()['checkpoint_selection_enabled'])
        self.assertEqual(attacker.parameters()['mask_schedule_count_per_image'], 0)
        attacker = ProgressiveRouteDisruptionAttacker(model, steps=1, input_diversity_groups=1,
                                                      device=torch.device('cpu'))
        choice = make_checkpoint('block0', 'block1', .1, (14, 14))
        with patch.object(model, 'select_contraction_checkpoints', return_value=(choice,)), \
             patch.object(attacker, '_attack_gradients', side_effect=RuntimeError('failure')):
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                attacker.attack_batch(attacker._normalize_output(pixels), torch.tensor([0]))
        self.assertIsNone(attacker._dynamic_mask_layout)

    def test_pit_stage_entry_drops_after_pooling_and_norm_and_preserves_cls(self):
        torch.manual_seed(42)
        model = build_whitebox_model('pit_ti_224', pretrained=False, device=torch.device('cpu'))
        pixels = torch.linspace(.1, .9, 3 * 224 * 224).reshape(1, 3, 224, 224)
        with torch.no_grad():
            native = model(pixels)
            for index, stage in enumerate(model.model.transformers):
                boundary = f'stage{index + 1}_block0'
                state = model.advance_progressive_state(model.begin_progressive_forward(pixels), boundary)
                torch.testing.assert_close(model.finish_progressive_forward(state), native, rtol=0, atol=0)
                self.assertTrue(state.context['stage_prepared'])
                mask = torch.zeros(state.local_tokens.shape[:2], dtype=torch.bool)
                mask[:, :3] = True
                cls = state.context['cls_token'].clone()
                captured = []
                handle = stage.blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
                try:
                    model.finish_progressive_forward(model.apply_progressive_mask(state, mask))
                finally:
                    handle.remove()
                self.assertEqual(int(captured[0][:, 1:][mask].count_nonzero()), 0)
                torch.testing.assert_close(captured[0][:, :1], cls, rtol=0, atol=0)
        attacker = ProgressiveRouteDisruptionAttacker(model, device=torch.device('cpu'))
        attacker._prepare_dynamic_mask_layout((make_checkpoint('stage3_block3', 'stage3_block4', .1, (7, 7)),), 1)
        self.assertEqual(tuple(item.count for item in attacker._build_mask_schedule(pixels)), (8,))


if __name__ == '__main__':
    unittest.main()
