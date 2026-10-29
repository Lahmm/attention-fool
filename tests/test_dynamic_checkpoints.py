import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nets import build_whitebox_model
from nets.base import SampleCheckpoint, select_variance_peaks
from progressive_attack import ProgressiveMaskSchedule, ProgressiveMaskSelection, ProgressiveRouteDisruptionAttacker
from tests.test_progressive_attack import make_model


class DynamicCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_population_variance_selects_each_image_independently_and_breaks_ties_by_depth(self):
        reference = torch.tensor([[1., 0.], [1., 0.]])
        early = torch.tensor([[[1., 0.], [-1., 0.], [1., 0.]],
                              [[1., 0.], [1., 0.], [1., 0.]]])
        late = torch.tensor([[[1., 0.], [1., 0.], [1., 0.]],
                             [[1., 0.], [0., 1.], [-1., 0.]]])
        observations = [('block1', 'block0', (1, 3), early, reference),
                        ('block2', 'block1', (1, 3), late, reference)]
        choices = select_variance_peaks(observations)
        self.assertEqual([item.checkpoint for item in choices], ['block0', 'block1'])
        self.assertAlmostEqual(choices[0].variance, 8 / 9)
        self.assertAlmostEqual(choices[1].variance, 2 / 3)
        tied = select_variance_peaks([observations[0], ('block2', 'block1', (1, 3), early, reference)])
        self.assertEqual([item.checkpoint for item in tied], ['block0', 'block0'])

    def test_selection_rejects_nonfinite_and_zero_representations(self):
        reference = torch.tensor([[1., 0.]])
        for invalid, message in ((float('nan'), 'non-finite'), (float('inf'), 'non-finite'), (0., 'nonzero')):
            local = torch.full((1, 3, 2), invalid)
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, message):
                select_variance_peaks([('block1', 'block0', (1, 3), local, reference)])

    def test_vit_and_distilled_deit_use_final_normalized_classification_cls(self):
        pixels = torch.linspace(.1, .9, 2 * 3 * 28 * 28).reshape(2, 3, 28, 28)
        for prefixes in (1, 2):
            model = make_model(prefixes)
            model.model.norm = torch.nn.LayerNorm(4)
            with torch.no_grad():
                state = model.begin_progressive_forward(pixels)
                tokens = torch.cat((state.context['prefix_tokens'], state.local_tokens), dim=1)
                snapshots = []
                for block in model.model.blocks:
                    tokens = block(tokens)
                    snapshots.append(tokens[:, prefixes:].clone())
                cls = model.model.norm(tokens)[:, :1]
                variances = torch.stack([F.cosine_similarity(local, cls, dim=-1).double().var(dim=1, unbiased=False)
                                         for local in snapshots])
            rng = torch.get_rng_state().clone()
            choices = model.select_variance_checkpoints(pixels)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for sample, choice in enumerate(choices):
                peak = int(variances[:, sample].argmax())
                self.assertEqual(choice.checkpoint, f'block{peak}')
                self.assertAlmostEqual(choice.variance, float(variances[peak, sample]))
            self.assertFalse(model.training)

    def test_cait_uses_raw_final_cls_after_class_attention_without_final_layernorm(self):
        torch.manual_seed(42)
        adapter = build_whitebox_model('cait_s24_224', pretrained=False, device=torch.device('cpu'))
        pixels = torch.linspace(.1, .9, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
        with torch.no_grad():
            local = adapter.begin_progressive_forward(pixels).local_tokens
            snapshots = []
            for block in adapter.model.blocks:
                local = block(local)
                snapshots.append(local.clone())
            cls = adapter.model.cls_token.expand(2, -1, -1)
            for block in adapter.model.blocks_token_only:
                cls = block(local, cls)
        with patch('nets.cait.select_variance_peaks', wraps=select_variance_peaks) as scoring, \
             patch.object(adapter.model.norm, 'forward', side_effect=AssertionError('final LayerNorm during scoring')):
            choices = adapter.select_variance_checkpoints(pixels)
        observations = scoring.call_args.args[0]
        self.assertEqual(len(observations), 24)
        for index, observation in enumerate(observations):
            self.assertEqual(observation[:2], (f'block{index+1}', f'block{index}'))
            torch.testing.assert_close(observation[3], snapshots[index], rtol=0, atol=0)
            torch.testing.assert_close(observation[4], cls[:, 0], rtol=0, atol=0)
        variances = torch.stack([F.cosine_similarity(local, cls, dim=-1, eps=1e-8).clamp(-1, 1)
                                 .double().var(dim=1, unbiased=False) for local in snapshots])
        for sample, choice in enumerate(choices):
            peak = int(variances[:, sample].argmax())
            self.assertEqual(choice.checkpoint, f'block{peak}')
            self.assertAlmostEqual(choice.variance, float(variances[peak, sample]))
        attacker = ProgressiveRouteDisruptionAttacker(adapter, device=torch.device('cpu'))
        self.assertEqual(attacker.parameters()['patch_score_reference'], adapter.checkpoint_score_reference)

    def test_mixed_batch_selects_once_and_keeps_boundaries_across_steps_and_phase_views(self):
        model = make_model()
        model.dynamic_checkpoint = True
        pixels = torch.linspace(.1, .9, 3 * 3 * 28 * 28).reshape(3, 3, 28, 28)
        labels = torch.tensor([0, 1, 0])
        choices = (SampleCheckpoint('block10', 'block11', .3, (14, 14)),
                   SampleCheckpoint('block0', 'block1', .4, (14, 14)),
                   SampleCheckpoint('block10', 'block11', .2, (14, 14)))
        attacker = ProgressiveRouteDisruptionAttacker(model, steps=2, input_diversity_groups=2,
                                                      device=torch.device('cpu'))
        schedules = []
        build = attacker._build_mask_schedule
        def capture(values):
            schedule = build(values)
            schedules.append(schedule)
            return schedule
        with patch.object(model, 'select_variance_checkpoints', return_value=choices) as select, \
             patch.object(attacker, '_build_mask_schedule', side_effect=capture), \
             patch.object(model, 'begin_progressive_forward', wraps=model.begin_progressive_forward) as begin:
            output = attacker.attack_batch(attacker._normalize_output(pixels), labels)
        select.assert_called_once()
        torch.testing.assert_close(select.call_args.args[0], attacker._normalize(pixels))
        self.assertEqual(begin.call_count, 8)
        self.assertTrue(all(call.args[0].size(0) == 3 for call in begin.call_args_list))
        self.assertEqual(len(schedules), 4)
        self.assertEqual({item.checkpoints for item in schedules}, {('block0', 'block10')})
        for schedule in schedules:
            self.assertEqual(schedule.counts, (20, 20))
            schedule.validate(batch_size=3)
            self.assertEqual(schedule.masks[0].sum(dim=1).tolist(), [0, 20, 0])
            self.assertEqual(schedule.masks[1].sum(dim=1).tolist(), [20, 0, 20])
        for boundary in (0, 1):
            masks = [schedule.masks[boundary] for schedule in schedules]
            self.assertFalse(torch.equal(masks[0], masks[1]))
        self.assertEqual(model.checkpoints, ('block3', 'block10'))
        self.assertIsNone(attacker._dynamic_mask_layout)
        self.assertLessEqual(float((attacker._denormalize(output) - pixels).abs().max()), attacker.epsilon + 1e-6)
        params = attacker.parameters()
        self.assertEqual(params['checkpoint_histogram'], {'block10': 2, 'block0': 1})
        self.assertEqual([item['sample_index'] for item in params['sample_checkpoints']], [0, 1, 2])
        self.assertEqual(params['checkpoint_mask_selection_count_per_image'], 4)
        self.assertNotIn('checkpoints', params)

    def test_heterogeneous_batch_matches_individual_losses_gradients_noise_and_phase_masks(self):
        model = make_model()
        model.dynamic_checkpoint = True
        attacker = ProgressiveRouteDisruptionAttacker(model, device=torch.device('cpu'))
        pixels = torch.linspace(.1, .9, 3 * 3 * 28 * 28).reshape(3, 3, 28, 28).requires_grad_(True)
        labels = torch.tensor([0, 1, 0])
        choices = (SampleCheckpoint('block10', 'block11', .3, (14, 14)),
                   SampleCheckpoint('block0', 'block1', .4, (14, 14)),
                   SampleCheckpoint('block10', 'block11', .2, (14, 14)))
        attacker._prepare_dynamic_mask_layout(choices, 3)
        with patch.object(model, 'begin_progressive_forward', side_effect=AssertionError('mask forward')):
            schedule = attacker._build_mask_schedule(pixels)
        phases = [(4, 4), (-8, 8), (12, -12)]
        shifted = attacker._phase_mask_schedule(schedule, phases, height=28, width=28)
        shifted.validate(batch_size=3)
        for original, paired in zip(schedule.selections, shifted.selections):
            torch.testing.assert_close(paired.mask.sum(dim=1), original.mask.sum(dim=1))
        with patch.object(attacker, '_strict_opponent_feature_noise',
                          side_effect=lambda state: .01 * state.local_tokens.detach().sin()):
            for is_phase, combined in ((False, schedule), (True, shifted)):
                view = attacker._apply_samplewise_phase_shifts(pixels, phases) if is_phase else pixels
                full = attacker._forward_with_schedule(view, labels, combined)
                expected = torch.autograd.grad(full, pixels)[0]
                actual = torch.zeros_like(pixels)
                losses = []
                for index, choice in enumerate(choices):
                    selected = next(item for item in combined.selections if item.checkpoint == choice.checkpoint)
                    individual = ProgressiveMaskSchedule((ProgressiveMaskSelection(
                        selected.checkpoint, selected.mask[index:index+1], selected.count, selected.grid_size,
                    ),))
                    part = pixels[index:index+1].detach().requires_grad_(True)
                    part_view = attacker._apply_phase_shift(part, *phases[index]) if is_phase else part
                    loss = attacker._forward_with_schedule(part_view, labels[index:index+1], individual) / 3
                    actual[index:index+1] = torch.autograd.grad(loss, part)[0]
                    losses.append(loss.detach())
                torch.testing.assert_close(sum(losses), full.detach())
                torch.testing.assert_close(actual, expected)

    def test_drop_off_skips_clean_selection_and_exception_clears_active_boundary(self):
        model = make_model()
        model.dynamic_checkpoint = True
        pixels = torch.linspace(.1, .9, 3 * 28 * 28).reshape(1, 3, 28, 28)
        attacker = ProgressiveRouteDisruptionAttacker(model, progressive_drop=False, drop_ratios=(.1,),
                                                      steps=1, input_diversity_groups=1, device=torch.device('cpu'))
        with patch.object(model, 'select_variance_checkpoints', side_effect=AssertionError('selection called')):
            attacker.attack_batch(attacker._normalize_output(pixels), torch.tensor([0]))
        self.assertFalse(attacker.parameters()['checkpoint_selection_enabled'])
        self.assertEqual(attacker.parameters()['mask_schedule_count_per_image'], 0)
        attacker = ProgressiveRouteDisruptionAttacker(model, steps=1, input_diversity_groups=1,
                                                      device=torch.device('cpu'))
        choice = SampleCheckpoint('block0', 'block1', .1, (14, 14))
        with patch.object(model, 'select_variance_checkpoints', return_value=(choice,)), \
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
        attacker._prepare_dynamic_mask_layout((SampleCheckpoint('stage3_block3', 'stage3_block4', .1, (7, 7)),), 1)
        self.assertEqual(attacker._build_mask_schedule(pixels).counts, (8,))


if __name__ == '__main__':
    unittest.main()
