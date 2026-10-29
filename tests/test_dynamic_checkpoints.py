import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nets import build_whitebox_model
from nets.base import SampleCheckpoint, select_variance_peaks
from progressive_attack import ProgressiveRouteDisruptionAttacker
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
             patch.object(attacker, '_build_mask_schedule', side_effect=capture):
            output = attacker.attack_batch(attacker._normalize_output(pixels), labels)
        select.assert_called_once()
        torch.testing.assert_close(select.call_args.args[0], attacker._normalize(pixels))
        self.assertEqual(len(schedules), 8)
        self.assertEqual({item.checkpoints for item in schedules}, {('block0',), ('block10',)})
        for schedule in schedules:
            self.assertEqual(schedule.counts, (20,))
        for checkpoint in ('block0', 'block10'):
            masks = [schedule.masks[0] for schedule in schedules if schedule.checkpoints == (checkpoint,)]
            self.assertFalse(torch.equal(masks[0], masks[1]))
        self.assertEqual(model.checkpoints, ('block3', 'block10'))
        self.assertIsNone(attacker._active_checkpoint)
        self.assertIsNone(attacker._loss_batch_size)
        self.assertLessEqual(float((attacker._denormalize(output) - pixels).abs().max()), attacker.epsilon + 1e-6)
        params = attacker.parameters()
        self.assertEqual(params['checkpoint_histogram'], {'block10': 2, 'block0': 1})
        self.assertEqual([item['sample_index'] for item in params['sample_checkpoints']], [0, 1, 2])
        self.assertEqual(params['checkpoint_mask_selection_count_per_image'], 4)
        self.assertNotIn('checkpoints', params)

    def test_group_losses_preserve_full_batch_mean_and_pixel_gradients(self):
        model = make_model()
        attacker = ProgressiveRouteDisruptionAttacker(model, opponent_noise_strength=0,
                                                      device=torch.device('cpu'))
        pixels = torch.linspace(.1, .9, 3 * 3 * 28 * 28).reshape(3, 3, 28, 28).requires_grad_(True)
        labels = torch.tensor([0, 1, 0])
        full = attacker._forward_with_schedule(pixels, labels, None)
        expected = torch.autograd.grad(full, pixels)[0]
        attacker._loss_batch_size = 3
        groups = [(0, 2), (1,)]
        actual = torch.zeros_like(pixels)
        losses = []
        for group in groups:
            indices = torch.tensor(group)
            part = pixels.detach().index_select(0, indices).requires_grad_(True)
            loss = attacker._forward_with_schedule(part, labels.index_select(0, indices), None)
            actual.index_copy_(0, indices, torch.autograd.grad(loss, part)[0])
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
        self.assertIsNone(attacker._active_checkpoint)
        self.assertIsNone(attacker._loss_batch_size)

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
        attacker._active_checkpoint = 'stage3_block3'
        self.assertEqual(attacker._build_mask_schedule(pixels).counts, (8,))


if __name__ == '__main__':
    unittest.main()
