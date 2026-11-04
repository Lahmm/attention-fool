import os
import unittest

import torch

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from nets.cait import CaiTAdapter
from nets.pit import PiTAdapter
from nets.vit import ViTAdapter
from progressive_attack import ProgressiveRouteDisruptionAttacker
from utils import SEED


class AdapterTests(unittest.TestCase):
    def test_sources_and_checkpoint_profiles(self):
        self.assertEqual(set(WHITEBOX_MODEL_CHOICES), {
            'vit_base_patch16_224', 'cait_s24_224', 'pit_ti_224',
            'deit_base_distilled_patch16_224',
        })
        self.assertEqual(ViTAdapter.drop_ratio, 0.102040816326)
        self.assertEqual(CaiTAdapter.drop_ratio, 0.15306122449)
        self.assertEqual(PiTAdapter.drop_ratio, 0.15306122449)
        with self.assertRaises(ValueError):
            build_whitebox_model('visformer_small')

    @unittest.skipUnless(os.environ.get('RUN_REAL_PRD_ADAPTERS') == '1',
                         'set RUN_REAL_PRD_ADAPTERS=1 for real-model CPU validation')
    def test_all_real_sources_preserve_forward_masks_and_input_gradients(self):
        torch.set_num_threads(2)
        for name in WHITEBOX_MODEL_CHOICES:
            with self.subTest(source=name):
                torch.manual_seed(SEED)
                model = build_whitebox_model(name, pretrained=False, device=torch.device('cpu'))
                pixels = torch.linspace(0.1, 0.9, 3 * 224 * 224).view(1, 3, 224, 224)
                with torch.no_grad():
                    initial = model.begin_progressive_forward(pixels)
                    native = model(pixels)
                    resumed = model.finish_progressive_forward(initial)
                    torch.testing.assert_close(resumed, native, rtol=0, atol=0)
                    if name == 'deit_base_distilled_patch16_224':
                        self.assertEqual(initial.context['prefix_tokens'].size(1), 2)
                attacker = ProgressiveRouteDisruptionAttacker(model, steps=1, input_diversity_groups=1,
                                                              device=torch.device('cpu'))
                choice = model.select_contraction_checkpoints(pixels)[0]
                attacker._prepare_dynamic_mask_layout((choice,), 1)
                expected_grid = (choice.grid_size,)
                count = max(1, int(round(choice.grid_size[0] * choice.grid_size[1] * attacker.drop_ratio)))
                expected_counts = (count,)
                with torch.no_grad():
                    boundary = model.advance_progressive_state(model.begin_progressive_forward(pixels), choice.checkpoint)
                    torch.testing.assert_close(model.finish_progressive_forward(boundary), native, rtol=0, atol=0)
                schedule = attacker._build_mask_schedule(pixels)
                schedule.validate(batch_size=1)
                self.assertEqual((schedule.grid_sizes, schedule.counts), (expected_grid, expected_counts))
                for item in schedule.selections:
                    self.assertEqual(model.checkpoint_grid_size(item.checkpoint), item.grid_size)
                state = model.begin_progressive_forward(pixels)
                for selection in schedule.selections:
                    state = model.advance_progressive_state(state, selection.checkpoint)
                    prefix_key = next((key for key in ('prefix_tokens', 'cls_token') if key in state.context), None)
                    prefix = state.context[prefix_key].clone() if prefix_key else None
                    masked = model.apply_progressive_mask(state, selection.mask)
                    self.assertEqual(int(torch.count_nonzero(masked.local_tokens[selection.mask])), 0)
                    if prefix_key:
                        torch.testing.assert_close(masked.context[prefix_key], prefix, rtol=0, atol=0)
                    state = masked
                del initial, native, resumed, state, masked
                probe = pixels.requires_grad_(True)
                loss = attacker._forward_with_schedule(probe, torch.tensor([1]), schedule)
                gradient = torch.autograd.grad(loss, probe)[0]
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.norm()), 0)
                print(f'{name}: grids={schedule.grid_sizes}, counts={schedule.counts}, finite gradients', flush=True)
                del model, attacker, probe, gradient, loss, pixels, schedule
