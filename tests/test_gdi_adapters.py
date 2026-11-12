import os
import unittest
from unittest.mock import patch

import torch

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from nets.cait import CaiTAdapter
from nets.pit import PiTAdapter
from nets.vit import ViTAdapter
from attack import GDIAttacker
from utils import SEED
from tests.test_attack import validate_schedule


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

    @unittest.skipUnless(os.environ.get('RUN_REAL_GDI_ADAPTERS') == '1',
                         'set RUN_REAL_GDI_ADAPTERS=1 for real-model CPU validation')
    def test_all_real_sources_preserve_forward_masks_and_input_gradients(self):
        torch.set_num_threads(2)
        for name in WHITEBOX_MODEL_CHOICES:
            with self.subTest(source=name):
                torch.manual_seed(SEED)
                model = build_whitebox_model(name, pretrained=False, device=torch.device('cpu'))
                pixels = torch.linspace(0.1, 0.9, 3 * 224 * 224).view(1, 3, 224, 224)
                with torch.no_grad():
                    initial = model.begin_gdi_forward(pixels)
                    native = model(pixels)
                    resumed = model.finish_gdi_forward(initial)
                    torch.testing.assert_close(resumed, native, rtol=0, atol=0)
                    if name == 'deit_base_distilled_patch16_224':
                        self.assertEqual(initial.context['prefix_tokens'].size(1), 2)
                attacker = GDIAttacker(model, steps=1, input_diversity_groups=1,
                                                              device=torch.device('cpu'))
                choice = model.select_contraction_checkpoints(pixels)[0]
                attacker._prepare_dynamic_mask_layout((choice,), 1)
                expected_grid = (choice.grid_size,)
                count = max(1, int(round(choice.grid_size[0] * choice.grid_size[1] * attacker.drop_ratio)))
                expected_counts = (count,)
                with torch.no_grad():
                    boundary = model.advance_gdi_state(model.begin_gdi_forward(pixels), choice.checkpoint)
                    torch.testing.assert_close(model.finish_gdi_forward(boundary), native, rtol=0, atol=0)
                schedule = attacker._build_mask_schedule(pixels)
                validate_schedule(schedule, batch_size=1)
                grids = tuple(item.grid_size for item in schedule)
                counts = tuple(item.count for item in schedule)
                self.assertEqual((grids, counts), (expected_grid, expected_counts))
                for item in schedule:
                    self.assertEqual(model.checkpoint_grid_size(item.checkpoint), item.grid_size)
                state = model.begin_gdi_forward(pixels)
                for selection in schedule:
                    state = model.advance_gdi_state(state, selection.checkpoint)
                    prefix_key = next((key for key in ('prefix_tokens', 'cls_token') if key in state.context), None)
                    prefix = state.context[prefix_key].clone() if prefix_key else None
                    masked = model.apply_gdi_mask(state, selection.mask)
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
                print(f'{name}: grids={grids}, counts={counts}, finite gradients', flush=True)
                del model, attacker, probe, gradient, loss, pixels, schedule

    @unittest.skipUnless(os.environ.get('RUN_REAL_GDI_ADAPTERS') == '1',
                         'set RUN_REAL_GDI_ADAPTERS=1 for real-model CPU validation')
    def test_remaining_sources_fixed_sweep_uses_every_native_input_and_grid(self):
        torch.set_num_threads(2)
        for name, count in (('cait_s24_224', 24), ('pit_ti_224', 12),
                            ('deit_base_distilled_patch16_224', 12)):
            torch.manual_seed(SEED)
            model = build_whitebox_model(name, pretrained=False, device=torch.device('cpu'))
            expected = (tuple(f'stage{s + 1}_block{b}' for s, depth in enumerate((2, 6, 4))
                              for b in range(depth)) if name == 'pit_ti_224'
                        else tuple(f'block{b}' for b in range(count)))
            self.assertEqual(model.block_input_checkpoints, expected)
            pixels = torch.linspace(.1, .9, 3 * 224 * 224).view(1, 3, 224, 224)
            for number, checkpoint in enumerate(expected, 1):
                with self.subTest(source=name, block=number):
                    attacker = GDIAttacker(model, fixed_block=number, steps=1, input_diversity_groups=1,
                                           device=torch.device('cpu'))
                    grid = ((27, 27) if number <= 2 else (14, 14) if number <= 8 else (7, 7)) if name == 'pit_ti_224' else (14, 14)
                    self.assertEqual(model.checkpoint_grid_size(checkpoint), grid)
                    drop_count = ((112 if number <= 2 else 30 if number <= 8 else 8)
                                  if name == 'pit_ti_224' else 30 if name == 'cait_s24_224' else 20)
                    block = (model.model.transformers[model.checkpoint_order(checkpoint)[0]].blocks[model.checkpoint_order(checkpoint)[1]]
                             if name == 'pit_ti_224' else model.model.blocks[number - 1])
                    masked_inputs, captured = [], []
                    apply_mask = model.apply_gdi_mask
                    def capture_mask(state, mask):
                        masked = apply_mask(state, mask)
                        self.assertEqual(int(masked.local_tokens[mask].count_nonzero()), 0)
                        self.assertEqual(mask.sum(1).tolist(), [drop_count])
                        prefix = next((masked.context[key] for key in ('prefix_tokens', 'cls_token')
                                       if key in masked.context), None)
                        tokens = torch.cat((prefix, masked.local_tokens), 1) if prefix is not None else masked.local_tokens
                        masked_inputs.append(tokens.detach().clone())
                        return masked
                    def capture_input(module, args):
                        torch.testing.assert_close(args[0], masked_inputs[len(captured)], rtol=0, atol=0)
                        captured.append(True)
                    handle = block.register_forward_pre_hook(capture_input)
                    try:
                        with patch.object(model, 'select_contraction_checkpoints', side_effect=AssertionError('selection called')), \
                             patch.object(model, 'apply_gdi_mask', side_effect=capture_mask):
                            output = attacker.attack_batch(attacker._normalize_output(pixels), torch.tensor([1]))
                    finally:
                        handle.remove()
                    self.assertEqual(len(captured), 2)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertGreater(float((output - attacker._normalize_output(pixels)).abs().max()), 0)
                    self.assertLessEqual(float((attacker._denormalize(output) - pixels).abs().max()), attacker.epsilon + 1e-6)
                    params = attacker.parameters()
                    self.assertEqual(params['fixed_checkpoint'], checkpoint)
                    self.assertEqual(params['checkpoint_histogram'], {checkpoint: 1})
                    self.assertEqual(params['sample_checkpoints'][0]['drop_count'], drop_count)
            with self.assertRaises(ValueError):
                GDIAttacker(model, fixed_block=count + 1, device=torch.device('cpu'))
            print(f'{name}: all {count} fixed block inputs, masks, grids and gradients verified', flush=True)
