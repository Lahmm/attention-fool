from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

import main
from tests.test_attack import make_model, make_checkpoint
from utils import SEED


class IndexedDataset(Dataset):
    samples = [{'image_name': f'image_{i}.png'} for i in range(4)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.zeros(3, 28, 28), 1, index


class MainTests(unittest.TestCase):
    def test_cli_preserves_view_count_ablation(self):
        for views in (1, 2):
            with patch.object(sys, 'argv', ['main.py', '--input-diversity-views-per-group', str(views)]):
                args = main.parse_args()
                self.assertEqual(args.input_diversity_views_per_group, views)
                self.assertEqual(args.output_dir, 'outputs/attack/gdi')
        self.assertEqual(SEED, 42)

    def test_mechanism_ablation_switches_are_available(self):
        with patch.object(sys, 'argv', ['main.py', '--no-gdi-drop',
                                       '--noise-strength', '0']):
            args = main.parse_args()
        self.assertTrue(args.no_gdi_drop)
        self.assertEqual(args.noise_strength, 0.0)


    def test_cli_uses_one_drop_ratio_and_records_augmentation(self):
        with patch.object(sys, 'argv', ['main.py', '--drop-ratio', '0.15', '--noise-strength', '0.3']):
            args = main.parse_args()
        self.assertEqual(args.drop_ratio, .15)
        self.assertEqual(args.noise_strength, 0.3)

    def test_removed_parameters_are_rejected(self):
        for option in ('--seed', '--attack-method', '--mi', '--no-mi', '--mi-decay',
                       '--checkpoints', '--gaussian-alpha',
                       '--vit-analysis-checkpoints', '--drop-ratios', '--noise-structure'):
            with self.subTest(option=option), redirect_stderr(StringIO()), \
                 patch.object(sys, 'argv', ['main.py', option]), self.assertRaises(SystemExit):
                main.parse_args()

    def test_main_records_fixed_seed_and_saves_exact_sample_limit(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(sys, 'argv', ['main.py', '--steps', '1', '--input-diversity-groups', '1',
                                      '--input-diversity-views-per-group', '1', '--max-attacked-samples', '2']):
            output = Path(directory)
            loader = DataLoader(IndexedDataset(), batch_size=4)
            with patch.object(main, 'validate_output_dir', return_value=output), \
                 patch.object(main, 'load_data', return_value=loader), \
                 patch.object(main, 'DEVICE', torch.device('cpu')), \
                 patch.object(main, 'build_whitebox_model', return_value=make_model()) as build:
                choices = tuple(make_checkpoint('block0', 'block1', .1, (14, 14)) for _ in range(2))
                build.return_value.select_contraction_checkpoints = lambda inputs: choices
                main.main(main.parse_args())
            params = json.loads((output / 'attack_params.json').read_text())
            self.assertEqual(params['attack_method'], 'geometry_guided_depth_intervention')
            self.assertTrue(params['gdi_drop_enabled'])
            self.assertEqual(params['seed'], 42)
            self.assertEqual(params['input_diversity_total_views'], 1)
            self.assertEqual(params['checkpoint_histogram'], {'block0': 2})
            self.assertNotIn('checkpoints', params)
            self.assertEqual(params['noise_structure'], 'isotropic_rgb')
            self.assertEqual(params['noise_strength'], 0.2)
            self.assertEqual({p.name for p in output.iterdir()},
                             {'adv_image_0.png', 'adv_image_1.png', 'attack_params.json'})

    def test_output_guard_rejects_repository_and_attack_root(self):
        for directory in ('.', 'outputs/attack', '/private/tmp'):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                main.validate_output_dir(directory)

    def test_dynamic_metadata_tracks_each_image_across_batches(self):
        model = make_model()
        choices = [(make_checkpoint('block0', 'block1', .1, (14, 14)),
                    make_checkpoint('block10', 'block11', .2, (14, 14))),
                   (make_checkpoint('block2', 'block3', .3, (14, 14)),)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(sys, 'argv', ['main.py', '--steps', '1', '--input-diversity-groups', '1',
                                      '--input-diversity-views-per-group', '1', '--max-attacked-samples', '3']):
            with patch.object(main, 'validate_output_dir', return_value=Path(directory)), \
                 patch.object(main, 'load_data', return_value=DataLoader(IndexedDataset(), batch_size=2)), \
                 patch.object(main, 'DEVICE', torch.device('cpu')), \
                 patch.object(main, 'build_whitebox_model', return_value=model), \
                 patch.object(model, 'select_contraction_checkpoints', side_effect=choices) as select:
                main.main(main.parse_args())
            self.assertEqual(select.call_count, 2)
            params = json.loads((Path(directory) / 'attack_params.json').read_text())
            self.assertEqual(params['num_images'], 3)
            self.assertEqual([item['image_name'] for item in params['sample_checkpoints']],
                             ['image_0.png', 'image_1.png', 'image_2.png'])
            self.assertEqual([item['checkpoint'] for item in params['sample_checkpoints']],
                             ['block0', 'block10', 'block2'])
            self.assertEqual(params['checkpoint_mask_selection_count_per_image'], 1)
            self.assertEqual(params['checkpoint_selection_metric'],
                             'global_direction_variance_ratio_times_positive_fractional_contraction')
            self.assertNotIn('experiment_only', params)
            for row in params['sample_checkpoints']:
                self.assertEqual(row['peak_contraction_score'],
                                 row['peak_direction_variance_ratio'] * row['contraction_fraction'])
                self.assertEqual(row['contraction_fraction'],
                                 1 - row['output_total_variance'] / row['input_total_variance'])
