import argparse
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
from tests.test_progressive_attack import make_model
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
                self.assertEqual(main.parse_args().input_diversity_views_per_group, views)
        self.assertEqual(SEED, 42)

    def test_mechanism_ablation_switches_are_available(self):
        with patch.object(sys, 'argv', ['main.py', '--no-progressive-drop',
                                       '--opponent-noise-strength', '0']):
            args = main.parse_args()
        self.assertTrue(args.no_progressive_drop)
        self.assertEqual(args.opponent_noise_strength, 0.0)

    def test_analysis_cli_parses_checkpoints_and_noise_structure(self):
        with patch.object(sys, 'argv', ['main.py', '--vit-analysis-checkpoints',
                                       'block5,block8', '--drop-ratios', '0.05,0.05',
                                       '--noise-structure', 'isotropic_rgb']):
            args = main.parse_args()
        self.assertEqual(args.vit_analysis_checkpoints, ('block5', 'block8'))
        self.assertEqual(args.noise_structure, 'isotropic_rgb')
        self.assertEqual(main.parse_checkpoint_list('block01,block4'), ('block1', 'block4'))
        for value in ('', 'block3,block3', 'block5,block3', 'layer3,block4', 'block0,block4'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                main.parse_checkpoint_list(value)

    def test_analysis_checkpoint_override_validates_model_boundaries_and_ratios(self):
        model = make_model()
        main.apply_vit_analysis_checkpoints(model, ('block5', 'block8'), (0.05, 0.05))
        self.assertEqual(model.checkpoints, ('block5', 'block8'))
        with self.assertRaises(ValueError):
            main.apply_vit_analysis_checkpoints(model, ('block13',), (0.05,))
        with self.assertRaises(ValueError):
            main.apply_vit_analysis_checkpoints(model, ('block3',), None)
        with self.assertRaises(ValueError):
            main.apply_vit_analysis_checkpoints(make_model(prefixes=2), ('block3',), (0.05,))

    def test_removed_parameters_are_rejected(self):
        for option in ('--seed', '--attack-method', '--mi', '--no-mi', '--mi-decay',
                       '--opponent-noise-grad', '--checkpoints', '--gaussian-alpha'):
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
                 patch.object(main, 'build_whitebox_model', return_value=make_model()):
                main.main(main.parse_args())
            params = json.loads((output / 'attack_params.json').read_text())
            self.assertEqual(params['seed'], 42)
            self.assertEqual(params['input_diversity_total_views'], 1)
            self.assertEqual(params['checkpoints'], ['block3', 'block10'])
            self.assertEqual(params['noise_structure'], 'opponent')
            self.assertEqual(params['noise_direction_variances'], [0.5, 1.25, 1.25])
            self.assertNotIn('opponent_noise_grad', params)
            self.assertEqual({p.name for p in output.iterdir()},
                             {'adv_image_0.png', 'adv_image_1.png', 'attack_params.json'})

    def test_output_guard_rejects_repository_and_attack_root(self):
        for directory in ('.', 'outputs/attack', '/private/tmp'):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                main.validate_output_dir(directory)
