from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from main import ATTACK_METHODS, attack_all_samples, parse_args


class IndexedDataset(Dataset):
    def __init__(self):
        self.samples = [{"image_name": f"image_{index}.png"} for index in range(6)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.zeros(3, 2, 2), 0, index


class RecordingAttacker:
    def __init__(self):
        self.ids = []

    def attack_batch(self, images, _labels, replay=None, sample_ids=None):
        self.ids.extend(sample_ids or [])
        return images


class MainTests(unittest.TestCase):
    def test_progressive_is_the_only_attack_method(self):
        self.assertEqual(ATTACK_METHODS, ("progressive",))

    def test_cli_defaults_match_progressive_configuration(self):
        old_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.attack_method, "progressive")
        self.assertEqual(args.progressive_patch_selector, "high")
        self.assertEqual(args.input_diversity_views_per_group, 2)

    def test_cli_accepts_rank_transition_selector(self):
        old_argv = sys.argv
        sys.argv = [
            "main.py",
            "--progressive-patch-selector",
            "rank-transition",
        ]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.progressive_patch_selector, "rank-transition")

    def test_attack_sample_limit_is_exact(self):
        dataloader = DataLoader(IndexedDataset(), batch_size=4, shuffle=False)
        attacker = RecordingAttacker()
        with tempfile.TemporaryDirectory() as directory:
            ids = attack_all_samples(
                dataloader,
                attacker,
                Path(directory),
                max_attacked_samples=2,
                replay=object(),
            )
        self.assertEqual(ids, ["image_0.png", "image_1.png"])
        self.assertEqual(attacker.ids, ids)


if __name__ == "__main__":
    unittest.main()
