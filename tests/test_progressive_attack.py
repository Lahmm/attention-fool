import ast
from pathlib import Path
import unittest

import torch

from gradient_replay import GradientReplay
from progressive_attack import ProgressivePatchScoreAttacker
from tests.vit_progressive_patch_score_attack_cases import (
    TinyViTWrapper,
    ViTProgressivePatchScoreAttacker,
)


class ProgressiveIndependenceTests(unittest.TestCase):
    def make_pair(self, **overrides):
        torch.manual_seed(7)
        legacy_model = TinyViTWrapper()
        new_model = TinyViTWrapper()
        new_model.load_state_dict(legacy_model.state_dict())
        common = {
            "checkpoints": (3, 6, 9),
            "drop_ratios": (0.25, 0.25, 0.25),
            "score_cls_noise_strength": 0.2,
            "opponent_noise_strength": 0.2,
            "steps": 1,
            "input_diversity_groups": 1,
            "input_diversity_views_per_group": 2,
            "input_diversity_phase_shift_set": ((0, 0),),
            "use_momentum": False,
            "gaussian_alpha": 0.0,
            "device": torch.device("cpu"),
        }
        common.update(overrides)
        return (
            ViTProgressivePatchScoreAttacker(legacy_model, **common),
            ProgressivePatchScoreAttacker(new_model, **common),
        )

    def test_module_has_no_attack_import_or_inheritance(self):
        source_path = Path(__file__).resolve().parents[1] / "progressive_attack.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertNotIn("attack", imported)
        progressive_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ProgressivePatchScoreAttacker"
        )
        self.assertEqual(progressive_class.bases, [])

    def test_vit_schedule_is_golden_equivalent(self):
        legacy, independent = self.make_pair()
        pixels = torch.linspace(0.01, 0.99, 48).view(1, 3, 4, 4)

        def schedule(attacker):
            replay = GradientReplay(20260903)
            replay.begin_batch(["golden.png"])
            replay.set_context(step=2, group=4, view=-1)
            attacker._gradient_replay = replay
            try:
                return attacker._build_mask_schedule(pixels)
            finally:
                attacker._gradient_replay = None

        legacy_schedule = schedule(legacy)
        independent_schedule = schedule(independent)
        self.assertEqual(legacy_schedule.counts, independent_schedule.counts)
        for expected, actual in zip(legacy_schedule.masks, independent_schedule.masks):
            self.assertTrue(torch.equal(expected, actual))

    def test_vit_gradient_and_adversarial_output_are_golden_equivalent(self):
        legacy, independent = self.make_pair()
        pixels = torch.linspace(0.01, 0.99, 48).view(1, 3, 4, 4)
        labels = torch.tensor([1])
        legacy_probe = legacy.probe_attack_gradients(
            pixels,
            labels,
            replay=GradientReplay(20260903),
            sample_ids=["golden.png"],
        )
        independent_probe = independent.probe_attack_gradients(
            pixels,
            labels,
            replay=GradientReplay(20260903),
            sample_ids=["golden.png"],
        )
        for key in ("view_gradients", "raw_mean", "processed"):
            self.assertTrue(
                torch.equal(legacy_probe[key], independent_probe[key]),
                f"golden mismatch for {key}",
            )
        legacy_adv = legacy.attack_batch(
            pixels, labels, replay=GradientReplay(99), sample_ids=["golden.png"]
        )
        independent_adv = independent.attack_batch(
            pixels, labels, replay=GradientReplay(99), sample_ids=["golden.png"]
        )
        self.assertTrue(torch.equal(legacy_adv, independent_adv))

    def test_score_noise_metadata_is_boolean_and_canonical_strength_is_numeric(self):
        _, attacker = self.make_pair(score_cls_noise_strength=0.0)
        metadata = attacker.mainline_metadata()
        self.assertIs(metadata["score_global_noise_active"], False)
        self.assertIsInstance(metadata["score_global_noise_active"], bool)
        self.assertEqual(metadata["score_global_noise_strength"], 0.0)

    def test_cross_grid_phase_and_union_preserve_each_budget(self):
        _, attacker = self.make_pair()
        from progressive_attack import ProgressiveMaskSchedule, ProgressiveMaskSelection

        first = ProgressiveMaskSelection(
            "block3", torch.tensor([[True, False, False, False]]), 1, (2, 2)
        )
        second = ProgressiveMaskSelection(
            "block6", torch.tensor([[True, False, False, False, False, False]]), 1, (2, 3)
        )
        schedule = ProgressiveMaskSchedule((first, second))
        shifted = attacker._phase_mask_schedule(schedule, [(1, 1)], height=8, width=8)
        self.assertEqual(shifted.counts, (1, 1))
        union = attacker._schedule_image_union(shifted, 8, 8)
        self.assertEqual(tuple(union.shape), (1, 1, 8, 8))


if __name__ == "__main__":
    unittest.main()
