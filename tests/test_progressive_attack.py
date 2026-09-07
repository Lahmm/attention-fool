import ast
import hashlib
from pathlib import Path
import unittest

import torch

from gradient_replay import GradientReplay
from progressive_attack import ProgressivePatchScoreAttacker
from tests.vit_progressive_patch_score_attack_cases import (
    TinyViTWrapper,
)


class ProgressiveIndependenceTests(unittest.TestCase):
    def make_attacker(self, **overrides):
        torch.manual_seed(7)
        model = TinyViTWrapper()
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
        return ProgressivePatchScoreAttacker(model, **common)

    @staticmethod
    def digest(tensor):
        return hashlib.sha256(
            tensor.detach().contiguous().numpy().tobytes()
        ).hexdigest()

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
        independent = self.make_attacker()
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

        independent_schedule = schedule(independent)
        self.assertEqual(independent_schedule.counts, (1, 1, 1))
        self.assertEqual(
            [self.digest(mask) for mask in independent_schedule.masks],
            [
                "b40711a88c7039756fb8a73827eabe2c0fe5a0346ca7e0a104adc0fc764f528d",
                "bf5e8ffa51a9e748985800c1d3d7f1a2a6ae7435136593ca8d9637e3f87c699c",
                "b40711a88c7039756fb8a73827eabe2c0fe5a0346ca7e0a104adc0fc764f528d",
            ],
        )

    def test_vit_gradient_and_adversarial_output_are_golden_equivalent(self):
        independent = self.make_attacker()
        pixels = torch.linspace(0.01, 0.99, 48).view(1, 3, 4, 4)
        labels = torch.tensor([1])
        independent_probe = independent.probe_attack_gradients(
            pixels,
            labels,
            replay=GradientReplay(20260903),
            sample_ids=["golden.png"],
        )
        self.assertEqual(
            {key: self.digest(value) for key, value in independent_probe.items()},
            {
                "view_gradients": "e97a182ec0d9da128e8bc76c4b814664b4c7ea9547fcbaac2f39a95d773b8a85",
                "raw_mean": "df11a2ddecceaf795868b2d882cdbf37be6c1c01ba2dda73986c63c257daee7d",
                "processed": "df11a2ddecceaf795868b2d882cdbf37be6c1c01ba2dda73986c63c257daee7d",
            },
        )
        independent_adv = independent.attack_batch(
            pixels, labels, replay=GradientReplay(99), sample_ids=["golden.png"]
        )
        self.assertEqual(
            self.digest(independent_adv),
            "df31fe9923f1e00f0a40bc8c8bab0af159eb3f6e3e3d310cbdef9e1f7c8c1463",
        )

    def test_score_noise_metadata_is_boolean_and_canonical_strength_is_numeric(self):
        attacker = self.make_attacker(score_cls_noise_strength=0.0)
        metadata = attacker.mainline_metadata()
        self.assertIs(metadata["score_global_noise_active"], False)
        self.assertIsInstance(metadata["score_global_noise_active"], bool)
        self.assertEqual(metadata["score_global_noise_strength"], 0.0)

    def test_gaussian_feature_noise_is_rms_matched_and_noise_off_is_explicit(self):
        attacker = self.make_attacker(
            feature_noise_type="gaussian", opponent_noise_strength=0.25
        )
        state = attacker.model.prepare_attack_feature_state(torch.rand(1, 3, 4, 4))
        noise = attacker._kept_feature_noise(state)
        token_rms = state.local_tokens.square().mean(dim=(1, 2)).sqrt()
        noise_rms = noise.square().mean(dim=(1, 2)).sqrt()
        self.assertTrue(torch.allclose(noise_rms, 0.25 * token_rms, rtol=1e-5))
        self.assertEqual(attacker._feature_noise_type, "feature_iid_gaussian")

        disabled = self.make_attacker(opponent_noise_strength=0.0)
        metadata = disabled.mainline_metadata()
        self.assertEqual(metadata["opponent_noise"], "disabled")

    def test_cross_grid_phase_and_union_preserve_each_budget(self):
        attacker = self.make_attacker()
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
