import ast
import hashlib
from pathlib import Path
import unittest

import torch

from gradient_replay import GradientReplay
from nets.base import PatchScoreFeatures
from progressive_attack import ProgressivePatchScoreAttacker
from tests.test_progressive_vit import TinyViTWrapper


class ProgressiveIndependenceTests(unittest.TestCase):
    def make_attacker(self, **overrides):
        torch.manual_seed(7)
        model = TinyViTWrapper()
        common = {
            "checkpoints": (3, 10),
            "drop_ratios": (0.25, 0.25),
            "score_global_noise_strength": 0.2,
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
        self.assertEqual(independent_schedule.counts, (1, 1))
        self.assertEqual(
            [self.digest(mask) for mask in independent_schedule.masks],
            [
                "b40711a88c7039756fb8a73827eabe2c0fe5a0346ca7e0a104adc0fc764f528d",
                "bf5e8ffa51a9e748985800c1d3d7f1a2a6ae7435136593ca8d9637e3f87c699c",
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
                "view_gradients": "21d1dc53cbb3d48afacecc2ce9d650460f1001808afa778a5f1358bb7cd1777d",
                "raw_mean": "b4b69f2bfd55216daaa2d47244a49cfc206e55b08704af26bfd61bf37b7be560",
                "processed": "b4b69f2bfd55216daaa2d47244a49cfc206e55b08704af26bfd61bf37b7be560",
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
        attacker = self.make_attacker(score_global_noise_strength=0.0)
        metadata = attacker.mainline_metadata()
        self.assertIs(metadata["score_global_noise_active"], False)
        self.assertIsInstance(metadata["score_global_noise_active"], bool)
        self.assertEqual(metadata["score_global_noise_strength"], 0.0)

    def test_gap_score_modes_match_their_definitions(self):
        local = torch.tensor(
            [[[2.0, 0.0], [0.0, 1.0], [1.0, 2.0]]], dtype=torch.float32
        )
        global_token = local.mean(dim=1, keepdim=True)
        features = PatchScoreFeatures(
            local_tokens=local,
            global_token=global_token,
            grid_size=(1, 3),
            source_name="gap-test",
            layer_id="gap-test",
            global_mode="gap",
        )
        cosine = self.make_attacker(
            progressive_score_mode="cosine", score_global_noise_strength=0.0
        )._score_at_checkpoint(features)
        self.assertTrue(
            torch.allclose(
                cosine,
                torch.nn.functional.cosine_similarity(
                    local, global_token.expand_as(local), dim=-1
                ),
            )
        )

        leave_one_out = self.make_attacker(
            progressive_score_mode="gap_leave_one_out_cosine",
            score_global_noise_strength=0.0,
        )._score_at_checkpoint(features)
        expected_loo_global = (local.size(1) * global_token - local) / (local.size(1) - 1)
        self.assertTrue(
            torch.allclose(
                leave_one_out,
                torch.nn.functional.cosine_similarity(local, expected_loo_global, dim=-1),
            )
        )

        projection = self.make_attacker(
            progressive_score_mode="gap_projection", score_global_noise_strength=0.0
        )._score_at_checkpoint(features)
        self.assertTrue(
            torch.allclose(
                projection,
                (local * global_token).sum(dim=-1) / (local.size(-1) ** 0.5),
            )
        )

        channel_rms = local.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        rms_cosine = self.make_attacker(
            progressive_score_mode="gap_channel_rms_cosine",
            score_global_noise_strength=0.0,
        )._score_at_checkpoint(features)
        self.assertTrue(
            torch.allclose(
                rms_cosine,
                torch.nn.functional.cosine_similarity(
                    local / channel_rms,
                    global_token.expand_as(local) / channel_rms,
                    dim=-1,
                ),
            )
        )

    def test_gap_only_score_modes_reject_cls_features(self):
        attacker = self.make_attacker(
            progressive_score_mode="gap_projection", score_global_noise_strength=0.0
        )
        local = torch.ones(1, 2, 3)
        features = PatchScoreFeatures(
            local_tokens=local,
            global_token=local[:, :1],
            grid_size=(1, 2),
            source_name="cls-test",
            layer_id="cls-test",
            global_mode="cls",
        )
        with self.assertRaisesRegex(ValueError, "requires GAP"):
            attacker._score_at_checkpoint(features)

    def test_model_specific_default_drop_ratios_are_used_when_omitted(self):
        attacker = self.make_attacker(checkpoints=None, drop_ratios=None)
        self.assertEqual(
            attacker.progressive_drop_ratios,
            (0.051020408163, 0.051020408163),
        )

    def test_gaussian_feature_noise_is_rms_matched_and_noise_off_is_explicit(self):
        attacker = self.make_attacker(
            feature_noise_type="gaussian", opponent_noise_strength=0.25
        )
        state = attacker.model.prepare_progressive_input(torch.rand(1, 3, 4, 4))
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
            "block10", torch.tensor([[True, False, False, False, False, False]]), 1, (2, 3)
        )
        schedule = ProgressiveMaskSchedule((first, second))
        shifted = attacker._phase_mask_schedule(schedule, [(1, 1)], height=8, width=8)
        self.assertEqual(shifted.counts, (1, 1))
        union = attacker._schedule_image_union(shifted, 8, 8)
        self.assertEqual(tuple(union.shape), (1, 1, 8, 8))


if __name__ == "__main__":
    unittest.main()
