import hashlib
import unittest
from unittest.mock import patch

import torch

from gradient_replay import GradientReplay
from progressive_attack import ProgressiveRouteDisruptionAttacker
from tests.test_progressive_vit import TinyViTWrapper


class ProgressiveAttackTests(unittest.TestCase):
    def make_attacker(self, **overrides):
        torch.manual_seed(7)
        model = TinyViTWrapper()
        common = {
            "checkpoints": (3, 10),
            "drop_ratios": (0.25, 0.25),
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
        return ProgressiveRouteDisruptionAttacker(model, **common)

    @staticmethod
    def digest(tensor):
        return hashlib.sha256(
            tensor.detach().contiguous().numpy().tobytes()
        ).hexdigest()

    def test_original_only_ablation_uses_fresh_schedules_without_phase_views(self):
        for groups in (1, 10):
            with self.subTest(groups=groups):
                attacker = self.make_attacker(
                    input_diversity_groups=groups,
                    input_diversity_views_per_group=1,
                    input_diversity_phase_shift_set=(),
                )
                pixels = torch.rand(1, 3, 4, 4)
                with patch.object(attacker, "_build_mask_schedule", wraps=attacker._build_mask_schedule) as schedules, \
                     patch.object(attacker, "_pick_input_diversity_phases", side_effect=AssertionError("phase sampling")), \
                     patch.object(attacker, "_apply_samplewise_phase_shifts", side_effect=AssertionError("phase image")), \
                     patch.object(attacker, "_phase_mask_schedule", side_effect=AssertionError("phase masks")), \
                     patch.object(attacker, "_strict_opponent_feature_noise", wraps=attacker._strict_opponent_feature_noise) as noise:
                    probe = attacker.probe_attack_gradients(pixels, torch.tensor([1]))
                self.assertEqual(probe["view_gradients"].shape, (groups, 1, 3, 4, 4))
                self.assertEqual(schedules.call_count, groups)
                self.assertEqual(noise.call_count, groups)
                torch.testing.assert_close(probe["raw_mean"], probe["view_gradients"].mean(dim=0))
                metadata = attacker.mainline_metadata()
                self.assertEqual(metadata["mask_schedule_count_per_image"], groups)
                self.assertEqual(metadata["checkpoint_mask_selection_count_per_image"], groups * 2)
                self.assertEqual(metadata["mask_pair_sharing"], "not_applicable")
                self.assertEqual(metadata["phase_mask_transform"], "disabled")

    def test_gaussian_residual_is_disabled_by_default(self):
        attacker = ProgressiveRouteDisruptionAttacker(
            TinyViTWrapper(), device=torch.device("cpu")
        )
        gradient = torch.randn(1, 3, 4, 4)
        self.assertEqual(attacker.gaussian_alpha, 0.0)
        self.assertIs(attacker._apply_gaussian_residual(gradient), gradient)
        self.assertEqual(attacker.mainline_metadata()["gaussian_alpha"], 0.0)

    def test_vit_schedule_regression(self):
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
                "6b1e73a0094b7b812d3b9e22cffb4f8239319847522c4fa103753b6950020f93",
                "bf5e8ffa51a9e748985800c1d3d7f1a2a6ae7435136593ca8d9637e3f87c699c",
            ],
        )

    def test_vit_gradient_and_adversarial_output_regression(self):
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
                "view_gradients": "426bf61f8bf838c8e3ff3b46c07b5bbbd629a51167096d8816244726c5c74bc0",
                "raw_mean": "aea265664c55c9ce86b08d9ccf7fb58b01dd7c9cacb9afbb9f185ca7ebbd7489",
                "processed": "aea265664c55c9ce86b08d9ccf7fb58b01dd7c9cacb9afbb9f185ca7ebbd7489",
            },
        )
        independent_adv = independent.attack_batch(
            pixels, labels, replay=GradientReplay(99), sample_ids=["golden.png"]
        )
        self.assertEqual(
            self.digest(independent_adv),
            "df31fe9923f1e00f0a40bc8c8bab0af159eb3f6e3e3d310cbdef9e1f7c8c1463",
        )

    def test_metadata_records_uniform_random_route_disruption(self):
        attacker = self.make_attacker()
        metadata = attacker.mainline_metadata()
        self.assertEqual(metadata["attack_method"], "progressive_route_disruption")
        self.assertEqual(metadata["drop_map_policy"], "uniform_random_all_local_tokens")
        self.assertEqual(metadata["opponent_noise"], "initial_rgb_projection_all_local_tokens")

    def test_model_specific_default_drop_ratios_are_used_when_omitted(self):
        attacker = self.make_attacker(checkpoints=None, drop_ratios=None)
        self.assertEqual(
            attacker.progressive_drop_ratios,
            (0.051020408163, 0.051020408163),
        )

    def test_noise_off_is_explicit(self):
        disabled = self.make_attacker(opponent_noise_strength=0.0)
        metadata = disabled.mainline_metadata()
        self.assertEqual(metadata["opponent_noise"], "disabled")

    def test_cross_grid_phase_preserves_each_budget(self):
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
        self.assertEqual(shifted.grid_sizes, ((2, 2), (2, 3)))
        shifted.validate(batch_size=1)


if __name__ == "__main__":
    unittest.main()
