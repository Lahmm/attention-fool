import hashlib
import unittest

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
                "view_gradients": "4a6a7b6275aba41ef2e1ff6521b743202478f549d2c0a940e183a5c5100b4cd3",
                "raw_mean": "98632b0b70d1e50a550317135378b7e78a50ad28a6ebec253b078e1abb80899a",
                "processed": "98632b0b70d1e50a550317135378b7e78a50ad28a6ebec253b078e1abb80899a",
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
