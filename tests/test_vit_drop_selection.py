from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from progressive_attack import ProgressiveRouteDisruptionAttacker
from tests.test_progressive_attack import make_model
from tests.vit_drop_selection import (
    CASES, SelectionAttacker, clean_boundary_scores, configure_model, evaluate, generate, rng_state, same_rng,
)


class ViTDropSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.model = configure_model(make_model())
        self.scores = torch.arange(196, dtype=torch.float32).repeat(2, 1)
        self.pixels = torch.linspace(0.1, 0.9, 3 * 28 * 28).view(1, 3, 28, 28).repeat(2, 1, 1, 1)

    def attacker(self, case, **kwargs):
        return SelectionAttacker(self.model, self.scores, case, steps=2, input_diversity_groups=2,
                                 device=torch.device("cpu"), **kwargs)

    def test_all_six_cases_have_equal_budgets_and_correct_candidate_pools(self):
        for case in CASES:
            torch.manual_seed(42)
            attacker = self.attacker(case)
            schedules = [attacker._build_mask_schedule(self.pixels) for _ in range(4)]
            for schedule in schedules:
                schedule.validate(batch_size=2)
                self.assertEqual(schedule.counts, (20,))
                chosen = schedule.masks[0].nonzero()[:, 1]
                if case.startswith("high"):
                    self.assertTrue((chosen >= 98).all())
                elif case.startswith("low"):
                    self.assertTrue((chosen < 98).all())
            fixed = case.endswith("fix")
            self.assertEqual(all(torch.equal(schedules[0].masks[0], s.masks[0]) for s in schedules), fixed)
            if case == "high-fix":
                self.assertTrue(torch.equal(schedules[0].masks[0][:, 176:], torch.ones(2, 20, dtype=torch.bool)))
            if case == "low-fix":
                self.assertTrue(torch.equal(schedules[0].masks[0][:, :20], torch.ones(2, 20, dtype=torch.bool)))

    def test_ties_partition_into_disjoint_halves_without_using_absolute_scores(self):
        self.scores.fill_(-0.5)
        high = self.attacker("high-random")
        low = self.attacker("low-random")
        self.assertTrue(torch.equal(high.pool_indices, torch.arange(98, 196).repeat(2, 1)))
        self.assertTrue(torch.equal(low.pool_indices, torch.arange(98).repeat(2, 1)))
        self.scores = torch.linspace(-1, -0.01, 196).repeat(2, 1)
        high = self.attacker("high-fix")
        low = self.attacker("low-fix")
        self.assertTrue(high.fixed_mask[:, -20:].all())
        self.assertTrue(low.fixed_mask[:, :20].all())

    def test_random_case_matches_production_masks_and_rng_exactly(self):
        baseline = ProgressiveRouteDisruptionAttacker(self.model, dynamic_checkpoint=False, device=torch.device("cpu"))
        attacker = self.attacker("random")
        torch.manual_seed(42)
        expected = [baseline._build_mask_schedule(self.pixels).masks[0] for _ in range(3)]
        expected_rng = rng_state()
        torch.manual_seed(42)
        observed = [attacker._build_mask_schedule(self.pixels).masks[0] for _ in range(3)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, observed)))
        self.assertTrue(same_rng(expected_rng, rng_state()))

    def test_clean_scoring_uses_exact_drop_boundary_and_final_normalized_cls(self):
        images = (self.pixels - 0.5) / 0.5
        actual = clean_boundary_scores(self.model, images)
        with torch.no_grad():
            mean = self.pixels.new_tensor(self.model.model_mean).view(1, 3, 1, 1)
            std = self.pixels.new_tensor(self.model.model_std).view(1, 3, 1, 1)
            state = self.model.begin_progressive_forward((self.pixels - mean) / std)
            state = self.model.advance_progressive_state(state, "block10")
            local = state.local_tokens.clone()
            tokens = torch.cat((state.context["prefix_tokens"], local), dim=1)
            for block in self.model.model.blocks[10:]:
                tokens = block(tokens)
            reference = self.model.model.norm(tokens)[:, 0]
            expected = torch.nn.functional.cosine_similarity(local, reference[:, None], dim=-1, eps=1e-8)
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)

    def test_full_attacks_align_noise_phase_draws_and_projected_updates(self):
        streams = []
        images, labels = (self.pixels - 0.5) / 0.5, torch.tensor([1, 0])
        for case in CASES:
            attacker = self.attacker(case)
            phases, noises = [], []
            original_pick = attacker._pick_input_diversity_phases
            original_randn = torch.randn

            def pick(*args, **kwargs):
                result = original_pick(*args, **kwargs)
                phases.append(result)
                return result

            def randn(*args, **kwargs):
                result = original_randn(*args, **kwargs)
                noises.append(result.clone())
                return result

            torch.manual_seed(42)
            with patch.object(attacker, "_pick_input_diversity_phases", side_effect=pick), \
                    patch("torch.randn", side_effect=randn):
                adversarial = attacker.attack_batch(images, labels)
            end = rng_state()
            pixels = (adversarial + 1) / 2
            self.assertTrue(torch.isfinite(pixels).all())
            self.assertGreaterEqual(float(pixels.min()), 0)
            self.assertLessEqual(float(pixels.max()), 1)
            self.assertLessEqual(float((pixels - self.pixels).abs().max()), attacker.epsilon + 1e-6)
            self.assertEqual(len(attacker.mask_indices), 4)
            self.assertEqual(attacker._actual_forward_view_count, 4)
            streams.append((phases, noises, end))
        for phases, noises, end in streams[1:]:
            self.assertEqual(phases, streams[0][0])
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(noises, streams[0][1])))
            self.assertTrue(same_rng(end, streams[0][2]))

    def test_phase_masks_preserve_budget_and_are_spatial_counterparts(self):
        for case in CASES:
            attacker = self.attacker(case)
            schedule = attacker._build_mask_schedule(self.pixels)
            shifted = attacker._phase_mask_schedule(schedule, [(4, 4), (8, 8)], height=224, width=224)
            shifted.validate(batch_size=2)
            self.assertEqual(shifted.counts, (20,))
            occupancy = torch.nn.functional.adaptive_avg_pool2d(
                attacker._apply_samplewise_phase_shifts(
                    attacker._mask_to_image(schedule.masks[0], (14, 14), 224, 224), [(4, 4), (8, 8)]
                ), (14, 14),
            ).flatten(1)
            expected_indices = torch.argsort(occupancy, dim=1, descending=True, stable=True)[:, :20]
            expected = torch.zeros(2, 196, dtype=torch.bool).scatter_(1, expected_indices, True)
            self.assertTrue(torch.equal(shifted.masks[0], expected))

    def test_driver_saves_multiple_batches_and_aggregates_fpr13_results(self):
        from transfer_eval import FPR_MODELS

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image_dir = directory / "images"
            image_dir.mkdir()
            annotations = {}
            for index in range(3):
                name = f"sample_{index}.png"
                Image.new("RGB", (28, 28), (50 + index * 40, 100, 150)).save(image_dir / name)
                annotations[name] = {"class_id": index % 2}
            annotation_path = directory / "labels.json"
            annotation_path.write_text(json.dumps(annotations))
            args = SimpleNamespace(
                threads=2, device="cpu", image_dir=str(image_dir), annotations_path=str(annotation_path),
                batch_size=2, eval_batch_size=2, num_workers=0, max_attacked_samples=3,
                cases=list(CASES), drop_ratio=20 / 196, steps=1, epsilon=16 / 255,
                opponent_noise_strength=0.2, input_diversity_groups=1, input_diversity_views_per_group=2,
            )
            root = directory / "experiment"
            with patch("tests.vit_drop_selection.build_whitebox_model", return_value=self.model), \
                    redirect_stdout(StringIO()):
                generate(args, root)
            manifest = json.loads((root / "experiment.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["random_streams_aligned"])
            for case in CASES:
                self.assertEqual(len(list((root / case).glob("adv_*.png"))), 3)
                trace = torch.load(root / case / "selection_traces/batch_00001.pt", weights_only=True)
                self.assertEqual(trace["image_names"], ["sample_2.png"])
                self.assertEqual(tuple(trace["original_mask_indices"].shape), (1, 1, 20))

            def fake_evaluator(command, **kwargs):
                case_root = Path(command[command.index("--image-dir") + 1])
                self.assertEqual(command[command.index("--protocol") + 1], "fpr")
                self.assertIn("--no-tf32", command)
                asr = float(case_root.name in ("high-fix", "low-random"))
                with (case_root / "transfer.csv").open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(("target", "correct", "total", "acc", "asr"))
                    for target in FPR_MODELS:
                        writer.writerow((target, int(3 * (1 - asr)), 3, 1 - asr, asr))
                (case_root / "transfer.json").write_text(json.dumps({
                    "summary": {"overall_asr": asr, "strict_black_box_asr": asr}
                }))

            with patch("tests.vit_drop_selection.subprocess.run", side_effect=fake_evaluator), \
                    redirect_stdout(StringIO()):
                evaluate(args, root)
            with (root / "asr_comparison.csv").open() as handle:
                rows = {row["case"]: row for row in csv.DictReader(handle)}
            self.assertEqual(float(rows["high-fix"]["strict_black_box_asr_delta_pp"]), 100)
            self.assertEqual(rows["high-fix"]["reference"], "random-fix")
            self.assertEqual(float(rows["low-random"]["strict_black_box_asr_delta_pp"]), 100)
            self.assertEqual(rows["low-random"]["reference"], "random")
            with (root / "asr_per_target.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 13)
            with self.assertRaisesRegex(ValueError, "not empty"):
                generate(args, root)


if __name__ == "__main__":
    unittest.main()
