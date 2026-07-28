from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from attack import ATTACK_METHODS, PATCH_SELECTORS, PatchScoreAttacker
from main import attack_all_samples, parse_args


class DummyModel:
    model_mean = (0.5, 0.5, 0.5)
    model_std = (0.5, 0.5, 0.5)

    def eval(self):
        return self


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


def make_attacker(**kwargs):
    return PatchScoreAttacker(model=DummyModel(), device=torch.device("cpu"), **kwargs)


class AttackPipelineTests(unittest.TestCase):
    def test_raw_view_mean_keeps_all_twenty_gradients(self):
        attacker = make_attacker()
        pixels = torch.zeros(2, 3, 2, 2, requires_grad=True)
        losses = [(pixels * float(index + 1)).sum() for index in range(20)]
        attacker._iter_attack_losses = lambda _pixels, _labels: iter(losses)

        actual = attacker._attack_grad(pixels, torch.zeros(2, dtype=torch.long))

        self.assertEqual(
            attacker.input_diversity_groups * attacker.input_diversity_views_per_group,
            20,
        )
        self.assertTrue(torch.equal(actual, torch.full_like(pixels, 10.5)))

    def test_gaussian_residual_is_identity_when_disabled(self):
        attacker = make_attacker(gaussian_alpha=0.0)
        gradient = torch.randn(2, 3, 16, 16)
        self.assertTrue(torch.equal(attacker._apply_gaussian_residual(gradient), gradient))

    def test_gaussian_residual_retains_original_and_adds_smooth_component(self):
        attacker = make_attacker(gaussian_sigma=1.0, gaussian_alpha=0.75)
        gradient = torch.zeros(1, 3, 15, 15)
        gradient[:, :, 7, 7] = 1.0

        result = attacker._apply_gaussian_residual(gradient)

        self.assertGreater(float(result[0, 0, 7, 7]), 1.0)
        self.assertGreater(float(result[0, 0, 7, 6]), 0.0)
        self.assertTrue(torch.equal(result[:, 0], result[:, 1]))

    def test_attack_batch_projects_to_epsilon_ball(self):
        attacker = make_attacker(
            steps=1,
            epsilon=0.1,
            step_size=0.2,
            use_momentum=False,
            gaussian_alpha=0.0,
        )
        attacker._compute_generic_patch_scores = lambda _pixels: None

        def raw_gradient(pixels, _labels):
            attacker._actual_forward_view_count = 20
            return torch.full_like(pixels, 2.0)

        attacker._attack_grad = raw_gradient
        images = torch.zeros(1, 3, 2, 2)
        adversarial = attacker.attack_batch(images, torch.zeros(1, dtype=torch.long))
        clean_pixels = attacker._denormalize(images)
        adversarial_pixels = attacker._denormalize(adversarial)
        self.assertLessEqual(
            float((adversarial_pixels - clean_pixels).abs().max()),
            0.1 + 1e-6,
        )

    def test_gradient_probe_returns_views_raw_mean_and_processed_direction(self):
        attacker = make_attacker(
            attack_method="none",
            gaussian_alpha=0.0,
            use_momentum=False,
        )
        attacker._attack_loss_for_pixels = lambda pixels, _labels: (2.0 * pixels).sum()
        pixels = torch.zeros(2, 3, 4, 4)
        result = attacker.probe_attack_gradients(
            pixels,
            torch.zeros(2, dtype=torch.long),
        )
        self.assertEqual(result["view_gradients"].shape, (1, 2, 3, 4, 4))
        self.assertTrue(torch.equal(result["raw_mean"], torch.full_like(pixels, 2.0)))
        self.assertTrue(torch.equal(result["processed"], result["raw_mean"]))

    def test_retained_attack_methods_and_cli_defaults(self):
        self.assertEqual(
            set(ATTACK_METHODS),
            {"none", "patch_dropout", "token_patch_dropout", "original_score_postdrop_phase_pair"},
        )
        old_argv = sys.argv
        sys.argv = ["main.py", "--dim", "--ni", "--ti-sigma", "1.0"]
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.max_attacked_samples, 1000)
        self.assertTrue(args.dim)
        self.assertTrue(args.ni)
        self.assertEqual(args.ti_sigma, 1.0)
        self.assertEqual(args.gaussian_sigma, 4.0)
        self.assertEqual(args.gaussian_alpha, 0.75)
        self.assertEqual(args.post_dropout_feature_noise_type, "opponent_projected")
        self.assertEqual(args.sample_offset, 0)
        self.assertEqual(args.patch_selector, "patch_score")
        self.assertEqual(args.gradcam_target_mode, "true")
        self.assertEqual(args.gradcam_zero_policy, "error")
        self.assertEqual(
            set(PATCH_SELECTORS),
            {"patch_score", "gradcam_relu", "random", "deviation", "no_drop"},
        )

    def test_dim_and_ti_keep_image_shape(self):
        attacker = make_attacker(
            attack_method="none",
            input_diversity=True,
            ti_sigma=1.0,
            gaussian_alpha=0.0,
        )
        pixels = torch.randn(2, 3, 16, 16)
        self.assertEqual(attacker._input_diversity(pixels).shape, pixels.shape)
        self.assertEqual(attacker._smooth_grad(pixels).shape, pixels.shape)

    def test_none_patch_and_token_dropout_paths_keep_expected_view_counts(self):
        pixels = torch.zeros(1, 3, 4, 4)
        labels = torch.zeros(1, dtype=torch.long)

        none_attacker = make_attacker(attack_method="none", gaussian_alpha=0.0)
        none_attacker._attack_loss_for_pixels = lambda view, _labels: view.sum()
        self.assertEqual(len(list(none_attacker._iter_attack_losses(pixels, labels))), 1)
        self.assertEqual(none_attacker._actual_forward_view_count, 1)

        patch_attacker = make_attacker(
            attack_method="patch_dropout",
            guide_aug_copies=3,
            gaussian_alpha=0.0,
        )
        patch_attacker._patch_dropout_pixels = lambda view: view
        patch_attacker._attack_loss_for_pixels = lambda view, _labels: view.sum()
        self.assertEqual(len(list(patch_attacker._iter_attack_losses(pixels, labels))), 3)
        self.assertEqual(patch_attacker._actual_forward_view_count, 3)

        token_attacker = make_attacker(
            attack_method="token_patch_dropout",
            input_diversity_groups=2,
            input_diversity_views_per_group=2,
            input_diversity_phase_shift_set=((1, 1),),
            gaussian_alpha=0.0,
        )
        token_attacker._attack_loss_for_token_patch_dropout = (
            lambda view, _labels: view.sum()
        )
        self.assertEqual(len(list(token_attacker._iter_attack_losses(pixels, labels))), 4)
        self.assertEqual(token_attacker._actual_forward_view_count, 4)

    def test_invalid_mainline_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            make_attacker(input_diversity_groups=11, input_diversity_views_per_group=2)
        with self.assertRaises(ValueError):
            make_attacker(gaussian_sigma=0.0, gaussian_alpha=0.75)
        with self.assertRaises(ValueError):
            make_attacker(nesterov=True, use_momentum=False)

    def test_attack_sample_offset_is_disjoint_and_exact(self):
        dataloader = DataLoader(IndexedDataset(), batch_size=4, shuffle=False)
        attacker = RecordingAttacker()
        with tempfile.TemporaryDirectory() as directory:
            ids = attack_all_samples(
                dataloader,
                attacker,
                Path(directory),
                max_attacked_samples=2,
                sample_offset=3,
                replay=object(),
            )
        self.assertEqual(ids, ["image_3.png", "image_4.png"])
        self.assertEqual(attacker.ids, ids)


if __name__ == "__main__":
    unittest.main()
