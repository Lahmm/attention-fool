import unittest

import torch

from gradient_replay import GradientReplay
from nets.base import AttackFeatureState, conv2d_attack_metadata
from experiments.patch_shuffle_pair_attack import PatchShufflePairAttacker


class TinyResumableModel(torch.nn.Module):
    model_mean = (0.5, 0.5, 0.5)
    model_std = (0.5, 0.5, 0.5)

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)
        self.head = torch.nn.Linear(4, 2, bias=False)

    def patch_score_layer_candidates(self):
        return ()

    def extract_patch_score_features(self, *_args, **_kwargs):
        raise AssertionError("patch-shuffle attack must not compute patch scores")

    def prepare_attack_feature_state(self, pixels):
        spatial = self.projection(pixels)
        state = AttackFeatureState(
            local_tokens=spatial.flatten(2).transpose(1, 2),
            grid_size=tuple(spatial.shape[-2:]),
            context=None,
            **conv2d_attack_metadata(self.projection),
        )
        state.validate()
        return state

    def forward_from_attack_feature_state(self, state, local_tokens):
        state.validate()
        return self.head(local_tokens.mean(dim=1))


def make_attacker(**kwargs):
    defaults = {
        "steps": 1,
        "input_diversity_groups": 2,
        "input_diversity_views_per_group": 2,
        "use_momentum": False,
        "gaussian_alpha": 0.0,
        "post_dropout_feature_noise_type": "opponent_projected",
        "post_dropout_feature_noise_strength": 0.2,
        "device": torch.device("cpu"),
    }
    defaults.update(kwargs)
    return PatchShufflePairAttacker(TinyResumableModel(), **defaults)


def image_patches(pixels):
    batch, channels, height, width = pixels.shape
    patch_h, patch_w = height // 14, width // 14
    return pixels.reshape(batch, channels, 14, patch_h, 14, patch_w).permute(
        0, 2, 4, 1, 3, 5
    ).reshape(batch, 196, channels, patch_h, patch_w)


class PatchShufflePairAttackTests(unittest.TestCase):
    def test_shuffle_preserves_intact_rgb_patch_multiset(self):
        attacker = make_attacker()
        patch_ids = torch.arange(196, dtype=torch.float32).reshape(1, 1, 14, 1, 14, 1)
        pixels = patch_ids.expand(1, 3, 14, 2, 14, 2).reshape(1, 3, 28, 28)
        pixels[:, 1] += 1000
        pixels[:, 2] += 2000

        torch.manual_seed(7)
        shuffled = attacker._shuffle_image_patches(pixels)
        before = image_patches(pixels).flatten(2)
        after = image_patches(shuffled).flatten(2)
        before_order = before[:, :, 0].argsort(dim=1)
        after_order = after[:, :, 0].argsort(dim=1)
        before_sorted = before.gather(1, before_order[:, :, None].expand_as(before))
        after_sorted = after.gather(1, after_order[:, :, None].expand_as(after))

        self.assertTrue(torch.equal(before_sorted, after_sorted))
        self.assertFalse(torch.equal(pixels, shuffled))
        self.assertEqual(attacker.mainline_metadata()["shuffle_patch_size"], [2, 2])

    def test_shuffle_is_differentiable_and_routes_back_to_input(self):
        attacker = make_attacker()
        attacker._sample_patch_permutation = (
            lambda count, _sample_index, device: torch.arange(count - 1, -1, -1, device=device)
        )
        pixels = torch.randn(1, 3, 28, 28, requires_grad=True)
        shuffled = attacker._shuffle_image_patches(pixels)
        destination_weights = torch.arange(196, dtype=pixels.dtype).reshape(1, 196, 1, 1, 1)
        loss = (image_patches(shuffled) * destination_weights).sum()
        gradient = torch.autograd.grad(loss, pixels)[0]
        original_patch_gradient = image_patches(gradient)[:, :, 0, 0, 0]

        self.assertTrue(
            torch.equal(
                original_patch_gradient.squeeze(0),
                torch.arange(195, -1, -1, dtype=pixels.dtype),
            )
        )

    def test_replay_permutations_are_sample_step_and_group_scoped(self):
        attacker = make_attacker()
        replay = GradientReplay(1234)
        replay.begin_batch(["sample-a", "sample-b"])
        attacker._gradient_replay = replay
        permutations = {}
        for step in (0, 1):
            for group in (0, 1):
                replay.set_context(step=step, group=group, view=1)
                permutations[(step, group, 0)] = attacker._sample_patch_permutation(
                    196, 0, torch.device("cpu")
                )
                permutations[(step, group, 1)] = attacker._sample_patch_permutation(
                    196, 1, torch.device("cpu")
                )

        values = list(permutations.values())
        for index, value in enumerate(values):
            for other in values[index + 1 :]:
                self.assertFalse(torch.equal(value, other))

        replay_again = GradientReplay(1234)
        replay_again.begin_batch(["sample-a"])
        replay_again.set_context(step=1, group=1, view=1)
        attacker._gradient_replay = replay_again
        repeated = attacker._sample_patch_permutation(196, 0, torch.device("cpu"))
        self.assertTrue(torch.equal(repeated, permutations[(1, 1, 0)]))

    def test_each_group_yields_current_original_then_fresh_shuffle(self):
        attacker = make_attacker(input_diversity_groups=3)
        replay = GradientReplay(9)
        replay.begin_batch(["sample"])
        replay.set_context(step=2, group=-1, view=-1)
        attacker._gradient_replay = replay
        pixels = torch.arange(3 * 28 * 28, dtype=torch.float32).reshape(1, 3, 28, 28)

        views = list(attacker._iter_patch_shuffle_pair(pixels))

        self.assertEqual(len(views), 6)
        self.assertTrue(all(torch.equal(views[index], pixels) for index in (0, 2, 4)))
        self.assertTrue(all(not torch.equal(views[index], pixels) for index in (1, 3, 5)))
        self.assertFalse(torch.equal(views[1], views[3]))
        self.assertFalse(torch.equal(views[3], views[5]))

    def test_gradient_probe_uses_no_score_no_drop_and_no_phase(self):
        attacker = make_attacker(input_diversity_groups=2)
        masks = []
        original_builder = attacker._build_post_dropout_feature_noise

        def recording_builder(feature_tokens, state, image_mask):
            masks.append(image_mask.detach().clone())
            return original_builder(feature_tokens, state, image_mask)

        attacker._build_post_dropout_feature_noise = recording_builder
        pixels = torch.rand(1, 3, 28, 28)
        result = attacker.probe_attack_gradients(
            pixels,
            torch.zeros(1, dtype=torch.long),
            replay=GradientReplay(77),
            sample_ids=["sample"],
        )

        self.assertEqual(result["view_gradients"].shape, (4, 1, 3, 28, 28))
        self.assertEqual(len(masks), 4)
        self.assertTrue(all(int(mask.count_nonzero()) == 0 for mask in masks))
        metadata = attacker.mainline_metadata()
        self.assertEqual(metadata["patch_score"], "disabled")
        self.assertEqual(metadata["patch_drop"], "disabled")
        self.assertEqual(metadata["phase_shift"], "disabled")
        self.assertEqual(metadata["feature_noise_scope"], "all_initial_local_tokens")

    def test_attack_projects_complete_output_to_epsilon_ball(self):
        attacker = make_attacker(
            steps=2,
            input_diversity_groups=1,
            epsilon=0.1,
            step_size=0.08,
        )
        images = torch.zeros(1, 3, 28, 28)
        clean_pixels = attacker._denormalize(images)
        adversarial = attacker.attack_batch(
            images,
            torch.zeros(1, dtype=torch.long),
            replay=GradientReplay(11),
            sample_ids=["sample"],
        )
        adversarial_pixels = attacker._denormalize(adversarial)

        self.assertEqual(adversarial.shape, images.shape)
        self.assertLessEqual(
            float((adversarial_pixels - clean_pixels).abs().max()),
            0.1 + 1e-6,
        )

    def test_raw_and_gaussian_processed_gradients_are_both_available(self):
        pixels = torch.rand(1, 3, 28, 28)
        labels = torch.zeros(1, dtype=torch.long)
        raw = make_attacker(input_diversity_groups=1, gaussian_alpha=0.0)
        raw_result = raw.probe_attack_gradients(
            pixels,
            labels,
            replay=GradientReplay(31),
            sample_ids=["sample"],
        )
        self.assertTrue(torch.equal(raw_result["processed"], raw_result["raw_mean"]))

        gaussian = make_attacker(
            input_diversity_groups=1,
            gaussian_sigma=1.0,
            gaussian_alpha=0.75,
        )
        gaussian_result = gaussian.probe_attack_gradients(
            pixels,
            labels,
            replay=GradientReplay(31),
            sample_ids=["sample"],
        )
        expected = gaussian._apply_gaussian_residual(gaussian_result["raw_mean"])
        self.assertTrue(torch.allclose(gaussian_result["processed"], expected))

    def test_invalid_grid_or_disabled_noise_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed 14 x 14"):
            make_attacker(shuffle_grid=(7, 7))
        with self.assertRaisesRegex(ValueError, "strength must be positive"):
            make_attacker(post_dropout_feature_noise_strength=0.0)
        attacker = make_attacker()
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            attacker._shuffle_image_patches(torch.zeros(1, 3, 30, 28))


if __name__ == "__main__":
    unittest.main()
