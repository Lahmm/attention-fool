import unittest

import torch
from torch import nn

from nets.base import AttackFeatureState, conv2d_attack_metadata
from vit_progressive_patch_score_attack import (
    MODEL_NAME,
    ViTProgressivePatchScoreAttacker,
    _parse_float_list,
    _parse_int_list,
)


class TinyPatchEmbed(nn.Module):
    def __init__(self, dimension: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(3, dimension, kernel_size=2, stride=2, bias=False)
        self.grid_size = (2, 2)

    def forward(self, x):
        local = self.proj(x).flatten(2).transpose(1, 2)
        cls = torch.zeros(
            x.size(0), 1, local.size(-1), device=x.device, dtype=x.dtype
        )
        return torch.cat((cls, local), dim=1)


class TinyBlock(nn.Module):
    def forward(self, tokens):
        # Make CLS and local tokens interact so that a preceding drop affects
        # the later checkpoint trajectory.
        return tokens + 0.1 * tokens.mean(dim=1, keepdim=True)


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = TinyPatchEmbed()
        self._pos_embed = nn.Identity()
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(12))
        self.norm = nn.Identity()
        self.head = nn.Linear(4, 2, bias=False)

    def forward_head(self, tokens):
        return self.head(tokens[:, 0])


class TinyViTWrapper(nn.Module):
    model_name = MODEL_NAME
    model_mean = (0.0, 0.0, 0.0)
    model_std = (1.0, 1.0, 1.0)

    def __init__(self):
        super().__init__()
        self.model = TinyViT()

    def patch_score_layer_candidates(self):
        return ("block3", "block6", "block9", "block12")

    def prepare_attack_feature_state(self, x):
        tokens = self.model.patch_embed(x)
        return AttackFeatureState(
            local_tokens=tokens[:, 1:],
            grid_size=(2, 2),
            context={"prefix_tokens": tokens[:, :1]},
            **conv2d_attack_metadata(self.model.patch_embed.proj),
        )

    def eval(self):
        super().eval()
        return self


class ProgressiveViTTests(unittest.TestCase):
    def make_attacker(self, **kwargs):
        return ViTProgressivePatchScoreAttacker(
            TinyViTWrapper(),
            checkpoints=(3, 6, 9),
            drop_ratios=(0.25, 0.25, 0.25),
            score_cls_noise_strength=0.0,
            opponent_noise_strength=0.0,
            steps=1,
            input_diversity_groups=1,
            input_diversity_views_per_group=2,
            input_diversity_phase_shift_set=((0, 0),),
            use_momentum=False,
            gaussian_alpha=0.0,
            device=torch.device("cpu"),
            **kwargs,
        )

    def test_list_parsers(self):
        self.assertEqual(_parse_int_list("3, 6,9"), (3, 6, 9))
        self.assertEqual(_parse_float_list(".05, 0.1, .2"), (0.05, 0.1, 0.2))

    def test_schedule_has_three_native_masks_and_allows_repeats(self):
        attacker = self.make_attacker()
        pixels = torch.rand(2, 3, 4, 4)
        schedule = attacker._build_mask_schedule(pixels)
        self.assertEqual(schedule.checkpoints, (3, 6, 9))
        self.assertEqual(schedule.counts, (1, 1, 1))
        schedule.validate(batch_size=2, token_count=4)
        self.assertTrue(all(mask.dtype == torch.bool for mask in schedule.masks))

    def test_phase_schedule_preserves_counts(self):
        attacker = self.make_attacker()
        schedule = attacker._build_mask_schedule(torch.rand(1, 3, 4, 4))
        shifted = attacker._phase_mask_schedule(
            schedule, [(1, 1)], height=4, width=4
        )
        self.assertEqual(shifted.counts, schedule.counts)
        for original, phase in zip(schedule.masks, shifted.masks):
            self.assertEqual(int(original.sum()), int(phase.sum()))

    def test_two_views_backpropagate_and_attack_respects_epsilon(self):
        attacker = self.make_attacker()
        pixels = torch.rand(1, 3, 4, 4, requires_grad=True)
        labels = torch.zeros(1, dtype=torch.long)
        losses = list(attacker._iter_attack_losses(pixels, labels))
        self.assertEqual(len(losses), 2)
        gradients = [torch.autograd.grad(loss, pixels)[0] for loss in losses]
        self.assertEqual(attacker._actual_forward_view_count, 2)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

        clean = torch.rand(1, 3, 4, 4)
        adversarial = attacker.attack_batch(clean, labels)
        clean_pixels = attacker._denormalize(clean)
        adversarial_pixels = attacker._denormalize(adversarial)
        self.assertLessEqual(
            float((adversarial_pixels - clean_pixels).abs().max()),
            attacker.epsilon + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
