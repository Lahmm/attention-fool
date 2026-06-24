import unittest

import torch
import torch.nn as nn

from attack import LMDSSAttacker


class TinyModel(nn.Module):
    def forward(self, x, return_attn=False):
        del return_attn
        return x.mean(dim=(2, 3))


def make_attacker(copies=9, area="all"):
    return LMDSSAttacker(
        TinyModel(),
        epsilon=0.1,
        steps=1,
        ti_sigma=0,
        layers=(-1,),
        guide_aug=True,
        guide_aug_area=area,
        guide_aug_methods=("antithetic_transport",),
        guide_aug_copies=copies,
        guide_aug_strength=0.2,
        device=torch.device("cpu"),
    )


class AntitheticTransportTests(unittest.TestCase):
    def test_emits_exact_budget_and_identity_for_odd_budget(self):
        torch.manual_seed(3)
        pixels = torch.rand(2, 3, 16, 16)
        views = list(make_attacker(copies=9)._iter_forward_pixels(pixels, None))
        self.assertEqual(len(views), 9)
        torch.testing.assert_close(views[-1], pixels)
        for view in views:
            self.assertEqual(view.shape, pixels.shape)
            self.assertGreaterEqual(float(view.min()), 0.0)
            self.assertLessEqual(float(view.max()), 1.0)

    def test_pair_uses_opposite_directions_and_is_differentiable(self):
        torch.manual_seed(5)
        attacker = make_attacker(copies=2)
        pixels = torch.rand(1, 3, 16, 16, requires_grad=True)
        positive, negative = list(attacker._iter_forward_pixels(pixels, None))
        self.assertGreater(float((positive - negative).abs().mean().detach()), 0.0)
        (positive.mean() + negative.mean()).backward()
        self.assertIsNotNone(pixels.grad)
        self.assertTrue(torch.isfinite(pixels.grad).all())

    def test_background_blending_preserves_fully_guided_pixels(self):
        torch.manual_seed(7)
        attacker = make_attacker(copies=2, area="background")
        pixels = torch.rand(1, 3, 16, 16)
        guide = torch.ones(1, 1, 16, 16)
        views = list(attacker._iter_forward_pixels(pixels, guide))
        for view in views:
            torch.testing.assert_close(view, pixels)


if __name__ == "__main__":
    unittest.main()
