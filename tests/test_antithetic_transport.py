import unittest

import torch
import torch.nn as nn

from attack import LMDSSAttacker


class TinyModel(nn.Module):
    def forward(self, x, return_attn=False):
        del return_attn
        return x.mean(dim=(2, 3))


def make_attacker(copies=9, area="all", method="antithetic_transport"):
    return LMDSSAttacker(
        TinyModel(),
        epsilon=0.1,
        steps=1,
        ti_sigma=0,
        layers=(-1,),
        guide_aug=True,
        guide_aug_area=area,
        guide_aug_methods=(method,),
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

    def test_filter_bank_is_antithetic_and_uses_exact_budget(self):
        torch.manual_seed(13)
        attacker = make_attacker(copies=9, method="antithetic_filter_bank")
        pixels = (torch.rand(2, 3, 16, 16) * 0.4 + 0.3).requires_grad_(True)
        views = list(attacker._iter_forward_pixels(pixels, None))
        self.assertEqual(len(views), 9)
        torch.testing.assert_close(views[-1], pixels)
        torch.testing.assert_close(
            (views[0] - pixels) + (views[1] - pixels), torch.zeros_like(pixels), atol=1e-6, rtol=1e-6
        )
        sum(view.mean() for view in views).backward()
        self.assertTrue(torch.isfinite(pixels.grad).all())

        probe = (torch.rand(2, 3, 16, 16) * 0.4 + 0.3).requires_grad_(True)
        gradient, terms = attacker._attack_grad_terms(probe, torch.tensor([0, 1]), None)
        self.assertEqual(len(terms), 9)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_multiscale_adjoint_is_forward_identity_with_distinct_view_gradients(self):
        attacker = make_attacker(copies=9, method="multiscale_adjoint_ensemble")
        pixels = torch.rand(2, 3, 16, 16)
        views = list(attacker._iter_forward_pixels(pixels, None))
        self.assertEqual(len(views), 9)
        for view in views:
            torch.testing.assert_close(view, pixels)
        probe = pixels.clone().requires_grad_(True)
        gradient, terms = attacker._attack_grad_terms(probe, torch.tensor([0, 1]), None)
        self.assertEqual(len(terms), 9)
        self.assertFalse(torch.equal(terms[0], terms[-1]))

    def test_natural_spectrum_transport_preserves_dc_and_backward_path(self):
        torch.manual_seed(11)
        attacker = make_attacker(copies=1, method="natural_spectrum_transport")
        pixels = (torch.rand(4, 3, 16, 16) * 0.4 + 0.3).requires_grad_(True)
        transformed = next(attacker._iter_forward_pixels(pixels, None))
        self.assertGreater(float((transformed - pixels).abs().mean().detach()), 0.0)
        torch.testing.assert_close(
            transformed.mean(dim=(2, 3)), pixels.mean(dim=(2, 3)), atol=1e-5, rtol=1e-5
        )
        transformed.square().mean().backward()
        self.assertIsNotNone(pixels.grad)
        self.assertTrue(torch.isfinite(pixels.grad).all())


if __name__ == "__main__":
    unittest.main()
