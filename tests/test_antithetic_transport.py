import unittest

import torch
import torch.nn as nn

from attack import LMDSSAttacker


class TinyModel(nn.Module):
    def forward(self, x, return_attn=False):
        del return_attn
        return x.mean(dim=(2, 3))


def make_attacker(copies=9, method="antithetic_transport"):
    return LMDSSAttacker(
        TinyModel(),
        epsilon=0.1,
        steps=1,
        ti_sigma=0,
        guide_aug=True,
        guide_aug_methods=(method,),
        guide_aug_copies=copies,
        guide_aug_strength=0.2,
        device=torch.device("cpu"),
    )


class AntitheticTransportTests(unittest.TestCase):
    def test_emits_exact_budget_and_identity_for_odd_budget(self):
        torch.manual_seed(3)
        pixels = torch.rand(2, 3, 16, 16)
        views = list(make_attacker(copies=9)._iter_forward_pixels(pixels))
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
        positive, negative = list(attacker._iter_forward_pixels(pixels))
        self.assertGreater(float((positive - negative).abs().mean().detach()), 0.0)
        (positive.mean() + negative.mean()).backward()
        self.assertIsNotNone(pixels.grad)
        self.assertTrue(torch.isfinite(pixels.grad).all())

    def test_filter_bank_is_antithetic_and_uses_exact_budget(self):
        torch.manual_seed(13)
        attacker = make_attacker(copies=9, method="antithetic_filter_bank")
        pixels = (torch.rand(2, 3, 16, 16) * 0.4 + 0.3).requires_grad_(True)
        views = list(attacker._iter_forward_pixels(pixels))
        self.assertEqual(len(views), 9)
        torch.testing.assert_close(views[-1], pixels)
        torch.testing.assert_close(
            (views[0] - pixels) + (views[1] - pixels), torch.zeros_like(pixels), atol=1e-6, rtol=1e-6
        )
        sum(view.mean() for view in views).backward()
        self.assertTrue(torch.isfinite(pixels.grad).all())

        probe = (torch.rand(2, 3, 16, 16) * 0.4 + 0.3).requires_grad_(True)
        gradient, terms = attacker._attack_grad_terms(probe, torch.tensor([0, 1]))
        self.assertEqual(len(terms), 9)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_multiscale_adjoint_is_forward_identity_with_distinct_view_gradients(self):
        attacker = make_attacker(copies=9, method="multiscale_adjoint_ensemble")
        pixels = torch.rand(2, 3, 16, 16)
        views = list(attacker._iter_forward_pixels(pixels))
        self.assertEqual(len(views), 9)
        for view in views:
            torch.testing.assert_close(view, pixels)
        probe = pixels.clone().requires_grad_(True)
        gradient, terms = attacker._attack_grad_terms(probe, torch.tensor([0, 1]))
        self.assertEqual(len(terms), 9)
        self.assertFalse(torch.equal(terms[0], terms[-1]))

    def test_dim_stable_edge_is_forward_identity_with_sampled_dim_gradients(self):
        torch.manual_seed(31)
        attacker = make_attacker(copies=5, method="dim_stable_edge")
        pixels = (torch.rand(2, 3, 16, 16) * 0.6 + 0.2).requires_grad_(True)
        views = list(attacker._iter_forward_pixels(pixels))
        self.assertEqual(len(views), 5)
        for view in views:
            torch.testing.assert_close(view, pixels)
        gradient, terms = attacker._attack_grad_terms(pixels, torch.tensor([0, 1]))
        self.assertEqual(len(terms), 5)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(sum((term - terms[0]).abs().mean() for term in terms[1:]).detach()), 0.0)

    def test_dim_stable_edge_mix_moves_forward_views_and_is_differentiable(self):
        torch.manual_seed(37)
        attacker = make_attacker(copies=5, method="dim_stable_edge_mix")
        pixels = (torch.rand(2, 3, 16, 16) * 0.6 + 0.2).requires_grad_(True)
        views = list(attacker._iter_forward_pixels(pixels))
        self.assertEqual(len(views), 5)
        self.assertGreater(float(sum((view - pixels).abs().mean() for view in views).detach()), 0.0)
        gradient, terms = attacker._attack_grad_terms(pixels, torch.tensor([0, 1]))
        self.assertEqual(len(terms), 5)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_natural_spectrum_transport_preserves_dc_and_backward_path(self):
        torch.manual_seed(11)
        attacker = make_attacker(copies=1, method="natural_spectrum_transport")
        pixels = (torch.rand(4, 3, 16, 16) * 0.4 + 0.3).requires_grad_(True)
        transformed = next(attacker._iter_forward_pixels(pixels))
        self.assertGreater(float((transformed - pixels).abs().mean().detach()), 0.0)
        torch.testing.assert_close(
            transformed.mean(dim=(2, 3)), pixels.mean(dim=(2, 3)), atol=1e-5, rtol=1e-5
        )
        transformed.square().mean().backward()
        self.assertIsNotNone(pixels.grad)
        self.assertTrue(torch.isfinite(pixels.grad).all())

    def test_orthogonal_photometric_ensemble_is_paired_and_differentiable(self):
        attacker = make_attacker(copies=9, method="orthogonal_photometric_ensemble")
        pixels = (torch.rand(2, 3, 16, 16) * 0.6 + 0.2).requires_grad_(True)
        views = list(attacker._iter_forward_pixels(pixels))
        self.assertEqual(len(views), 9)
        torch.testing.assert_close(views[-1], pixels)
        for pair_index in range(4):
            positive, negative = views[2 * pair_index:2 * pair_index + 2]
            self.assertGreater(float((positive - negative).abs().mean().detach()), 0.0)
        sum(view.mean() for view in views).backward()
        self.assertTrue(torch.isfinite(pixels.grad).all())

    def test_orthogonal_spherical_smoothing_has_antithetic_orthogonal_pairs(self):
        torch.manual_seed(17)
        attacker = make_attacker(copies=9, method="orthogonal_spherical_smoothing")
        attacker.guide_aug_strength = 0.02
        pixels = torch.full((2, 3, 16, 16), 0.5, requires_grad=True)
        views = list(attacker._iter_forward_pixels(pixels))
        self.assertEqual(len(views), 9)
        torch.testing.assert_close(views[-1], pixels)
        directions = []
        for pair_index in range(4):
            positive, negative = views[2 * pair_index:2 * pair_index + 2]
            torch.testing.assert_close((positive + negative) / 2.0, pixels)
            directions.append((positive - negative).flatten(1))
        for index, direction in enumerate(directions):
            for previous in directions[:index]:
                torch.testing.assert_close((direction * previous).sum(1), torch.zeros(2), atol=1e-5, rtol=1e-5)
        sum(view.square().mean() for view in views).backward()
        self.assertTrue(torch.isfinite(pixels.grad).all())


    def test_antithetic_jitter_cubature_pairs_brightness_noise_views(self):
        torch.manual_seed(29)
        attacker = make_attacker(copies=9, method="antithetic_jitter_cubature")
        attacker.guide_aug_strength = 0.02
        pixels = torch.full((2, 3, 16, 16), 0.5, requires_grad=True)
        views = list(attacker._iter_forward_pixels(pixels))
        self.assertEqual(len(views), 9)
        torch.testing.assert_close(views[-1], pixels)
        for pair_index in range(4):
            positive, negative = views[2 * pair_index:2 * pair_index + 2]
            torch.testing.assert_close((positive + negative) / 2.0, pixels, atol=1e-6, rtol=1e-6)
            self.assertGreater(float((positive - negative).flatten(1).norm(dim=1).min().detach()), 0.0)
        sum(view.mean() for view in views).backward()
        self.assertTrue(torch.isfinite(pixels.grad).all())


if __name__ == "__main__":
    unittest.main()
