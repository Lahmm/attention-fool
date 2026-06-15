import unittest
import torch
import torch.nn as nn
from attack import LMDSSAttacker
from experiments.gradient_analysis import (
    fft_decompose,
    haar_packet_paths,
    haar_packet_project,
    haar_packet_region_project,
    parse_component,
    run_analyzed_attack,
    screening_component_specs,
)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(3*8*8, 4)
    def forward(self, x, return_attn=False): return self.linear(x.flatten(1))

class ProjectionTests(unittest.TestCase):
    def test_fft_parseval_and_reconstruction(self):
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64); parts = fft_decompose(x)
        self.assertLess((sum(parts)-x).abs().max().item(), 1e-10)
        self.assertAlmostEqual(sum(p.square().sum().item() for p in parts), x.square().sum().item(), places=8)
    def test_haar_packet_parseval_and_reconstruction(self):
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        parts = [haar_packet_project(x, p) for p in haar_packet_paths()]
        self.assertLess((sum(parts)-x).abs().max().item(), 1e-10)
        self.assertAlmostEqual(sum(p.square().sum().item() for p in parts), x.square().sum().item(), places=8)
    def test_local_haar_regions_are_orthogonal_and_reconstruct_path(self):
        x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        regions = [haar_packet_region_project(x, "LHD", row, col) for row in range(4) for col in range(4)]
        expected = haar_packet_project(x, "LHD")
        self.assertLess((sum(regions) - expected).abs().max().item(), 1e-10)
        self.assertAlmostEqual(sum(part.square().sum().item() for part in regions), expected.square().sum().item(), places=8)
        self.assertLess((regions[0] * regions[1]).sum().abs().item(), 1e-10)
        self.assertTrue(torch.equal(parse_component("haar:LHD:0:0")(x), regions[0]))

    def test_fixed_screening_candidate_set(self):
        specs = screening_component_specs()
        self.assertEqual(len(specs), 24 + 1024)
        self.assertEqual(len(set(specs)), len(specs))
        self.assertEqual(sum(spec.startswith("fft:") for spec in specs), 24)
        self.assertEqual(sum(spec.startswith("haar:") for spec in specs), 1024)

    def test_unobserved_runner_matches_current_attack(self):
        torch.manual_seed(7); model = TinyModel()
        attacker = LMDSSAttacker(model, epsilon=0.1, steps=3, ti_sigma=0, input_diversity=True,
            use_momentum=True, guide_aug=True, guide_aug_area="all", guide_aug_methods=("dropout","jitter","freq"),
            guide_aug_copies=2, device=torch.device("cpu"))
        images, labels = torch.randn(2,3,8,8), torch.tensor([1,2])
        torch.manual_seed(123); expected = attacker.attack_batch(images, labels)
        torch.manual_seed(123); actual = run_analyzed_attack(attacker, images, labels)
        self.assertTrue(torch.equal(actual, expected))
        torch.manual_seed(123); traced = run_analyzed_attack(attacker, images, labels, diagnostics=True)
        self.assertTrue(torch.equal(traced, expected))

    def test_dim_forward_and_backward_paths_are_independently_selectable(self):
        model = TinyModel(); images = torch.randn(2, 3, 8, 8, requires_grad=True)
        attacker = LMDSSAttacker(model, steps=1, ti_sigma=0, input_diversity=True,
            dim_resize_range=(0.5, 0.5), dim_mode="full-fixed", device=torch.device("cpu"))
        torch.manual_seed(4); full = attacker._input_diversity(images); full.sum().backward(); full_grad = images.grad.clone()
        images.grad.zero_(); attacker.dim_mode = "forward-only"; forward = attacker._input_diversity(images); forward.sum().backward()
        self.assertTrue(torch.allclose(forward, full)); self.assertTrue(torch.equal(images.grad, torch.ones_like(images)))
        images.grad.zero_(); attacker.dim_mode = "backward-fixed"; backward = attacker._input_diversity(images); backward.sum().backward()
        self.assertTrue(torch.allclose(backward, images)); self.assertTrue(torch.equal(images.grad, full_grad))

    def test_fixed_dim_reuses_transform_and_random_dim_resamples(self):
        model = TinyModel(); images = torch.randn(1, 3, 8, 8)
        attacker = LMDSSAttacker(model, steps=1, ti_sigma=0, input_diversity=True,
            dim_resize_range=(0.5, 0.5), dim_mode="full-fixed", device=torch.device("cpu"))
        torch.manual_seed(9); first = attacker._input_diversity(images); second = attacker._input_diversity(images)
        self.assertTrue(torch.equal(first, second))
        attacker.dim_mode = "full-random"; draws = [attacker._input_diversity(images) for _ in range(4)]
        self.assertTrue(any(not torch.equal(draws[0], draw) for draw in draws[1:]))
if __name__ == "__main__": unittest.main()
