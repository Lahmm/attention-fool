import unittest
import torch
import torch.nn as nn
from attack import LazyAggregationAttacker
from gradient_analysis import fft_decompose, haar_packet_paths, haar_packet_project, run_analyzed_attack

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
    def test_unobserved_runner_matches_current_attack(self):
        torch.manual_seed(7); model = TinyModel()
        attacker = LazyAggregationAttacker(model, epsilon=0.1, steps=3, ti_sigma=0, input_diversity=True,
            use_momentum=True, guide_aug=True, guide_aug_area="all", guide_aug_methods=("dropout","jitter","freq"),
            guide_aug_copies=2, device=torch.device("cpu"))
        images, labels = torch.randn(2,3,8,8), torch.tensor([1,2])
        torch.manual_seed(123); expected = attacker.attack_batch(images, labels)
        torch.manual_seed(123); actual = run_analyzed_attack(attacker, images, labels)
        self.assertTrue(torch.equal(actual, expected))
        torch.manual_seed(123); traced = run_analyzed_attack(attacker, images, labels, diagnostics=True)
        self.assertTrue(torch.equal(traced, expected))
if __name__ == "__main__": unittest.main()
