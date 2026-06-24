import unittest

import torch

from experiments.jitter_information_mechanism import derive_seed, effective_rank, span_residual, view_consensus


class JitterInformationMechanismTests(unittest.TestCase):
    def test_derived_seed_is_deterministic_and_numpy_compatible(self):
        seed = derive_seed(20260623, 10_007, 303)
        self.assertEqual(seed, derive_seed(20260623, 10_007, 303))
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32 - 1)

    def test_effective_rank_distinguishes_duplicate_and_orthogonal_views(self):
        base = torch.tensor([[[[1.0, 0.0]]]])
        duplicate = (base, base.clone(), base.clone())
        orthogonal = tuple(
            torch.eye(3)[index].view(1, 1, 1, 3) for index in range(3)
        )
        self.assertAlmostEqual(float(effective_rank(duplicate)), 1.0, places=5)
        self.assertAlmostEqual(float(effective_rank(orthogonal)), 3.0, places=5)
        self.assertAlmostEqual(float(view_consensus(duplicate)), 1.0, places=5)

    def test_span_residual_removes_basis_and_preserves_novel_direction(self):
        e1 = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float64)
        e2 = torch.tensor([[[[0.0, 1.0, 0.0]]]], dtype=torch.float64)
        vector = torch.tensor([[[[2.0, -3.0, 4.0]]]], dtype=torch.float64)
        residual = span_residual(vector, (e1, e2))
        expected = torch.tensor([[[[0.0, 0.0, 4.0]]]], dtype=torch.float64)
        torch.testing.assert_close(residual, expected, atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
