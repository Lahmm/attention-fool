import unittest
import torch

from randomness import AttackRandomness


class RandomnessTests(unittest.TestCase):
    def test_sample_streams_do_not_depend_on_batch_position(self):
        batch = AttackRandomness(['a', 'b'])
        single = AttackRandomness(['b'])
        for rng in (batch, single):
            rng.set_context(step=2, group=3, view=1)
        x = batch.randn_like(torch.empty(2, 4), 'noise')
        y = single.randn_like(torch.empty(1, 4), 'noise')
        torch.testing.assert_close(x[1], y[0], rtol=0, atol=0)
        torch.testing.assert_close(batch.randperm(196, 'mask', 1, device=torch.device('cpu')),
                                   single.randperm(196, 'mask', 0, device=torch.device('cpu')))
        self.assertEqual(batch.randint(3, 'phase', 1, device=torch.device('cpu')),
                         single.randint(3, 'phase', 0, device=torch.device('cpu')))
