import unittest

import torch

from gradient_observer import GradientObserver
from gradient_replay import GradientReplay
from gradient_study import GroupRemovalProbe, SpatialPatchProbe


class GradientReplayTests(unittest.TestCase):
    def test_sample_randomness_is_independent_of_batch_order(self):
        tensor = torch.empty(2, 3)
        first = GradientReplay(17)
        first.begin_batch(["a", "b"])
        first.set_context(step=2, group=3, view=1)
        values = first.randn_like(tensor, "noise")

        reordered = GradientReplay(17)
        reordered.begin_batch(["b", "a"])
        reordered.set_context(step=2, group=3, view=1)
        reordered_values = reordered.randn_like(tensor, "noise")

        self.assertTrue(torch.equal(values[0], reordered_values[1]))
        self.assertTrue(torch.equal(values[1], reordered_values[0]))

    def test_event_context_changes_randomness(self):
        replay = GradientReplay(17)
        replay.begin_batch(["a"])
        replay.set_context(step=0, group=0, view=0)
        first = replay.randn_like(torch.empty(1, 8), "noise")
        replay.set_context(view=1)
        second = replay.randn_like(torch.empty(1, 8), "noise")
        self.assertFalse(torch.equal(first, second))


class GradientObserverTests(unittest.TestCase):
    def test_pairwise_cosine_excludes_diagonal(self):
        views = torch.zeros(2, 1, 1, 1, 2)
        views[0, 0, 0, 0, 0] = 1.0
        views[1, 0, 0, 0, 1] = 1.0
        observer = GradientObserver(sample_ids=["sample"])
        observer.record_raw_views(views)
        summary = observer.per_sample_summary()[0]
        self.assertAlmostEqual(summary["view_pairwise_cosine"], 0.0, places=6)

    def test_pre_momentum_and_step_to_step_are_recorded(self):
        observer = GradientObserver(sample_ids=["sample"])
        first = torch.ones(1, 1, 2, 2)
        observer.record_normalized(first)
        observer.record_pre_momentum(torch.zeros_like(first), first)
        observer.record_momentum(first)
        observer.record_sign_update(first.sign())
        observer.close_step()

        second = -first
        observer.record_normalized(second)
        observer.record_pre_momentum(first, second)
        observer.record_momentum(torch.zeros_like(first))
        observer.record_sign_update(second.sign())
        observer.close_step()

        records = observer._records
        self.assertIsNone(records[0]["per_sample"][0]["pre_momentum_to_grad_cosine"])
        self.assertAlmostEqual(records[1]["per_sample"][0]["pre_momentum_to_grad_cosine"], -1.0)
        self.assertAlmostEqual(records[1]["per_sample"][0]["step_to_step_grad_cosine"], -1.0)

    def test_reference_sign_flip_rate(self):
        reference = [torch.tensor([[[[1.0, -1.0]]]])]
        observer = GradientObserver(sample_ids=["sample"], reference_signs=reference)
        observer.record_sign_update(torch.tensor([[[[1.0, 1.0]]]]))
        self.assertAlmostEqual(observer._records[0]["per_sample"][0]["update_sign_flip_rate"], 0.5)


class GradientProbeTests(unittest.TestCase):
    def test_group_probe_removes_one_group_per_sample(self):
        gradients = torch.arange(4 * 2 * 1 * 1 * 2, dtype=torch.float32).reshape(4, 2, 1, 1, 2)
        probe = GroupRemovalProbe("random")
        result = probe.apply(gradients, ["a", "b"], step=0)
        self.assertEqual(result.shape, gradients.shape[1:])

    def test_spatial_probe_preserves_shape(self):
        gradients = torch.ones(20, 2, 3, 224, 224)
        probe = SpatialPatchProbe("highest", ratio=0.10)
        result = probe.apply(gradients, ["a", "b"], step=0)
        self.assertEqual(result.shape, gradients.shape[1:])
        self.assertGreater(float(result.eq(0).float().mean()), 0.05)


if __name__ == "__main__":
    unittest.main()
