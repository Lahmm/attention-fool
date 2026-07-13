import unittest

import torch

from gradient_observer import GradientObserver
from gradient_replay import GradientReplay
from gradient_study import (
    AmplitudeProbe,
    AmplitudePowerProbe,
    CoordinateWienerProbe,
    CovarianceTransportProbe,
    EnergyEqualizationProbe,
    FrequencyGainProbe,
    GroupRemovalProbe,
    SpatialPatchProbe,
    SpectralWienerProbe,
    SpectralAmplitudePowerProbe,
    build_probe,
)


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

    def test_amplitude_remove_and_clip_preserve_signs(self):
        gradient = torch.tensor([[[[[1.0, -2.0, 3.0, -100.0]]]]])
        removed = AmplitudeProbe("remove_high", 0.75).apply(gradient, ["a"], 0)
        clipped = AmplitudeProbe("clip_high", 0.75).apply(gradient, ["a"], 0)
        self.assertEqual(float(removed[0, 0, 0, 3]), 0.0)
        self.assertLess(float(clipped[0, 0, 0, 3]), 0.0)
        self.assertLess(abs(float(clipped[0, 0, 0, 3])), 100.0)
        self.assertTrue(torch.equal(clipped.sign(), gradient.mean(dim=0).sign()))

    def test_coordinate_wiener_preserves_stable_and_shrinks_noisy_coordinates(self):
        views = torch.tensor(
            [
                [[[[2.0, 1.0]]]],
                [[[[2.0, 3.0]]]],
            ]
        )
        result = CoordinateWienerProbe(0.0).apply(views, ["a"], 0)
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(result[0, 0, 0, 1]), 1.75, places=6)

    def test_fixed_frequency_gain_leaves_dc_and_attenuates_checkerboard(self):
        constant = torch.ones(2, 1, 1, 8, 8)
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        high = checkerboard.view(1, 1, 1, 8, 8).repeat(2, 1, 1, 1, 1)
        probe = FrequencyGainProbe(0.5)
        self.assertTrue(torch.allclose(probe.apply(constant, ["a"], 0), constant.mean(0)))
        self.assertTrue(torch.allclose(probe.apply(high, ["a"], 0), high.mean(0) * 0.5))

    def test_spectral_wiener_uses_cross_view_coherence(self):
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        views = torch.stack((checkerboard, checkerboard * 3.0)).view(2, 1, 1, 8, 8)
        result = SpectralWienerProbe(0.0).apply(views, ["a"], 0)
        self.assertTrue(torch.allclose(result, checkerboard.view(1, 1, 8, 8) * 1.75, atol=1e-5))

    def test_amplitude_power_emphasizes_large_coordinates(self):
        views = torch.tensor([[[[[1.0, -2.0]]]]])
        result = AmplitudePowerProbe(2.0).apply(views, ["a"], 0)
        ratio = abs(float(result[0, 0, 0, 1] / result[0, 0, 0, 0]))
        self.assertAlmostEqual(ratio, 4.0)
        self.assertTrue(torch.equal(result.sign(), views.mean(0).sign()))

    def test_spectral_amplitude_power_preserves_single_frequency_shape(self):
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        views = checkerboard.view(1, 1, 1, 8, 8)
        result = SpectralAmplitudePowerProbe(1.5).apply(views, ["a"], 0)
        cosine = torch.nn.functional.cosine_similarity(result.flatten(), checkerboard.flatten(), dim=0)
        self.assertAlmostEqual(float(cosine), 1.0, places=5)

    def test_covariance_transport_uses_structured_view_variation(self):
        views = torch.tensor(
            [
                [[[[3.0, 0.0]]]],
                [[[[1.0, 2.0]]]],
                [[[[2.0, 1.0]]]],
            ]
        )
        mean = views.mean(0)
        result = CovarianceTransportProbe(0.5).apply(views, ["a"], 0)
        self.assertEqual(result.shape, mean.shape)
        self.assertGreater(float(torch.nn.functional.cosine_similarity(result.flatten(), mean.flatten(), dim=0)), 0.0)
        self.assertFalse(torch.allclose(result, mean))

    def test_patch_energy_equalization_preserves_signs_and_reduces_energy_ratio(self):
        gradient = torch.ones(1, 1, 1, 32, 32)
        gradient[..., :16, :16] = 4.0
        probe = EnergyEqualizationProbe(0.5, scope="patch")
        result = probe.apply(gradient, ["a"], 0)
        before = gradient[..., :16, :16].square().mean() / gradient[..., 16:, 16:].square().mean()
        after = result[..., :16, :16].square().mean() / result[..., 16:, 16:].square().mean()
        self.assertLess(float(after), float(before))
        self.assertTrue(torch.equal(result.sign(), gradient.mean(0).sign()))

    def test_local_energy_equalization_preserves_shape(self):
        gradient = torch.randn(2, 1, 3, 32, 32)
        result = EnergyEqualizationProbe(0.25, scope="local").apply(gradient, ["a"], 0)
        self.assertEqual(result.shape, gradient.shape[1:])

    def test_component_probe_names_round_trip(self):
        names = (
            "amplitude_remove_low_q20",
            "amplitude_clip_high_q99",
            "coordinate_wiener_floor25",
            "frequency_high_gain50",
            "spectral_wiener_all_floor50",
            "spectral_wiener_high_floor25",
            "amplitude_power125",
            "spectral_amplitude_power150",
            "covariance_transport_view_a25",
            "covariance_transport_group_a50",
            "energy_equalize_patch_a25",
            "energy_equalize_local_a50",
        )
        for name in names:
            self.assertEqual(build_probe(name).name, name)


if __name__ == "__main__":
    unittest.main()
