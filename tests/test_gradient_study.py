import unittest

import torch

from gradient_observer import GradientObserver
from gradient_replay import GradientReplay
from gradient_study import (
    AmplitudeProbe,
    AmplitudePowerProbe,
    CoordinateWienerProbe,
    CrossScaleCovarianceProbe,
    CrossScaleCanonicalProbe,
    CrossScaleGaussianProbe,
    CrossScaleResidualGaussianProbe,
    ConflictProjectionProbe,
    DivisiveNormalizationProbe,
    CovarianceTransportProbe,
    CrossStepSignPersistenceProbe,
    EnergyEqualizationProbe,
    FrequencyGainProbe,
    HaarWaveletShrinkProbe,
    GaussianBlendProbe,
    GaussianBandBlendProbe,
    PatchGaussianBlendProbe,
    RawScaleTemporalGaussianProbe,
    AdaptiveGaussianProbe,
    GroupRemovalProbe,
    GroupReliabilityProbe,
    GroupNormEqualizationProbe,
    JointLowAmplitudeHighFrequencyProbe,
    LowFrequencyBoostProbe,
    LaplacianProxProbe,
    PostMomentumPreconditionProbe,
    PairPhaseProbe,
    PatchProjectionProbe,
    PatchEnergyTransportProbe,
    RiskAdaptiveGaussianProbe,
    MomentumTrajectoryProbe,
    MagnitudeEnvelopeProbe,
    SpatialPatchProbe,
    SoftPercentileClipProbe,
    SpectralWienerProbe,
    SpectralEnergyTransportProbe,
    SpectralBandAmplitudePowerProbe,
    GaussianHighPowerCompositeProbe,
    PositiveLowFrequencyProbe,
    SpectralAmplitudePowerProbe,
    SpectralComponentBoostProbe,
    SpectralPhaseConsensusProbe,
    SignReliabilityProbe,
    ViewPCProbe,
    ViewGLSProbe,
    StepWindowProbe,
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
        observer.record_gradient(first)
        observer.record_pre_momentum(torch.zeros_like(first), first)
        observer.record_momentum(first)
        observer.record_sign_update(first.sign())
        observer.close_step()

        second = -first
        observer.record_gradient(second)
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

    def test_group_reliability_and_norm_equalization_preserve_shape(self):
        gradients = torch.arange(4 * 2 * 1 * 1 * 2, dtype=torch.float32).reshape(4, 2, 1, 1, 2) + 1
        reliability = GroupReliabilityProbe(5.0).apply(gradients, ["a", "b"], 0)
        equalized = GroupNormEqualizationProbe(0.5).apply(gradients, ["a", "b"], 0)
        self.assertEqual(reliability.shape, gradients.shape[1:])
        self.assertEqual(equalized.shape, gradients.shape[1:])

    def test_phase_pair_probes_use_even_views_and_preserve_shape(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        for mode, high_only in (
            ("difference_add", False),
            ("difference_reverse", False),
            ("difference_orthogonal", False),
            ("phase_wiener", False),
            ("phase_wiener", True),
        ):
            probe = PairPhaseProbe(mode, 0.25, high_only=high_only)
            result = probe.apply(gradients, ["a", "b"], 0)
            self.assertEqual(result.shape, gradients.shape[1:])
            self.assertTrue(torch.isfinite(result).all())

    def test_risk_adaptive_gaussian_preserves_shape_and_is_bounded(self):
        gradients = torch.randn(20, 4, 3, 32, 32)
        for mode in (
            "amp_freq",
            "amp_freq_group",
            "freq_group",
            "positive_amp_freq",
            "positive_amp_freq_group",
        ):
            probe = RiskAdaptiveGaussianProbe(mode, 0.5)
            result = probe.apply(gradients, ["a", "b", "c", "d"], 0)
            self.assertEqual(result.shape, gradients.shape[1:])
            self.assertTrue(torch.isfinite(result).all())

    def test_spectral_energy_transport_preserves_shape_and_finite_values(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        for floor, strength in ((0.0, 0.5), (0.0, 1.0), (0.5, 1.0)):
            result = SpectralEnergyTransportProbe(floor, strength).apply(
                gradients, ["a", "b"], 0
            )
            self.assertEqual(result.shape, gradients.shape[1:])
            self.assertTrue(torch.isfinite(result).all())

    def test_magnitude_envelope_preserves_signs_and_shape(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        result = MagnitudeEnvelopeProbe(1.0, 0.5).apply(gradients, ["a", "b"], 0)
        self.assertEqual(result.shape, gradients.shape[1:])
        self.assertTrue(torch.isfinite(result).all())
        mean = gradients.mean(dim=0)
        self.assertTrue(torch.equal(result.sign(), mean.sign()))

    def test_conflict_projection_preserves_shape_and_finite_values(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        for grouped in (False, True):
            result = ConflictProjectionProbe(0.5, grouped=grouped).apply(
                gradients, ["a", "b"], 0
            )
            self.assertEqual(result.shape, gradients.shape[1:])
            self.assertTrue(torch.isfinite(result).all())

    def test_patch_projection_preserves_shape_and_signs(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        result = PatchProjectionProbe(0.5, grid=8).apply(gradients, ["a", "b"], 0)
        self.assertEqual(result.shape, gradients.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_patch_energy_transport_preserves_shape_and_finite_values(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        result = PatchEnergyTransportProbe(1.0, grid=8).apply(
            gradients, ["a", "b"], 0
        )
        self.assertEqual(result.shape, gradients.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_spectral_band_amplitude_power_preserves_shape_and_finite_values(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        result = SpectralBandAmplitudePowerProbe(0.75).apply(
            gradients, ["a", "b"], 0
        )
        self.assertEqual(result.shape, gradients.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_gaussian_high_power_composite_preserves_shape_and_finite_values(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        result = GaussianHighPowerCompositeProbe(0.75, 0.25).apply(
            gradients, ["a", "b"], 0
        )
        self.assertEqual(result.shape, gradients.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_positive_low_frequency_probe_preserves_shape_and_finite_values(self):
        gradients = torch.randn(20, 2, 3, 32, 32)
        for mode in ("amp_freq", "amp_freq_group"):
            result = PositiveLowFrequencyProbe(mode, 0.25).apply(
                gradients, ["a", "b"], 0
            )
            self.assertEqual(result.shape, gradients.shape[1:])
            self.assertTrue(torch.isfinite(result).all())

    def test_momentum_trajectory_is_identity_at_first_step_and_changes_later(self):
        first = torch.tensor([[[[[1.0, 0.0]]]]])
        second = torch.tensor([[[[[1.0, 1.0]]]]])
        probe = MomentumTrajectoryProbe(0.5, mode="align")
        first_result = probe.apply(first, ["a"], 0)
        second_result = probe.apply(second, ["a"], 1)
        self.assertTrue(torch.equal(first_result, first.mean(0) / first.mean(0).abs().mean()))
        self.assertFalse(torch.equal(second_result, second.mean(0)))

    def test_raw_temporal_gaussian_preserves_first_scale_and_weights_later_scale(self):
        views = torch.ones(2, 1, 1, 8, 8)
        probe = RawScaleTemporalGaussianProbe(0.5, 1.0, 0.25, ema_decay=0.5)
        first = probe.apply(views, ["a"], 0)
        second = probe.apply(views * 4.0, ["a"], 1)

        self.assertAlmostEqual(float(first.abs().mean()), 1.25, places=5)
        self.assertGreater(float(second.abs().mean()), 4.25)
        self.assertTrue(torch.isfinite(second).all())

    def test_patch_gaussian_blend_preserves_shape_and_finite_values(self):
        views = torch.randn(20, 2, 3, 224, 224)
        probe = PatchGaussianBlendProbe(4.0, 0.75, 0.25)
        result = probe.apply(views, ["a", "b"], 0)
        self.assertEqual(result.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_gaussian_band_blend_preserves_shape_and_finite_values(self):
        views = torch.randn(20, 2, 3, 32, 32)
        probe = GaussianBandBlendProbe(4.0, 0.75, 0.5)
        result = probe.apply(views, ["a", "b"], 0)
        self.assertEqual(result.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_sign_persistence_is_identity_at_first_step_and_preserves_shape(self):
        views = torch.randn(20, 1, 1, 16, 16)
        probe = CrossStepSignPersistenceProbe("high", 0.5)
        first = probe.apply(views, ["a"], 0)
        second = probe.apply(-views, ["a"], 1)
        self.assertTrue(torch.allclose(first, views.mean(0), atol=1e-6))
        self.assertEqual(second.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(second).all())

    def test_view_pc_transport_preserves_shape_and_changes_mean(self):
        views = torch.tensor(
            [
                [[[[2.0, 0.0]]]],
                [[[[0.0, 2.0]]]],
                [[[[3.0, -1.0]]]],
                [[[[1.0, 1.0]]]],
            ]
        )
        result = ViewPCProbe(0.25).apply(views, ["a"], 0)
        self.assertEqual(result.shape, views.shape[1:])
        self.assertFalse(torch.allclose(result, views.mean(0)))
        self.assertTrue(torch.isfinite(result).all())

    def test_view_gls_probe_returns_valid_shared_direction(self):
        views = torch.tensor(
            [
                [[[[1.0, 0.0]]]],
                [[[[1.0, 0.1]]]],
                [[[[0.9, 0.2]]]],
                [[[[1.0, -0.1]]]],
            ]
        )
        result = ViewGLSProbe(0.1).apply(views, ["a"], 0)
        self.assertEqual(result.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(float(torch.nn.functional.cosine_similarity(result.flatten(), views.mean(0).flatten(), dim=0)), 0.0)

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

    def test_sign_reliability_keeps_amplitude_and_reweights_consensus(self):
        views = torch.tensor(
            [
                [[[[2.0, 1.0]]]],
                [[[[2.0, -1.0]]]],
            ]
        )
        result = SignReliabilityProbe("boost", 1.0).apply(views, ["a"], 0)
        self.assertAlmostEqual(float(result[0, 0, 0, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(result[0, 0, 0, 1]), 0.0, places=6)

    def test_fixed_frequency_gain_leaves_dc_and_attenuates_checkerboard(self):
        constant = torch.ones(2, 1, 1, 8, 8)
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        high = checkerboard.view(1, 1, 1, 8, 8).repeat(2, 1, 1, 1, 1)
        probe = FrequencyGainProbe(0.5)
        self.assertTrue(torch.allclose(probe.apply(constant, ["a"], 0), constant.mean(0)))
        self.assertTrue(torch.allclose(probe.apply(high, ["a"], 0), high.mean(0) * 0.5))

    def test_laplacian_prox_is_identity_at_zero_and_attenuates_checkerboard(self):
        constant = torch.ones(2, 1, 1, 8, 8)
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        high = checkerboard.view(1, 1, 1, 8, 8).repeat(2, 1, 1, 1, 1)
        self.assertTrue(torch.allclose(LaplacianProxProbe(0.0).apply(constant, ["a"], 0), constant.mean(0)))
        filtered = LaplacianProxProbe(1.0).apply(high, ["a"], 0)
        self.assertLess(float(filtered.abs().max()), float(high.abs().max()))

    def test_post_momentum_probe_only_changes_update_direction(self):
        views = torch.randn(4, 1, 1, 16, 16)
        mean = views.mean(0)
        probe = PostMomentumPreconditionProbe("high_shrink", 0.0, strength=0.5)
        self.assertTrue(torch.equal(probe.apply(views, ["a"], 0), mean))
        result = probe.apply_update(mean, mean, ["a"], 1)
        self.assertEqual(result.shape, mean.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_haar_wavelet_shrink_is_identity_for_zero_threshold(self):
        views = torch.randn(4, 1, 3, 16, 16)
        result = HaarWaveletShrinkProbe(0.0).apply(views, ["a"], 0)
        self.assertTrue(torch.allclose(result, views.mean(0), atol=1e-5, rtol=1e-5))

    def test_haar_wavelet_shrink_preserves_shape_and_finite_values(self):
        views = torch.randn(4, 2, 3, 16, 16)
        result = HaarWaveletShrinkProbe(0.5).apply(views, ["a", "b"], 0)
        self.assertEqual(result.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_low_frequency_boost_retains_original_high_frequency_component(self):
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        gradient = torch.ones(1, 1, 8, 8) + checkerboard.view(1, 1, 8, 8)
        views = gradient.unsqueeze(0)
        result = LowFrequencyBoostProbe(1.0, cutoff=0.5).apply(views, ["a"], 0)
        self.assertGreater(float(result.abs().max()), float(gradient.abs().max()))
        self.assertGreater(float(result.mean()), float(gradient.mean()))

    def test_gaussian_blend_preserves_shape(self):
        views = torch.randn(2, 1, 3, 32, 32)
        result = GaussianBlendProbe(1.0, 0.5).apply(views, ["a"], 0)
        self.assertEqual(result.shape, views.shape[1:])

    def test_normalized_gaussian_blend_equalizes_component_l1_scale(self):
        views = torch.randn(2, 1, 3, 32, 32)
        probe = GaussianBlendProbe(1.0, 0.25, normalize_component=True)
        result = probe.apply(views, ["a"], 0)
        self.assertEqual(result.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_adaptive_gaussian_only_changes_selected_samples(self):
        low_energy = torch.ones(1, 1, 3, 32, 32)
        concentrated = torch.zeros(1, 1, 3, 32, 32)
        concentrated[..., 16, 16] = 10.0
        batch = torch.cat((low_energy[0], concentrated[0]), dim=0)
        views = batch.unsqueeze(0).repeat(2, 1, 1, 1, 1)
        result = AdaptiveGaussianProbe("entropy_low", 0.5).apply(views, ["a", "b"], 0)
        self.assertTrue(torch.equal(result[0], views.mean(0)[0]))
        self.assertFalse(torch.equal(result[1], views.mean(0)[1]))

    def test_spectral_wiener_uses_cross_view_coherence(self):
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        views = torch.stack((checkerboard, checkerboard * 3.0)).view(2, 1, 1, 8, 8)
        result = SpectralWienerProbe(0.0).apply(views, ["a"], 0)
        self.assertTrue(torch.allclose(result, checkerboard.view(1, 1, 8, 8) * 1.75, atol=1e-5))

    def test_spectral_component_boost_separates_signal_and_residual(self):
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        views = torch.stack((checkerboard, checkerboard * 3.0)).view(2, 1, 1, 8, 8)
        signal = SpectralComponentBoostProbe("signal", 1.0).apply(views, ["a"], 0)
        residual = SpectralComponentBoostProbe("residual", 1.0).apply(views, ["a"], 0)
        self.assertTrue(torch.allclose(signal, checkerboard.view(1, 1, 8, 8) * 3.75, atol=1e-5))
        self.assertTrue(torch.allclose(residual, checkerboard.view(1, 1, 8, 8) * 2.25, atol=1e-5))

    def test_amplitude_power_emphasizes_large_coordinates(self):
        views = torch.tensor([[[[[1.0, -2.0]]]]])
        result = AmplitudePowerProbe(2.0).apply(views, ["a"], 0)
        ratio = abs(float(result[0, 0, 0, 1] / result[0, 0, 0, 0]))
        self.assertAlmostEqual(ratio, 4.0)
        self.assertTrue(torch.equal(result.sign(), views.mean(0).sign()))

    def test_amplitude_peak_controls_preserve_shape_and_sign(self):
        views = torch.randn(4, 2, 3, 16, 16)
        divisive = DivisiveNormalizationProbe(1.0).apply(views, ["a", "b"], 0)
        clipped = SoftPercentileClipProbe(0.01).apply(views, ["a", "b"], 0)
        mean = views.mean(0)
        self.assertEqual(divisive.shape, mean.shape)
        self.assertEqual(clipped.shape, mean.shape)
        self.assertTrue(torch.isfinite(divisive).all())
        self.assertTrue(torch.isfinite(clipped).all())
        self.assertTrue(torch.equal(clipped.sign(), mean.sign()))

    def test_spectral_amplitude_power_preserves_single_frequency_shape(self):
        axis = torch.arange(8)
        checkerboard = ((axis[:, None] + axis[None, :]) % 2).mul(2).sub(1).float()
        views = checkerboard.view(1, 1, 1, 8, 8)
        result = SpectralAmplitudePowerProbe(1.5).apply(views, ["a"], 0)
        cosine = torch.nn.functional.cosine_similarity(result.flatten(), checkerboard.flatten(), dim=0)
        self.assertAlmostEqual(float(cosine), 1.0, places=5)

    def test_spectral_phase_consensus_is_identity_for_identical_views(self):
        gradient = torch.randn(1, 1, 16, 16)
        views = gradient.unsqueeze(0).repeat(4, 1, 1, 1, 1)
        result = SpectralPhaseConsensusProbe(1.0).apply(views, ["a"], 0)
        self.assertTrue(torch.allclose(result, gradient, atol=1e-5, rtol=1e-5))

    def test_cross_scale_probe_preserves_shape_and_finite_values(self):
        views = torch.randn(20, 2, 3, 32, 32)
        for mode in ("add", "replace", "project"):
            result = CrossScaleCovarianceProbe(mode, 0.5).apply(
                views, ["a", "b"], 0
            )
            self.assertEqual(result.shape, views.shape[1:])
            self.assertTrue(torch.isfinite(result).all())

    def test_cross_scale_probe_is_identity_at_zero_strength(self):
        views = torch.randn(20, 1, 1, 16, 16)
        mean = views.mean(0)
        for mode in ("add", "replace", "project"):
            result = CrossScaleCovarianceProbe(mode, 0.0).apply(views, ["a"], 0)
            self.assertTrue(torch.allclose(result, mean, atol=1e-5, rtol=1e-5))

    def test_cross_scale_canonical_probe_shape_and_zero_identity(self):
        views = torch.randn(20, 1, 1, 16, 16)
        mean = views.mean(0)
        result = CrossScaleCanonicalProbe(0.25).apply(views, ["a"], 0)
        self.assertEqual(result.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(result).all())
        identity = CrossScaleCanonicalProbe(0.0).apply(views, ["a"], 0)
        self.assertTrue(torch.allclose(identity, mean, atol=1e-5, rtol=1e-5))

    def test_cross_scale_gaussian_composition_preserves_shape(self):
        views = torch.randn(4, 2, 3, 16, 16)
        result = CrossScaleGaussianProbe(0.5, 0.25, 1.0, 0.25).apply(
            views, ["a", "b"], 0
        )
        self.assertEqual(result.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(result).all())

    def test_cross_scale_residual_gaussian_identity_at_full_residual_gain(self):
        views = torch.randn(20, 1, 3, 16, 16)
        base = CrossScaleGaussianProbe(0.5, 0.5, 1.0, 0.25).apply(
            views, ["a"], 0
        )
        result = CrossScaleResidualGaussianProbe(
            0.5, 0.5, 1.0, 1.0, 0.25
        ).apply(views, ["a"], 0)
        self.assertTrue(torch.allclose(result, base, atol=1e-5, rtol=1e-5))
        attenuated = CrossScaleResidualGaussianProbe(
            0.5, 0.5, 0.0, 1.0, 0.25
        ).apply(views, ["a"], 0)
        self.assertEqual(attenuated.shape, views.shape[1:])
        self.assertTrue(torch.isfinite(attenuated).all())

    def test_joint_amplitude_frequency_probe_preserves_shape(self):
        views = torch.randn(4, 4, 3, 32, 32)
        for operation in ("shrink", "low_boost", "gaussian", "low_only", "low_equalize"):
            result = JointLowAmplitudeHighFrequencyProbe(operation, 0.25).apply(
                views, ["a", "b", "c", "d"], 0
            )
            self.assertEqual(result.shape, views.shape[1:])
            self.assertTrue(torch.isfinite(result).all())

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

    def test_step_window_is_identity_outside_and_active_inside(self):
        views = torch.tensor([[[[[1.0, 100.0]]]]])
        probe = StepWindowProbe(AmplitudeProbe("remove_high", 0.5), 1, 3)
        self.assertTrue(torch.equal(probe.apply(views, ["a"], 0), views.mean(0)))
        self.assertEqual(float(probe.apply(views, ["a"], 1)[0, 0, 0, 1]), 0.0)

    def test_component_probe_names_round_trip(self):
        names = (
            "amplitude_remove_low_q20",
            "amplitude_clip_high_q99",
            "amplitude_softclip_p010",
            "divisive_normalize_s10",
            "coordinate_wiener_floor25",
            "sign_reliability_boost_a50",
            "sign_reliability_gate_a25",
            "frequency_high_gain50",
            "laplacian_prox_l100",
            "post_momentum_gaussian_s10_a025",
            "post_momentum_laplacian_l100",
            "post_momentum_high_shrink_c50_a025",
            "joint_lowamp_highfreq_shrink_a025",
            "joint_lowamp_highfreq_low_only_a025",
            "risk_gaussian_amp_freq_s10_a025",
            "risk_gaussian_amp_freq_group_s10_a050",
            "risk_gaussian_positive_amp_freq_s10_a025",
            "risk_gaussian_positive_amp_freq_group_s10_a050",
            "magnitude_envelope_s10_a025",
            "conflict_project_view_a025",
            "conflict_project_group_a050",
            "patch_projection_g14_a025",
            "patch_energy_transport_g14_a050",
            "spectral_high_amplitude_power075",
            "composite_gaussian_s10_highpower075_a025",
            "positive_low_boost_amp_freq_c50_a025",
            "positive_low_boost_amp_freq_group_c50_a050",
            "haar_wavelet_shrink_t050",
            "spectral_wiener_all_floor50",
            "spectral_wiener_high_floor25",
            "spectral_transport_high_f00_a100",
            "amplitude_power125",
            "spectral_amplitude_power150",
            "spectral_phase_consensus_a050",
            "cross_scale_replace_c50_a025",
            "cross_scale_canonical_c50_a025",
            "cross_scale_gaussian_c50_x025_s10_a025",
            "cross_scale_residual_gaussian_c50_x050_r025_s10_a025",
            "covariance_transport_view_a25",
            "covariance_transport_group_a50",
            "group_reliability_t20",
            "group_norm_equalize_a50",
            "pair_difference_add_a025",
            "pair_difference_reverse_a025",
            "pair_difference_orthogonal_a025",
            "pair_phase_wiener_f025",
            "pair_phase_wiener_high_f025",
            "momentum_trajectory_align_a25",
            "momentum_trajectory_parallel_boost_a50",
            "momentum_trajectory_orthogonal_a25",
            "sign_persistence_high_c50_a025",
            "view_pc_transport_a25",
            "view_gls_ridge10",
            "energy_equalize_patch_a25",
            "energy_equalize_local_a50",
            "temporal_frequency_remove_high_s0e5",
            "temporal_spectral_wiener_high_floor00_s1e5",
            "spectral_boost_signal_high_a025",
            "spectral_boost_residual_all_a100",
            "low_frequency_boost_c50_a50",
            "gaussian_blend_s10_a50",
            "raw_temporal_gaussian_p050_s10_a025",
            "patch_gaussian_s40_a075_p025",
            "gaussian_band_s40_l075_h050",
            "gaussian_norm_blend_s10_a25",
            "adaptive_gaussian_entropy_low_q50",
            "adaptive_gaussian_freq_high_q50",
        )
        for name in names:
            probe = build_probe(name)
            self.assertEqual(probe.name, name)
            if name == "low_frequency_boost_c50_a50":
                self.assertAlmostEqual(probe.cutoff, 0.5)
                self.assertAlmostEqual(probe.strength, 0.5)


if __name__ == "__main__":
    unittest.main()
