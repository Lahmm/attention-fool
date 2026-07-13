from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


FEATURE_FAMILIES = {
    "group": (
        "group_within_cosine_mean",
        "group_within_cosine_min",
        "group_to_rest_cosine_mean",
        "group_to_rest_cosine_min",
        "group_norm_cv",
        "group_loo_influence_max",
    ),
    "spatial": ("spatial_entropy", "spatial_gini", "spatial_top5_energy", "agg_kurtosis"),
    "frequency": ("freq_low", "freq_mid", "freq_high"),
}


class GradientProbe(Protocol):
    name: str

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor: ...


def _stable_index(sample_id: str, step: int, count: int, salt: str) -> int:
    digest = hashlib.blake2b(f"{sample_id}|{step}|{salt}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % count


@dataclass
class GroupRemovalProbe:
    selection: str

    @property
    def name(self) -> str:
        return f"group_remove_{self.selection}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        view_count, batch_size = view_gradients.shape[:2]
        if view_count % 2:
            raise ValueError("group probes require paired views.")
        group_count = view_count // 2
        group_means = view_gradients.view(group_count, 2, *view_gradients.shape[1:]).mean(dim=1)
        flat = group_means.flatten(2).permute(1, 0, 2)  # [B,G,D]
        reliability = []
        for group_index in range(group_count):
            rest = (flat.sum(dim=1) - flat[:, group_index]) / (group_count - 1)
            reliability.append(F.cosine_similarity(flat[:, group_index], rest, dim=1))
        reliability_tensor = torch.stack(reliability, dim=1)
        outputs = []
        for batch_index, sample_id in enumerate(sample_ids):
            if self.selection == "lowest":
                removed = int(reliability_tensor[batch_index].argmin())
            elif self.selection == "highest":
                removed = int(reliability_tensor[batch_index].argmax())
            elif self.selection == "random":
                removed = _stable_index(sample_id, step, group_count, self.name)
            else:
                raise ValueError(f"unsupported group selection: {self.selection}")
            keep = [index for index in range(group_count) if index != removed]
            outputs.append(group_means[keep, batch_index].mean(dim=0))
        return torch.stack(outputs, dim=0)


@dataclass
class GroupReliabilityProbe:
    """Softly upweight groups whose direction agrees with the other groups."""

    temperature: float

    @property
    def name(self) -> str:
        return f"group_reliability_t{int(round(self.temperature * 10)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if view_gradients.size(0) % 2:
            raise ValueError("group reliability requires paired views.")
        group_means = view_gradients.view(view_gradients.size(0) // 2, 2, *view_gradients.shape[1:]).mean(dim=1)
        flat = group_means.flatten(2).permute(1, 0, 2)  # [B,G,D]
        total = flat.sum(dim=1, keepdim=True)
        rest = (total - flat) / max(1, flat.size(1) - 1)
        reliability = F.cosine_similarity(flat, rest, dim=2)
        weights = torch.softmax(self.temperature * reliability, dim=1)
        return torch.einsum("bg,bgchw->bchw", weights, group_means.permute(1, 0, 2, 3, 4))


@dataclass
class GroupNormEqualizationProbe:
    """Equalize group contribution norms with a bounded power-law gain."""

    strength: float

    @property
    def name(self) -> str:
        return f"group_norm_equalize_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if view_gradients.size(0) % 2:
            raise ValueError("group norm equalization requires paired views.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("group norm equalization strength must be in [0, 1].")
        groups = view_gradients.view(view_gradients.size(0) // 2, 2, *view_gradients.shape[1:]).mean(dim=1)
        norms = groups.flatten(2).norm(dim=2, keepdim=True)
        target = norms.mean(dim=0, keepdim=True)
        gains = (target / norms.clamp_min(1e-20)).pow(self.strength).clamp(0.5, 2.0)
        return (groups * gains.view(groups.size(0), groups.size(1), 1, 1, 1)).mean(dim=0)


@dataclass
class SpatialPatchProbe:
    selection: str
    ratio: float = 0.10

    @property
    def name(self) -> str:
        return f"spatial_patch_{self.selection}_{self.ratio:.2f}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        gradient = view_gradients.mean(dim=0)
        batch_size, _, height, width = gradient.shape
        grid = 14
        if height % grid or width % grid:
            raise ValueError("spatial probes require dimensions divisible by 14.")
        patch_energy = F.avg_pool2d(
            gradient.pow(2).sum(dim=1, keepdim=True),
            kernel_size=(height // grid, width // grid),
            stride=(height // grid, width // grid),
        ).flatten(1)
        count = max(1, int(round(patch_energy.size(1) * self.ratio)))
        outputs = []
        for index, sample_id in enumerate(sample_ids):
            if self.selection == "highest":
                selected = patch_energy[index].topk(count, largest=True).indices
            elif self.selection == "lowest":
                selected = patch_energy[index].topk(count, largest=False).indices
            elif self.selection == "random":
                generator = torch.Generator(device=gradient.device)
                generator.manual_seed(_stable_index(sample_id, step, 2**31 - 1, self.name))
                selected = torch.randperm(patch_energy.size(1), device=gradient.device, generator=generator)[:count]
            else:
                raise ValueError(f"unsupported spatial selection: {self.selection}")
            patch_mask = torch.zeros(patch_energy.size(1), device=gradient.device, dtype=gradient.dtype)
            patch_mask[selected] = 1.0
            image_mask = (
                patch_mask.view(1, grid, grid)
                .repeat_interleave(height // grid, dim=1)
                .repeat_interleave(width // grid, dim=2)
            )
            outputs.append(gradient[index] * (1.0 - image_mask))
        return torch.stack(outputs, dim=0)


@dataclass
class FrequencyBandProbe:
    band: str

    @property
    def name(self) -> str:
        return f"frequency_remove_{self.band}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.arange(height, device=gradient.device, dtype=torch.float32),
            torch.arange(width, device=gradient.device, dtype=torch.float32),
            indexing="ij",
        )
        radius = ((yy - height / 2.0).square() + (xx - width / 2.0).square()).sqrt()
        maximum = radius.max()
        masks = {
            "low": radius <= maximum * 0.25,
            "mid": (radius > maximum * 0.25) & (radius <= maximum * 0.50),
            "high": radius > maximum * 0.50,
        }
        if self.band not in (*masks, "random"):
            raise ValueError(f"unsupported frequency band: {self.band}")
        spectrum = torch.fft.fftshift(torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1))
        if self.band == "random":
            masked = []
            bands = ("low", "mid", "high")
            for index, sample_id in enumerate(sample_ids):
                band = bands[_stable_index(sample_id, step, len(bands), self.name)]
                masked.append(
                    spectrum[index].masked_fill(masks[band].view(1, height, width), 0)
                )
            spectrum = torch.stack(masked, dim=0)
        else:
            spectrum = spectrum.masked_fill(masks[self.band].view(1, 1, height, width), 0)
        return torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)).real


def _sample_quantile(values: torch.Tensor, quantile: float) -> torch.Tensor:
    """Return one quantile per sample with broadcastable image dimensions."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")
    return torch.quantile(values.flatten(1), quantile, dim=1).view(-1, 1, 1, 1)


@dataclass
class AmplitudeProbe:
    """Causal interventions on spatial-coordinate gradient magnitudes.

    The intervention keeps each retained coordinate's sign and therefore tests
    relative amplitude information before the existing MI normalization.
    """

    operation: str
    quantile: float

    @property
    def name(self) -> str:
        return f"amplitude_{self.operation}_q{int(round(self.quantile * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        gradient = view_gradients.mean(dim=0)
        magnitude = gradient.abs()
        threshold = _sample_quantile(magnitude, self.quantile)
        if self.operation == "remove_low":
            return gradient.masked_fill(magnitude <= threshold, 0.0)
        if self.operation == "remove_high":
            return gradient.masked_fill(magnitude >= threshold, 0.0)
        if self.operation == "clip_high":
            return gradient.sign() * magnitude.minimum(threshold)
        raise ValueError(f"unsupported amplitude operation: {self.operation}")


@dataclass
class AmplitudePowerProbe:
    """Redistribute spatial-coordinate magnitude while retaining every sign."""

    power: float

    @property
    def name(self) -> str:
        return f"amplitude_power{int(round(self.power * 100)):03d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.power <= 0:
            raise ValueError("amplitude power must be positive.")
        gradient = view_gradients.mean(dim=0)
        scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-20)
        return gradient.sign() * (gradient.abs() / scale).pow(self.power)


@dataclass
class CoordinateWienerProbe:
    """Empirical-Bayes shrinkage of coordinates unreliable across views.

    Views are treated as repeated noisy measurements g_v = s + e_v.  The
    variance of their mean is var(g_v)/V; subtracting it from |mean|^2 gives a
    non-negative signal-power estimate.  The resulting Wiener gain changes
    gradient composition without changing augmentation or update mechanics.
    """

    floor: float

    @property
    def name(self) -> str:
        return f"coordinate_wiener_floor{int(round(self.floor * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("Wiener floor must be in [0, 1].")
        mean = view_gradients.mean(dim=0)
        noise = view_gradients.var(dim=0, unbiased=False) / view_gradients.size(0)
        signal = (mean.square() - noise).clamp_min(0.0)
        gain = signal / (signal + noise).clamp_min(1e-20)
        gain = self.floor + (1.0 - self.floor) * gain
        return mean * gain


def _frequency_radius(height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return ((yy - height / 2.0).square() + (xx - width / 2.0).square()).sqrt()


@dataclass
class FrequencyGainProbe:
    """Fixed high-frequency gain used as a matched component-mixing control."""

    high_gain: float

    @property
    def name(self) -> str:
        return f"frequency_high_gain{int(round(self.high_gain * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.high_gain <= 1.0:
            raise ValueError("high-frequency gain must be in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        radius = _frequency_radius(height, width, gradient.device)
        high = radius > radius.max() * 0.50
        spectrum = torch.fft.fftshift(torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1))
        gain = torch.ones_like(radius)
        gain[high] = self.high_gain
        spectrum = spectrum * gain.view(1, 1, height, width)
        return torch.fft.ifft2(
            torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)
        ).real


@dataclass
class LowFrequencyBoostProbe:
    """Add a low-frequency projection while retaining the complete gradient."""

    strength: float
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return f"low_frequency_boost_c{int(round(self.cutoff * 100)):02d}_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.strength < 0 or not 0.0 < self.cutoff <= 1.0:
            raise ValueError("low-frequency boost requires strength >= 0 and cutoff in (0, 1].")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        radius = _frequency_radius(height, width, gradient.device)
        low = radius <= radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1))
        low_component = spectrum * low.view(1, 1, height, width)
        low_component = torch.fft.ifft2(
            torch.fft.ifftshift(low_component, dim=(-2, -1)), dim=(-2, -1)
        ).real
        return gradient + self.strength * low_component


@dataclass
class GaussianBlendProbe:
    """Add a spatially smoothed projection while retaining the original detail."""

    sigma: float
    strength: float

    @property
    def name(self) -> str:
        return f"gaussian_blend_s{int(round(self.sigma * 10)):02d}_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.sigma <= 0 or self.strength < 0:
            raise ValueError("Gaussian blend requires sigma > 0 and strength >= 0.")
        gradient = view_gradients.mean(dim=0)
        radius = max(1, int(round(3 * self.sigma)))
        axis = torch.arange(-radius, radius + 1, device=gradient.device, dtype=gradient.dtype)
        kernel_1d = torch.exp(-0.5 * (axis / self.sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(
            1, 1, 2 * radius + 1, 2 * radius + 1
        )
        kernel = kernel.repeat(gradient.size(1), 1, 1, 1)
        smoothed = F.conv2d(
            F.pad(gradient, (radius, radius, radius, radius), mode="reflect"),
            kernel,
            groups=gradient.size(1),
        )
        return gradient + self.strength * smoothed


@dataclass
class SpectralWienerProbe:
    """Retain Fourier components supported coherently by independent views.

    Unlike a radial low-pass filter, this estimates signal and noise power for
    every complex Fourier coefficient.  ``high_only`` leaves low/mid frequency
    components unchanged and decomposes only the empirically harmful band.
    """

    floor: float
    high_only: bool = False

    @property
    def name(self) -> str:
        scope = "high" if self.high_only else "all"
        return f"spectral_wiener_{scope}_floor{int(round(self.floor * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("Wiener floor must be in [0, 1].")
        spectra = torch.fft.fftshift(
            torch.fft.fft2(view_gradients, dim=(-2, -1)), dim=(-2, -1)
        )
        mean = spectra.mean(dim=0)
        noise = (spectra - mean.unsqueeze(0)).abs().square().mean(dim=0) / spectra.size(0)
        signal = (mean.abs().square() - noise).clamp_min(0.0)
        gain = signal / (signal + noise).clamp_min(1e-20)
        gain = self.floor + (1.0 - self.floor) * gain
        if self.high_only:
            height, width = mean.shape[-2:]
            radius = _frequency_radius(height, width, mean.device)
            high = radius > radius.max() * 0.50
            gain = torch.where(high.view(1, 1, height, width), gain, torch.ones_like(gain))
        filtered = mean * gain
        return torch.fft.ifft2(
            torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)
        ).real


@dataclass
class SpectralComponentBoostProbe:
    """Amplify coherent signal or uncertain residual Fourier components."""

    component: str
    strength: float
    high_only: bool = True

    @property
    def name(self) -> str:
        scope = "high" if self.high_only else "all"
        return f"spectral_boost_{self.component}_{scope}_a{int(round(self.strength * 100)):03d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.component not in ("signal", "residual"):
            raise ValueError("spectral boost component must be signal or residual.")
        if self.strength < 0:
            raise ValueError("spectral boost strength must be non-negative.")
        spectra = torch.fft.fftshift(
            torch.fft.fft2(view_gradients, dim=(-2, -1)), dim=(-2, -1)
        )
        mean = spectra.mean(dim=0)
        noise = (spectra - mean.unsqueeze(0)).abs().square().mean(dim=0) / spectra.size(0)
        signal = (mean.abs().square() - noise).clamp_min(0.0)
        wiener = signal / (signal + noise).clamp_min(1e-20)
        component_weight = wiener if self.component == "signal" else 1.0 - wiener
        gain = 1.0 + self.strength * component_weight
        if self.high_only:
            height, width = mean.shape[-2:]
            radius = _frequency_radius(height, width, mean.device)
            high = radius > radius.max() * 0.50
            gain = torch.where(high.view(1, 1, height, width), gain, torch.ones_like(gain))
        transformed = mean * gain
        return torch.fft.ifft2(
            torch.fft.ifftshift(transformed, dim=(-2, -1)), dim=(-2, -1)
        ).real


@dataclass
class SpectralAmplitudePowerProbe:
    """Power-law Fourier magnitudes while preserving the mean gradient phase.

    This combines amplitude and frequency without assuming that radial high
    frequencies are uniformly harmful.  Large Fourier components are amplified
    for power > 1 and compressed for power < 1, regardless of their radius.
    """

    power: float

    @property
    def name(self) -> str:
        return f"spectral_amplitude_power{int(round(self.power * 100)):03d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.power <= 0:
            raise ValueError("spectral amplitude power must be positive.")
        gradient = view_gradients.mean(dim=0)
        spectrum = torch.fft.fft2(gradient, dim=(-2, -1))
        magnitude = spectrum.abs()
        scale = magnitude.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-20)
        phase = spectrum / magnitude.clamp_min(1e-20)
        transformed = phase * (magnitude / scale).pow(self.power)
        return torch.fft.ifft2(transformed, dim=(-2, -1)).real


@dataclass
class CovarianceTransportProbe:
    """Add the cross-view covariance component supported along the mean.

    C*mean is the first power-iteration direction of the centered view-gradient
    covariance.  It emphasizes structured augmentation variation aligned with
    the attack direction instead of discarding view disagreement as noise.
    """

    strength: float
    grouped: bool = False

    @property
    def name(self) -> str:
        scope = "group" if self.grouped else "view"
        return f"covariance_transport_{scope}_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.strength < 0:
            raise ValueError("covariance transport strength must be non-negative.")
        components = view_gradients
        if self.grouped:
            if view_gradients.size(0) % 2:
                raise ValueError("group covariance requires paired views.")
            components = view_gradients.view(
                view_gradients.size(0) // 2, 2, *view_gradients.shape[1:]
            ).mean(dim=1)
        mean = view_gradients.mean(dim=0)
        flat = components.flatten(2).permute(1, 0, 2)  # [B,K,D]
        mean_flat = mean.flatten(1)
        centered = flat - flat.mean(dim=1, keepdim=True)
        projection = torch.einsum("bkd,bd->bk", centered, mean_flat)
        transported = torch.einsum("bkd,bk->bd", centered, projection) / components.size(0)
        transported_norm = transported.norm(dim=1, keepdim=True).clamp_min(1e-20)
        mean_norm = mean_flat.norm(dim=1, keepdim=True)
        transported = transported * (mean_norm / transported_norm)
        return (mean_flat + self.strength * transported).view_as(mean)


@dataclass
class EnergyEqualizationProbe:
    """Redistribute gradient energy without deleting signs or components.

    ``patch`` uses the ViT-B/16 token grid and assigns one bounded gain to each
    16x16 patch. ``local`` uses a smooth 17x17 RMS envelope.  Both preserve the
    internal direction of energetic regions while increasing spatial entropy.
    """

    strength: float
    scope: str = "patch"

    @property
    def name(self) -> str:
        return f"energy_equalize_{self.scope}_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("energy equalization strength must be in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        global_rms = gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-20)
        if self.scope == "patch":
            if gradient.size(-2) % 16 or gradient.size(-1) % 16:
                raise ValueError("patch energy equalization requires dimensions divisible by 16.")
            local_rms = F.avg_pool2d(
                gradient.square().mean(dim=1, keepdim=True), kernel_size=16, stride=16
            ).sqrt().clamp_min(1e-20)
            gain = (global_rms / local_rms).pow(self.strength).clamp(0.5, 2.0)
            gain = gain.repeat_interleave(16, dim=2).repeat_interleave(16, dim=3)
        elif self.scope == "local":
            local_rms = F.avg_pool2d(
                F.pad(
                    gradient.square().mean(dim=1, keepdim=True),
                    (8, 8, 8, 8),
                    mode="reflect",
                ),
                kernel_size=17,
                stride=1,
            ).sqrt().clamp_min(1e-20)
            gain = (global_rms / local_rms).pow(self.strength).clamp(0.5, 2.0)
        else:
            raise ValueError(f"unsupported energy equalization scope: {self.scope}")
        return gradient * gain


@dataclass
class StepWindowProbe:
    """Apply a component intervention only on a half-open step interval."""

    inner: GradientProbe
    start: int
    end: int

    @property
    def name(self) -> str:
        return f"temporal_{self.inner.name}_s{self.start}e{self.end}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        if self.start <= step < self.end:
            return self.inner.apply(view_gradients, sample_ids, step)
        return view_gradients.mean(dim=0)


def build_probe(name: str) -> GradientProbe:
    if name.startswith("temporal_"):
        inner_name, encoded_window = name.removeprefix("temporal_").rsplit("_s", 1)
        encoded_start, encoded_end = encoded_window.split("e", 1)
        start, end = int(encoded_start), int(encoded_end)
        if start < 0 or end <= start:
            raise ValueError("temporal probe requires 0 <= start < end.")
        return StepWindowProbe(build_probe(inner_name), start, end)
    if name.startswith("group_remove_"):
        return GroupRemovalProbe(name.removeprefix("group_remove_"))
    if name.startswith("group_reliability_t"):
        return GroupReliabilityProbe(int(name.removeprefix("group_reliability_t")) / 10.0)
    if name.startswith("group_norm_equalize_a"):
        return GroupNormEqualizationProbe(int(name.removeprefix("group_norm_equalize_a")) / 100.0)
    if name.startswith("spatial_patch_"):
        return SpatialPatchProbe(name.removeprefix("spatial_patch_"), ratio=0.10)
    if name.startswith("frequency_remove_"):
        return FrequencyBandProbe(name.removeprefix("frequency_remove_"))
    if name.startswith("amplitude_"):
        if name.startswith("amplitude_power"):
            power = int(name.removeprefix("amplitude_power")) / 100.0
            return AmplitudePowerProbe(power)
        operation, encoded_quantile = name.removeprefix("amplitude_").rsplit("_q", 1)
        return AmplitudeProbe(operation, int(encoded_quantile) / 100.0)
    if name.startswith("coordinate_wiener_floor"):
        floor = int(name.removeprefix("coordinate_wiener_floor")) / 100.0
        return CoordinateWienerProbe(floor)
    if name.startswith("frequency_high_gain"):
        gain = int(name.removeprefix("frequency_high_gain")) / 100.0
        return FrequencyGainProbe(gain)
    if name.startswith("low_frequency_boost_"):
        encoded = name.removeprefix("low_frequency_boost_")
        cutoff, strength = encoded.split("_a", 1)
        if not cutoff.startswith("c"):
            raise ValueError(f"unsupported low-frequency boost name: {name}")
        return LowFrequencyBoostProbe(
            int(strength) / 100.0,
            cutoff=int(cutoff.removeprefix("c")) / 100.0,
        )
    if name.startswith("gaussian_blend_"):
        encoded = name.removeprefix("gaussian_blend_")
        sigma, strength = encoded.split("_a", 1)
        if not sigma.startswith("s"):
            raise ValueError(f"unsupported Gaussian blend name: {name}")
        return GaussianBlendProbe(int(sigma.removeprefix("s")) / 10.0, int(strength) / 100.0)
    if name.startswith("spectral_wiener_"):
        scope, encoded_floor = name.removeprefix("spectral_wiener_").split("_floor", 1)
        if scope not in ("all", "high"):
            raise ValueError(f"unsupported spectral Wiener scope: {scope}")
        return SpectralWienerProbe(int(encoded_floor) / 100.0, high_only=scope == "high")
    if name.startswith("spectral_boost_"):
        component, scope, encoded_strength = name.removeprefix("spectral_boost_").split("_", 2)
        if scope not in ("all", "high") or not encoded_strength.startswith("a"):
            raise ValueError(f"unsupported spectral boost name: {name}")
        return SpectralComponentBoostProbe(
            component,
            int(encoded_strength.removeprefix("a")) / 100.0,
            high_only=scope == "high",
        )
    if name.startswith("spectral_amplitude_power"):
        power = int(name.removeprefix("spectral_amplitude_power")) / 100.0
        return SpectralAmplitudePowerProbe(power)
    if name.startswith("covariance_transport_"):
        scope, encoded_strength = name.removeprefix("covariance_transport_").split("_a", 1)
        if scope not in ("view", "group"):
            raise ValueError(f"unsupported covariance scope: {scope}")
        return CovarianceTransportProbe(int(encoded_strength) / 100.0, grouped=scope == "group")
    if name.startswith("energy_equalize_"):
        scope, encoded_strength = name.removeprefix("energy_equalize_").split("_a", 1)
        return EnergyEqualizationProbe(int(encoded_strength) / 100.0, scope=scope)
    raise ValueError(f"unsupported probe: {name}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _permutation_pvalue(x: np.ndarray, y: np.ndarray, observed: float, seed: int, count: int = 2000) -> float:
    rng = np.random.default_rng(seed)
    extreme = 1
    for _ in range(count):
        if abs(_spearman(x, rng.permutation(y))) >= abs(observed):
            extreme += 1
    return extreme / (count + 1)


def _bh_qvalues(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    qvalues = np.ones(len(pvalues), dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = len(pvalues) - reverse_rank + 1
        running = min(running, pvalues[index] * len(pvalues) / rank)
        qvalues[index] = running
    return qvalues.tolist()


def analyze_features(records: list[dict[str, object]], output_path: Path) -> dict[str, object]:
    feature_rows = []
    for family, features in FEATURE_FAMILIES.items():
        for feature in features:
            discovery = [row for row in records if row["split"] == "discovery" and feature in row]
            validation = [row for row in records if row["split"] == "validation" and feature in row]
            if len(discovery) < 10 or len(validation) < 10:
                continue
            x_discovery = np.asarray([float(row[feature]) for row in discovery])
            y_discovery = np.asarray([float(row["transfer_overall"]) for row in discovery])
            x_validation = np.asarray([float(row[feature]) for row in validation])
            y_validation = np.asarray([float(row["transfer_overall"]) for row in validation])
            rho_discovery = _spearman(x_discovery, y_discovery)
            rho_validation = _spearman(x_validation, y_validation)
            feature_rows.append(
                {
                    "family": family,
                    "feature": feature,
                    "rho_discovery": rho_discovery,
                    "rho_validation": rho_validation,
                    "p_discovery": _permutation_pvalue(
                        x_discovery, y_discovery, rho_discovery, seed=20260710 + len(feature_rows)
                    ),
                }
            )
    qvalues = _bh_qvalues([float(row["p_discovery"]) for row in feature_rows]) if feature_rows else []
    for row, qvalue in zip(feature_rows, qvalues):
        row["q_discovery"] = qvalue
        row["passes_gate"] = bool(
            qvalue < 0.10
            and abs(float(row["rho_discovery"])) >= 0.20
            and abs(float(row["rho_validation"])) >= 0.15
            and float(row["rho_discovery"]) * float(row["rho_validation"]) > 0
        )
        row["score"] = min(abs(float(row["rho_discovery"])), abs(float(row["rho_validation"])))

    family_best = []
    for family in FEATURE_FAMILIES:
        passing = [row for row in feature_rows if row["family"] == family and row["passes_gate"]]
        if passing:
            family_best.append(max(passing, key=lambda row: float(row["score"])))
    family_best.sort(key=lambda row: float(row["score"]), reverse=True)
    selected = [str(row["family"]) for row in family_best[:2]]
    result = {"features": feature_rows, "selected_families": selected, "selected_evidence": family_best[:2]}
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def probe_names_for_family(family: str, evidence: dict[str, object] | None = None) -> list[str]:
    if family == "group":
        return ["group_remove_lowest", "group_remove_highest", "group_remove_random"]
    if family == "spatial":
        return ["spatial_patch_highest", "spatial_patch_lowest", "spatial_patch_random"]
    if family == "frequency":
        harmful = "low"
        if evidence and str(evidence.get("feature", "")).startswith("freq_"):
            band = str(evidence["feature"]).removeprefix("freq_")
            if float(evidence.get("rho_validation", 0.0)) < 0:
                harmful = band
        opposite = "high" if harmful == "low" else "low"
        return [f"frequency_remove_{harmful}", f"frequency_remove_{opposite}", "frequency_remove_random"]
    raise ValueError(f"unsupported feature family: {family}")
