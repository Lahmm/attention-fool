from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
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
class MomentumTrajectoryProbe:
    """Project the current gradient toward its own previous MI trajectory."""

    strength: float
    mode: str = "align"
    _momentum: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"momentum_trajectory_{self.mode}_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        if self.mode not in ("align", "parallel_boost"):
            raise ValueError("trajectory mode must be align or parallel_boost.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("trajectory strength must be in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        normalized = gradient / gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-20)
        outputs = []
        for index, sample_id in enumerate(sample_ids):
            if step == 0:
                self._momentum.pop(sample_id, None)
            current = normalized[index]
            previous = self._momentum.get(sample_id)
            if previous is not None:
                current_flat = current.flatten()
                previous_flat = previous.flatten()
                projection = (
                    current_flat.dot(previous_flat)
                    / previous_flat.dot(previous_flat).clamp_min(1e-20)
                ) * previous_flat
                if self.mode == "align":
                    current = (
                        (1.0 - self.strength) * current_flat
                        + self.strength * projection
                    ).view_as(current)
                else:
                    current = (current_flat + self.strength * projection).view_as(current)
            self._momentum[sample_id] = previous.mul(1.0).add(current) if previous is not None else current
            outputs.append(current)
        return torch.stack(outputs, dim=0)


@dataclass
class CrossStepSignPersistenceProbe:
    """Boost sign-persistent gradient amplitudes across attack steps.

    A coordinate that flips sign between consecutive gradients cannot build a
    stable sign trajectory under MI.  Persistent coordinates receive a gain
    of ``1 + strength`` and flip coordinates receive ``1 - strength``.
    ``low``/``high`` restrict the same operation to an orthogonal Fourier band.
    """

    band: str
    strength: float
    cutoff: float = 0.50
    _previous: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return (
            f"sign_persistence_{self.band}_c{int(round(self.cutoff * 100)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def _select_band(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.band == "all":
            return gradient
        if self.band not in ("low", "high"):
            raise ValueError("sign persistence band must be all, low, or high.")
        height, width = gradient.shape[-2:]
        radius = _frequency_radius(height, width, gradient.device)
        low = radius <= radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
        )
        mask = low if self.band == "low" else ~low
        return torch.fft.ifft2(
            torch.fft.ifftshift(
                spectrum * mask.view(1, 1, height, width), dim=(-2, -1)
            ),
            dim=(-2, -1),
        ).real

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if self.band not in ("all", "low", "high"):
            raise ValueError("sign persistence band must be all, low, or high.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("sign persistence strength must be in [0, 1].")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("sign persistence cutoff must be in (0, 1].")
        current = view_gradients.mean(dim=0)
        selected = self._select_band(current)
        outputs = []
        for index, sample_id in enumerate(sample_ids):
            if step == 0:
                self._previous.pop(sample_id, None)
            previous = self._previous.get(sample_id)
            if previous is None:
                transformed = current[index]
            else:
                previous_selected = self._select_band(previous.unsqueeze(0))[0]
                agreement = (selected[index] * previous_selected > 0).to(current.dtype)
                gain = 1.0 + self.strength * (2.0 * agreement - 1.0)
                transformed = current[index] + selected[index] * (gain - 1.0)
            self._previous[sample_id] = current[index].detach()
            outputs.append(transformed)
        return torch.stack(outputs, dim=0)


@dataclass
class ViewPCProbe:
    """Add a mean-aligned principal direction of cross-view variation."""

    strength: float
    iterations: int = 3

    @property
    def name(self) -> str:
        return f"view_pc_transport_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.strength < 0:
            raise ValueError("view PC transport strength must be non-negative.")
        if self.iterations <= 0:
            raise ValueError("view PC iterations must be positive.")
        mean = view_gradients.mean(dim=0)
        mean_flat = mean.flatten(1)
        centered = (view_gradients - mean.unsqueeze(0)).flatten(2).permute(1, 0, 2)
        direction = centered[:, 0]
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-20)
        for _ in range(self.iterations):
            coefficients = torch.einsum("bvd,bd->bv", centered, direction)
            direction = torch.einsum("bvd,bv->bd", centered, coefficients)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-20)
        alignment = (direction * mean_flat).sum(dim=1, keepdim=True)
        direction = direction * torch.where(alignment >= 0, 1.0, -1.0)
        direction = direction * (
            mean_flat.abs().mean(dim=1, keepdim=True)
            / direction.abs().mean(dim=1, keepdim=True).clamp_min(1e-20)
        )
        return (mean_flat + self.strength * direction).view_as(mean)


@dataclass
class ViewGLSProbe:
    """Generalized least-squares combination in the cross-view direction space."""

    ridge: float

    @property
    def name(self) -> str:
        return f"view_gls_ridge{int(round(self.ridge * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.ridge <= 0:
            raise ValueError("GLS ridge must be positive.")
        flat = view_gradients.flatten(2).permute(1, 0, 2)  # [B,V,D]
        unit = flat / flat.norm(dim=2, keepdim=True).clamp_min(1e-20)
        gram = torch.bmm(unit, unit.transpose(1, 2))
        identity = torch.eye(unit.size(1), device=unit.device, dtype=unit.dtype).expand_as(gram)
        rhs = torch.ones(unit.size(0), unit.size(1), 1, device=unit.device, dtype=unit.dtype)
        raw_weights = torch.linalg.solve(gram + self.ridge * identity, rhs).squeeze(-1)
        weights = raw_weights.clamp_min(0.0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-20)
        fallback = torch.full_like(weights, 1.0 / weights.size(1))
        valid = weights.sum(dim=1, keepdim=True) > 1e-20
        weights = torch.where(valid, weights, fallback)
        return torch.einsum("bv,bvchw->bchw", weights, unit.view_as(view_gradients.permute(1, 0, 2, 3, 4)))


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
class DivisiveNormalizationProbe:
    """Suppress local amplitude peaks using a smooth divisive envelope."""

    sigma: float

    @property
    def name(self) -> str:
        return f"divisive_normalize_s{int(round(self.sigma * 10)):02d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.sigma <= 0:
            raise ValueError("divisive normalization sigma must be positive.")
        gradient = view_gradients.mean(dim=0)
        radius = int(3 * self.sigma)
        axis = torch.arange(
            -radius, radius + 1, dtype=gradient.dtype, device=gradient.device
        )
        gaussian = torch.exp(-0.5 * (axis / self.sigma).square())
        gaussian = gaussian / gaussian.sum()
        kernel = (gaussian[:, None] @ gaussian[None, :]).view(
            1, 1, 2 * radius + 1, 2 * radius + 1
        )
        kernel = kernel.repeat(gradient.size(1), 1, 1, 1)
        local_scale = F.conv2d(
            F.pad(gradient.abs(), (radius, radius, radius, radius), mode="reflect"),
            kernel,
            groups=gradient.size(1),
        )
        return gradient / (1.0 + local_scale)


@dataclass
class SoftPercentileClipProbe:
    """Smoothly clip only the signed amplitude tails."""

    percentile: float

    @property
    def name(self) -> str:
        return f"amplitude_softclip_p{int(round(self.percentile * 1000)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 < self.percentile <= 0.5:
            raise ValueError("soft clip percentile must be in (0, 0.5].")
        gradient = view_gradients.mean(dim=0)
        flat = gradient.flatten(1)
        lower = torch.quantile(flat, self.percentile, dim=1, keepdim=True)
        upper = torch.quantile(flat, 1.0 - self.percentile, dim=1, keepdim=True)
        scale = (upper - lower).clamp_min(1e-8) * 0.5
        upper_value = upper + (flat - upper) / (
            1.0 + (flat - upper).abs() / scale
        )
        lower_value = lower - (lower - flat) / (
            1.0 + (lower - flat).abs() / scale
        )
        flat = torch.where(flat > upper, upper_value, flat)
        flat = torch.where(flat < lower, lower_value, flat)
        return flat.view_as(gradient)


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


@dataclass
class SignReliabilityProbe:
    """Use view sign reliability as a soft multiplier on original amplitudes."""

    mode: str
    strength: float

    @property
    def name(self) -> str:
        return f"sign_reliability_{self.mode}_a{int(round(self.strength * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.mode not in ("boost", "gate"):
            raise ValueError("sign reliability mode must be boost or gate.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("sign reliability strength must be in [0, 1].")
        mean = view_gradients.mean(dim=0)
        reliability = view_gradients.sign().mean(dim=0).abs()
        if self.mode == "boost":
            gain = 1.0 + self.strength * reliability
        else:
            gain = (1.0 - self.strength) + self.strength * reliability
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
class HaarWaveletShrinkProbe:
    """Robustly shrink low-amplitude high-frequency details.

    A one-level orthonormal Haar transform separates a low-pass approximation
    from three detail bands.  The detail threshold is estimated by the robust
    MAD noise estimator on HH and scaled by the universal threshold.  This is
    an amplitude-aware frequency intervention: small details are treated as
    noise while large high-frequency coefficients are retained.
    """

    threshold_strength: float

    @property
    def name(self) -> str:
        return f"haar_wavelet_shrink_t{int(round(self.threshold_strength * 100)):03d}"

    @staticmethod
    def _soft_threshold(values: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        return values.sign() * (values.abs() - threshold).clamp_min(0.0)

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.threshold_strength < 0:
            raise ValueError("Haar threshold strength must be non-negative.")
        gradient = view_gradients.mean(dim=0)
        if gradient.size(-2) % 2 or gradient.size(-1) % 2:
            raise ValueError("Haar shrinkage requires even spatial dimensions.")
        a = gradient[..., 0::2, 0::2]
        b = gradient[..., 0::2, 1::2]
        c = gradient[..., 1::2, 0::2]
        d = gradient[..., 1::2, 1::2]
        low = (a + b + c + d) * 0.5
        horizontal = (a - b + c - d) * 0.5
        vertical = (a + b - c - d) * 0.5
        diagonal = (a - b - c + d) * 0.5
        noise_scale = torch.median(diagonal.abs().flatten(1), dim=1).values
        noise_scale = noise_scale / 0.67448975
        coefficient_count = diagonal[0].numel() * 3
        threshold = (
            self.threshold_strength
            * noise_scale
            * (2.0 * np.log(max(2, coefficient_count))) ** 0.5
        ).view(-1, 1, 1, 1)
        horizontal = self._soft_threshold(horizontal, threshold)
        vertical = self._soft_threshold(vertical, threshold)
        diagonal = self._soft_threshold(diagonal, threshold)
        a = (low + horizontal + vertical + diagonal) * 0.5
        b = (low - horizontal + vertical - diagonal) * 0.5
        c = (low + horizontal - vertical - diagonal) * 0.5
        d = (low - horizontal - vertical + diagonal) * 0.5
        result = torch.zeros_like(gradient)
        result[..., 0::2, 0::2] = a
        result[..., 0::2, 1::2] = b
        result[..., 1::2, 0::2] = c
        result[..., 1::2, 1::2] = d
        return result


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
    normalize_component: bool = False

    @property
    def name(self) -> str:
        prefix = "gaussian_norm_blend" if self.normalize_component else "gaussian_blend"
        return f"{prefix}_s{int(round(self.sigma * 10)):02d}_a{int(round(self.strength * 100)):02d}"

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
        if self.normalize_component:
            original_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True)
            smooth_scale = smoothed.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-20)
            smoothed = smoothed * (original_scale / smooth_scale)
        return gradient + self.strength * smoothed


@dataclass
class AdaptiveGaussianProbe:
    """Apply a weak smooth blend only to samples with an extreme feature."""

    metric: str
    quantile: float
    sigma: float = 1.0
    strength: float = 0.25

    @property
    def name(self) -> str:
        return f"adaptive_gaussian_{self.metric}_q{int(round(self.quantile * 100)):02d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.metric not in ("entropy_low", "freq_high"):
            raise ValueError("adaptive Gaussian metric must be entropy_low or freq_high.")
        if not 0.0 < self.quantile <= 1.0:
            raise ValueError("adaptive Gaussian quantile must be in (0, 1].")
        gradient = view_gradients.mean(dim=0)
        if self.metric == "entropy_low":
            energy = gradient.detach().pow(2).sum(dim=1).flatten(1)
            probabilities = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-20)
            feature = -(probabilities * probabilities.clamp_min(1e-20).log()).sum(dim=1)
            threshold = torch.quantile(feature, self.quantile)
            selected = feature <= threshold
        else:
            height, width = gradient.shape[-2:]
            radius = _frequency_radius(height, width, gradient.device)
            spectrum_power = torch.fft.fftshift(
                torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
            ).abs().square().sum(dim=1)
            total = spectrum_power.sum(dim=(1, 2)).clamp_min(1e-20)
            feature = (spectrum_power * (radius > radius.max() * 0.50)).sum(dim=(1, 2)) / total
            threshold = torch.quantile(feature, 1.0 - self.quantile)
            selected = feature >= threshold
        smoothed = GaussianBlendProbe(self.sigma, self.strength).apply(
            view_gradients, ["adaptive"] * gradient.size(0), 0
        )
        return torch.where(selected.view(-1, 1, 1, 1), smoothed, gradient)


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
class SpectralPhaseConsensusProbe:
    """Separate Fourier magnitude averaging from circular phase averaging."""

    strength: float

    @property
    def name(self) -> str:
        return f"spectral_phase_consensus_a{int(round(self.strength * 100)):03d}"

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("phase consensus strength must be in [0, 1].")
        spectra = torch.fft.fft2(view_gradients, dim=(-2, -1))
        mean_spectrum = spectra.mean(dim=0)
        magnitudes = spectra.abs()
        phase_vectors = spectra / magnitudes.clamp_min(1e-20)
        phase_mean = phase_vectors.mean(dim=0)
        phase_unit = phase_mean / phase_mean.abs().clamp_min(1e-20)
        consensus = magnitudes.mean(dim=0) * phase_unit
        mean_scale = mean_spectrum.abs().mean(dim=(1, 2, 3), keepdim=True)
        consensus_scale = consensus.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-20)
        consensus = consensus * (mean_scale / consensus_scale)
        transformed = (1.0 - self.strength) * mean_spectrum + self.strength * consensus
        return torch.fft.ifft2(transformed, dim=(-2, -1)).real


@dataclass
class CrossScaleCovarianceProbe:
    """Use cross-view low/high covariance to separate useful high frequencies.

    Let ``l_v`` and ``h_v`` be orthogonal low/high Fourier projections of view
    ``v``.  The high-frequency direction ``C_hl l`` is the linear least-squares
    covariance transport from low-frequency variation into the high-frequency
    space.  This gives a mathematical test for whether a high-frequency
    component is coupled to the low-frequency signal, instead of assuming that
    every high-frequency coordinate is harmful.

    ``replace`` keeps the mean low-frequency component and interpolates the
    mean high-frequency component toward the covariance-transport direction.
    ``project`` removes only the part of the mean high-frequency component that
    is orthogonal to that direction.  Both operations preserve the original
    low/high split and do not change the augmentation or update rule.
    """

    mode: str
    strength: float
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return (
            f"cross_scale_{self.mode}_c{int(round(self.cutoff * 100)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.mode not in ("replace", "project", "add"):
            raise ValueError("cross-scale mode must be replace, project, or add.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("cross-scale strength must be in [0, 1].")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("cross-scale cutoff must be in (0, 1].")

        mean = view_gradients.mean(dim=0)
        height, width = mean.shape[-2:]
        radius = _frequency_radius(height, width, mean.device)
        low_mask = radius <= radius.max() * self.cutoff
        spectra = torch.fft.fftshift(
            torch.fft.fft2(view_gradients, dim=(-2, -1)), dim=(-2, -1)
        )
        low_views = torch.fft.ifft2(
            torch.fft.ifftshift(
                spectra * low_mask.view(1, 1, 1, height, width), dim=(-2, -1)
            ),
            dim=(-2, -1),
        ).real
        high_views = view_gradients - low_views
        low_mean = low_views.mean(dim=0)
        high_mean = high_views.mean(dim=0)

        low_flat = low_views.flatten(2).permute(1, 0, 2)  # [B,V,D]
        high_flat = high_views.flatten(2).permute(1, 0, 2)
        low_centered = low_flat - low_flat.mean(dim=1, keepdim=True)
        high_centered = high_flat - high_flat.mean(dim=1, keepdim=True)
        low_mean_flat = low_mean.flatten(1)
        high_mean_flat = high_mean.flatten(1)

        # C_hl l = (1/V) sum_v (h_v-h_bar)<l_v-l_bar,l_bar>.
        coefficients = torch.einsum(
            "bvd,bd->bv", low_centered, low_mean_flat
        )
        transported = torch.einsum(
            "bvd,bv->bd", high_centered, coefficients
        ) / max(1, view_gradients.size(0))
        transported_norm = transported.norm(dim=1, keepdim=True)
        high_norm = high_mean_flat.norm(dim=1, keepdim=True)
        transported = transported * (
            high_norm / transported_norm.clamp_min(1e-20)
        )
        valid = (transported_norm > 1e-20) & (high_norm > 1e-20)
        transported = torch.where(valid, transported, high_mean_flat)

        if self.mode == "add":
            transformed = low_mean_flat + high_mean_flat + self.strength * transported
        elif self.mode == "replace":
            transformed = low_mean_flat + (
                (1.0 - self.strength) * high_mean_flat
                + self.strength * transported
            )
        else:
            transport_norm_sq = transported.square().sum(dim=1, keepdim=True)
            aligned = (
                (high_mean_flat * transported).sum(dim=1, keepdim=True)
                / transport_norm_sq.clamp_min(1e-20)
            ) * transported
            aligned = torch.where(valid, aligned, high_mean_flat)
            transformed = low_mean_flat + high_mean_flat + self.strength * (
                aligned - high_mean_flat
            )
        return transformed.view_as(mean)


@dataclass
class CrossScaleCanonicalProbe:
    """Replace high-frequency content with the top low/high CCA direction.

    The 20 views are treated as repeated observations.  After removing each
    band's view mean and normalizing each residual view, the SVD of the
    low/high cross-view cosine matrix gives the strongest cross-scale coupled
    direction.  This is a low-rank mathematical intervention, not a radial
    frequency heuristic.
    """

    strength: float
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return (
            f"cross_scale_canonical_c{int(round(self.cutoff * 100)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("cross-scale canonical strength must be in [0, 1].")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("cross-scale canonical cutoff must be in (0, 1].")
        mean = view_gradients.mean(dim=0)
        height, width = mean.shape[-2:]
        radius = _frequency_radius(height, width, mean.device)
        low_mask = radius <= radius.max() * self.cutoff
        spectra = torch.fft.fftshift(
            torch.fft.fft2(view_gradients, dim=(-2, -1)), dim=(-2, -1)
        )
        low_views = torch.fft.ifft2(
            torch.fft.ifftshift(
                spectra * low_mask.view(1, 1, 1, height, width), dim=(-2, -1)
            ),
            dim=(-2, -1),
        ).real
        high_views = view_gradients - low_views
        low_mean = low_views.mean(dim=0)
        high_mean = high_views.mean(dim=0)
        low_flat = low_views.flatten(2).permute(1, 0, 2)
        high_flat = high_views.flatten(2).permute(1, 0, 2)
        low_centered = low_flat - low_flat.mean(dim=1, keepdim=True)
        high_centered = high_flat - high_flat.mean(dim=1, keepdim=True)
        low_unit = low_centered / low_centered.norm(dim=2, keepdim=True).clamp_min(1e-20)
        high_unit = high_centered / high_centered.norm(dim=2, keepdim=True).clamp_min(1e-20)
        cross_view_cosine = torch.bmm(low_unit, high_unit.transpose(1, 2))
        _, _, right_vectors = torch.linalg.svd(cross_view_cosine, full_matrices=False)
        right = right_vectors[:, 0, :]
        canonical_high = torch.einsum("bvd,bv->bd", high_unit, right)
        canonical_high_norm = canonical_high.norm(dim=1, keepdim=True)
        high_mean_flat = high_mean.flatten(1)
        high_mean_norm = high_mean_flat.norm(dim=1, keepdim=True)
        canonical_high = canonical_high * (
            high_mean_norm / canonical_high_norm.clamp_min(1e-20)
        )
        alignment = (canonical_high * high_mean_flat).sum(dim=1, keepdim=True)
        canonical_high = canonical_high * torch.where(
            alignment >= 0, torch.ones_like(alignment), -torch.ones_like(alignment)
        )
        valid = (canonical_high_norm > 1e-20) & (high_mean_norm > 1e-20)
        canonical_high = torch.where(valid, canonical_high, high_mean_flat)
        transformed = low_mean.flatten(1) + (
            (1.0 - self.strength) * high_mean_flat
            + self.strength * canonical_high
        )
        return transformed.view_as(mean)


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
    if name.startswith("momentum_trajectory_"):
        mode, encoded_strength = name.removeprefix("momentum_trajectory_").split("_a", 1)
        return MomentumTrajectoryProbe(int(encoded_strength) / 100.0, mode=mode)
    if name.startswith("sign_persistence_"):
        encoded = name.removeprefix("sign_persistence_")
        band, encoded = encoded.split("_c", 1)
        cutoff, strength = encoded.split("_a", 1)
        return CrossStepSignPersistenceProbe(
            band,
            int(strength) / 100.0,
            cutoff=int(cutoff) / 100.0,
        )
    if name.startswith("view_pc_transport_a"):
        return ViewPCProbe(int(name.removeprefix("view_pc_transport_a")) / 100.0)
    if name.startswith("view_gls_ridge"):
        return ViewGLSProbe(int(name.removeprefix("view_gls_ridge")) / 100.0)
    if name.startswith("spatial_patch_"):
        return SpatialPatchProbe(name.removeprefix("spatial_patch_"), ratio=0.10)
    if name.startswith("frequency_remove_"):
        return FrequencyBandProbe(name.removeprefix("frequency_remove_"))
    if name.startswith("amplitude_"):
        if name.startswith("amplitude_power"):
            power = int(name.removeprefix("amplitude_power")) / 100.0
            return AmplitudePowerProbe(power)
        if name.startswith("amplitude_softclip_p"):
            return SoftPercentileClipProbe(
                int(name.removeprefix("amplitude_softclip_p")) / 1000.0
            )
        operation, encoded_quantile = name.removeprefix("amplitude_").rsplit("_q", 1)
        return AmplitudeProbe(operation, int(encoded_quantile) / 100.0)
    if name.startswith("divisive_normalize_s"):
        return DivisiveNormalizationProbe(
            int(name.removeprefix("divisive_normalize_s")) / 10.0
        )
    if name.startswith("coordinate_wiener_floor"):
        floor = int(name.removeprefix("coordinate_wiener_floor")) / 100.0
        return CoordinateWienerProbe(floor)
    if name.startswith("sign_reliability_"):
        mode, encoded_strength = name.removeprefix("sign_reliability_").split("_a", 1)
        return SignReliabilityProbe(mode, int(encoded_strength) / 100.0)
    if name.startswith("frequency_high_gain"):
        gain = int(name.removeprefix("frequency_high_gain")) / 100.0
        return FrequencyGainProbe(gain)
    if name.startswith("haar_wavelet_shrink_t"):
        return HaarWaveletShrinkProbe(
            int(name.removeprefix("haar_wavelet_shrink_t")) / 100.0
        )
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
    if name.startswith("gaussian_norm_blend_"):
        encoded = name.removeprefix("gaussian_norm_blend_")
        sigma, strength = encoded.split("_a", 1)
        if not sigma.startswith("s"):
            raise ValueError(f"unsupported normalized Gaussian name: {name}")
        return GaussianBlendProbe(
            int(sigma.removeprefix("s")) / 10.0,
            int(strength) / 100.0,
            normalize_component=True,
        )
    if name.startswith("adaptive_gaussian_"):
        encoded = name.removeprefix("adaptive_gaussian_")
        metric, encoded_quantile = encoded.rsplit("_q", 1)
        return AdaptiveGaussianProbe(metric, int(encoded_quantile) / 100.0)
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
    if name.startswith("spectral_phase_consensus_a"):
        return SpectralPhaseConsensusProbe(
            int(name.removeprefix("spectral_phase_consensus_a")) / 100.0
        )
    if name.startswith("cross_scale_"):
        encoded = name.removeprefix("cross_scale_")
        if encoded.startswith("canonical_c"):
            cutoff, strength = encoded.removeprefix("canonical_c").split("_a", 1)
            return CrossScaleCanonicalProbe(
                int(strength) / 100.0,
                cutoff=int(cutoff) / 100.0,
            )
        mode, encoded = encoded.split("_c", 1)
        cutoff, strength = encoded.split("_a", 1)
        return CrossScaleCovarianceProbe(
            mode,
            int(strength) / 100.0,
            cutoff=int(cutoff) / 100.0,
        )
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
