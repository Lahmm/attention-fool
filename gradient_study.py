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
class PairPhaseProbe:
    """Use the A/B phase-pair decomposition as a noise or direction estimate."""

    mode: str
    strength: float
    high_only: bool = False

    @property
    def name(self) -> str:
        if self.mode in (
            "difference_add",
            "difference_reverse",
            "difference_orthogonal",
        ):
            return f"pair_{self.mode}_a{int(round(self.strength * 100)):03d}"
        scope = "high_" if self.high_only else ""
        return f"pair_phase_wiener_{scope}f{int(round(self.strength * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if view_gradients.size(0) % 2:
            raise ValueError("phase-pair probes require an even number of views.")
        if self.mode not in (
            "difference_add",
            "difference_reverse",
            "difference_orthogonal",
            "phase_wiener",
        ):
            raise ValueError("unsupported phase-pair probe mode.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("phase-pair probe strength must be in [0, 1].")
        grouped = view_gradients.view(
            view_gradients.size(0) // 2, 2, *view_gradients.shape[1:]
        )
        pair_mean = grouped.mean(dim=1)
        pair_difference = (grouped[:, 0] - grouped[:, 1]) * 0.5
        mean = pair_mean.mean(dim=0)
        if self.mode == "difference_add":
            return mean + self.strength * pair_difference.mean(dim=0)
        if self.mode == "difference_reverse":
            return mean - self.strength * pair_difference.mean(dim=0)
        if self.mode == "difference_orthogonal":
            difference = pair_difference.mean(dim=0)
            mean_flat = mean.flatten(1)
            difference_flat = difference.flatten(1)
            projection = (
                (mean_flat * difference_flat).sum(dim=1, keepdim=True)
                / difference_flat.square().sum(dim=1, keepdim=True).clamp_min(1e-20)
            ) * difference_flat
            return (
                mean_flat - self.strength * projection
            ).view_as(mean)

        if self.high_only:
            height, width = mean.shape[-2:]
            radius = _frequency_radius(height, width, mean.device)
            high = radius > radius.max() * 0.50
            mean_spectrum = torch.fft.fftshift(
                torch.fft.fft2(mean, dim=(-2, -1)), dim=(-2, -1)
            )
            difference_spectrum = torch.fft.fftshift(
                torch.fft.fft2(pair_difference, dim=(-2, -1)), dim=(-2, -1)
            )
            noise = difference_spectrum.abs().square().mean(dim=0)
            signal = (mean_spectrum.abs().square() - noise).clamp_min(0.0)
            gain = signal / (signal + noise).clamp_min(1e-20)
            gain = self.strength + (1.0 - self.strength) * gain
            gain = torch.where(
                high.view(1, 1, height, width), gain, torch.ones_like(gain)
            )
            filtered = mean_spectrum * gain
            return torch.fft.ifft2(
                torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)
            ).real

        noise = pair_difference.square().mean(dim=0)
        signal = (mean.square() - noise).clamp_min(0.0)
        gain = signal / (signal + noise).clamp_min(1e-20)
        gain = self.strength + (1.0 - self.strength) * gain
        return mean * gain


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
        if self.mode not in ("align", "parallel_boost", "orthogonal"):
            raise ValueError("trajectory mode must be align, parallel_boost, or orthogonal.")
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
                elif self.mode == "parallel_boost":
                    current = (current_flat + self.strength * projection).view_as(current)
                else:
                    current = (current_flat - self.strength * projection).view_as(current)
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
class AdaptiveTrajectoryBlendProbe:
    """Damp only abrupt raw-gradient innovations along the attack trajectory."""

    strength: float
    ema_decay: float = 0.9
    _ema: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return f"adaptive_trajectory_blend_a{int(round(self.strength * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("adaptive trajectory strength must be in [0, 1].")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("adaptive trajectory ema_decay must be in [0, 1).")
        current = view_gradients.mean(dim=0)
        outputs = []
        for index, sample_id in enumerate(sample_ids):
            if step == 0:
                self._ema.pop(sample_id, None)
            gradient = current[index]
            previous = self._ema.get(sample_id)
            if previous is None:
                transformed = gradient
                ema = gradient.detach()
            else:
                gradient_flat = gradient.flatten()
                previous_flat = previous.flatten()
                projection = (
                    (gradient_flat * previous_flat).sum()
                    / previous_flat.square().sum().clamp_min(1e-20)
                ) * previous_flat
                innovation = gradient_flat - projection
                cosine = F.cosine_similarity(
                    gradient_flat.unsqueeze(0), previous_flat.unsqueeze(0), dim=1
                )[0].clamp(-1.0, 1.0)
                disagreement = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)
                weight = self.strength * disagreement
                transformed = (gradient_flat - weight * innovation).view_as(gradient)
                ema = (
                    self.ema_decay * previous
                    + (1.0 - self.ema_decay) * gradient.detach()
                )
            self._ema[sample_id] = ema.detach()
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
class GeometricMedianProbe:
    """Robustly aggregate views with a few Weiszfeld iterations."""

    strength: float
    preserve_scale: bool = False
    iterations: int = 3

    @property
    def name(self) -> str:
        suffix = "scaled" if self.preserve_scale else "raw"
        return f"geometric_median_a{int(round(self.strength * 100)):03d}_{suffix}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("geometric median strength must be in [0, 1].")
        if self.iterations < 1:
            raise ValueError("geometric median requires at least one iteration.")
        batch_size = view_gradients.size(1)
        flat = view_gradients.flatten(2).permute(1, 0, 2)
        mean = flat.mean(dim=1)
        estimate = mean
        for _ in range(self.iterations):
            distances = (flat - estimate.unsqueeze(1)).norm(dim=2).clamp_min(1e-12)
            weights = distances.reciprocal()
            estimate = (weights.unsqueeze(-1) * flat).sum(dim=1) / weights.sum(
                dim=1, keepdim=True
            )
        estimate = mean + self.strength * (estimate - mean)
        if self.preserve_scale:
            source_scale = mean.abs().mean(dim=1, keepdim=True)
            estimate_scale = estimate.abs().mean(dim=1, keepdim=True)
            estimate = estimate * (source_scale / estimate_scale.clamp_min(1e-20))
        return estimate.view(batch_size, *view_gradients.shape[2:])


@dataclass
class PatchEnergyTransportProbe:
    """Move patch-residual energy into the patch mean direction."""

    strength: float
    grid: int = 14

    @property
    def name(self) -> str:
        return f"patch_energy_transport_g{self.grid}_a{int(round(self.strength * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("patch energy transport strength must be in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        if height % self.grid or width % self.grid:
            raise ValueError("patch energy transport requires dimensions divisible by grid.")
        kernel_size = (height // self.grid, width // self.grid)
        patch_mean = F.avg_pool2d(gradient, kernel_size=kernel_size, stride=kernel_size)
        projection = patch_mean.repeat_interleave(kernel_size[0], dim=2).repeat_interleave(
            kernel_size[1], dim=3
        )
        residual = gradient - projection
        patch_area = float(kernel_size[0] * kernel_size[1])
        mean_energy = (
            patch_mean.square().sum(dim=1, keepdim=True) * patch_area
        )
        residual_energy = F.avg_pool2d(
            residual.square().sum(dim=1, keepdim=True),
            kernel_size=kernel_size,
            stride=kernel_size,
        ) * patch_area
        ratio = (
            self.strength**2
            * residual_energy
            / mean_energy.clamp_min(1e-20)
        )
        mean_gain = torch.sqrt(1.0 + ratio)
        mean_gain = mean_gain.repeat_interleave(kernel_size[0], dim=2).repeat_interleave(
            kernel_size[1], dim=3
        )
        return projection * mean_gain + (1.0 - self.strength) * residual


@dataclass
class PatchEnergyTransportRescaleProbe:
    """Patch energy transport with explicit preservation of raw scale."""

    strength: float
    grid: int = 14
    scale: str = "l1"

    @property
    def name(self) -> str:
        return (
            f"patch_energy_transport_rescaled_{self.scale}"
            f"_g{self.grid}_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("rescaled patch energy strength must be in [0, 1].")
        if self.scale not in ("l1", "l2"):
            raise ValueError("rescaled patch energy scale must be l1 or l2.")
        gradient = view_gradients.mean(dim=0)
        transformed = PatchEnergyTransportProbe(
            self.strength, grid=self.grid
        ).apply(gradient.unsqueeze(0), ["patch_energy"] * gradient.size(0), 0)
        if self.scale == "l1":
            original_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True)
            transformed_scale = transformed.abs().mean(dim=(1, 2, 3), keepdim=True)
        else:
            original_scale = gradient.flatten(1).norm(dim=1, keepdim=True).view(
                -1, 1, 1, 1
            )
            transformed_scale = transformed.flatten(1).norm(dim=1, keepdim=True).view(
                -1, 1, 1, 1
            )
        return transformed * (original_scale / transformed_scale.clamp_min(1e-20))


@dataclass
class PatchProjectionProbe:
    """Blend the orthogonal projection onto the ViT patch-mean subspace."""

    strength: float
    grid: int = 14

    @property
    def name(self) -> str:
        return f"patch_projection_g{self.grid}_a{int(round(self.strength * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("patch projection strength must be in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        if height % self.grid or width % self.grid:
            raise ValueError("patch projection requires dimensions divisible by grid.")
        kernel_size = (height // self.grid, width // self.grid)
        patch_mean = F.avg_pool2d(gradient, kernel_size=kernel_size, stride=kernel_size)
        projection = patch_mean.repeat_interleave(kernel_size[0], dim=2).repeat_interleave(
            kernel_size[1], dim=3
        )
        return gradient + self.strength * projection


@dataclass
class PatchEmbeddingMetricProbe:
    """Precondition each ViT patch gradient with its embedding metric W^T W."""

    strength: float
    metric: torch.Tensor
    preserve_scale: bool = False
    patch_size: int = 16

    @property
    def name(self) -> str:
        suffix = "scaled" if self.preserve_scale else "raw"
        return f"patch_embedding_metric_a{int(round(self.strength * 100)):03d}_{suffix}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("patch embedding metric strength must be in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("patch embedding metric requires divisible image dimensions.")
        metric = self.metric.to(device=gradient.device, dtype=gradient.dtype)
        patches = F.unfold(
            gradient,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        transformed = torch.matmul(patches, metric.transpose(0, 1))
        transformed = F.fold(
            transformed.transpose(1, 2),
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        if self.preserve_scale:
            source_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True)
            transformed_scale = transformed.abs().mean(dim=(1, 2, 3), keepdim=True)
            transformed = transformed * (
                source_scale / transformed_scale.clamp_min(1e-20)
            )
        return gradient + self.strength * (transformed - gradient)


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
class GlobalGradientScaleProbe:
    """Change only the current raw-gradient weight entering MI."""

    scale: float

    @property
    def name(self) -> str:
        return f"raw_global_scale_g{self.scale:g}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.scale <= 0:
            raise ValueError("global raw gradient scale must be positive.")
        return view_gradients.mean(dim=0) * self.scale


@dataclass
class FixedChannelGainProbe:
    """Apply a fixed cross-ViT color-gradient gain learned offline."""

    strength: float
    inverse: bool = False

    @property
    def name(self) -> str:
        prefix = "vit_shared_color_inverse" if self.inverse else "vit_shared_color"
        return f"{prefix}_a{int(round(self.strength * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("fixed channel gain strength must be in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        gains = torch.tensor(
            (1.02, 0.76, 1.22) if self.inverse else (0.98, 1.24, 0.78),
            device=gradient.device,
            dtype=gradient.dtype,
        )
        gains = 1.0 + self.strength * (gains - 1.0)
        return gradient * gains.view(1, 3, 1, 1)


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
class FrequencyGainRescaleProbe:
    """Apply a high-frequency gain while preserving raw gradient scale."""

    high_gain: float
    scale: str = "l1"

    @property
    def name(self) -> str:
        return (
            f"frequency_rescaled_high_{self.scale}_g"
            f"{int(round(self.high_gain * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.high_gain <= 1.0:
            raise ValueError("rescaled high-frequency gain must be in [0, 1].")
        if self.scale not in ("l1", "l2"):
            raise ValueError("rescaled frequency scale must be l1 or l2.")
        gradient = view_gradients.mean(dim=0)
        filtered = FrequencyGainProbe(self.high_gain).apply(
            gradient.unsqueeze(0), ["frequency_rescaled"] * gradient.size(0), 0
        )
        if self.scale == "l1":
            original_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True)
            filtered_scale = filtered.abs().mean(dim=(1, 2, 3), keepdim=True)
        else:
            original_scale = gradient.flatten(1).norm(dim=1, keepdim=True).view(
                -1, 1, 1, 1
            )
            filtered_scale = filtered.flatten(1).norm(dim=1, keepdim=True).view(
                -1, 1, 1, 1
            )
        return filtered * (original_scale / filtered_scale.clamp_min(1e-20))


@dataclass
class AdaptiveFrequencyGainRescaleProbe:
    """Rescale high-frequency gain only for high-frequency-heavy samples."""

    high_gain: float
    quantile: float
    scale: str = "l1"
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return (
            f"adaptive_frequency_rescaled_{self.scale}"
            f"_q{int(round(self.quantile * 100)):02d}"
            f"_g{int(round(self.high_gain * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.high_gain <= 1.0:
            raise ValueError("adaptive high-frequency gain must be in [0, 1].")
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("adaptive frequency quantile must be in [0, 1].")
        if self.scale not in ("l1", "l2"):
            raise ValueError("adaptive frequency scale must be l1 or l2.")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("adaptive frequency cutoff must be in (0, 1].")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        radius = _frequency_radius(height, width, gradient.device)
        high = radius > radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
        )
        power = spectrum.abs().square().sum(dim=1)
        high_fraction = power.masked_select(high.view(1, height, width)).view(
            gradient.size(0), -1
        ).sum(dim=1) / power.sum(dim=(1, 2)).clamp_min(1e-20)
        threshold = torch.quantile(high_fraction, self.quantile)
        selected = high_fraction >= threshold
        transformed = FrequencyGainRescaleProbe(
            self.high_gain, scale=self.scale
        ).apply(gradient.unsqueeze(0), ["adaptive_frequency"] * gradient.size(0), 0)
        return torch.where(selected.view(-1, 1, 1, 1), transformed, gradient)


@dataclass
class LaplacianProxProbe:
    """Tikhonov/Laplacian proximal low-pass preconditioning.

    It solves ``argmin_x ||x-g||^2 + lambda ||∇x||^2`` in the Fourier domain,
    yielding the rational filter ``1 / (1 + lambda r^2)``.  This is a
    mathematically specified constructive low-frequency enhancement rather
    than hard band deletion.
    """

    regularization: float

    @property
    def name(self) -> str:
        return f"laplacian_prox_l{int(round(self.regularization * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.regularization < 0:
            raise ValueError("Laplacian regularization must be non-negative.")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        radius = _frequency_radius(height, width, gradient.device)
        normalized_radius = radius / radius.max().clamp_min(1e-20)
        filter_gain = 1.0 / (
            1.0 + self.regularization * normalized_radius.square()
        )
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
        )
        filtered = spectrum * filter_gain.view(1, 1, height, width)
        return torch.fft.ifft2(
            torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)
        ).real


@dataclass
class PostMomentumPreconditionProbe:
    """Precondition only the MI direction immediately before ``sign``.

    ``apply`` is intentionally the identity aggregation.  ``apply_update`` is
    called after the attack has updated its MI state and changes only the
    direction used for the current sign update; the unfiltered momentum is
    retained for the next step.  This isolates the question whether harmful
    frequency content is introduced or amplified by accumulation.
    """

    kind: str
    value: float
    strength: float = 0.0
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        if self.kind == "gaussian":
            return (
                f"post_momentum_gaussian_s{int(round(self.value * 10)):02d}"
                f"_a{int(round(self.strength * 100)):03d}"
            )
        if self.kind == "laplacian":
            return f"post_momentum_laplacian_l{int(round(self.value * 100)):03d}"
        return (
            f"post_momentum_high_shrink_c{int(round(self.cutoff * 100)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        return view_gradients.mean(dim=0)

    def apply_update(
        self,
        update: torch.Tensor,
        current_gradient: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del current_gradient, sample_ids, step
        if self.kind == "gaussian":
            if self.value <= 0 or self.strength < 0:
                raise ValueError("post-momentum Gaussian requires positive sigma and strength.")
            return GaussianBlendProbe(self.value, self.strength).apply(
                update.unsqueeze(0), ["update"] * update.size(0), 0
            )
        if self.kind == "laplacian":
            if self.value < 0:
                raise ValueError("post-momentum Laplacian regularization must be non-negative.")
            return LaplacianProxProbe(self.value).apply(
                update.unsqueeze(0), ["update"] * update.size(0), 0
            )
        if self.kind != "high_shrink":
            raise ValueError(f"unsupported post-momentum kind: {self.kind}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("post-momentum high shrink strength must be in [0, 1].")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("post-momentum cutoff must be in (0, 1].")
        height, width = update.shape[-2:]
        radius = _frequency_radius(height, width, update.device)
        high = radius > radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(update, dim=(-2, -1)), dim=(-2, -1)
        )
        gain = torch.ones_like(radius)
        gain[high] = 1.0 - self.strength
        filtered = spectrum * gain.view(1, 1, height, width)
        return torch.fft.ifft2(
            torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1)
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
class OrthogonalGaussianProbe:
    """Inject only the Gaussian component orthogonal to the raw gradient."""

    sigma: float
    strength: float

    @property
    def name(self) -> str:
        return (
            f"orthogonal_gaussian_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.sigma <= 0 or self.strength < 0:
            raise ValueError("orthogonal Gaussian requires sigma > 0 and strength >= 0.")
        gradient = view_gradients.mean(dim=0)
        radius = max(1, int(round(3 * self.sigma)))
        axis = torch.arange(-radius, radius + 1, device=gradient.device, dtype=gradient.dtype)
        kernel_1d = torch.exp(-0.5 * (axis / self.sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(
            1, 1, 2 * radius + 1, 2 * radius + 1
        ).repeat(gradient.size(1), 1, 1, 1)
        smoothed = F.conv2d(
            F.pad(gradient, (radius,) * 4, mode="reflect"),
            kernel,
            groups=gradient.size(1),
        )
        flat_gradient = gradient.flatten(1)
        flat_smoothed = smoothed.flatten(1)
        projection = (
            (flat_smoothed * flat_gradient).sum(dim=1, keepdim=True)
            / flat_gradient.square().sum(dim=1, keepdim=True).clamp_min(1e-20)
        ) * flat_gradient
        residual = (flat_smoothed - projection).view_as(gradient)
        return gradient + self.strength * residual


@dataclass
class RawScaleTemporalGaussianProbe:
    """Preserve raw temporal scale while adding a weak spatially smooth component.

    The per-sample raw gradient scale is compared with an EMA from earlier
    attack steps.  ``power=0`` keeps the raw mean unchanged, while larger
    powers give later high-magnitude gradients a bounded extra MI weight.
    Gaussian blending is applied after this temporal weighting.
    """

    power: float
    sigma: float
    strength: float
    ema_decay: float = 0.9
    _ema_by_sample: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return (
            f"raw_temporal_gaussian_p{int(round(self.power * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        if not 0.0 <= self.power <= 1.0:
            raise ValueError("raw temporal Gaussian power must be in [0, 1].")
        if self.sigma <= 0 or self.strength < 0:
            raise ValueError("raw temporal Gaussian requires sigma > 0 and strength >= 0.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("raw temporal Gaussian ema_decay must be in [0, 1).")
        gradient = view_gradients.mean(dim=0)
        raw_scale = gradient.detach().abs().mean(dim=(1, 2, 3), keepdim=True)
        weighted = []
        for index, sample_id in enumerate(sample_ids):
            if step == 0:
                self._ema_by_sample.pop(sample_id, None)
            current_scale = raw_scale[index]
            previous_ema = self._ema_by_sample.get(sample_id)
            if previous_ema is None:
                ema = current_scale
            else:
                ema = self.ema_decay * previous_ema + (1.0 - self.ema_decay) * current_scale
            self._ema_by_sample[sample_id] = ema.detach()
            relative_scale = (current_scale / ema.clamp_min(1e-20)).clamp(0.25, 4.0)
            weighted.append(gradient[index] * relative_scale.pow(self.power))
        weighted_gradient = torch.stack(weighted, dim=0)
        return GaussianBlendProbe(self.sigma, self.strength).apply(
            weighted_gradient.unsqueeze(0), sample_ids, step
        )


@dataclass
class RawScaleCrossScaleGaussianProbe:
    """Apply raw temporal weighting before cross-scale Gaussian processing."""

    power: float
    cutoff: float
    cross_strength: float
    sigma: float
    smooth_strength: float
    ema_decay: float = 0.9
    _ema_by_sample: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    @property
    def name(self) -> str:
        return (
            f"raw_temporal_cross_scale_p{int(round(self.power * 100)):03d}"
            f"_c{int(round(self.cutoff * 100)):02d}"
            f"_x{int(round(self.cross_strength * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.smooth_strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if not 0.0 <= self.power <= 1.0:
            raise ValueError("raw temporal cross-scale power must be in [0, 1].")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("raw temporal cross-scale cutoff must be in (0, 1].")
        if not 0.0 <= self.cross_strength <= 1.0:
            raise ValueError("raw temporal cross-scale strength must be in [0, 1].")
        if self.sigma <= 0 or self.smooth_strength < 0:
            raise ValueError("raw temporal cross-scale Gaussian parameters are invalid.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("raw temporal cross-scale ema_decay must be in [0, 1).")
        gradient = view_gradients.mean(dim=0)
        raw_scale = gradient.detach().abs().mean(dim=(1, 2, 3), keepdim=True)
        weighted = []
        for index, sample_id in enumerate(sample_ids):
            if step == 0:
                self._ema_by_sample.pop(sample_id, None)
            current_scale = raw_scale[index]
            previous_ema = self._ema_by_sample.get(sample_id)
            if previous_ema is None:
                ema = current_scale
            else:
                ema = self.ema_decay * previous_ema + (1.0 - self.ema_decay) * current_scale
            self._ema_by_sample[sample_id] = ema.detach()
            relative_scale = (current_scale / ema.clamp_min(1e-20)).clamp(0.25, 4.0)
            weighted.append(gradient[index] * relative_scale.pow(self.power))
        weighted_gradient = torch.stack(weighted, dim=0)
        return CrossScaleGaussianProbe(
            self.cutoff,
            self.cross_strength,
            self.sigma,
            self.smooth_strength,
        ).apply(weighted_gradient.unsqueeze(0), sample_ids, step)


@dataclass
class PatchGaussianBlendProbe:
    """Add Gaussian and soft ViT patch-scale residuals without projection."""

    sigma: float
    gaussian_strength: float
    patch_strength: float
    patch_size: int = 16

    @property
    def name(self) -> str:
        return (
            f"patch_gaussian_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.gaussian_strength * 100)):03d}"
            f"_p{int(round(self.patch_strength * 100)):03d}"
        )

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.sigma <= 0 or self.gaussian_strength < 0 or self.patch_strength < 0:
            raise ValueError("patch Gaussian strengths must be non-negative and sigma > 0.")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("patch Gaussian requires dimensions divisible by patch_size.")
        gaussian = GaussianBlendProbe(self.sigma, self.gaussian_strength).apply(
            gradient.unsqueeze(0), ["patch_gaussian"] * gradient.size(0), 0
        ) - gradient
        patch = F.avg_pool2d(
            gradient,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patch = F.interpolate(patch, size=(height, width), mode="nearest")
        return gradient + gaussian + self.patch_strength * patch


@dataclass
class CrossScalePatchGaussianProbe:
    """Add patch-grid DC structure to the cross-scale Gaussian direction.

    The cross-scale transport preserves Fourier low/high structure while the
    patch term tests whether a token-sized, architecture-agnostic spatial
    statistic contains complementary transfer information.  The original
    raw mean remains the base component; no view, mask, or update path is
    changed.
    """

    cutoff: float
    cross_strength: float
    sigma: float
    smooth_strength: float
    patch_strength: float
    patch_size: int = 16

    @property
    def name(self) -> str:
        return (
            f"cross_scale_patch_gaussian_c{int(round(self.cutoff * 100)):02d}"
            f"_x{int(round(self.cross_strength * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.smooth_strength * 100)):03d}"
            f"_p{int(round(self.patch_strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if self.patch_strength < 0:
            raise ValueError("patch strength must be non-negative.")
        gradient = view_gradients.mean(dim=0)
        base = CrossScaleGaussianProbe(
            self.cutoff,
            self.cross_strength,
            self.sigma,
            self.smooth_strength,
        ).apply(view_gradients, sample_ids, step)
        height, width = gradient.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("cross-scale patch Gaussian requires divisible dimensions.")
        patch = F.avg_pool2d(
            gradient,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        patch = F.interpolate(patch, size=(height, width), mode="nearest")
        return base + self.patch_strength * patch


@dataclass
class CrossScaleCoherentGaussianProbe:
    """Gate cross-scale transport by frequency-wise view coherence."""

    cutoff: float
    cross_strength: float
    coherence_threshold: float
    sigma: float
    smooth_strength: float

    @property
    def name(self) -> str:
        return (
            f"cross_scale_coherent_gaussian_c{int(round(self.cutoff * 100)):02d}"
            f"_x{int(round(self.cross_strength * 100)):03d}"
            f"_q{int(round(self.coherence_threshold * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.smooth_strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("coherent cross-scale cutoff must be in (0, 1].")
        if not 0.0 <= self.cross_strength <= 1.0:
            raise ValueError("coherent cross-scale strength must be in [0, 1].")
        if not 0.0 <= self.coherence_threshold < 1.0:
            raise ValueError("coherence threshold must be in [0, 1).")
        if self.sigma <= 0 or self.smooth_strength < 0:
            raise ValueError("coherent cross-scale Gaussian parameters are invalid.")

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
        low_mean_flat = low_mean.flatten(1)
        high_mean_flat = high_mean.flatten(1)
        coefficients = torch.einsum("bvd,bd->bv", low_centered, low_mean_flat)
        transported = torch.einsum("bvd,bv->bd", high_centered, coefficients)
        transported = transported / view_gradients.size(0)
        transported_norm = transported.norm(dim=1, keepdim=True)
        high_norm = high_mean_flat.norm(dim=1, keepdim=True)
        transported = transported * (high_norm / transported_norm.clamp_min(1e-20))
        valid = (transported_norm > 1e-20) & (high_norm > 1e-20)
        transported = torch.where(valid, transported, high_mean_flat)

        high_spectrum = torch.fft.fftshift(
            torch.fft.fft2(high_views, dim=(-2, -1)), dim=(-2, -1)
        )
        high_mean_spectrum = torch.fft.fftshift(
            torch.fft.fft2(high_mean, dim=(-2, -1)), dim=(-2, -1)
        )
        transported_spectrum = torch.fft.fftshift(
            torch.fft.fft2(transported.view_as(high_mean), dim=(-2, -1)), dim=(-2, -1)
        )
        centered_spectrum = high_spectrum - high_mean_spectrum.unsqueeze(0)
        covariance = torch.einsum(
            "vbchw,bv->bchw", centered_spectrum, coefficients.to(centered_spectrum.dtype)
        )
        covariance = covariance / view_gradients.size(0)
        variance_high = centered_spectrum.abs().square().mean(dim=0)
        variance_coeff = coefficients.square().mean(dim=1, keepdim=True).view(
            coefficients.size(0), 1, 1, 1
        )
        coherence = covariance.abs() / (
            variance_high * variance_coeff
        ).sqrt().clamp_min(1e-20)
        coherence = coherence.clamp(0.0, 1.0)
        coherence = (
            (coherence - self.coherence_threshold)
            / (1.0 - self.coherence_threshold)
        ).clamp(0.0, 1.0)
        transported_spectrum = high_mean_spectrum + coherence * (
            transported_spectrum - high_mean_spectrum
        )
        mixed_spectrum = (1.0 - self.cross_strength) * high_mean_spectrum
        mixed_spectrum = mixed_spectrum + self.cross_strength * transported_spectrum
        mixed_high = torch.fft.ifft2(
            torch.fft.ifftshift(mixed_spectrum, dim=(-2, -1)), dim=(-2, -1)
        ).real
        transformed = low_mean + mixed_high
        return GaussianBlendProbe(self.sigma, self.smooth_strength).apply(
            transformed.unsqueeze(0), ["coherent_cross_scale"] * transformed.size(0), 0
        )


@dataclass
class GaussianBandBlendProbe:
    """Adjust low/high spatial components while retaining the full gradient."""

    sigma: float
    low_strength: float
    high_strength: float

    @property
    def name(self) -> str:
        return (
            f"gaussian_band_s{int(round(self.sigma * 10)):02d}"
            f"_l{int(round(self.low_strength * 100)):03d}"
            f"_h{int(round(self.high_strength * 100)):03d}"
        )

    def apply(self, view_gradients: torch.Tensor, sample_ids: list[str], step: int) -> torch.Tensor:
        del sample_ids, step
        if self.sigma <= 0:
            raise ValueError("Gaussian band blend requires sigma > 0.")
        gradient = view_gradients.mean(dim=0)
        low = GaussianBlendProbe(self.sigma, 1.0).apply(
            gradient.unsqueeze(0), ["gaussian_band"] * gradient.size(0), 0
        ) - gradient
        high = gradient - low
        return gradient + self.low_strength * low + self.high_strength * high


@dataclass
class CrossScaleGaussianProbe:
    """Compose cross-scale high-frequency replacement with weak smoothing."""

    cutoff: float
    cross_strength: float
    sigma: float
    smooth_strength: float

    @property
    def name(self) -> str:
        return (
            f"cross_scale_gaussian_c{int(round(self.cutoff * 100)):02d}"
            f"_x{int(round(self.cross_strength * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.smooth_strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("cross-scale Gaussian cutoff must be in (0, 1].")
        if not 0.0 <= self.cross_strength <= 1.0:
            raise ValueError("cross-scale Gaussian strength must be in [0, 1].")
        base = CrossScaleCovarianceProbe(
            "replace", self.cross_strength, cutoff=self.cutoff
        ).apply(view_gradients, sample_ids, step)
        return GaussianBlendProbe(
            self.sigma, self.smooth_strength
        ).apply(base.unsqueeze(0), sample_ids, step)


@dataclass
class ConfidenceCrossScaleGaussianProbe:
    """Apply cross-scale transport only to samples with sharedness signals."""

    cutoff: float
    cross_strength: float
    sigma: float
    smooth_strength: float
    selected_fraction: float

    @property
    def name(self) -> str:
        return (
            f"confidence_cross_scale_gaussian_c{int(round(self.cutoff * 100)):02d}"
            f"_x{int(round(self.cross_strength * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.smooth_strength * 100)):03d}"
            f"_q{int(round(self.selected_fraction * 100)):03d}"
        )

    @staticmethod
    def _rank01(values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= 1:
            return torch.zeros_like(values)
        order = torch.argsort(values)
        ranks = torch.empty_like(values, dtype=torch.float32)
        ranks[order] = torch.arange(
            values.numel(), device=values.device, dtype=torch.float32
        )
        return ranks / float(values.numel() - 1)

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("confidence cross-scale cutoff must be in (0, 1].")
        if not 0.0 <= self.cross_strength <= 1.0:
            raise ValueError("confidence cross-scale strength must be in [0, 1].")
        if self.sigma <= 0 or self.smooth_strength < 0:
            raise ValueError("confidence cross-scale Gaussian parameters are invalid.")
        if not 0.0 < self.selected_fraction <= 1.0:
            raise ValueError("confidence selected fraction must be in (0, 1].")
        if view_gradients.size(0) % 2:
            raise ValueError("confidence cross-scale requires paired views.")

        mean = view_gradients.mean(dim=0)
        batch_size, _, height, width = mean.shape
        amplitude = mean.detach().abs().mean(dim=(1, 2, 3))

        spatial_power = mean.detach().square().sum(dim=1)
        spatial_probability = spatial_power / spatial_power.sum(
            dim=(1, 2), keepdim=True
        ).clamp_min(1e-20)
        spatial_entropy = -(
            spatial_probability * spatial_probability.clamp_min(1e-20).log()
        ).sum(dim=(1, 2))

        grouped = view_gradients.view(
            view_gradients.size(0) // 2, 2, batch_size, 3, height, width
        ).mean(dim=1)
        group_flat = grouped.flatten(2).permute(1, 0, 2)
        group_sum = group_flat.sum(dim=1, keepdim=True)
        group_rest = (group_sum - group_flat) / float(group_flat.size(1) - 1)
        group_agreement = F.cosine_similarity(
            group_flat, group_rest, dim=2
        ).mean(dim=1)

        radius = _frequency_radius(height, width, mean.device)
        high = radius > radius.max() * self.cutoff
        spectrum_power = torch.fft.fftshift(
            torch.fft.fft2(mean.detach(), dim=(-2, -1)), dim=(-2, -1)
        ).abs().square().sum(dim=1)
        high_fraction = spectrum_power[:, high].sum(dim=1) / spectrum_power.sum(
            dim=(1, 2)
        ).clamp_min(1e-20)

        confidence = (
            self._rank01(amplitude)
            * self._rank01(spatial_entropy)
            * self._rank01(group_agreement)
            * (1.0 - self._rank01(high_fraction))
        )
        threshold = torch.quantile(confidence, 1.0 - self.selected_fraction)
        selected = confidence >= threshold
        transformed = CrossScaleGaussianProbe(
            self.cutoff,
            self.cross_strength,
            self.sigma,
            self.smooth_strength,
        ).apply(view_gradients, sample_ids, step)
        return torch.where(selected.view(batch_size, 1, 1, 1), transformed, mean)


@dataclass
class LowFrequencyConsensusGaussianProbe:
    """Use view sign consensus only in the low-frequency ViT patch structure."""

    cutoff: float
    consensus_strength: float
    sigma: float
    smooth_strength: float

    @property
    def name(self) -> str:
        return (
            f"low_consensus_gaussian_c{int(round(self.cutoff * 100)):02d}"
            f"_x{int(round(self.consensus_strength * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.smooth_strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("low-frequency consensus cutoff must be in (0, 1].")
        if not 0.0 <= self.consensus_strength <= 1.0:
            raise ValueError("low-frequency consensus strength must be in [0, 1].")
        if self.sigma <= 0 or self.smooth_strength < 0:
            raise ValueError("low-frequency consensus Gaussian parameters are invalid.")
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
        low_mean = low_views.mean(dim=0)
        high_mean = mean - low_mean
        consensus = low_views.sign().mean(dim=0).sign()
        low_scale = low_mean.abs().mean(dim=(1, 2, 3), keepdim=True)
        consensus = consensus * low_scale
        transformed = (
            (1.0 - self.consensus_strength) * low_mean
            + self.consensus_strength * consensus
            + high_mean
        )
        return GaussianBlendProbe(
            self.sigma, self.smooth_strength
        ).apply(transformed.unsqueeze(0), sample_ids, step)


@dataclass
class CrossScaleResidualGaussianProbe:
    """Keep covariance-supported high frequency and attenuate its residual.

    ``CrossScaleCovarianceProbe`` replaces part of the mean high-frequency
    component with the direction supported by low/high cross-view covariance.
    This probe decomposes that result as

    ``transported_high + residual_high``

    and applies a bounded gain only to ``residual_high`` before the same weak
    Gaussian blend.  The gain of one is exactly the existing cross-scale
    Gaussian candidate; lower gains test whether unsupported high-frequency
    detail is harmful while retaining the structured component.
    """

    cutoff: float
    cross_strength: float
    residual_gain: float
    sigma: float
    smooth_strength: float

    @property
    def name(self) -> str:
        return (
            f"cross_scale_residual_gaussian_c{int(round(self.cutoff * 100)):02d}"
            f"_x{int(round(self.cross_strength * 100)):03d}"
            f"_r{int(round(self.residual_gain * 100)):03d}"
            f"_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.smooth_strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("cross-scale residual cutoff must be in (0, 1].")
        if not 0.0 < self.cross_strength <= 1.0:
            raise ValueError("cross-scale residual strength must be in (0, 1].")
        if not 0.0 <= self.residual_gain <= 1.0:
            raise ValueError("cross-scale residual gain must be in [0, 1].")

        mean = view_gradients.mean(dim=0)
        height, width = mean.shape[-2:]
        radius = _frequency_radius(height, width, mean.device)
        low_mask = radius <= radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(mean, dim=(-2, -1)), dim=(-2, -1)
        )
        low_spectrum = spectrum * low_mask.view(1, 1, height, width)
        low = torch.fft.ifft2(
            torch.fft.ifftshift(low_spectrum, dim=(-2, -1)), dim=(-2, -1)
        ).real
        high = mean - low

        cross_scale = CrossScaleCovarianceProbe(
            "replace", self.cross_strength, cutoff=self.cutoff
        ).apply(view_gradients, sample_ids, step)
        transported = (
            cross_scale - low - (1.0 - self.cross_strength) * high
        ) / self.cross_strength
        residual = cross_scale - low - transported
        transformed = low + transported + self.residual_gain * residual
        return GaussianBlendProbe(
            self.sigma, self.smooth_strength
        ).apply(transformed.unsqueeze(0), sample_ids, step)


@dataclass
class JointLowAmplitudeHighFrequencyProbe:
    """Condition a coarse spectral intervention on global gradient amplitude.

    The intervention is deliberately sample-level and binary: within the
    current batch, select samples below the median raw gradient amplitude and
    above the median high-frequency energy fraction.  This targets the
    interaction observed in the held-out per-sample study instead of changing
    every coordinate or every sample.
    """

    operation: str
    strength: float
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return (
            f"joint_lowamp_highfreq_{self.operation}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.operation not in (
            "shrink",
            "low_boost",
            "gaussian",
            "low_only",
            "low_equalize",
        ):
            raise ValueError(
                "joint operation must be shrink, low_boost, gaussian, low_only, or low_equalize."
            )
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("joint strength must be in [0, 1].")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("joint cutoff must be in (0, 1].")
        gradient = view_gradients.mean(dim=0)
        batch_size, _, height, width = gradient.shape
        radius = _frequency_radius(height, width, gradient.device)
        high = radius > radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
        )
        high_spectrum = spectrum * high.view(1, 1, height, width)
        high_component = torch.fft.ifft2(
            torch.fft.ifftshift(high_spectrum, dim=(-2, -1)), dim=(-2, -1)
        ).real
        low_component = gradient - high_component
        power = spectrum.abs().square().sum(dim=1)
        high_fraction = (
            power.masked_select(high.view(1, height, width)).view(batch_size, -1).sum(dim=1)
            / power.sum(dim=(1, 2)).clamp_min(1e-20)
        )
        amplitude = gradient.abs().mean(dim=(1, 2, 3)).log()
        selected = (amplitude <= amplitude.median()) & (
            high_fraction >= high_fraction.median()
        )
        if self.operation == "shrink":
            transformed = low_component + (1.0 - self.strength) * high_component
        elif self.operation == "low_boost":
            transformed = gradient + self.strength * low_component
        elif self.operation == "low_only":
            transformed = low_component
        elif self.operation == "low_equalize":
            low_scale = low_component.abs().mean(dim=(1, 2, 3), keepdim=True)
            full_scale = gradient.abs().mean(dim=(1, 2, 3), keepdim=True)
            transformed = (
                low_component * (full_scale / low_scale.clamp_min(1e-20))
                + high_component
            )
        else:
            radius_kernel = max(1, int(round(3.0)))
            axis = torch.arange(
                -radius_kernel,
                radius_kernel + 1,
                device=gradient.device,
                dtype=gradient.dtype,
            )
            kernel_1d = torch.exp(-0.5 * axis.square())
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(
                1, 1, 2 * radius_kernel + 1, 2 * radius_kernel + 1
            ).repeat(gradient.size(1), 1, 1, 1)
            smoothed = F.conv2d(
                F.pad(gradient, (radius_kernel,) * 4, mode="reflect"),
                kernel,
                groups=gradient.size(1),
            )
            transformed = gradient + self.strength * smoothed
        return torch.where(selected.view(batch_size, 1, 1, 1), transformed, gradient)


@dataclass
class ConflictProjectionProbe:
    """Project only pairwise-conflicting view components before averaging.

    For each view, negative inner products with the other views are removed
    by a PCGrad-style projection.  Positive agreement and non-conflicting
    diversity are retained.  ``grouped`` applies the same operation to the
    ten A/B pair means, so phase-pair diversity is not treated as independent
    conflict.  The operation is a gradient-space change only.
    """

    strength: float
    grouped: bool = False

    @property
    def name(self) -> str:
        scope = "group" if self.grouped else "view"
        return f"conflict_project_{scope}_a{int(round(self.strength * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("conflict projection strength must be in [0, 1].")
        components = view_gradients
        if self.grouped:
            if view_gradients.size(0) % 2:
                raise ValueError("group conflict projection requires paired views.")
            components = view_gradients.view(
                view_gradients.size(0) // 2, 2, *view_gradients.shape[1:]
            ).mean(dim=1)
        flat = components.flatten(2).permute(1, 0, 2)  # [B,K,D]
        adjusted = flat.clone()
        for view_index in range(flat.size(1)):
            current = adjusted[:, view_index]
            for reference_index in range(flat.size(1)):
                if view_index == reference_index:
                    continue
                reference = flat[:, reference_index]
                dot = (current * reference).sum(dim=1, keepdim=True)
                coefficient = (
                    self.strength * dot.clamp_max(0.0)
                    / reference.square().sum(dim=1, keepdim=True).clamp_min(1e-20)
                )
                current = current - coefficient * reference
            adjusted[:, view_index] = current
        return adjusted.mean(dim=1).view_as(components[0])


@dataclass
class MagnitudeEnvelopeProbe:
    """Diffuse the spatial log-magnitude envelope while preserving signs.

    Signed-gradient smoothing changes the phase and can remove useful detail.
    This probe instead smooths ``log(|g|)`` and uses the result as a bounded
    multiplicative gain on the original signed gradient.  Isolated amplitude
    spikes are reduced, while low-amplitude coordinates inside a coherent
    neighborhood are amplified.  The gain is normalized per sample only up
    to a global scalar, which is immaterial because the existing pipeline
    performs mean-absolute gradient normalization afterwards.
    """

    sigma: float
    strength: float

    @property
    def name(self) -> str:
        return (
            f"magnitude_envelope_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.sigma <= 0 or not 0.0 <= self.strength <= 1.0:
            raise ValueError("magnitude envelope requires sigma > 0 and strength in [0, 1].")
        gradient = view_gradients.mean(dim=0)
        radius = max(1, int(round(3.0 * self.sigma)))
        axis = torch.arange(
            -radius, radius + 1, device=gradient.device, dtype=gradient.dtype
        )
        kernel_1d = torch.exp(-0.5 * (axis / self.sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(
            1, 1, 2 * radius + 1, 2 * radius + 1
        ).repeat(gradient.size(1), 1, 1, 1)
        log_magnitude = gradient.abs().clamp_min(1e-12).log()
        smooth_log_magnitude = F.conv2d(
            F.pad(log_magnitude, (radius,) * 4, mode="reflect"),
            kernel,
            groups=gradient.size(1),
        )
        log_gain = self.strength * (smooth_log_magnitude - log_magnitude)
        gain = log_gain.exp().clamp(0.5, 2.0)
        return gradient * gain


@dataclass
class RiskAdaptiveGaussianProbe:
    """Use a source-only risk score to allocate a bounded smooth component.

    The score combines three observations from the gradient itself: low raw
    amplitude, high-frequency energy, and (optionally) low agreement between
    dropout groups.  Each feature is converted to a within-batch percentile
    rank, so the probe does not use black-box labels or running statistics.
    The geometric mean avoids allowing one feature to dominate the other two.
    Only the Gaussian residual is weighted; the original gradient is retained.
    """

    mode: str
    strength: float
    sigma: float = 1.0

    @property
    def name(self) -> str:
        return (
            f"risk_gaussian_{self.mode}_s{int(round(self.sigma * 10)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    @staticmethod
    def _rank01(values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= 1:
            return torch.zeros_like(values)
        order = torch.argsort(values)
        ranks = torch.empty_like(values, dtype=torch.float32)
        ranks[order] = torch.arange(
            values.numel(), device=values.device, dtype=torch.float32
        )
        return ranks / float(values.numel() - 1)

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.mode not in (
            "amp_freq",
            "amp_freq_group",
            "freq_group",
            "positive_amp_freq",
            "positive_amp_freq_group",
        ):
            raise ValueError(
                "unsupported risk Gaussian mode."
            )
        if self.strength < 0 or self.sigma <= 0:
            raise ValueError("risk Gaussian requires strength >= 0 and sigma > 0.")

        gradient = view_gradients.mean(dim=0)
        batch_size, _, height, width = gradient.shape
        scores = []

        if self.mode in ("amp_freq", "amp_freq_group"):
            amplitude = gradient.abs().mean(dim=(1, 2, 3))
            scores.append(1.0 - self._rank01(amplitude))
        elif self.mode in ("positive_amp_freq", "positive_amp_freq_group"):
            amplitude = gradient.abs().mean(dim=(1, 2, 3))
            scores.append(self._rank01(amplitude))

        radius = _frequency_radius(height, width, gradient.device)
        high = radius > radius.max() * 0.50
        spectrum_power = torch.fft.fftshift(
            torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
        ).abs().square().sum(dim=1)
        high_fraction = spectrum_power[:, high].sum(dim=1) / spectrum_power.sum(
            dim=(1, 2)
        ).clamp_min(1e-20)
        if self.mode in ("positive_amp_freq", "positive_amp_freq_group"):
            scores.append(1.0 - self._rank01(high_fraction))
        else:
            scores.append(self._rank01(high_fraction))

        if self.mode in (
            "amp_freq_group",
            "freq_group",
            "positive_amp_freq_group",
        ):
            if view_gradients.size(0) % 2:
                raise ValueError("risk Gaussian group score requires paired views.")
            groups = view_gradients.view(
                view_gradients.size(0) // 2, 2, *view_gradients.shape[1:]
            ).mean(dim=1)
            flat = groups.flatten(2).permute(1, 0, 2)
            rest = (flat.sum(dim=1, keepdim=True) - flat) / max(1, flat.size(1) - 1)
            agreement = F.cosine_similarity(flat, rest, dim=2).mean(dim=1)
            if self.mode == "positive_amp_freq_group":
                scores.append(self._rank01(agreement))
            else:
                scores.append(1.0 - self._rank01(agreement))

        risk = torch.stack(scores, dim=0).prod(dim=0).pow(1.0 / len(scores))
        smooth_delta = GaussianBlendProbe(self.sigma, 1.0).apply(
            view_gradients, ["risk"] * batch_size, 0
        ) - gradient
        return gradient + self.strength * risk.view(-1, 1, 1, 1) * smooth_delta


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
class PositiveLowFrequencyProbe:
    """Boost low-frequency components only for source-side positive-score samples."""

    mode: str
    strength: float
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return (
            f"positive_low_boost_{self.mode}_c{int(round(self.cutoff * 100)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    @staticmethod
    def _rank01(values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= 1:
            return torch.zeros_like(values)
        order = torch.argsort(values)
        ranks = torch.empty_like(values, dtype=torch.float32)
        ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
        return ranks / float(values.numel() - 1)

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.mode not in ("amp_freq", "amp_freq_group"):
            raise ValueError("positive low-frequency mode must be amp_freq or amp_freq_group.")
        if not 0.0 <= self.strength <= 1.0 or not 0.0 < self.cutoff <= 1.0:
            raise ValueError("invalid positive low-frequency parameters.")
        gradient = view_gradients.mean(dim=0)
        batch_size, _, height, width = gradient.shape
        radius = _frequency_radius(height, width, gradient.device)
        low = radius <= radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
        )
        low_component = torch.fft.ifft2(
            torch.fft.ifftshift(
                spectrum * low.view(1, 1, height, width), dim=(-2, -1)
            ),
            dim=(-2, -1),
        ).real
        amplitude = gradient.abs().mean(dim=(1, 2, 3))
        power = spectrum.abs().square().sum(dim=1)
        high_fraction = power[:, ~low].sum(dim=1) / power.sum(dim=(1, 2)).clamp_min(1e-20)
        scores = [self._rank01(amplitude), 1.0 - self._rank01(high_fraction)]
        if self.mode == "amp_freq_group":
            if view_gradients.size(0) % 2:
                raise ValueError("positive low-frequency group mode requires paired views.")
            groups = view_gradients.view(
                view_gradients.size(0) // 2, 2, *view_gradients.shape[1:]
            ).mean(dim=1)
            flat = groups.flatten(2).permute(1, 0, 2)
            rest = (flat.sum(dim=1, keepdim=True) - flat) / max(1, flat.size(1) - 1)
            agreement = F.cosine_similarity(flat, rest, dim=2).mean(dim=1)
            scores.append(self._rank01(agreement))
        positive_score = torch.stack(scores, dim=0).prod(dim=0).pow(1.0 / len(scores))
        return gradient + self.strength * positive_score.view(
            batch_size, 1, 1, 1
        ) * low_component


@dataclass
class GaussianHighPowerCompositeProbe:
    """Compose high-band magnitude shaping with a weak signed Gaussian blend."""

    power: float
    gaussian_strength: float
    sigma: float = 1.0

    @property
    def name(self) -> str:
        return (
            f"composite_gaussian_s{int(round(self.sigma * 10)):02d}"
            f"_highpower{int(round(self.power * 100)):03d}"
            f"_a{int(round(self.gaussian_strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        if self.power <= 0 or self.sigma <= 0 or self.gaussian_strength < 0:
            raise ValueError("invalid Gaussian/high-power composite parameters.")
        shaped = SpectralBandAmplitudePowerProbe(self.power).apply(
            view_gradients, sample_ids, step
        )
        return GaussianBlendProbe(self.sigma, self.gaussian_strength).apply(
            shaped.unsqueeze(0), sample_ids, step
        )


@dataclass
class SpectralBandAmplitudePowerProbe:
    """Apply a power-law magnitude map only to the high-frequency band."""

    power: float
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return f"spectral_high_amplitude_power{int(round(self.power * 100)):03d}"

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if self.power <= 0:
            raise ValueError("spectral band amplitude power must be positive.")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("spectral band amplitude cutoff must be in (0, 1].")
        gradient = view_gradients.mean(dim=0)
        height, width = gradient.shape[-2:]
        radius = _frequency_radius(height, width, gradient.device)
        high = radius > radius.max() * self.cutoff
        spectrum = torch.fft.fftshift(
            torch.fft.fft2(gradient, dim=(-2, -1)), dim=(-2, -1)
        )
        magnitude = spectrum.abs()
        high_scale = magnitude[:, :, high].mean(dim=2, keepdim=True).unsqueeze(-1)
        high_scale = high_scale.clamp_min(1e-20)
        transformed_magnitude = high_scale * (
            magnitude / high_scale
        ).pow(self.power)
        transformed = torch.where(
            high.view(1, 1, height, width),
            spectrum / magnitude.clamp_min(1e-20) * transformed_magnitude,
            spectrum,
        )
        return torch.fft.ifft2(
            torch.fft.ifftshift(transformed, dim=(-2, -1)), dim=(-2, -1)
        ).real


@dataclass
class SpectralEnergyTransportProbe:
    """Transport uncertain high-frequency energy into the low-frequency axis.

    The mean spectrum is split into low and high bands.  A view-wise Wiener
    estimate retains high-frequency coefficients supported across views; the
    rejected high-frequency residual is not discarded.  Instead, a bounded
    fraction of its L2 energy is transferred to the existing low-frequency
    direction.  At strength one the transfer conserves the L2 energy of the
    original low+high decomposition (up to numerical error), while preserving
    the phase and sign of the retained components.
    """

    floor: float
    strength: float
    cutoff: float = 0.50

    @property
    def name(self) -> str:
        return (
            f"spectral_transport_high_f{int(round(self.floor * 100)):02d}"
            f"_a{int(round(self.strength * 100)):03d}"
        )

    def apply(
        self,
        view_gradients: torch.Tensor,
        sample_ids: list[str],
        step: int,
    ) -> torch.Tensor:
        del sample_ids, step
        if not 0.0 <= self.floor <= 1.0:
            raise ValueError("spectral transport floor must be in [0, 1].")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("spectral transport strength must be in [0, 1].")
        if not 0.0 < self.cutoff <= 1.0:
            raise ValueError("spectral transport cutoff must be in (0, 1].")

        spectra = torch.fft.fftshift(
            torch.fft.fft2(view_gradients, dim=(-2, -1)), dim=(-2, -1)
        )
        mean_spectrum = spectra.mean(dim=0)
        height, width = mean_spectrum.shape[-2:]
        radius = _frequency_radius(height, width, mean_spectrum.device)
        low = radius <= radius.max() * self.cutoff
        high = ~low

        noise = (spectra - mean_spectrum.unsqueeze(0)).abs().square().mean(dim=0)
        signal = (mean_spectrum.abs().square() - noise).clamp_min(0.0)
        wiener = signal / (signal + noise).clamp_min(1e-20)
        high_gain = self.floor + (1.0 - self.floor) * wiener
        gain = torch.where(
            high.view(1, 1, height, width), high_gain, torch.ones_like(high_gain)
        )
        retained_spectrum = mean_spectrum * gain
        retained = torch.fft.ifft2(
            torch.fft.ifftshift(retained_spectrum, dim=(-2, -1)), dim=(-2, -1)
        ).real
        low_component = torch.fft.ifft2(
            torch.fft.ifftshift(mean_spectrum * low.view(1, 1, height, width), dim=(-2, -1)),
            dim=(-2, -1),
        ).real
        removed = retained.new_zeros(retained.shape)
        removed_spectrum = mean_spectrum * (1.0 - gain)
        removed = torch.fft.ifft2(
            torch.fft.ifftshift(removed_spectrum, dim=(-2, -1)), dim=(-2, -1)
        ).real
        low_norm = low_component.flatten(1).norm(dim=1, keepdim=True)
        removed_norm = removed.flatten(1).norm(dim=1, keepdim=True)
        energy_ratio = (
            self.strength * removed_norm / low_norm.clamp_min(1e-20)
        )
        low_gain = torch.sqrt(1.0 + energy_ratio.square())
        low_gain = torch.where(low_norm > 1e-20, low_gain, torch.ones_like(low_gain))
        return retained + low_component * (low_gain.view(-1, 1, 1, 1) - 1.0)


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


def build_probe(name: str, model: torch.nn.Module | None = None) -> GradientProbe:
    if name.startswith("temporal_"):
        inner_name, encoded_window = name.removeprefix("temporal_").rsplit("_s", 1)
        encoded_start, encoded_end = encoded_window.split("e", 1)
        start, end = int(encoded_start), int(encoded_end)
        if start < 0 or end <= start:
            raise ValueError("temporal probe requires 0 <= start < end.")
        return StepWindowProbe(build_probe(inner_name, model=model), start, end)
    if name.startswith("group_remove_"):
        return GroupRemovalProbe(name.removeprefix("group_remove_"))
    if name.startswith("group_reliability_t"):
        return GroupReliabilityProbe(int(name.removeprefix("group_reliability_t")) / 10.0)
    if name.startswith("group_norm_equalize_a"):
        return GroupNormEqualizationProbe(int(name.removeprefix("group_norm_equalize_a")) / 100.0)
    if name.startswith("conflict_project_"):
        encoded = name.removeprefix("conflict_project_")
        scope, strength = encoded.split("_a", 1)
        if scope not in ("view", "group"):
            raise ValueError(f"unsupported conflict projection scope: {scope}")
        return ConflictProjectionProbe(
            int(strength) / 100.0,
            grouped=scope == "group",
        )
    if name.startswith("pair_difference_add_a"):
        return PairPhaseProbe(
            "difference_add",
            int(name.removeprefix("pair_difference_add_a")) / 100.0,
        )
    if name.startswith("pair_difference_reverse_a"):
        return PairPhaseProbe(
            "difference_reverse",
            int(name.removeprefix("pair_difference_reverse_a")) / 100.0,
        )
    if name.startswith("pair_difference_orthogonal_a"):
        return PairPhaseProbe(
            "difference_orthogonal",
            int(name.removeprefix("pair_difference_orthogonal_a")) / 100.0,
        )
    if name.startswith("pair_phase_wiener_high_f"):
        return PairPhaseProbe(
            "phase_wiener",
            int(name.removeprefix("pair_phase_wiener_high_f")) / 100.0,
            high_only=True,
        )
    if name.startswith("pair_phase_wiener_f"):
        return PairPhaseProbe(
            "phase_wiener",
            int(name.removeprefix("pair_phase_wiener_f")) / 100.0,
        )
    if name.startswith("momentum_trajectory_"):
        mode, encoded_strength = name.removeprefix("momentum_trajectory_").split("_a", 1)
        return MomentumTrajectoryProbe(int(encoded_strength) / 100.0, mode=mode)
    if name.startswith("adaptive_trajectory_blend_a"):
        return AdaptiveTrajectoryBlendProbe(
            int(name.removeprefix("adaptive_trajectory_blend_a")) / 100.0
        )
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
    if name.startswith("geometric_median_a"):
        encoded = name.removeprefix("geometric_median_a")
        strength, suffix = encoded.split("_", 1)
        if suffix not in ("raw", "scaled"):
            raise ValueError(f"unsupported geometric median name: {name}")
        return GeometricMedianProbe(
            int(strength) / 100.0,
            preserve_scale=suffix == "scaled",
        )
    if name.startswith("spatial_patch_"):
        return SpatialPatchProbe(name.removeprefix("spatial_patch_"), ratio=0.10)
    if name.startswith("patch_projection_g"):
        encoded = name.removeprefix("patch_projection_g")
        grid, strength = encoded.split("_a", 1)
        return PatchProjectionProbe(
            int(strength) / 100.0,
            grid=int(grid),
        )
    if name.startswith("patch_embedding_metric_a"):
        encoded = name.removeprefix("patch_embedding_metric_a")
        strength, suffix = encoded.split("_", 1)
        if suffix not in ("raw", "scaled"):
            raise ValueError(f"unsupported patch embedding metric name: {name}")
        if model is None:
            raise ValueError("patch embedding metric requires the whitebox model.")
        patch_embed = getattr(getattr(model, "model", model), "patch_embed", None)
        projection = getattr(patch_embed, "proj", None)
        weight = getattr(projection, "weight", None)
        if weight is None or weight.ndim != 4 or tuple(weight.shape[1:]) != (3, 16, 16):
            raise ValueError("whitebox model has no compatible 16x16 patch embedding.")
        flat_weight = weight.detach().float().flatten(1)
        metric = flat_weight.transpose(0, 1) @ flat_weight
        metric = metric / metric.trace().clamp_min(1e-20) * metric.size(0)
        return PatchEmbeddingMetricProbe(
            int(strength) / 100.0,
            metric,
            preserve_scale=suffix == "scaled",
        )
    if name.startswith("patch_energy_transport_g"):
        encoded = name.removeprefix("patch_energy_transport_g")
        grid, strength = encoded.split("_a", 1)
        return PatchEnergyTransportProbe(
            int(strength) / 100.0,
            grid=int(grid),
        )
    if name.startswith("patch_energy_transport_rescaled_"):
        encoded = name.removeprefix("patch_energy_transport_rescaled_")
        scale, encoded = encoded.split("_g", 1)
        grid, strength = encoded.split("_a", 1)
        return PatchEnergyTransportRescaleProbe(
            int(strength) / 100.0,
            grid=int(grid),
            scale=scale,
        )
    if name.startswith("frequency_remove_"):
        return FrequencyBandProbe(name.removeprefix("frequency_remove_"))
    if name.startswith("frequency_rescaled_high_"):
        encoded = name.removeprefix("frequency_rescaled_high_")
        scale, encoded_gain = encoded.split("_g", 1)
        return FrequencyGainRescaleProbe(
            int(encoded_gain) / 100.0,
            scale=scale,
        )
    if name.startswith("adaptive_frequency_rescaled_"):
        encoded = name.removeprefix("adaptive_frequency_rescaled_")
        scale, encoded = encoded.split("_q", 1)
        quantile, encoded_gain = encoded.split("_g", 1)
        return AdaptiveFrequencyGainRescaleProbe(
            int(encoded_gain) / 100.0,
            int(quantile) / 100.0,
            scale=scale,
        )
    if name.startswith("raw_global_scale_g"):
        return GlobalGradientScaleProbe(
            float(name.removeprefix("raw_global_scale_g"))
        )
    if name.startswith("vit_shared_color_inverse_a"):
        return FixedChannelGainProbe(
            int(name.removeprefix("vit_shared_color_inverse_a")) / 100.0,
            inverse=True,
        )
    if name.startswith("vit_shared_color_a"):
        return FixedChannelGainProbe(
            int(name.removeprefix("vit_shared_color_a")) / 100.0
        )
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
    if name.startswith("laplacian_prox_l"):
        return LaplacianProxProbe(
            int(name.removeprefix("laplacian_prox_l")) / 100.0
        )
    if name.startswith("post_momentum_gaussian_s"):
        encoded = name.removeprefix("post_momentum_gaussian_s")
        sigma, strength = encoded.split("_a", 1)
        return PostMomentumPreconditionProbe(
            "gaussian", int(sigma) / 10.0, strength=int(strength) / 100.0
        )
    if name.startswith("post_momentum_laplacian_l"):
        return PostMomentumPreconditionProbe(
            "laplacian",
            int(name.removeprefix("post_momentum_laplacian_l")) / 100.0,
        )
    if name.startswith("post_momentum_high_shrink_c"):
        encoded = name.removeprefix("post_momentum_high_shrink_c")
        cutoff, strength = encoded.split("_a", 1)
        return PostMomentumPreconditionProbe(
            "high_shrink",
            0.0,
            strength=int(strength) / 100.0,
            cutoff=int(cutoff) / 100.0,
        )
    if name.startswith("joint_lowamp_highfreq_"):
        operation, encoded_strength = name.removeprefix(
            "joint_lowamp_highfreq_"
        ).rsplit("_a", 1)
        return JointLowAmplitudeHighFrequencyProbe(
            operation,
            int(encoded_strength) / 100.0,
        )
    if name.startswith("magnitude_envelope_s"):
        encoded = name.removeprefix("magnitude_envelope_s")
        sigma, strength = encoded.split("_a", 1)
        return MagnitudeEnvelopeProbe(
            int(sigma) / 10.0,
            int(strength) / 100.0,
        )
    if name.startswith("risk_gaussian_"):
        encoded = name.removeprefix("risk_gaussian_")
        mode, encoded = encoded.split("_s", 1)
        sigma, strength = encoded.split("_a", 1)
        return RiskAdaptiveGaussianProbe(
            mode,
            int(strength) / 100.0,
            sigma=int(sigma) / 10.0,
        )
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
    if name.startswith("orthogonal_gaussian_s"):
        encoded = name.removeprefix("orthogonal_gaussian_s")
        sigma, strength = encoded.split("_a", 1)
        return OrthogonalGaussianProbe(
            int(sigma) / 10.0,
            int(strength) / 100.0,
        )
    if name.startswith("raw_temporal_gaussian_"):
        encoded = name.removeprefix("raw_temporal_gaussian_")
        power, sigma, strength = encoded.split("_")
        if not power.startswith("p") or not sigma.startswith("s") or not strength.startswith("a"):
            raise ValueError(f"unsupported raw temporal Gaussian name: {name}")
        return RawScaleTemporalGaussianProbe(
            int(power.removeprefix("p")) / 100.0,
            int(sigma.removeprefix("s")) / 10.0,
            int(strength.removeprefix("a")) / 100.0,
        )
    if name.startswith("raw_temporal_cross_scale_"):
        encoded = name.removeprefix("raw_temporal_cross_scale_p")
        power, encoded = encoded.split("_c", 1)
        cutoff, encoded = encoded.split("_x", 1)
        cross_strength, encoded = encoded.split("_s", 1)
        sigma, smooth_strength = encoded.split("_a", 1)
        return RawScaleCrossScaleGaussianProbe(
            int(power) / 100.0,
            int(cutoff) / 100.0,
            int(cross_strength) / 100.0,
            int(sigma) / 10.0,
            int(smooth_strength) / 100.0,
        )
    if name.startswith("patch_gaussian_s"):
        encoded = name.removeprefix("patch_gaussian_s")
        sigma, encoded = encoded.split("_a", 1)
        gaussian_strength, patch_strength = encoded.split("_p", 1)
        return PatchGaussianBlendProbe(
            int(sigma) / 10.0,
            int(gaussian_strength) / 100.0,
            int(patch_strength) / 100.0,
        )
    if name.startswith("gaussian_band_s"):
        encoded = name.removeprefix("gaussian_band_s")
        sigma, encoded = encoded.split("_l", 1)
        low_strength, high_strength = encoded.split("_h", 1)
        return GaussianBandBlendProbe(
            int(sigma) / 10.0,
            int(low_strength) / 100.0,
            int(high_strength) / 100.0,
        )
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
    if name.startswith("spectral_transport_high_f"):
        encoded = name.removeprefix("spectral_transport_high_f")
        floor, strength = encoded.split("_a", 1)
        return SpectralEnergyTransportProbe(
            int(floor) / 100.0,
            int(strength) / 100.0,
        )
    if name.startswith("spectral_high_amplitude_power"):
        return SpectralBandAmplitudePowerProbe(
            int(name.removeprefix("spectral_high_amplitude_power")) / 100.0
        )
    if name.startswith("composite_gaussian_s"):
        encoded = name.removeprefix("composite_gaussian_s")
        sigma, encoded = encoded.split("_highpower", 1)
        power, strength = encoded.split("_a", 1)
        return GaussianHighPowerCompositeProbe(
            int(power) / 100.0,
            int(strength) / 100.0,
            sigma=int(sigma) / 10.0,
        )
    if name.startswith("positive_low_boost_"):
        encoded = name.removeprefix("positive_low_boost_")
        mode, encoded = encoded.split("_c", 1)
        cutoff, strength = encoded.split("_a", 1)
        return PositiveLowFrequencyProbe(
            mode,
            int(strength) / 100.0,
            cutoff=int(cutoff) / 100.0,
        )
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
    if name.startswith("confidence_cross_scale_gaussian_"):
        encoded = name.removeprefix("confidence_cross_scale_gaussian_c")
        cutoff, encoded = encoded.split("_x", 1)
        cross_strength, encoded = encoded.split("_s", 1)
        sigma, encoded = encoded.split("_a", 1)
        smooth_strength, selected_fraction = encoded.split("_q", 1)
        return ConfidenceCrossScaleGaussianProbe(
            int(cutoff) / 100.0,
            int(cross_strength) / 100.0,
            int(sigma) / 10.0,
            int(smooth_strength) / 100.0,
            int(selected_fraction) / 100.0,
        )
    if name.startswith("low_consensus_gaussian_"):
        encoded = name.removeprefix("low_consensus_gaussian_c")
        cutoff, encoded = encoded.split("_x", 1)
        consensus_strength, encoded = encoded.split("_s", 1)
        sigma, smooth_strength = encoded.split("_a", 1)
        return LowFrequencyConsensusGaussianProbe(
            int(cutoff) / 100.0,
            int(consensus_strength) / 100.0,
            int(sigma) / 10.0,
            int(smooth_strength) / 100.0,
        )
    if name.startswith("cross_scale_"):
        encoded = name.removeprefix("cross_scale_")
        if encoded.startswith("coherent_gaussian_c"):
            cutoff, encoded = encoded.removeprefix("coherent_gaussian_c").split("_x", 1)
            cross_strength, encoded = encoded.split("_q", 1)
            threshold, encoded = encoded.split("_s", 1)
            sigma, smooth_strength = encoded.split("_a", 1)
            return CrossScaleCoherentGaussianProbe(
                cutoff=int(cutoff) / 100.0,
                cross_strength=int(cross_strength) / 100.0,
                coherence_threshold=int(threshold) / 100.0,
                sigma=int(sigma) / 10.0,
                smooth_strength=int(smooth_strength) / 100.0,
            )
        if encoded.startswith("patch_gaussian_c"):
            cutoff, encoded = encoded.removeprefix("patch_gaussian_c").split("_x", 1)
            cross_strength, encoded = encoded.split("_s", 1)
            sigma, encoded = encoded.split("_a", 1)
            smooth_strength, patch_strength = encoded.split("_p", 1)
            return CrossScalePatchGaussianProbe(
                cutoff=int(cutoff) / 100.0,
                cross_strength=int(cross_strength) / 100.0,
                sigma=int(sigma) / 10.0,
                smooth_strength=int(smooth_strength) / 100.0,
                patch_strength=int(patch_strength) / 100.0,
            )
        if encoded.startswith("residual_gaussian_c"):
            encoded = encoded.removeprefix("residual_gaussian_c")
            cutoff, encoded = encoded.split("_x", 1)
            cross_strength, encoded = encoded.split("_r", 1)
            residual_gain, encoded = encoded.split("_s", 1)
            sigma, smooth_strength = encoded.split("_a", 1)
            return CrossScaleResidualGaussianProbe(
                cutoff=int(cutoff) / 100.0,
                cross_strength=int(cross_strength) / 100.0,
                residual_gain=int(residual_gain) / 100.0,
                sigma=int(sigma) / 10.0,
                smooth_strength=int(smooth_strength) / 100.0,
            )
        if encoded.startswith("gaussian_c"):
            cutoff, encoded = encoded.removeprefix("gaussian_c").split("_x", 1)
            cross_strength, encoded = encoded.split("_s", 1)
            sigma, smooth_strength = encoded.split("_a", 1)
            return CrossScaleGaussianProbe(
                cutoff=int(cutoff) / 100.0,
                cross_strength=int(cross_strength) / 100.0,
                sigma=int(sigma) / 10.0,
                smooth_strength=int(smooth_strength) / 100.0,
            )
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
