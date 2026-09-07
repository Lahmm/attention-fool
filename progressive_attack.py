"""Architecture-neutral progressive patch-score mainline.

This module is deliberately independent from :mod:`attack`.  It owns the
complete attack lifecycle and relies only on the progressive adapter contract
implemented by ``nets/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from gradient_replay import GradientReplay


PROGRESSIVE_PATCH_SELECTORS = ("patch_score", "high", "low", "random")
PROGRESSIVE_FEATURE_NOISE_TYPES = ("opponent_projected", "gaussian")
DEFAULT_DROP_RATIOS = (0.05, 0.05, 0.05)


@dataclass(frozen=True)
class ProgressiveMaskSelection:
    checkpoint: str
    mask: torch.Tensor
    count: int
    grid_size: tuple[int, int]

    def validate(self, *, batch_size: int) -> None:
        token_count = self.grid_size[0] * self.grid_size[1]
        if self.mask.shape != (batch_size, token_count):
            raise ValueError(
                f"mask at {self.checkpoint} must have shape "
                f"[{batch_size}, {token_count}], got {tuple(self.mask.shape)}."
            )
        if self.mask.dtype != torch.bool:
            raise ValueError("progressive masks must be boolean.")
        expected = torch.full(
            (batch_size,), self.count, device=self.mask.device, dtype=torch.long
        )
        if not torch.equal(self.mask.sum(dim=1), expected):
            raise ValueError("progressive mask counts differ within a batch.")


@dataclass(frozen=True)
class ProgressiveMaskSchedule:
    selections: tuple[ProgressiveMaskSelection, ...]

    def validate(self, *, batch_size: int, token_count: int | None = None) -> None:
        if not self.selections:
            raise ValueError("a progressive schedule must contain selections.")
        for selection in self.selections:
            selection.validate(batch_size=batch_size)
            if token_count is not None and selection.mask.size(1) != token_count:
                raise ValueError("schedule token count does not match the requested count.")

    @property
    def checkpoints(self) -> tuple[str, ...]:
        return tuple(item.checkpoint for item in self.selections)

    @property
    def masks(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.mask for item in self.selections)

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(item.count for item in self.selections)

    @property
    def grid_sizes(self) -> tuple[tuple[int, int], ...]:
        return tuple(item.grid_size for item in self.selections)

    @property
    def grid_size(self) -> tuple[int, int]:
        grids = self.grid_sizes
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError("cross-scale schedules do not have one grid_size.")
        return grids[0]


class ProgressivePatchScoreAttacker:
    """Complete progressive attack implementation with model-owned traversal."""

    def __init__(
        self,
        model,
        *,
        checkpoints: tuple[str | int, ...] | None = None,
        drop_ratios: tuple[float, ...] = DEFAULT_DROP_RATIOS,
        patch_selector: str = "patch_score",
        score_global_noise_strength: float | None = None,
        score_cls_noise_strength: float | None = None,
        opponent_noise_strength: float = 0.2,
        feature_noise_type: str = "opponent_projected",
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        use_momentum: bool = True,
        momentum_decay: float = 1.0,
        nesterov: bool = False,
        ti_sigma: float = 0.0,
        input_diversity_groups: int = 10,
        input_diversity_views_per_group: int = 2,
        input_diversity_phase_shift_set: tuple[tuple[int, int], ...] = (
            (4, 4),
            (8, 8),
            (12, 12),
        ),
        post_dropout_phase_token_noise: bool = True,
        gaussian_sigma: float = 4.0,
        gaussian_alpha: float = 0.75,
        device: torch.device | None = None,
        attack_method: str | None = None,
        input_diversity: bool = False,
        **unsupported,
    ) -> None:
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unsupported progressive attack arguments: {names}.")
        if attack_method not in (None, "progressive", "original_score_postdrop_phase_pair"):
            raise ValueError("the progressive attacker does not execute a legacy attack method.")
        if input_diversity:
            raise ValueError("the progressive phase-pair mainline does not combine with DIM.")
        if epsilon < 0 or steps <= 0:
            raise ValueError("epsilon must be non-negative and steps must be positive.")
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive.")
        if nesterov and not use_momentum:
            raise ValueError("Nesterov requires momentum.")
        if ti_sigma < 0:
            raise ValueError("ti_sigma must be non-negative.")
        if input_diversity_groups <= 0 or input_diversity_views_per_group != 2:
            raise ValueError("progressive requires positive groups and exactly two views per group.")
        if input_diversity_groups * input_diversity_views_per_group > 20:
            raise ValueError("actual progressive views must be <= 20.")
        if not input_diversity_phase_shift_set:
            raise ValueError("phase shift set cannot be empty.")
        if gaussian_sigma < 0 or gaussian_alpha < 0:
            raise ValueError("Gaussian parameters must be non-negative.")
        if gaussian_alpha > 0 and gaussian_sigma == 0:
            raise ValueError("gaussian_sigma must be positive when gaussian_alpha is enabled.")
        if patch_selector not in PROGRESSIVE_PATCH_SELECTORS:
            raise ValueError(
                f"patch_selector must be one of {PROGRESSIVE_PATCH_SELECTORS}, "
                f"got {patch_selector!r}."
            )
        if score_global_noise_strength is not None and score_cls_noise_strength is not None:
            if float(score_global_noise_strength) != float(score_cls_noise_strength):
                raise ValueError("score noise aliases disagree.")
        resolved_score_noise = (
            score_global_noise_strength
            if score_global_noise_strength is not None
            else score_cls_noise_strength
            if score_cls_noise_strength is not None
            else 0.2
        )
        if resolved_score_noise < 0 or opponent_noise_strength < 0:
            raise ValueError("score and opponent noise strengths must be non-negative.")
        if feature_noise_type not in PROGRESSIVE_FEATURE_NOISE_TYPES:
            raise ValueError(
                f"feature_noise_type must be one of {PROGRESSIVE_FEATURE_NOISE_TYPES}."
            )

        candidates = tuple(model.progressive_checkpoint_candidates())
        requested = checkpoints or tuple(model.default_progressive_checkpoints())
        canonical = tuple(self._canonical_checkpoint(model, value) for value in requested)
        ratios = tuple(float(value) for value in drop_ratios)
        if len(canonical) != 3 or len(ratios) != 3:
            raise ValueError("exactly three progressive checkpoints and ratios are required.")
        if any(item not in candidates for item in canonical):
            raise ValueError(
                f"checkpoints {canonical} must belong to adapter candidates {candidates}."
            )
        positions = tuple(candidates.index(item) for item in canonical)
        if positions != tuple(sorted(positions)) or len(set(positions)) != 3:
            raise ValueError("progressive checkpoints must be strictly increasing.")
        if any(ratio <= 0.0 or ratio > 0.5 for ratio in ratios):
            raise ValueError("each progressive drop ratio must satisfy 0 < ratio <= 0.5.")

        self.model = model
        self.model.eval()
        self.device = device if device is not None else DEVICE
        self.progressive_checkpoints = canonical
        self.progressive_drop_ratios = ratios
        self.progressive_patch_selector = patch_selector
        self.score_global_noise_strength = float(resolved_score_noise)
        # Compatibility name is metadata-only; the canonical setting is global
        # because GAP-based adapters do not have a CLS token.
        self.score_cls_noise_strength = self.score_global_noise_strength
        self.opponent_noise_strength = float(opponent_noise_strength)
        self.feature_noise_type = feature_noise_type
        self.post_dropout_phase_token_noise = bool(post_dropout_phase_token_noise)
        self.post_dropout_feature_noise_strength = self.opponent_noise_strength
        self.patch_dropout_ratio = 0.0
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size) if step_size is not None else self.epsilon / self.steps
        self.use_momentum = bool(use_momentum)
        self.decay = float(momentum_decay)
        self.nesterov = bool(nesterov)
        self.ti_sigma = float(ti_sigma)
        self.input_diversity_groups = int(input_diversity_groups)
        self.input_diversity_views_per_group = 2
        self.input_diversity_phase_shift_set = tuple(
            tuple(int(value) for value in shift) for shift in input_diversity_phase_shift_set
        )
        self.gaussian_sigma = float(gaussian_sigma)
        self.gaussian_alpha = float(gaussian_alpha)
        self.pixel_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        model_mean = getattr(model, "model_mean", IMAGENET_MEAN)
        model_std = getattr(model, "model_std", IMAGENET_STD)
        self.model_mean = torch.tensor(model_mean, device=self.device).view(1, 3, 1, 1)
        self.model_std = torch.tensor(model_std, device=self.device).view(1, 3, 1, 1)
        self._gradient_replay: GradientReplay | None = None
        self._actual_forward_view_count = 0
        self._progressive_mask_counts: tuple[int, ...] = ()
        self._progressive_mask_grids: tuple[tuple[int, int], ...] = ()
        self._progressive_global_modes: tuple[str, ...] = ()
        self._progressive_schedule_count = 0
        self._progressive_checkpoint_selection_count = 0
        self._feature_noise_type = ""
        self._gradient_diagnostics: dict[str, list[float]] = {
            "view_cosine_to_final": [],
            "sign_agreement": [],
            "effective_rank": [],
            "mi_cumulative_cosine": [],
        }
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma) if self.ti_sigma > 0 else None

    @staticmethod
    def _canonical_checkpoint(model, value: str | int) -> str:
        if isinstance(value, int):
            if getattr(model, "model_name", "") != "vit_base_patch16_224":
                raise ValueError("integer checkpoints are supported only by the ViT compatibility API.")
            return f"block{value}"
        return str(value)

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.model_mean) / self.model_std

    def _normalize_output(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _randn_like(self, tensor: torch.Tensor, event: str) -> torch.Tensor:
        if self._gradient_replay is not None:
            return self._gradient_replay.randn_like(tensor, event)
        return torch.randn_like(tensor)

    def _score_at_checkpoint(self, features) -> torch.Tensor:
        local = features.local_tokens
        global_token = features.global_token
        if self.score_global_noise_strength > 0:
            token_rms = local.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            noise = self._randn_like(
                global_token, f"progressive_score_cls_{features.layer_id}"
            )
            global_token = global_token + self.score_global_noise_strength * token_rms * noise
        return F.cosine_similarity(local, global_token.expand_as(local), dim=-1)

    def _sample_score_mask(
        self, scores: torch.Tensor, ratio: float, checkpoint: str, *, largest: bool
    ) -> torch.Tensor:
        batch_size, token_count = scores.shape
        candidate_count = max(1, token_count // 2)
        drop_count = max(1, int(round(token_count * ratio)))
        if drop_count > candidate_count:
            raise ValueError("drop budget exceeds the selected score half.")
        candidates = torch.topk(scores, candidate_count, dim=1, largest=largest).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for batch_index in range(batch_size):
            if self._gradient_replay is None:
                order = torch.randperm(candidate_count, device=scores.device)
            else:
                order = self._gradient_replay.randperm(
                    candidate_count,
                    f"progressive_drop_{checkpoint}",
                    batch_index,
                    device=scores.device,
                )
            mask[batch_index, candidates[batch_index, order[:drop_count]]] = True
        return mask.detach()

    def _sample_random_mask(
        self,
        *,
        batch_size: int,
        token_count: int,
        ratio: float,
        checkpoint: str,
        device: torch.device,
    ) -> torch.Tensor:
        drop_count = max(1, int(round(token_count * ratio)))
        mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
        for batch_index in range(batch_size):
            if self._gradient_replay is None:
                order = torch.randperm(token_count, device=device)
            else:
                order = self._gradient_replay.randperm(
                    token_count,
                    f"progressive_random_drop_{checkpoint}",
                    batch_index,
                    device=device,
                )
            mask[batch_index, order[:drop_count]] = True
        return mask.detach()

    def _sample_high_mask(
        self, scores: torch.Tensor, ratio: float, *, checkpoint: str | int
    ) -> torch.Tensor:
        checkpoint_id = self._canonical_checkpoint(self.model, checkpoint)
        return self._sample_score_mask(scores, ratio, checkpoint_id, largest=True)

    def _sample_low_mask(
        self, scores: torch.Tensor, ratio: float, *, checkpoint: str | int
    ) -> torch.Tensor:
        checkpoint_id = self._canonical_checkpoint(self.model, checkpoint)
        return self._sample_score_mask(scores, ratio, checkpoint_id, largest=False)

    def _build_mask_schedule(self, pixels: torch.Tensor) -> ProgressiveMaskSchedule:
        with torch.no_grad():
            state = self.model.begin_progressive_forward(self._normalize(pixels.detach()))
            selections: list[ProgressiveMaskSelection] = []
            global_modes: list[str] = []
            for checkpoint, ratio in zip(
                self.progressive_checkpoints, self.progressive_drop_ratios
            ):
                state = self.model.advance_progressive_state(state, checkpoint)
                features = self.model.progressive_score_features(state, checkpoint)
                features.validate()
                global_modes.append(features.global_mode)
                if self.progressive_patch_selector == "random":
                    mask = self._sample_random_mask(
                        batch_size=features.local_tokens.size(0),
                        token_count=features.local_tokens.size(1),
                        ratio=ratio,
                        checkpoint=checkpoint,
                        device=features.local_tokens.device,
                    )
                else:
                    scores = self._score_at_checkpoint(features)
                    mask = self._sample_score_mask(
                        scores,
                        ratio,
                        checkpoint,
                        largest=self.progressive_patch_selector != "low",
                    )
                count = int(mask.sum(dim=1)[0].item())
                selections.append(
                    ProgressiveMaskSelection(checkpoint, mask, count, features.grid_size)
                )
                state = self.model.apply_progressive_mask(state, mask)
        schedule = ProgressiveMaskSchedule(tuple(selections))
        schedule.validate(batch_size=pixels.size(0))
        self._progressive_mask_counts = schedule.counts
        self._progressive_mask_grids = schedule.grid_sizes
        self._progressive_global_modes = tuple(global_modes)
        self._progressive_schedule_count += 1
        self._progressive_checkpoint_selection_count += len(selections)
        return schedule

    @staticmethod
    def _mask_to_image(
        mask: torch.Tensor,
        grid_size: tuple[int, int],
        height: int,
        width: int,
    ) -> torch.Tensor:
        return F.interpolate(
            mask[:, None].to(torch.float32).view(mask.size(0), 1, *grid_size),
            size=(height, width),
            mode="nearest",
        )

    @staticmethod
    def _apply_phase_shift(tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        if dx == 0 and dy == 0:
            return tensor
        padded = F.pad(
            tensor,
            (max(0, dx), max(0, -dx), max(0, dy), max(0, -dy)),
            mode="reflect",
        )
        return padded[
            ...,
            max(0, -dy) : max(0, -dy) + tensor.size(-2),
            max(0, -dx) : max(0, -dx) + tensor.size(-1),
        ]

    def _apply_samplewise_phase_shifts(
        self, tensor: torch.Tensor, phases: list[tuple[int, int]]
    ) -> torch.Tensor:
        return torch.cat(
            [self._apply_phase_shift(tensor[index : index + 1], *phase) for index, phase in enumerate(phases)],
            dim=0,
        )

    def _pick_input_diversity_phases(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[int, int]]:
        if self._gradient_replay is None:
            index = int(torch.randint(len(self.input_diversity_phase_shift_set), (1,)).item())
            return [self.input_diversity_phase_shift_set[index]] * batch_size
        phases = [
            self.input_diversity_phase_shift_set[
                self._gradient_replay.randint(
                    len(self.input_diversity_phase_shift_set), "phase", index, device=device
                )
            ]
            for index in range(batch_size)
        ]
        for index, phase in enumerate(phases):
            self._gradient_replay.record_phase(index, phase)
        return phases

    def _phase_mask_schedule(
        self,
        schedule: ProgressiveMaskSchedule,
        phases: list[tuple[int, int]],
        *,
        height: int,
        width: int,
    ) -> ProgressiveMaskSchedule:
        selections = []
        for item in schedule.selections:
            image_mask = self._mask_to_image(item.mask, item.grid_size, height, width)
            shifted_image = self._apply_samplewise_phase_shifts(image_mask, phases)
            occupancy = F.adaptive_avg_pool2d(shifted_image, item.grid_size).flatten(1)
            indices = torch.argsort(occupancy, dim=1, descending=True, stable=True)[:, : item.count]
            shifted = torch.zeros_like(item.mask)
            shifted.scatter_(1, indices, True)
            selections.append(
                ProgressiveMaskSelection(item.checkpoint, shifted.detach(), item.count, item.grid_size)
            )
        shifted_schedule = ProgressiveMaskSchedule(tuple(selections))
        shifted_schedule.validate(batch_size=schedule.selections[0].mask.size(0))
        return shifted_schedule

    def _schedule_image_union(
        self, schedule: ProgressiveMaskSchedule, height: int, width: int
    ) -> torch.Tensor:
        image_masks = [
            self._mask_to_image(item.mask, item.grid_size, height, width)
            for item in schedule.selections
        ]
        return torch.stack(image_masks, dim=0).amax(dim=0)

    @staticmethod
    def _image_mask_to_projection_drop_mask(image_mask: torch.Tensor, state) -> torch.Tensor:
        state.validate()
        kernel = torch.ones(
            1, 1, *state.projection_kernel, device=image_mask.device, dtype=image_mask.dtype
        )
        kwargs = {
            "stride": state.projection_stride,
            "padding": state.projection_padding,
            "dilation": state.projection_dilation,
        }
        dropped_area = F.conv2d(image_mask, kernel, **kwargs)
        valid_area = F.conv2d(torch.ones_like(image_mask), kernel, **kwargs).clamp_min(1.0)
        fraction = dropped_area / valid_area
        if tuple(fraction.shape[-2:]) != state.grid_size:
            raise ValueError("projection mask grid does not match initial attack feature grid.")
        return fraction.flatten(1).gt(0.5)

    def _strict_opponent_feature_noise(self, state) -> torch.Tensor:
        state.validate()
        local = state.local_tokens
        weight = state.rgb_projection_weight
        kernel_h, kernel_w = state.projection_kernel
        batch, count, dimension = local.shape
        coefficients = self._randn_like(
            torch.empty(
                batch, count, 3, kernel_h, kernel_w, device=local.device, dtype=local.dtype
            ),
            "mainline_opponent_token",
        )
        luma = 0.5**0.5 * coefficients[:, :, 0:1]
        red_green = 1.25**0.5 * coefficients[:, :, 1:2]
        yellow_blue = 1.25**0.5 * coefficients[:, :, 2:3]
        pixel_noise = torch.cat(
            (
                3**-0.5 * luma + 2**-0.5 * red_green + 6**-0.5 * yellow_blue,
                3**-0.5 * luma - 2**-0.5 * red_green + 6**-0.5 * yellow_blue,
                3**-0.5 * luma - 2 * 6**-0.5 * yellow_blue,
            ),
            dim=2,
        ).flatten(2)
        projection = weight.detach().to(local).reshape(dimension, -1)
        if projection.size(1) != pixel_noise.size(2):
            raise ValueError("opponent noise and RGB projection dimensions do not match.")
        feature_noise = pixel_noise.matmul(projection.t())
        token_rms = local.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        noise_rms = feature_noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        self._feature_noise_type = "opponent_channel_rgb_projection"
        return self.opponent_noise_strength * feature_noise * (token_rms / noise_rms)

    def _kept_feature_noise(self, state) -> torch.Tensor:
        if self.feature_noise_type == "opponent_projected":
            return self._strict_opponent_feature_noise(state)
        raw_noise = self._randn_like(state.local_tokens, "mainline_feature_gaussian")
        token_rms = (
            state.local_tokens.detach()
            .square()
            .mean(dim=(1, 2), keepdim=True)
            .sqrt()
            .clamp_min(1e-6)
        )
        noise_rms = raw_noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        self._feature_noise_type = "feature_iid_gaussian"
        return self.opponent_noise_strength * raw_noise * (token_rms / noise_rms)

    def _forward_with_schedule(
        self, pixels: torch.Tensor, labels: torch.Tensor, schedule: ProgressiveMaskSchedule
    ) -> torch.Tensor:
        normalized = self._normalize(pixels)
        initial = self.model.prepare_attack_feature_state(normalized)
        initial.validate()
        state = self.model.begin_progressive_forward(normalized)
        if state.local_tokens.shape != initial.local_tokens.shape:
            raise ValueError("progressive and RGB-projection initial states disagree.")
        local = state.local_tokens
        if self.post_dropout_phase_token_noise and self.opponent_noise_strength > 0:
            image_union = self._schedule_image_union(
                schedule, pixels.size(-2), pixels.size(-1)
            ).to(pixels)
            initial_drop = self._image_mask_to_projection_drop_mask(image_union, initial)
            noise = self._kept_feature_noise(initial)
            local = torch.where((~initial_drop).unsqueeze(-1), local + noise, local)
        state = self.model.replace_progressive_local_tokens(state, local)
        for item in schedule.selections:
            state = self.model.advance_progressive_state(state, item.checkpoint)
            if state.grid_size != item.grid_size:
                raise ValueError(
                    f"replay grid mismatch at {item.checkpoint}: {state.grid_size} != {item.grid_size}."
                )
            state = self.model.apply_progressive_mask(state, item.mask)
        logits = self.model.finish_progressive_forward(state)
        return F.cross_entropy(logits, labels)

    def _iter_attack_losses(
        self, pixels: torch.Tensor, labels: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        for group_index in range(self.input_diversity_groups):
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(group=group_index, view=-1)
            schedule = self._build_mask_schedule(pixels.detach())
            phases = self._pick_input_diversity_phases(pixels.size(0), pixels.device)
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=0)
            self._actual_forward_view_count += 1
            yield self._forward_with_schedule(pixels, labels, schedule)
            shifted_schedule = self._phase_mask_schedule(
                schedule, phases, height=pixels.size(-2), width=pixels.size(-1)
            )
            shifted_pixels = self._apply_samplewise_phase_shifts(pixels, phases)
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=1)
            self._actual_forward_view_count += 1
            yield self._forward_with_schedule(shifted_pixels, labels, shifted_schedule)

    @staticmethod
    def _aggregate_gradients(view_gradients: torch.Tensor) -> torch.Tensor:
        if view_gradients.ndim != 5 or view_gradients.size(0) == 0:
            raise ValueError("view gradients must have shape [V,B,C,H,W] with V > 0.")
        return view_gradients.mean(dim=0)

    def _record_gradient_diagnostics(
        self, view_gradients: torch.Tensor, final_gradient: torch.Tensor
    ) -> None:
        with torch.no_grad():
            views = view_gradients.detach().flatten(2).transpose(0, 1)
            final = final_gradient.detach().flatten(1)
            cosines = F.cosine_similarity(views, final.unsqueeze(1), dim=-1)
            final_sign = final.sign()
            valid = final_sign.ne(0).unsqueeze(1)
            agreement = (views.sign().eq(final_sign.unsqueeze(1)) & valid).float()
            denominator = valid.expand_as(agreement).sum().clamp_min(1.0)
            normalized = views / views.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            eigenvalues = torch.linalg.eigvalsh(
                torch.bmm(normalized, normalized.transpose(1, 2))
            ).clamp_min(0.0)
            probabilities = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            effective_rank = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            )
            self._gradient_diagnostics["view_cosine_to_final"].append(float(cosines.mean().cpu()))
            self._gradient_diagnostics["sign_agreement"].append(
                float(agreement.sum().cpu() / denominator.cpu())
            )
            self._gradient_diagnostics["effective_rank"].append(
                float(effective_rank.mean().cpu())
            )

    def gradient_diagnostics_summary(self) -> dict[str, float | int]:
        summary: dict[str, float | int] = {
            "num_gradient_batches": len(self._gradient_diagnostics["effective_rank"])
        }
        for name, values in self._gradient_diagnostics.items():
            if values:
                summary[name] = sum(values) / len(values)
        return summary

    @staticmethod
    def _build_ti_kernel(sigma: float) -> torch.Tensor:
        radius = int(3 * sigma)
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        gaussian = torch.exp(-0.5 * (axis / sigma).square())
        gaussian = gaussian / gaussian.sum()
        return (gaussian[:, None] @ gaussian[None, :]).view(1, 1, -1, gaussian.numel())

    def _smooth_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self._ti_kernel is None:
            return grad
        kernel = self._ti_kernel.to(grad.device, grad.dtype).repeat(grad.size(1), 1, 1, 1)
        padding = kernel.size(-1) // 2
        return F.conv2d(
            F.pad(grad, (padding, padding, padding, padding), mode="reflect"),
            kernel,
            groups=grad.size(1),
        )

    def _apply_gaussian_residual(self, grad: torch.Tensor) -> torch.Tensor:
        if self.gaussian_alpha == 0:
            return grad
        radius = max(1, int(round(3 * self.gaussian_sigma)))
        axis = torch.arange(-radius, radius + 1, device=grad.device, dtype=grad.dtype)
        kernel_1d = torch.exp(-0.5 * (axis / self.gaussian_sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(
            1, 1, 2 * radius + 1, 2 * radius + 1
        ).repeat(grad.size(1), 1, 1, 1)
        smoothed = F.conv2d(
            F.pad(grad, (radius, radius, radius, radius), mode="reflect"),
            kernel,
            groups=grad.size(1),
        )
        return grad + self.gaussian_alpha * smoothed

    def _processed_gradient(
        self, pixels: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gradients = [
            torch.autograd.grad(loss, pixels, retain_graph=False)[0]
            for loss in self._iter_attack_losses(pixels, labels)
        ]
        if not gradients:
            raise RuntimeError("no progressive losses were generated.")
        views = torch.stack(gradients, dim=0)
        raw_mean = self._aggregate_gradients(views)
        self._record_gradient_diagnostics(views, raw_mean)
        processed = self._smooth_grad(self._apply_gaussian_residual(raw_mean))
        return views, raw_mean, processed

    def _check_view_count(self) -> None:
        expected = self.input_diversity_groups * 2
        if self._actual_forward_view_count != expected:
            raise RuntimeError(
                f"view count mismatch: {self._actual_forward_view_count} != {expected}."
            )

    def probe_attack_gradients(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        *,
        replay: GradientReplay | None = None,
        sample_ids: list[str] | None = None,
        step_index: int = 0,
    ) -> dict[str, torch.Tensor]:
        if replay is not None:
            if sample_ids is None or len(sample_ids) != pixels.size(0):
                raise ValueError("sample_ids must match pixels when replay is enabled.")
            replay.begin_batch(sample_ids)
            replay.set_context(step=step_index, group=-1, view=-1)
        self._gradient_replay = replay
        self._actual_forward_view_count = 0
        probe = pixels.to(self.device).detach().requires_grad_(True)
        labels = labels.to(self.device)
        try:
            views, raw_mean, processed = self._processed_gradient(probe, labels)
            self._check_view_count()
            return {
                "view_gradients": views.detach(),
                "raw_mean": raw_mean.detach(),
                "processed": processed.detach(),
            }
        finally:
            self._gradient_replay = None

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        replay: GradientReplay | None = None,
        sample_ids: list[str] | None = None,
    ) -> torch.Tensor:
        if replay is not None:
            if sample_ids is None or len(sample_ids) != images.size(0):
                raise ValueError("sample_ids must match the batch when replay is enabled.")
            replay.begin_batch(sample_ids)
        self._gradient_replay = replay
        images = images.to(self.device)
        labels = labels.to(self.device)
        clean = self._denormalize(images).detach()
        adversarial = clean.clone()
        momentum = torch.zeros_like(adversarial)
        try:
            for step_index in range(self.steps):
                if replay is not None:
                    replay.set_context(step=step_index, group=-1, view=-1)
                self._actual_forward_view_count = 0
                gradient_pixels = adversarial.detach()
                if self.nesterov and step_index > 0:
                    gradient_pixels = gradient_pixels + self.decay * self.step_size * momentum.sign()
                    delta = torch.clamp(gradient_pixels - clean, -self.epsilon, self.epsilon)
                    gradient_pixels = torch.clamp(clean + delta, 0.0, 1.0)
                gradient_pixels = gradient_pixels.detach().requires_grad_(True)
                _, _, gradient = self._processed_gradient(gradient_pixels, labels)
                self._check_view_count()
                if self.use_momentum:
                    momentum = self.decay * momentum + gradient
                    update = momentum
                else:
                    update = gradient
                with torch.no_grad():
                    self._gradient_diagnostics["mi_cumulative_cosine"].append(
                        float(F.cosine_similarity(momentum, gradient, dim=1).mean().cpu())
                        if self.use_momentum
                        else 1.0
                    )
                adversarial = adversarial + self.step_size * update.sign()
                delta = torch.clamp(adversarial - clean, -self.epsilon, self.epsilon)
                adversarial = torch.clamp(clean + delta, 0.0, 1.0).detach()
            return self._normalize_output(adversarial)
        finally:
            self._gradient_replay = None

    def mainline_metadata(self) -> dict[str, object]:
        score_based = self.progressive_patch_selector != "random"
        score_noise_active = score_based and self.score_global_noise_strength > 0
        if score_noise_active:
            score_reference = "current_global_plus_checkpoint_gaussian_noise"
        elif score_based:
            score_reference = "current_global_without_noise"
        else:
            score_reference = "none_uniform_all_local_tokens"
        return {
            "attack_method": "progressive_patch_score",
            "whitebox_model": getattr(self.model, "model_name", "unknown"),
            "patch_selector": self.progressive_patch_selector,
            "progressive_checkpoints": list(self.progressive_checkpoints),
            "progressive_drop_ratios": list(self.progressive_drop_ratios),
            "progressive_drop_counts": list(self._progressive_mask_counts),
            "progressive_grids": [list(grid) for grid in self._progressive_mask_grids],
            "progressive_global_modes": list(self._progressive_global_modes),
            "progressive_repeated_positions": True,
            "score_reference": score_reference,
            "score_global_noise_active": score_noise_active,
            "score_global_noise_strength": self.score_global_noise_strength,
            "score_cls_noise_active": score_noise_active,
            "score_cls_noise_strength": self.score_global_noise_strength,
            "token_intervention": "local_patch_tokens_hard_zero_after_checkpoint",
            "mask_schedule_policy": "current_attack_iterate_per_step_group",
            "mask_schedule_count_per_image": self.steps * self.input_diversity_groups,
            "checkpoint_mask_selection_count_per_image": (
                self.steps * self.input_diversity_groups * len(self.progressive_checkpoints)
            ),
            "mask_pair_sharing": "same_schedule_with_phase_transformed_masks",
            "phase_mask_transform": "image_reflect_shift_then_native_grid_occupancy_topk",
            "opponent_noise": (
                "initial_rgb_projection_kept_image_union_only"
                if self.feature_noise_type == "opponent_projected"
                and self.opponent_noise_strength > 0
                else "disabled"
            ),
            "opponent_noise_strength": self.opponent_noise_strength,
            "feature_noise_type": self.feature_noise_type,
            "feature_noise_cls": False,
            "asr_definition": "1 - adversarial accuracy over all evaluated samples",
            "model_mean": self.model_mean.flatten().tolist(),
            "model_std": self.model_std.flatten().tolist(),
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_alpha": self.gaussian_alpha,
        }
