from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

from utils import DEVICE, SEED, IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class GDIMaskSelection:
    checkpoint: str
    mask: torch.Tensor
    count: int
    grid_size: tuple[int, int]
    active_samples: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.mask.ndim != 2 or self.mask.size(1) != math.prod(self.grid_size)
                or self.mask.dtype != torch.bool):
            raise ValueError("mask must be boolean [B,N] on its native token grid.")
        if not 0 < self.count <= self.mask.size(1):
            raise ValueError("drop count must fit the local grid.")
        if self.active_samples is not None and (
            self.active_samples.shape != (self.mask.size(0),)
            or self.active_samples.dtype != torch.bool
            or self.active_samples.device != self.mask.device
        ):
            raise ValueError("active samples must be a boolean vector on the mask device.")


MaskSchedule = tuple[GDIMaskSelection, ...]


class GDIAttacker:
    """Use one input boundary per image and resample token masks, with fixed-block ablations."""

    def __init__(
        self,
        model,
        *,
        gdi_drop: bool = True,
        fixed_block: int | None = None,
        drop_ratio: float | None = None,
        noise_strength: float | None = None,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        input_diversity_groups: int = 10,
        input_diversity_views_per_group: int = 2,
        input_diversity_phase_shift_set: tuple[tuple[int, int], ...] = (
            (4, 4),
            (8, 8),
            (12, 12),
        ),
        device: torch.device | None = None,
    ) -> None:
        fixed_checkpoint = None
        if fixed_block is not None:
            candidates = model.block_input_checkpoints
            if type(fixed_block) is not int or not 1 <= fixed_block <= len(candidates):
                raise ValueError(f"fixed block must be an integer from 1 to {len(candidates)}.")
            if not gdi_drop:
                raise ValueError("fixed block cannot be combined with disabled GDI drops.")
            fixed_checkpoint = candidates[fixed_block - 1]
        if epsilon < 0 or steps <= 0:
            raise ValueError("epsilon must be non-negative and steps must be positive.")
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive.")
        if input_diversity_groups <= 0 or input_diversity_views_per_group not in (1, 2):
            raise ValueError("gdi requires positive groups and one or two views per group.")
        if input_diversity_groups * input_diversity_views_per_group > 20:
            raise ValueError("actual gdi views must be <= 20.")
        if input_diversity_views_per_group == 2 and not input_diversity_phase_shift_set:
            raise ValueError("phase shift set cannot be empty.")
        if noise_strength is None:
            noise_strength = model.noise_strength
        if noise_strength < 0:
            raise ValueError("noise strength must be non-negative.")
        ratio = float(model.drop_ratio if drop_ratio is None else drop_ratio)
        if not math.isfinite(ratio) or not 0 < ratio <= 0.5:
            raise ValueError("drop ratio must satisfy 0 < ratio <= 0.5.")
        ratio = round(ratio, 12)
        if ratio <= 0:
            raise ValueError("drop ratio must remain positive after rounding to 12 decimals.")

        self.model = model
        self.model.eval()
        self.gdi_drop = bool(gdi_drop)
        self.fixed_block = fixed_block
        self.fixed_checkpoint = fixed_checkpoint
        self.device = device if device is not None else DEVICE
        self.drop_ratio = ratio
        self.noise_strength = float(noise_strength)
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size) if step_size is not None else self.epsilon / self.steps
        self.input_diversity_groups = int(input_diversity_groups)
        self.input_diversity_views_per_group = int(input_diversity_views_per_group)
        self.input_diversity_phase_shift_set = tuple(
            tuple(int(value) for value in shift) for shift in input_diversity_phase_shift_set
        )
        self.pixel_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        model_mean = getattr(model, "model_mean", IMAGENET_MEAN)
        model_std = getattr(model, "model_std", IMAGENET_STD)
        self.model_mean = torch.tensor(model_mean, device=self.device).view(1, 3, 1, 1)
        self.model_std = torch.tensor(model_std, device=self.device).view(1, 3, 1, 1)
        self._dynamic_mask_layout = None
        self.checkpoint_selections = []

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.model_mean) / self.model_std

    def _normalize_output(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _sample_random_mask(
        self,
        *,
        batch_size: int,
        token_count: int,
        ratio: float,
        device: torch.device,
    ) -> torch.Tensor:
        drop_count = max(1, int(round(token_count * ratio)))
        mask = torch.zeros(batch_size, token_count, dtype=torch.bool, device=device)
        for batch_index in range(batch_size):
            order = torch.randperm(token_count, device=device)
            mask[batch_index, order[:drop_count]] = True
        return mask.detach()

    def _build_mask_schedule(self, pixels: torch.Tensor) -> MaskSchedule:
        layout = self._dynamic_mask_layout
        if not layout:
            raise ValueError("select clean-image checkpoints before building attack schedules")
        selections = []
        for checkpoint, grid, sample_indices, active in layout:
            token_count = math.prod(grid)
            sampled = self._sample_random_mask(
                batch_size=sample_indices.numel(), token_count=token_count,
                ratio=self.drop_ratio, device=pixels.device,
            )
            mask = torch.zeros(pixels.size(0), token_count, dtype=torch.bool, device=pixels.device)
            mask.index_copy_(0, sample_indices, sampled)
            selections.append(GDIMaskSelection(
                checkpoint, mask, max(1, round(token_count * self.drop_ratio)), grid, active,
            ))
        return tuple(selections)

    def _prepare_dynamic_mask_layout(self, choices, batch_size: int) -> None:
        if len(choices) != batch_size:
            raise ValueError("one selected checkpoint is required for each clean sample")
        groups = {}
        grids = {}
        for index, choice in enumerate(choices):
            grid = self.model.checkpoint_grid_size(choice.checkpoint)
            if choice.grid_size != grid:
                raise ValueError("selected checkpoint grid differs from native model geometry")
            groups.setdefault(choice.checkpoint, []).append(index)
            grids[choice.checkpoint] = grid
        layout = []
        for checkpoint in sorted(groups, key=self.model.checkpoint_order):
            indices = torch.tensor(groups[checkpoint], device=self.device, dtype=torch.long)
            active = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
            active.index_fill_(0, indices, True)
            layout.append((checkpoint, grids[checkpoint], indices, active))
        self._dynamic_mask_layout = tuple(layout)

    def _prepare_fixed_mask_layout(self, batch_size: int) -> None:
        checkpoint = self.fixed_checkpoint
        grid = self.model.checkpoint_grid_size(checkpoint)
        indices = torch.arange(batch_size, device=self.device)
        active = torch.ones(batch_size, device=self.device, dtype=torch.bool)
        self._dynamic_mask_layout = ((checkpoint, grid, indices, active),)
        count = max(1, round(math.prod(grid) * self.drop_ratio))
        offset = len(self.checkpoint_selections)
        self.checkpoint_selections.extend({
            "sample_index": offset + index, "checkpoint": checkpoint,
            "grid_size": list(grid), "drop_count": count,
            "actual_drop_ratio": count / math.prod(grid),
        } for index in range(batch_size))

    def _apply_samplewise_phase_shifts(
        self, tensor: torch.Tensor, phases: list[tuple[int, int]]
    ) -> torch.Tensor:
        if len(phases) != tensor.size(0):
            raise ValueError("one phase is required per sample")
        height, width = tensor.shape[-2:]
        if any(abs(dx) >= width or abs(dy) >= height for dx, dy in phases):
            raise ValueError("reflection phase must be smaller than the image dimensions")
        shifts = torch.tensor(phases, device=tensor.device, dtype=torch.long)
        rows = torch.arange(height, device=tensor.device)[None, :] - shifts[:, 1:2]
        cols = torch.arange(width, device=tensor.device)[None, :] - shifts[:, 0:1]
        if height > 1:
            rows = rows.remainder(2 * (height - 1))
            rows = torch.minimum(rows, 2 * (height - 1) - rows)
        if width > 1:
            cols = cols.remainder(2 * (width - 1))
            cols = torch.minimum(cols, 2 * (width - 1) - cols)
        indices = (rows[:, :, None] * width + cols[:, None, :]).flatten(1)
        flat = tensor.flatten(2)
        return flat.gather(2, indices[:, None].expand(-1, flat.size(1), -1)).view_as(tensor)

    def _pick_input_diversity_phases(
        self, batch_size: int, device: torch.device
    ) -> list[tuple[int, int]]:
        indices = torch.randint(
            len(self.input_diversity_phase_shift_set), (batch_size,), device="cpu"
        ).tolist()
        return [self.input_diversity_phase_shift_set[index] for index in indices]

    def _phase_mask_schedule(
        self,
        schedule: MaskSchedule,
        phases: list[tuple[int, int]],
        *,
        height: int,
        width: int,
    ) -> MaskSchedule:
        selections = []
        for item in schedule:
            image_mask = F.interpolate(
                item.mask[:, None].to(torch.float32).view(item.mask.size(0), 1, *item.grid_size),
                size=(height, width), mode="nearest",
            )
            shifted_image = self._apply_samplewise_phase_shifts(image_mask, phases)
            occupancy = F.adaptive_avg_pool2d(shifted_image, item.grid_size).flatten(1)
            indices = torch.argsort(occupancy, dim=1, descending=True, stable=True)[:, : item.count]
            shifted = torch.zeros_like(item.mask)
            shifted.scatter_(1, indices, True)
            if item.active_samples is not None:
                shifted &= item.active_samples[:, None]
            selections.append(
                GDIMaskSelection(item.checkpoint, shifted.detach(), item.count, item.grid_size,
                                         item.active_samples)
            )
        return tuple(selections)

    def _projected_gaussian_noise(self, state) -> torch.Tensor:
        state.validate()
        local = state.local_tokens
        weight = state.rgb_projection_weight
        kernel_h, kernel_w = weight.shape[-2:]
        batch, count, dimension = local.shape
        rgb_noise = torch.randn(
            batch, count, 3, kernel_h, kernel_w, device=local.device, dtype=local.dtype
        )
        pixel_noise = rgb_noise.flatten(2)
        projection = weight.detach().to(local).reshape(dimension, -1)
        if projection.size(1) != pixel_noise.size(2):
            raise ValueError("Gaussian noise and RGB projection dimensions do not match.")
        feature_noise = pixel_noise.matmul(projection.t())
        token_rms = local.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        noise_rms = feature_noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        return self.noise_strength * feature_noise * (token_rms / noise_rms)

    def _forward_with_schedule(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        schedule: MaskSchedule | None,
    ) -> torch.Tensor:
        normalized = self._normalize(pixels)
        state = self.model.begin_gdi_forward(normalized)
        state.validate()
        local = state.local_tokens
        if self.noise_strength > 0:
            # Noise reaches every initial local position before checkpoint drops.
            local = local + self._projected_gaussian_noise(state)
        state = replace(state, local_tokens=local)
        if schedule is not None:
            for item in schedule:
                state = self.model.advance_gdi_state(state, item.checkpoint)
                if state.grid_size != item.grid_size:
                    raise ValueError(
                        f"replay grid mismatch at {item.checkpoint}: {state.grid_size} != {item.grid_size}."
                    )
                state = self.model.apply_gdi_mask(state, item.mask)
        logits = self.model.finish_gdi_forward(state)
        return F.cross_entropy(logits, labels)

    def _iter_attack_losses(
        self, pixels: torch.Tensor, labels: torch.Tensor
    ) -> Iterator[torch.Tensor]:
        for _ in range(self.input_diversity_groups):
            schedule = (
                self._build_mask_schedule(pixels.detach())
                if self.gdi_drop else None
            )
            phases = (
                self._pick_input_diversity_phases(pixels.size(0), pixels.device)
                if self.input_diversity_views_per_group == 2
                else None
            )
            yield self._forward_with_schedule(pixels, labels, schedule)
            if self.input_diversity_views_per_group == 1:
                continue
            shifted_schedule = (
                self._phase_mask_schedule(
                    schedule, phases, height=pixels.size(-2), width=pixels.size(-1)
                )
                if schedule is not None else None
            )
            shifted_pixels = self._apply_samplewise_phase_shifts(pixels, phases)
            yield self._forward_with_schedule(shifted_pixels, labels, shifted_schedule)

    def _attack_gradients(
        self, pixels: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        gradients = [
            torch.autograd.grad(loss, pixels, retain_graph=False)[0]
            for loss in self._iter_attack_losses(pixels, labels)
        ]
        expected = self.input_diversity_groups * self.input_diversity_views_per_group
        if len(gradients) != expected:
            raise RuntimeError(f"view count mismatch: {len(gradients)} != {expected}.")
        return torch.stack(gradients, dim=0).mean(dim=0)

    def attack_batch(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        clean = self._denormalize(images).detach()
        adversarial = clean.clone()
        momentum = torch.zeros_like(adversarial)
        try:
            if self.fixed_block is not None:
                self._prepare_fixed_mask_layout(clean.size(0))
            elif self.gdi_drop:
                with torch.no_grad():
                    choices = self.model.select_contraction_checkpoints(self._normalize(clean))
                self._prepare_dynamic_mask_layout(choices, clean.size(0))
                offset = len(self.checkpoint_selections)
                for index, choice in enumerate(choices):
                    count = max(1, round(math.prod(choice.grid_size) * self.drop_ratio))
                    self.checkpoint_selections.append({
                        "sample_index": offset + index, "checkpoint": choice.checkpoint,
                        "peak_layer": choice.peak_layer, "peak_variance": choice.variance,
                        "peak_direction_variance_ratio": choice.direction_variance_ratio,
                        "peak_contraction_score": choice.contraction_score,
                        "contraction_fraction": choice.contraction_fraction,
                        "input_total_variance": choice.input_total_variance,
                        "output_total_variance": choice.output_total_variance,
                        "grid_size": list(choice.grid_size), "drop_count": count,
                        "actual_drop_ratio": count / math.prod(choice.grid_size),
                    })
            for _ in range(self.steps):
                pixels = adversarial.detach().requires_grad_(True)
                gradient = self._attack_gradients(pixels, labels)
                momentum = momentum + gradient
                adversarial = adversarial + self.step_size * momentum.sign()
                delta = torch.clamp(adversarial - clean, -self.epsilon, self.epsilon)
                adversarial = torch.clamp(clean + delta, 0, 1).detach()
        finally:
            self._dynamic_mask_layout = None
        return self._normalize_output(adversarial)

    def parameters(self):
        schedules = self.steps * self.input_diversity_groups if self.gdi_drop else 0
        mode = "disabled" if not self.gdi_drop else "fixed" if self.fixed_block is not None else "gdi"
        params = {
            "attack_method": "geometry_guided_depth_intervention",
            "seed": SEED,
            "whitebox_model": self.model.model_name,
            "epsilon": self.epsilon,
            "step_size": self.step_size,
            "steps": self.steps,
            "momentum_decay": 1.0,
            "gdi_drop_enabled": self.gdi_drop,
            "checkpoint_selection_mode": mode,
            "fixed_block": self.fixed_block,
            "fixed_checkpoint": self.fixed_checkpoint,
            "noise_enabled": self.noise_strength > 0,
            "noise_strength": self.noise_strength,
            "noise_structure": "isotropic_rgb" if self.noise_strength > 0 else "none",
            "input_diversity_groups": self.input_diversity_groups,
            "input_diversity_views_per_group": self.input_diversity_views_per_group,
            "input_diversity_phase_shift_set": (
                [list(shift) for shift in self.input_diversity_phase_shift_set]
                if self.input_diversity_views_per_group == 2 else []
            ),
            "input_diversity_total_views": self.input_diversity_groups * self.input_diversity_views_per_group,
            "mask_schedule_count_per_image": schedules,
            "checkpoint_mask_selection_count_per_image": schedules,
            "model_mean": self.model_mean.flatten().tolist(),
            "model_std": self.model_std.flatten().tolist(),
        }
        params.update({
            "checkpoint_selection_enabled": self.gdi_drop,
            "checkpoint_selection": "per-image maximum output global-direction variance ratio times max(0, 1 - normalized-local output dispersion / input dispersion); native block input/output; select once; earliest native tie; drop at selected block input",
            "checkpoint_selection_metric": "global_direction_variance_ratio_times_positive_fractional_contraction",
            "checkpoint_score_normalization": {
                "local_features": "per-token L2 normalization",
                "reference": "L2 normalization",
                "denominator": "total local population variance / native feature dimension",
                "precision": "float64 normalized geometry from float32 features",
                "zero_total_variance_score": 0.0,
                "contraction": "max(0, 1 - output total normalized-local population variance / input total normalized-local population variance)",
                "zero_input_variance_score": 0.0,
            },
            "checkpoint_depth_weighting": False,
            "checkpoint_tie_break": "earliest block in native forward order",
            "total_drop_ratio": self.drop_ratio,
            "drop_count_rounding": "total ratio rounded to 12 decimals; max(1, round(native token count * ratio))",
            "checkpoint_histogram": dict(Counter(item["checkpoint"] for item in self.checkpoint_selections)),
            "sample_checkpoints": list(self.checkpoint_selections),
        })
        if mode != "gdi":
            params.update({
                "checkpoint_selection_enabled": False,
                "checkpoint_selection": (
                    "fixed block input; shared by all images; resample token positions per step/group"
                    if mode == "fixed" else "disabled; no token drops or clean-image layer selection"
                ),
                "checkpoint_selection_metric": None,
                "checkpoint_score_normalization": None,
                "checkpoint_tie_break": None,
            })
        reference = getattr(self.model, "checkpoint_score_reference", None)
        if reference is not None:
            params["patch_score_reference"] = reference
        return params
