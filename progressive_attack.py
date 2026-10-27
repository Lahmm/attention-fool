from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from utils import DEVICE, SEED, IMAGENET_MEAN, IMAGENET_STD


NOISE_DIRECTION_VARIANCES = {
    "opponent": (0.5, 1.25, 1.25),
    "isotropic_rgb": (1.0, 1.0, 1.0),
}


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

    def validate(self, *, batch_size: int) -> None:
        if not self.selections:
            raise ValueError("a progressive schedule must contain selections.")
        for selection in self.selections:
            selection.validate(batch_size=batch_size)

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


class ProgressiveRouteDisruptionAttacker:
    """Randomly disrupt evolving token routes at sequential checkpoints."""

    def __init__(
        self,
        model,
        *,
        progressive_drop: bool = True,
        drop_ratios: tuple[float, ...] | None = None,
        opponent_noise_strength: float | None = None,
        noise_structure: str = "opponent",
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
        if epsilon < 0 or steps <= 0:
            raise ValueError("epsilon must be non-negative and steps must be positive.")
        if step_size is not None and step_size <= 0:
            raise ValueError("step_size must be positive.")
        if input_diversity_groups <= 0 or input_diversity_views_per_group not in (1, 2):
            raise ValueError("progressive requires positive groups and one or two views per group.")
        if input_diversity_groups * input_diversity_views_per_group > 20:
            raise ValueError("actual progressive views must be <= 20.")
        if input_diversity_views_per_group == 2 and not input_diversity_phase_shift_set:
            raise ValueError("phase shift set cannot be empty.")
        if noise_structure not in NOISE_DIRECTION_VARIANCES:
            raise ValueError(f"unsupported noise structure: {noise_structure!r}.")
        if opponent_noise_strength is None:
            opponent_noise_strength = model.opponent_noise_strength
        if opponent_noise_strength < 0:
            raise ValueError("opponent noise strength must be non-negative.")
        ratios = tuple(model.drop_ratios if drop_ratios is None else drop_ratios)
        if len(ratios) != len(model.checkpoints):
            raise ValueError("checkpoint/drop-ratio count mismatch.")
        if progressive_drop and any(ratio <= 0 or ratio > 0.5 for ratio in ratios):
            raise ValueError("each drop ratio must satisfy 0 < ratio <= 0.5.")

        self.model = model
        self.model.eval()
        self.progressive_drop = bool(progressive_drop)
        self.device = device if device is not None else DEVICE
        self.progressive_checkpoints = model.checkpoints
        self.progressive_drop_ratios = tuple(float(value) for value in ratios)
        self.opponent_noise_strength = float(opponent_noise_strength)
        self.noise_structure = noise_structure
        self.noise_direction_variances = NOISE_DIRECTION_VARIANCES[noise_structure]
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
        self._actual_forward_view_count = 0
        self._progressive_mask_counts: tuple[int, ...] = ()
        self._progressive_mask_grids: tuple[tuple[int, int], ...] = ()

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

    def _build_mask_schedule(self, pixels: torch.Tensor) -> ProgressiveMaskSchedule:
        with torch.no_grad():
            state = self.model.begin_progressive_forward(self._normalize(pixels.detach()))
            selections: list[ProgressiveMaskSelection] = []
            for checkpoint, ratio in zip(
                self.progressive_checkpoints, self.progressive_drop_ratios
            ):
                state = self.model.advance_progressive_state(state, checkpoint)
                state.validate()
                mask = self._sample_random_mask(
                    batch_size=state.local_tokens.size(0),
                    token_count=state.local_tokens.size(1),
                    ratio=ratio,
                    device=state.local_tokens.device,
                )
                count = int(mask.sum(dim=1)[0].item())
                selections.append(
                    ProgressiveMaskSelection(checkpoint, mask, count, state.grid_size)
                )
                state = self.model.apply_progressive_mask(state, mask)
        schedule = ProgressiveMaskSchedule(tuple(selections))
        schedule.validate(batch_size=pixels.size(0))
        self._progressive_mask_counts = schedule.counts
        self._progressive_mask_grids = schedule.grid_sizes
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
        indices = torch.randint(
            len(self.input_diversity_phase_shift_set), (batch_size,), device=device
        ).tolist()
        return [self.input_diversity_phase_shift_set[index] for index in indices]

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

    def _strict_opponent_feature_noise(self, state) -> torch.Tensor:
        state.validate()
        local = state.local_tokens
        weight = state.rgb_projection_weight
        kernel_h, kernel_w = weight.shape[-2:]
        batch, count, dimension = local.shape
        coefficients = torch.randn(
            batch, count, 3, kernel_h, kernel_w, device=local.device, dtype=local.dtype
        )
        luma_variance, red_green_variance, yellow_blue_variance = self.noise_direction_variances
        luma = luma_variance**0.5 * coefficients[:, :, 0:1]
        red_green = red_green_variance**0.5 * coefficients[:, :, 1:2]
        yellow_blue = yellow_blue_variance**0.5 * coefficients[:, :, 2:3]
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
        return self.opponent_noise_strength * feature_noise * (token_rms / noise_rms)

    def _forward_with_schedule(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        schedule: ProgressiveMaskSchedule | None,
    ) -> torch.Tensor:
        normalized = self._normalize(pixels)
        state = self.model.begin_progressive_forward(normalized)
        state.validate()
        local = state.local_tokens
        if self.opponent_noise_strength > 0:
            # Noise reaches every initial local position before checkpoint drops.
            local = local + self._strict_opponent_feature_noise(state)
        state = replace(state, local_tokens=local)
        if schedule is not None:
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
        for _ in range(self.input_diversity_groups):
            schedule = (
                self._build_mask_schedule(pixels.detach())
                if self.progressive_drop else None
            )
            phases = (
                self._pick_input_diversity_phases(pixels.size(0), pixels.device)
                if self.input_diversity_views_per_group == 2
                else None
            )
            self._actual_forward_view_count += 1
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
            self._actual_forward_view_count += 1
            yield self._forward_with_schedule(shifted_pixels, labels, shifted_schedule)

    @staticmethod
    def _aggregate_gradients(view_gradients: torch.Tensor) -> torch.Tensor:
        if view_gradients.ndim != 5 or view_gradients.size(0) == 0:
            raise ValueError("view gradients must have shape [V,B,C,H,W] with V > 0.")
        return view_gradients.mean(dim=0)

    def _attack_gradients(
        self, pixels: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        gradients = [
            torch.autograd.grad(loss, pixels, retain_graph=False)[0]
            for loss in self._iter_attack_losses(pixels, labels)
        ]
        if not gradients:
            raise RuntimeError("no progressive losses were generated.")
        views = torch.stack(gradients, dim=0)
        return self._aggregate_gradients(views)

    def _check_view_count(self) -> None:
        expected = self.input_diversity_groups * self.input_diversity_views_per_group
        if self._actual_forward_view_count != expected:
            raise RuntimeError(
                f"view count mismatch: {self._actual_forward_view_count} != {expected}."
            )

    def attack_batch(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        clean = self._denormalize(images).detach()
        adversarial = clean.clone()
        momentum = torch.zeros_like(adversarial)
        for _ in range(self.steps):
            self._actual_forward_view_count = 0
            pixels = adversarial.detach().requires_grad_(True)
            gradient = self._attack_gradients(pixels, labels)
            self._check_view_count()
            momentum = momentum + gradient
            adversarial = adversarial + self.step_size * momentum.sign()
            delta = torch.clamp(adversarial - clean, -self.epsilon, self.epsilon)
            adversarial = torch.clamp(clean + delta, 0, 1).detach()
        return self._normalize_output(adversarial)

    def parameters(self):
        return {
            "attack_method": "progressive_route_disruption",
            "seed": SEED,
            "whitebox_model": self.model.model_name,
            "epsilon": self.epsilon,
            "step_size": self.step_size,
            "steps": self.steps,
            "momentum_decay": 1.0,
            "progressive_drop_enabled": self.progressive_drop,
            "opponent_noise_enabled": self.opponent_noise_strength > 0,
            "checkpoints": list(self.progressive_checkpoints),
            "drop_ratios": list(self.progressive_drop_ratios),
            "drop_counts": list(self._progressive_mask_counts),
            "grids": [list(grid) for grid in self._progressive_mask_grids],
            "opponent_noise_strength": self.opponent_noise_strength,
            "noise_structure": self.noise_structure if self.opponent_noise_strength > 0 else "none",
            "noise_direction_variances": (
                list(self.noise_direction_variances) if self.opponent_noise_strength > 0 else []
            ),
            "input_diversity_groups": self.input_diversity_groups,
            "input_diversity_views_per_group": self.input_diversity_views_per_group,
            "input_diversity_phase_shift_set": (
                [list(shift) for shift in self.input_diversity_phase_shift_set]
                if self.input_diversity_views_per_group == 2 else []
            ),
            "input_diversity_total_views": self.input_diversity_groups * self.input_diversity_views_per_group,
            "mask_schedule_count_per_image": self.steps * self.input_diversity_groups,
            "checkpoint_mask_selection_count_per_image": self.steps * self.input_diversity_groups * len(self.progressive_checkpoints),
            "model_mean": self.model_mean.flatten().tolist(),
            "model_std": self.model_std.flatten().tolist(),
        }
