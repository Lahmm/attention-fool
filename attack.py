from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD

if TYPE_CHECKING:
    from gradient_replay import GradientReplay


ATTACK_METHODS = (
    "none",
    "patch_dropout",
    "token_patch_dropout",
    "original_score_postdrop_phase_pair",
)
PATCH_SCORE_LAYER_MODES = ("final",)
PATCH_SELECTORS = (
    "patch_score",
    "gradcam_relu",
    "random",
    "deviation",
    "no_drop",
)
POST_DROPOUT_NOISE_TYPES = (
    "gaussian",
    "opponent_projected",
)


class PatchScoreAttacker:
    """MI/NI attack with the retained patch-score and input-diversity paths."""

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        *,
        attack_method: str = "original_score_postdrop_phase_pair",
        use_momentum: bool = True,
        momentum_decay: float = 1.0,
        nesterov: bool = False,
        ti_sigma: float = 0.0,
        input_diversity: bool = False,
        dim_resize_range: tuple[float, float] = (0.85, 1.0),
        guide_aug_copies: int = 20,
        input_diversity_groups: int = 10,
        input_diversity_views_per_group: int = 2,
        input_diversity_phase_shift: tuple[int, int] = (0, 0),
        input_diversity_phase_shift_set: tuple[tuple[int, int], ...] | None = (
            (4, 4),
            (8, 8),
            (12, 12),
        ),
        guide_aug_strength: float = 0.2,
        patch_dropout_ratio: float = 0.3,
        patch_dropout_score_mode: str = "high",
        patch_dropout_sampling_mode: str = "random",
        patch_dropout_score_quantile_jitter: float = 0.0,
        patch_dropout_score_noise: float = 0.0,
        patch_dropout_noise_mode: str = "opponent_channel_gaussian",
        token_cls_noise: bool = False,
        token_score_cls_noise: bool = True,
        token_score_cls_mode: str = "learned",
        token_score_patch_noise: bool = False,
        token_cls_noise_mode: str = "gaussian",
        token_cls_noise_strength: float | None = None,
        post_dropout_phase_token_noise: bool = True,
        post_dropout_feature_noise_strength: float | None = None,
        post_dropout_feature_noise_type: str = "opponent_projected",
        feature_layer: int = 12,
        patch_score_layer: str = "final",
        patch_selector: str = "patch_score",
        gradcam_target_mode: str = "true",
        gradcam_zero_policy: str = "error",
        gaussian_sigma: float = 4.0,
        gaussian_alpha: float = 0.75,
        device: torch.device | None = None,
    ) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        if step_size is not None and step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        if attack_method not in ATTACK_METHODS:
            raise ValueError(f"attack_method must be one of {ATTACK_METHODS}, got {attack_method!r}.")
        if nesterov and not use_momentum:
            raise ValueError("Nesterov input diversity requires momentum.")
        if ti_sigma < 0:
            raise ValueError(f"ti_sigma must be non-negative, got {ti_sigma}.")
        lo, hi = dim_resize_range
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError("dim_resize_range must satisfy 0 < low <= high <= 1.")
        if guide_aug_copies <= 0:
            raise ValueError("guide_aug_copies must be positive.")
        if input_diversity_groups <= 0 or input_diversity_views_per_group <= 0:
            raise ValueError("input-diversity groups and views must be positive.")
        if input_diversity_views_per_group not in (1, 2):
            raise ValueError("input_diversity_views_per_group must be 1 or 2.")
        total_views = input_diversity_groups * input_diversity_views_per_group
        if total_views > 20:
            raise ValueError(f"actual input-diversity views must be <= 20, got {total_views}.")
        if len(input_diversity_phase_shift) != 2:
            raise ValueError("input_diversity_phase_shift must be (dx, dy).")
        if input_diversity_phase_shift_set is not None:
            if not input_diversity_phase_shift_set:
                raise ValueError("input_diversity_phase_shift_set cannot be empty.")
            if any(len(shift) != 2 for shift in input_diversity_phase_shift_set):
                raise ValueError("every phase shift must be (dx, dy).")
        if guide_aug_strength < 0:
            raise ValueError("guide_aug_strength must be non-negative.")
        if post_dropout_feature_noise_strength is not None and post_dropout_feature_noise_strength < 0:
            raise ValueError("post_dropout_feature_noise_strength must be non-negative.")
        if post_dropout_feature_noise_type not in POST_DROPOUT_NOISE_TYPES:
            raise ValueError(
                "post_dropout_feature_noise_type must be one of "
                f"{POST_DROPOUT_NOISE_TYPES}, got {post_dropout_feature_noise_type!r}."
            )
        if not 0.0 <= patch_dropout_ratio <= 1.0:
            raise ValueError("patch_dropout_ratio must be in [0, 1].")
        if patch_dropout_score_mode not in ("high", "low", "all"):
            raise ValueError("patch_dropout_score_mode must be high, low, or all.")
        if patch_dropout_sampling_mode not in ("random", "bernoulli", "extreme", "score_weighted"):
            raise ValueError("unsupported patch_dropout_sampling_mode.")
        if not 0.0 <= patch_dropout_score_quantile_jitter < 0.5:
            raise ValueError("patch_dropout_score_quantile_jitter must be in [0, 0.5).")
        if patch_dropout_score_noise < 0:
            raise ValueError("patch_dropout_score_noise must be non-negative.")
        if patch_dropout_noise_mode not in ("gaussian", "opponent_channel_gaussian"):
            raise ValueError("patch_dropout_noise_mode must be gaussian or opponent_channel_gaussian.")
        if token_score_cls_mode not in ("learned", "gaussian"):
            raise ValueError("token_score_cls_mode must be learned or gaussian.")
        if token_cls_noise_mode not in ("gaussian", "mahalanobis"):
            raise ValueError("token_cls_noise_mode must be gaussian or mahalanobis.")
        available_layers = tuple(
            getattr(model, "patch_score_layer_candidates", lambda: ())()
        )
        if patch_score_layer not in ("final", *available_layers):
            raise ValueError(
                "patch_score_layer must be 'final' or one of the model's registered "
                f"checkpoints {available_layers}, got {patch_score_layer!r}."
            )
        if patch_selector not in PATCH_SELECTORS:
            raise ValueError(
                f"patch_selector must be one of {PATCH_SELECTORS}, got {patch_selector!r}."
            )
        if gradcam_target_mode not in ("true", "predicted"):
            raise ValueError("gradcam_target_mode must be true or predicted.")
        if gradcam_zero_policy not in ("error", "random"):
            raise ValueError("gradcam_zero_policy must be error or random.")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be non-negative.")
        if gaussian_alpha < 0:
            raise ValueError("gaussian_alpha must be non-negative.")
        if gaussian_alpha > 0 and gaussian_sigma == 0:
            raise ValueError("gaussian_sigma must be positive when gaussian_alpha is enabled.")

        self.model = model
        self.model.eval()
        self.device = device if device is not None else DEVICE
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size) if step_size is not None else self.epsilon / self.steps
        self.attack_method = attack_method
        self.use_momentum = bool(use_momentum)
        self.decay = float(momentum_decay)
        self.nesterov = bool(nesterov)
        self.ti_sigma = float(ti_sigma)
        self.input_diversity = bool(input_diversity)
        self.dim_resize_range = (float(lo), float(hi))
        self.guide_aug_copies = int(guide_aug_copies)
        self.input_diversity_groups = int(input_diversity_groups)
        self.input_diversity_views_per_group = int(input_diversity_views_per_group)
        self.input_diversity_phase_shift = tuple(int(value) for value in input_diversity_phase_shift)
        self.input_diversity_phase_shift_set = (
            tuple(tuple(int(value) for value in shift) for shift in input_diversity_phase_shift_set)
            if input_diversity_phase_shift_set is not None
            else None
        )
        self.guide_aug_strength = float(guide_aug_strength)
        self.patch_dropout_ratio = float(patch_dropout_ratio)
        self.patch_dropout_score_mode = patch_dropout_score_mode
        self.patch_dropout_sampling_mode = patch_dropout_sampling_mode
        self.patch_dropout_score_quantile_jitter = float(patch_dropout_score_quantile_jitter)
        self.patch_dropout_score_noise = float(patch_dropout_score_noise)
        self.patch_dropout_noise_mode = patch_dropout_noise_mode
        self.token_cls_noise = bool(token_cls_noise)
        self.token_score_cls_noise = bool(token_score_cls_noise)
        self.token_score_cls_mode = token_score_cls_mode
        self.token_score_patch_noise = bool(token_score_patch_noise)
        self.token_cls_noise_mode = token_cls_noise_mode
        self.token_cls_noise_strength = (
            float(token_cls_noise_strength)
            if token_cls_noise_strength is not None
            else self.guide_aug_strength
        )
        self.post_dropout_phase_token_noise = bool(post_dropout_phase_token_noise)
        self.post_dropout_feature_noise_strength = (
            float(post_dropout_feature_noise_strength)
            if post_dropout_feature_noise_strength is not None
            else self.guide_aug_strength
        )
        self.post_dropout_feature_noise_type = post_dropout_feature_noise_type
        self.feature_layer = int(feature_layer)
        self.patch_score_layer = patch_score_layer
        self.patch_selector = patch_selector
        self.gradcam_target_mode = gradcam_target_mode
        self.gradcam_zero_policy = gradcam_zero_policy
        self.gaussian_sigma = float(gaussian_sigma)
        self.gaussian_alpha = float(gaussian_alpha)
        self.pixel_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        model_mean = getattr(model, "model_mean", IMAGENET_MEAN)
        model_std = getattr(model, "model_std", IMAGENET_STD)
        self.model_mean = torch.tensor(model_mean, device=self.device).view(1, 3, 1, 1)
        self.model_std = torch.tensor(model_std, device=self.device).view(1, 3, 1, 1)
        self._patch_scores: torch.Tensor | None = None
        self._score_grid_size: tuple[int, int] | None = None
        self._last_drop_count = 0
        self._last_drop_ratio = 0.0
        self._score_source = ""
        self._resolved_score_layer = ""
        self._score_global_mode = ""
        self._gradcam_activation_source = ""
        self._gradcam_zero_fraction: list[float] = []
        self._feature_noise_type = ""
        self._fixed_mainline_drop_mask: torch.Tensor | None = None
        self._fixed_mainline_grid_size: tuple[int, int] | None = None
        self._gradient_replay: GradientReplay | None = None
        self._actual_forward_view_count = 0
        self._gradient_diagnostics = {
            "view_cosine_to_final": [],
            "sign_agreement": [],
            "effective_rank": [],
            "mi_cumulative_cosine": [],
        }
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma) if self.ti_sigma > 0 else None

        if self.attack_method == "original_score_postdrop_phase_pair":
            if self.input_diversity:
                raise ValueError("the post-dropout phase-pair mainline does not combine with DIM.")
            if self.input_diversity_views_per_group != 2:
                raise ValueError("the post-dropout phase-pair mainline requires two views per group.")
            if self.token_score_patch_noise:
                raise ValueError(
                    "score-layer patch noise is not supported by the strict cross-architecture mainline."
                )

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.model_mean) / self.model_std

    def _normalize_output(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def mainline_metadata(self) -> dict[str, object]:
        return {
            "score_source": self._score_source,
            "patch_score_layer": self.patch_score_layer,
            "resolved_patch_score_layer": self._resolved_score_layer,
            "score_global_mode": self._score_global_mode,
            "patch_selector": self.patch_selector,
            "gradcam_target_mode": self.gradcam_target_mode,
            "gradcam_zero_policy": self.gradcam_zero_policy,
            "gradcam_activation_source": self._gradcam_activation_source or None,
            "gradcam_zero_fraction": (
                sum(self._gradcam_zero_fraction) / len(self._gradcam_zero_fraction)
                if self._gradcam_zero_fraction
                else None
            ),
            "score_grid": list(self._score_grid_size) if self._score_grid_size else None,
            "target_patch_drop_ratio": (
                0.0
                if self.patch_selector == "no_drop"
                else self.patch_dropout_ratio
                if self.patch_dropout_score_mode == "all"
                else 0.5 * self.patch_dropout_ratio
            ),
            "actual_patch_drop_count": self._last_drop_count,
            "actual_patch_drop_ratio": self._last_drop_ratio,
            "feature_noise_type": self._feature_noise_type,
            "post_dropout_feature_noise_strength": self.post_dropout_feature_noise_strength,
            "post_dropout_feature_noise_type": self.post_dropout_feature_noise_type,
            "post_dropout_feature_noise_position": "initial",
            "patch_mask_policy": "clean_fixed_per_attack",
            "patch_mask_reference": "clean_pixels",
            "patch_mask_selections_per_attack": 1,
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_alpha": self.gaussian_alpha,
            "model_mean": self.model_mean.flatten().tolist(),
            "model_std": self.model_std.flatten().tolist(),
        }

    @staticmethod
    def _aggregate_gradients(view_gradients: torch.Tensor) -> torch.Tensor:
        """Return the raw mean over all actual model views."""
        if view_gradients.ndim != 5:
            raise ValueError(
                "view_gradients must have shape [num_views, batch, channels, height, width]."
            )
        if view_gradients.size(0) == 0:
            raise ValueError("view_gradients must contain at least one view.")
        return view_gradients.mean(dim=0)

    def _record_gradient_diagnostics(
        self,
        view_gradients: torch.Tensor,
        final_gradient: torch.Tensor,
    ) -> None:
        """Record compact, batch-averaged diagnostics without changing gradients."""
        with torch.no_grad():
            views = view_gradients.detach().flatten(2).transpose(0, 1)
            final = final_gradient.detach().flatten(1)
            view_cosines = F.cosine_similarity(views, final.unsqueeze(1), dim=-1)
            final_sign = final.sign()
            view_sign = views.sign()
            valid = final_sign.ne(0).unsqueeze(1)
            sign_agreement = (view_sign.eq(final_sign.unsqueeze(1)) & valid).float()
            sign_denominator = valid.expand_as(sign_agreement).sum().clamp_min(1.0)

            normalized = views / views.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            gram = torch.bmm(normalized, normalized.transpose(1, 2))
            eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            probabilities = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            effective_rank = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
            )

            self._gradient_diagnostics["view_cosine_to_final"].append(
                float(view_cosines.mean().cpu())
            )
            self._gradient_diagnostics["sign_agreement"].append(
                float(sign_agreement.sum().cpu() / sign_denominator.cpu())
            )
            self._gradient_diagnostics["effective_rank"].append(
                float(effective_rank.mean().cpu())
            )

    def gradient_diagnostics_summary(self) -> dict[str, float | int]:
        summary: dict[str, float | int] = {
            "num_gradient_batches": len(self._gradient_diagnostics["effective_rank"]),
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
        padded = F.pad(grad, (padding, padding, padding, padding), mode="reflect")
        return F.conv2d(padded, kernel, groups=grad.size(1))

    def _apply_gaussian_residual(self, grad: torch.Tensor) -> torch.Tensor:
        """Add a channel-wise Gaussian-smoothed residual without rescaling it."""
        if self.gaussian_alpha == 0:
            return grad
        radius = max(1, int(round(3 * self.gaussian_sigma)))
        axis = torch.arange(
            -radius,
            radius + 1,
            device=grad.device,
            dtype=grad.dtype,
        )
        kernel_1d = torch.exp(-0.5 * (axis / self.gaussian_sigma).square())
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).view(
            1,
            1,
            2 * radius + 1,
            2 * radius + 1,
        )
        kernel = kernel.repeat(grad.size(1), 1, 1, 1)
        smoothed = F.conv2d(
            F.pad(grad, (radius, radius, radius, radius), mode="reflect"),
            kernel,
            groups=grad.size(1),
        )
        return grad + self.gaussian_alpha * smoothed

    def _input_diversity(self, pixels: torch.Tensor) -> torch.Tensor:
        if not self.input_diversity:
            return pixels
        _, _, height, width = pixels.shape
        lo, hi = self.dim_resize_range
        scale = lo + (hi - lo) * torch.rand((), device=pixels.device)
        new_h = max(1, min(height, int(round(height * float(scale)))))
        new_w = max(1, min(width, int(round(width * float(scale)))))
        resized = F.interpolate(pixels, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h, pad_w = height - new_h, width - new_w
        top = int(torch.randint(pad_h + 1, (), device=pixels.device)) if pad_h else 0
        left = int(torch.randint(pad_w + 1, (), device=pixels.device)) if pad_w else 0
        return F.pad(resized, (left, pad_w - left, top, pad_h - top), value=0.0)

    def _randn_like(self, tensor: torch.Tensor, event: str) -> torch.Tensor:
        if self._gradient_replay is not None:
            return self._gradient_replay.randn_like(tensor, event)
        return torch.randn_like(tensor)

    @staticmethod
    def _apply_phase_shift(pixels: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        if dx == 0 and dy == 0:
            return pixels
        padded = F.pad(
            pixels,
            (max(0, dx), max(0, -dx), max(0, dy), max(0, -dy)),
            mode="reflect",
        )
        start_y = max(0, -dy)
        start_x = max(0, -dx)
        return padded[
            ...,
            start_y : start_y + pixels.size(-2),
            start_x : start_x + pixels.size(-1),
        ]

    def _pick_input_diversity_phase(self) -> tuple[int, int]:
        if self.input_diversity_phase_shift_set is None:
            return self.input_diversity_phase_shift
        index = int(torch.randint(len(self.input_diversity_phase_shift_set), (1,)).item())
        return self.input_diversity_phase_shift_set[index]

    def _pick_input_diversity_phases(self, batch_size: int, device: torch.device) -> list[tuple[int, int]]:
        if self._gradient_replay is None:
            phase = self._pick_input_diversity_phase()
            return [phase] * batch_size
        if self.input_diversity_phase_shift_set is None:
            phases = [self.input_diversity_phase_shift] * batch_size
        else:
            phases = [
                self.input_diversity_phase_shift_set[
                    self._gradient_replay.randint(
                        len(self.input_diversity_phase_shift_set),
                        "phase",
                        index,
                        device=device,
                    )
                ]
                for index in range(batch_size)
            ]
        for index, phase in enumerate(phases):
            self._gradient_replay.record_phase(index, phase)
        return phases

    def _apply_samplewise_phase_shifts(
        self,
        tensor: torch.Tensor,
        phases: list[tuple[int, int]],
    ) -> torch.Tensor:
        return torch.cat(
            [self._apply_phase_shift(tensor[index : index + 1], *phase) for index, phase in enumerate(phases)],
            dim=0,
        )

    @staticmethod
    def _run_vit_blocks(blocks, tokens: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for index in range(start, end):
            tokens = blocks[index](tokens)
        return tokens

    @staticmethod
    def _vit_base_model(model):
        base_model = getattr(model, "model", None)
        required = ("patch_embed", "_pos_embed", "patch_drop", "norm_pre", "blocks", "norm", "forward_head")
        if base_model is None or any(not hasattr(base_model, name) for name in required):
            raise ValueError("this attack method requires a ViT-style timm model.")
        return base_model

    def _embed_vit_tokens(self, base_model, pixels: torch.Tensor) -> torch.Tensor:
        tokens = base_model.patch_embed(self._normalize(pixels))
        tokens = base_model._pos_embed(tokens)
        tokens = base_model.patch_drop(tokens)
        tokens = base_model.norm_pre(tokens)
        if tokens.ndim != 3 or tokens.size(1) < 2:
            raise ValueError(f"expected CLS + patch tokens, got {tuple(tokens.shape)}.")
        return tokens

    def _make_cls_noise(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        token_rms = patch_tokens.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        if self.token_cls_noise_mode == "mahalanobis":
            centered = patch_tokens.detach() - patch_tokens.detach().mean(dim=1, keepdim=True)
            coefficients = self._randn_like(patch_tokens[:, :, :1], "cls_mahalanobis")
            raw = torch.bmm(centered.transpose(1, 2), coefficients).squeeze(-1)
            raw = raw / patch_tokens.size(1) ** 0.5
            return self.token_cls_noise_strength * token_rms * raw.unsqueeze(1)
        return self.token_cls_noise_strength * token_rms * self._randn_like(
            patch_tokens[:, :1], "cls_gaussian"
        )

    def _opponent_channel_noise_like(self, pixels: torch.Tensor, event: str = "opponent_pixel") -> torch.Tensor:
        if pixels.size(1) != 3:
            return self._randn_like(pixels, event)
        coefficients = self._randn_like(pixels, event)
        luma = 0.5**0.5 * coefficients[:, 0:1]
        red_green = 1.25**0.5 * coefficients[:, 1:2]
        yellow_blue = 1.25**0.5 * coefficients[:, 2:3]
        return torch.cat(
            (
                3**-0.5 * luma + 2**-0.5 * red_green + 6**-0.5 * yellow_blue,
                3**-0.5 * luma - 2**-0.5 * red_green + 6**-0.5 * yellow_blue,
                3**-0.5 * luma - 2 * 6**-0.5 * yellow_blue,
            ),
            dim=1,
        )

    def _pixel_noise_like(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.patch_dropout_noise_mode == "opponent_channel_gaussian":
            return self.guide_aug_strength * self._opponent_channel_noise_like(pixels)
        return self.guide_aug_strength * self._randn_like(pixels, "pixel_gaussian")

    def _token_patch_dropout_noise(self, patch_tokens: torch.Tensor, base_model) -> torch.Tensor:
        token_rms = patch_tokens.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        if self.patch_dropout_noise_mode != "opponent_channel_gaussian":
            return self.guide_aug_strength * token_rms * self._randn_like(
                patch_tokens, "token_gaussian"
            )

        weight = getattr(getattr(getattr(base_model, "patch_embed", None), "proj", None), "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 4:
            return self.guide_aug_strength * token_rms * self._randn_like(
                patch_tokens, "token_gaussian_no_projection"
            )
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        if in_channels != 3 or out_channels != patch_tokens.size(-1):
            return self.guide_aug_strength * token_rms * self._randn_like(
                patch_tokens, "token_gaussian_shape_mismatch"
            )

        batch, count, dimension = patch_tokens.shape
        coefficients = self._randn_like(
            torch.empty(
                batch,
                count,
                3,
                kernel_h,
                kernel_w,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            ),
            "token_opponent",
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
        ).view(
            batch * count,
            3,
            kernel_h,
            kernel_w,
        )
        pixel_noise = pixel_noise.flatten(1)
        projection = weight.detach().to(patch_tokens).reshape(dimension, -1)
        token_noise = pixel_noise.matmul(projection.t()).view(batch, count, dimension)
        noise_rms = token_noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        return self.guide_aug_strength * token_noise * (token_rms / noise_rms)

    def _strict_opponent_feature_noise(self, state) -> torch.Tensor:
        """Project the original opponent-channel construction through an RGB Conv2d."""
        state.validate()
        local_tokens = state.local_tokens
        weight = state.rgb_projection_weight
        kernel_h, kernel_w = state.projection_kernel
        if tuple(weight.shape[2:]) != (kernel_h, kernel_w):
            raise ValueError("RGB projection weight and attack-state kernel do not match.")

        batch, count, dimension = local_tokens.shape
        coefficients = self._randn_like(
            torch.empty(
                batch,
                count,
                3,
                kernel_h,
                kernel_w,
                device=local_tokens.device,
                dtype=local_tokens.dtype,
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
        projection = weight.detach().to(local_tokens).reshape(dimension, -1)
        if projection.size(1) != pixel_noise.size(2):
            raise ValueError("opponent noise and RGB projection dimensions do not match.")
        feature_noise = pixel_noise.matmul(projection.t())
        token_rms = (
            local_tokens.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        )
        noise_rms = feature_noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        self._feature_noise_type = "opponent_channel_rgb_projection"
        return self.post_dropout_feature_noise_strength * feature_noise * (token_rms / noise_rms)

    def _match_feature_noise_rms(
        self,
        feature_tokens: torch.Tensor,
        raw_noise: torch.Tensor,
        event: str,
    ) -> torch.Tensor:
        token_rms = feature_tokens.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        noise_rms = raw_noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        return self.post_dropout_feature_noise_strength * raw_noise * (token_rms / noise_rms)

    def _build_post_dropout_feature_noise(
        self,
        feature_tokens: torch.Tensor,
        state,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build one of the two retained kept-only noises at the RGB projection."""
        feature_drop_mask = self._image_mask_to_projection_drop_mask(image_mask, state)
        if self.post_dropout_feature_noise_type == "opponent_projected":
            raw_noise = self._strict_opponent_feature_noise(state)
        elif self.post_dropout_feature_noise_type == "gaussian":
            self._feature_noise_type = "feature_iid_gaussian"
            raw_noise = self._randn_like(feature_tokens, "mainline_feature_gaussian")
        else:
            raise ValueError(
                f"unsupported post-dropout noise type: {self.post_dropout_feature_noise_type!r}"
            )

        matched = self._match_feature_noise_rms(
            feature_tokens,
            raw_noise,
            self.post_dropout_feature_noise_type,
        )
        return torch.where((~feature_drop_mask).unsqueeze(-1), matched, torch.zeros_like(matched))

    @staticmethod
    def _image_mask_to_projection_drop_mask(
        image_mask: torch.Tensor,
        state,
    ) -> torch.Tensor:
        """Map a pixel mask through the first RGB convolution's true receptive fields."""
        state.validate()
        if image_mask.ndim != 4 or image_mask.size(1) != 1:
            raise ValueError("image_mask must have shape [B,1,H,W].")
        kernel = torch.ones(
            1,
            1,
            *state.projection_kernel,
            device=image_mask.device,
            dtype=image_mask.dtype,
        )
        kwargs = {
            "stride": state.projection_stride,
            "padding": state.projection_padding,
            "dilation": state.projection_dilation,
        }
        dropped_area = F.conv2d(image_mask, kernel, **kwargs)
        valid_area = F.conv2d(torch.ones_like(image_mask), kernel, **kwargs).clamp_min(1.0)
        drop_fraction = dropped_area / valid_area
        if tuple(drop_fraction.shape[-2:]) != state.grid_size:
            raise ValueError(
                "RGB receptive-field mask grid does not match the attack feature grid: "
                f"{tuple(drop_fraction.shape[-2:])} != {state.grid_size}."
            )
        return drop_fraction.flatten(1).gt(0.5)

    def _patch_score_candidate_mask(self, scores: torch.Tensor) -> torch.Tensor:
        working_scores = scores
        if self.patch_dropout_score_noise > 0:
            score_std = scores.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
            working_scores = scores + self.patch_dropout_score_noise * score_std * self._randn_like(
                scores, "score_noise"
            )
        if self.patch_dropout_score_quantile_jitter == 0:
            if self.patch_dropout_score_mode == "all":
                return torch.ones_like(scores, dtype=torch.bool)
            candidate_count = max(1, scores.size(1) // 2)
            candidate_indices = torch.topk(
                working_scores,
                candidate_count,
                dim=1,
                largest=self.patch_dropout_score_mode == "high",
            ).indices
            candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
            return candidate_mask.scatter(1, candidate_indices, True)
        else:
            jitter = self.patch_dropout_score_quantile_jitter
            if self._gradient_replay is None:
                quantile_random = torch.rand(scores.size(0), device=scores.device)
            else:
                quantile_random = torch.stack(
                    [
                        self._gradient_replay.rand_scalar(
                            "score_quantile", index, device=scores.device
                        )
                        for index in range(scores.size(0))
                    ]
                )
            quantiles = 0.5 + (quantile_random * 2 - 1) * jitter
            threshold = torch.stack(
                [torch.quantile(working_scores[index], quantiles[index]) for index in range(scores.size(0))]
            ).unsqueeze(1)
        if self.patch_dropout_score_mode == "high":
            return working_scores > threshold
        if self.patch_dropout_score_mode == "low":
            return working_scores < threshold
        return torch.ones_like(scores, dtype=torch.bool)

    @staticmethod
    def _deviation_candidate_mask(scores: torch.Tensor) -> torch.Tensor:
        deviation = (scores - scores.median(dim=1, keepdim=True).values).abs()
        candidate_count = max(1, scores.size(1) // 2)
        indices = deviation.topk(candidate_count, dim=1).indices
        return torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)

    @staticmethod
    def _top_candidate_mask(scores: torch.Tensor) -> torch.Tensor:
        candidate_count = max(1, scores.size(1) // 2)
        indices = scores.topk(candidate_count, dim=1).indices
        return torch.zeros_like(scores, dtype=torch.bool).scatter(1, indices, True)

    @staticmethod
    def _local_activation_for_grid(
        activation: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> torch.Tensor:
        patch_count = grid_size[0] * grid_size[1]
        if activation.ndim == 4:
            local = activation.flatten(2).transpose(1, 2)
        elif activation.ndim == 3:
            if activation.size(1) < patch_count:
                raise ValueError(
                    "Grad-CAM activation has fewer tokens than the routing grid."
                )
            local = activation[:, -patch_count:]
        else:
            raise ValueError(
                f"unsupported Grad-CAM activation shape: {tuple(activation.shape)}."
            )
        if local.size(1) != patch_count:
            raise ValueError("Grad-CAM activation does not match the routing grid.")
        return local

    def _gradcam_scores(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        features,
    ) -> torch.Tensor:
        """Return a true/predicted-class Grad-CAM map at the routing checkpoint."""
        capture_getter = getattr(self.model, "patch_score_activation_capture", None)
        if capture_getter is None:
            raise ValueError("Grad-CAM routing requires a checkpoint activation module adapter.")
        capture = capture_getter(self.patch_score_layer)
        capture.validate()
        self._gradcam_activation_source = capture.source_name
        captured: dict[str, torch.Tensor] = {}

        def output_hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor):
                raise ValueError("Grad-CAM checkpoint did not produce a tensor activation.")
            captured["activation"] = value

        def input_hook(_module, inputs):
            value = inputs[0]
            if not isinstance(value, torch.Tensor):
                raise ValueError("Grad-CAM checkpoint did not receive a tensor activation.")
            captured["activation"] = value

        handle = (
            capture.module.register_forward_pre_hook(input_hook)
            if capture.hook_type == "input"
            else capture.module.register_forward_hook(output_hook)
        )
        try:
            logits = self.model(self._normalize(pixels))
        finally:
            handle.remove()
        if "activation" not in captured:
            raise RuntimeError("failed to capture the Grad-CAM routing activation.")
        target = labels if self.gradcam_target_mode == "true" else logits.argmax(dim=1)
        activation = captured["activation"]
        target_logits = logits.gather(1, target[:, None]).sum()
        gradient = torch.autograd.grad(
            target_logits,
            activation,
            retain_graph=False,
            create_graph=False,
        )[0]
        local = self._local_activation_for_grid(activation, features.grid_size)
        gradient_local = self._local_activation_for_grid(gradient, features.grid_size)
        alpha = gradient_local.mean(dim=1)
        scores = F.relu((local * alpha[:, None]).sum(dim=2)).detach()
        zero_fraction = scores.abs().sum(dim=1).eq(0).float()
        self._gradcam_zero_fraction.append(float(zero_fraction.mean().cpu()))
        if bool(zero_fraction.any()):
            if self.gradcam_zero_policy == "error":
                raise RuntimeError(
                    "Grad-CAM produced an all-zero map for at least one sample at "
                    f"{capture.source_name}; use the explicit random zero-map policy "
                    "to preserve the matched drop budget."
                )
            # ReLU Grad-CAM contains no ranking information for these samples.
            # A sample-keyed random ranking preserves the exact candidate/drop
            # budget without silently changing the saliency definition.
            fallback = self._randn_like(scores, "gradcam_zero_random_ranking")
            scores = torch.where(zero_fraction[:, None].bool(), fallback, scores)
        return scores

    def _sample_patch_dropout_mask(
        self,
        scores: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        drop_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        if self.patch_dropout_ratio <= 0:
            return drop_mask
        for batch_index in range(scores.size(0)):
            candidates = candidate_mask[batch_index].nonzero(as_tuple=True)[0]
            if candidates.numel() == 0:
                continue
            target_ratio = (
                self.patch_dropout_ratio
                if self.patch_dropout_score_mode == "all"
                else 0.5 * self.patch_dropout_ratio
            )
            count = max(1, int(round(scores.size(1) * target_ratio)))
            if candidates.numel() < count:
                raise ValueError(
                    "the score candidate set is smaller than the model-independent "
                    "native-patch dropout budget."
                )
            if self.patch_dropout_sampling_mode == "bernoulli":
                if self._gradient_replay is None:
                    random_values = torch.rand(candidates.numel(), device=scores.device)
                else:
                    generator = self._gradient_replay._generator(
                        scores.device,
                        self._gradient_replay._seed("dropout_bernoulli", self._gradient_replay.sample_ids[batch_index]),
                    )
                    random_values = torch.rand(candidates.numel(), device=scores.device, generator=generator)
                selected = candidates[torch.topk(random_values, count, largest=False).indices]
            elif self.patch_dropout_sampling_mode == "extreme":
                order = torch.argsort(
                    scores[batch_index, candidates],
                    descending=self.patch_dropout_score_mode == "high",
                )
                selected = candidates[order[:count]]
            elif self.patch_dropout_sampling_mode == "score_weighted":
                median = scores[batch_index].median()
                weights = (scores[batch_index, candidates] - median).abs().clamp_min(1e-6)
                if self._gradient_replay is None:
                    sampled = torch.multinomial(weights, count, replacement=False)
                else:
                    sampled = self._gradient_replay.multinomial(
                        weights, count, "dropout_weighted", batch_index
                    )
                selected = candidates[sampled]
            else:
                if self._gradient_replay is None:
                    order = torch.randperm(candidates.numel(), device=scores.device)
                else:
                    order = self._gradient_replay.randperm(
                        candidates.numel(), "dropout_random", batch_index, device=scores.device
                    )
                selected = candidates[order[:count]]
            drop_mask[batch_index, selected] = True
        if drop_mask.numel():
            per_sample_counts = drop_mask.sum(dim=1)
            if not torch.equal(per_sample_counts, per_sample_counts[:1].expand_as(per_sample_counts)):
                raise RuntimeError("mainline patch dropout counts differ within a batch.")
            self._last_drop_count = int(per_sample_counts[0].item())
            self._last_drop_ratio = self._last_drop_count / drop_mask.size(1)
        return drop_mask

    def _score_cls_and_patches(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        base_model,
    ) -> torch.Tensor:
        score_cls = cls_token
        score_patches = patch_tokens
        if self.token_score_cls_mode == "gaussian":
            score_cls = self._randn_like(cls_token, "score_cls_gaussian")
            score_cls = score_cls / score_cls.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            score_cls = score_cls * cls_token.norm(dim=-1, keepdim=True)
        if self.token_score_cls_noise and self.token_cls_noise_strength > 0:
            score_cls = score_cls + self._make_cls_noise(score_patches)
        if self.token_score_patch_noise and self.guide_aug_strength > 0:
            score_patches = score_patches + self._token_patch_dropout_noise(score_patches, base_model)
        return F.cosine_similarity(score_patches, score_cls.expand_as(score_patches), dim=-1)

    def _compute_mainline_drop_mask(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        extract = getattr(self.model, "extract_patch_score_features", None)
        if extract is None:
            raise ValueError("the mainline requires a white-box patch-score adapter.")
        with torch.no_grad():
            features = extract(
                self._normalize(pixels.detach()),
                score_layer=self.patch_score_layer,
            )
            features.validate()
        self._resolved_score_layer = features.layer_id
        self._score_global_mode = features.global_mode
        if self.patch_selector == "no_drop":
            drop_mask = torch.zeros(
                features.local_tokens.size(0),
                features.local_tokens.size(1),
                device=pixels.device,
                dtype=torch.bool,
            )
            self._last_drop_count = 0
            self._last_drop_ratio = 0.0
        else:
            with torch.no_grad():
                patch_scores = self._score_cls_and_patches(
                    features.global_token,
                    features.local_tokens,
                    getattr(self.model, "model", None),
                )
            if self.patch_selector == "gradcam_relu":
                if labels is None:
                    raise ValueError("Grad-CAM routing requires labels.")
                scores = self._gradcam_scores(pixels, labels, features)
                # Grad-CAM always drops its high-saliency half.  The globally
                # frozen patch-score polarity does not redefine Grad-CAM.
                candidates = self._top_candidate_mask(scores)
            elif self.patch_selector == "random":
                scores = patch_scores
                candidates = torch.ones_like(scores, dtype=torch.bool)
            elif self.patch_selector == "deviation":
                scores = patch_scores
                candidates = self._deviation_candidate_mask(scores)
            else:
                scores = patch_scores
                candidates = self._patch_score_candidate_mask(scores)
            drop_mask = self._sample_patch_dropout_mask(scores, candidates).detach()
        self._score_grid_size = features.grid_size
        self._score_source = features.source_name
        return drop_mask, features.grid_size

    def _clear_fixed_mainline_drop_mask(self) -> None:
        self._fixed_mainline_drop_mask = None
        self._fixed_mainline_grid_size = None

    def _initialize_fixed_mainline_drop_mask(
        self,
        clean_pixels: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> None:
        """Select the exact native-patch mask once from the clean input."""
        if self.attack_method != "original_score_postdrop_phase_pair":
            return
        if self._fixed_mainline_drop_mask is not None:
            raise RuntimeError("the fixed mainline mask was initialized more than once.")
        drop_mask, grid_size = self._compute_mainline_drop_mask(clean_pixels, labels)
        self._fixed_mainline_drop_mask = drop_mask.detach()
        self._fixed_mainline_grid_size = grid_size

    def _fixed_mainline_mask(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        # Direct view/probe callers may not enter through attack_batch.  Their
        # first input is therefore treated as the clean reference and cached
        # before any group is emitted.
        if self._fixed_mainline_drop_mask is None:
            self._initialize_fixed_mainline_drop_mask(pixels.detach(), labels)
        drop_mask = self._fixed_mainline_drop_mask
        grid_size = self._fixed_mainline_grid_size
        if drop_mask is None or grid_size is None:
            raise RuntimeError("the fixed mainline mask was not initialized.")
        if drop_mask.size(0) != pixels.size(0):
            raise ValueError("the fixed mainline mask batch does not match the attack batch.")
        return drop_mask, grid_size

    @staticmethod
    def _patch_drop_mask_to_image(
        drop_mask: torch.Tensor,
        grid_size: tuple[int, int],
        height: int,
        width: int,
    ) -> torch.Tensor:
        if drop_mask.size(1) != grid_size[0] * grid_size[1]:
            raise ValueError("drop mask token count does not match its explicit grid.")
        return F.interpolate(
            drop_mask.view(drop_mask.size(0), 1, *grid_size).to(torch.float32),
            size=(height, width),
            mode="nearest",
        )

    def _forward_vit_tokens(self, base_model, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tokens = self._run_vit_blocks(base_model.blocks, tokens, 0, len(base_model.blocks))
        logits = base_model.forward_head(base_model.norm(tokens))
        return F.cross_entropy(logits, labels)

    def _attack_loss_for_original_score_postdrop_phase_view(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        prepare = getattr(self.model, "prepare_attack_feature_state", None)
        resume = getattr(self.model, "forward_from_attack_feature_state", None)
        if prepare is None or resume is None:
            raise ValueError("the mainline requires a resumable white-box attack adapter.")
        state = prepare(self._normalize(pixels))
        state.validate()
        local_tokens = state.local_tokens
        if not self.post_dropout_phase_token_noise:
            logits = resume(state, local_tokens)
        else:
            noise = self._build_post_dropout_feature_noise(
                local_tokens,
                state,
                image_mask,
            )
            logits = resume(state, local_tokens + noise)
        return F.cross_entropy(logits, labels)

    def _iter_original_score_postdrop_phase_pair(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # One exact mask is selected from the clean image before the attack
        # loop and reused by every group at every iterative attack step.
        drop_mask, grid_size = self._fixed_mainline_mask(pixels, labels)
        for group_index in range(self.input_diversity_groups):
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(group=group_index, view=-1)
            image_mask = self._patch_drop_mask_to_image(
                drop_mask,
                grid_size,
                pixels.size(-2),
                pixels.size(-1),
            )
            image_mask = image_mask.to(device=pixels.device, dtype=pixels.dtype)
            dropped_pixels = pixels * (1.0 - image_mask)
            phases = self._pick_input_diversity_phases(pixels.size(0), pixels.device)
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=0)
            self._actual_forward_view_count += 1
            yield dropped_pixels, image_mask
            # Rebuild the shared-mask pixel branch after view A autograd frees its graph.
            dropped_pixels = pixels * (1.0 - image_mask)
            shifted_pixels = self._apply_samplewise_phase_shifts(dropped_pixels, phases)
            shifted_mask = self._apply_samplewise_phase_shifts(image_mask, phases)
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=1)
            self._actual_forward_view_count += 1
            yield shifted_pixels, shifted_mask

    def _attack_loss_for_token_patch_dropout(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        base_model = self._vit_base_model(self.model)
        model_pixels = self._input_diversity(pixels)
        tokens = self._embed_vit_tokens(base_model, model_pixels)
        num_blocks = len(base_model.blocks)
        score_layer = self.feature_layer if self.feature_layer >= 0 else num_blocks + self.feature_layer
        score_layer = max(0, min(score_layer, num_blocks))
        with torch.no_grad():
            score_tokens = self._run_vit_blocks(base_model.blocks, tokens, 0, score_layer)
            scores = self._score_cls_and_patches(
                score_tokens[:, :1],
                score_tokens[:, 1:],
                base_model,
            )
            drop_mask = self._sample_patch_dropout_mask(
                scores,
                self._patch_score_candidate_mask(scores),
            )

        cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
        if self.token_cls_noise and self.token_cls_noise_strength > 0:
            cls_token = cls_token + self._make_cls_noise(patch_tokens)
        noisy_patches = patch_tokens + self._token_patch_dropout_noise(patch_tokens, base_model)
        patch_tokens = torch.where(
            drop_mask.unsqueeze(-1),
            torch.zeros_like(patch_tokens),
            noisy_patches,
        )
        return self._forward_vit_tokens(
            base_model,
            torch.cat((cls_token, patch_tokens), dim=1),
            labels,
        )

    def _compute_generic_patch_scores(self, pixels: torch.Tensor) -> None:
        if self.attack_method != "patch_dropout":
            self._patch_scores = None
            return
        with torch.no_grad():
            outputs = self.model(self._normalize(pixels.detach()), return_tokens=True)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise ValueError("patch_dropout requires a white-box model with token hooks.")
            _, layers = outputs
            layer_index = self.feature_layer if self.feature_layer >= 0 else len(layers) + self.feature_layer
            if layer_index == len(layers):
                layer_index -= 1
            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(f"feature_layer {self.feature_layer} is invalid for {len(layers)} layers.")
            tokens = layers[layer_index]
            if tokens.ndim != 3 or tokens.size(1) < 2:
                raise ValueError("patch_dropout requires CLS + patch token features.")
            self._patch_scores = F.cosine_similarity(tokens[:, 1:], tokens[:, :1].expand_as(tokens[:, 1:]), dim=-1)

    def _patch_dropout_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if self._patch_scores is None:
            raise RuntimeError("patch scores were not initialized.")
        count = self._patch_scores.size(1)
        grid = int(round(count**0.5))
        if grid * grid != count:
            raise ValueError("patch_dropout requires a square patch grid.")
        drop_mask = self._sample_patch_dropout_mask(
            self._patch_scores,
            self._patch_score_candidate_mask(self._patch_scores),
        )
        image_mask = self._patch_drop_mask_to_image(
            drop_mask,
            (grid, grid),
            pixels.size(-2),
            pixels.size(-1),
        )
        image_mask = image_mask.to(device=pixels.device, dtype=pixels.dtype)
        noised = torch.clamp(pixels + self._pixel_noise_like(pixels), 0.0, 1.0)
        return torch.where(image_mask > 0.5, torch.zeros_like(pixels), noised)

    def _attack_loss_for_pixels(self, pixels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        model_pixels = self._input_diversity(pixels)
        logits = self.model(self._normalize(model_pixels))
        return F.cross_entropy(logits, labels)

    def _iter_attack_losses(self, pixels: torch.Tensor, labels: torch.Tensor) -> Iterator[torch.Tensor]:
        if self.attack_method == "original_score_postdrop_phase_pair":
            for view_pixels, image_mask in self._iter_original_score_postdrop_phase_pair(
                pixels, labels
            ):
                yield self._attack_loss_for_original_score_postdrop_phase_view(
                    view_pixels,
                    labels,
                    image_mask,
                )
            return
        if self.attack_method == "token_patch_dropout":
            for _group_index in range(self.input_diversity_groups):
                phase = self._pick_input_diversity_phase()
                for view_index in range(self.input_diversity_views_per_group):
                    view_pixels = self._apply_phase_shift(pixels, *phase) if view_index == 1 else pixels
                    self._actual_forward_view_count += 1
                    yield self._attack_loss_for_token_patch_dropout(view_pixels, labels)
            return
        if self.attack_method == "patch_dropout":
            for _copy_index in range(self.guide_aug_copies):
                self._actual_forward_view_count += 1
                yield self._attack_loss_for_pixels(self._patch_dropout_pixels(pixels), labels)
            return
        self._actual_forward_view_count += 1
        yield self._attack_loss_for_pixels(pixels, labels)

    def _attack_grad(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        gradients = []
        for loss in self._iter_attack_losses(pixels, labels):
            gradients.append(torch.autograd.grad(loss, pixels, retain_graph=False)[0])
        if not gradients:
            raise RuntimeError("no attack losses were generated.")
        view_gradients = torch.stack(gradients, dim=0)
        aggregated = self._aggregate_gradients(view_gradients)
        self._record_gradient_diagnostics(view_gradients, aggregated)
        return aggregated

    def probe_attack_gradients(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        *,
        replay: "GradientReplay | None" = None,
        sample_ids: list[str] | None = None,
        step_index: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Run one production gradient step without updating the image.

        This is the mechanism-diagnostic interface: it returns every actual
        augmentation-view gradient, their raw mean, and the fully processed
        direction immediately before MI accumulation.  Inputs are raw pixels
        in ``[0, 1]`` rather than repository-normalized image tensors.
        """
        if replay is not None:
            if sample_ids is None or len(sample_ids) != pixels.size(0):
                raise ValueError("sample_ids must match pixels when replay is enabled.")
            replay.begin_batch(sample_ids)
            replay.set_context(step=-1, group=-1, view=-1)
        self._gradient_replay = replay
        self._clear_fixed_mainline_drop_mask()
        self._actual_forward_view_count = 0
        probe_pixels = pixels.to(self.device).detach().requires_grad_(True)
        labels = labels.to(self.device)
        try:
            if self.attack_method == "original_score_postdrop_phase_pair":
                self._initialize_fixed_mainline_drop_mask(probe_pixels.detach(), labels)
            if replay is not None:
                replay.set_context(step=step_index, group=-1, view=-1)
            gradients = [
                torch.autograd.grad(loss, probe_pixels, retain_graph=False)[0]
                for loss in self._iter_attack_losses(probe_pixels, labels)
            ]
            if not gradients:
                raise RuntimeError("no attack losses were generated for the gradient probe.")
            view_gradients = torch.stack(gradients, dim=0)
            raw_mean = self._aggregate_gradients(view_gradients)
            self._record_gradient_diagnostics(view_gradients, raw_mean)
            processed = self._smooth_grad(self._apply_gaussian_residual(raw_mean))
            expected_views = (
                self.guide_aug_copies
                if self.attack_method == "patch_dropout"
                else self.input_diversity_groups * self.input_diversity_views_per_group
                if self.attack_method in ("token_patch_dropout", "original_score_postdrop_phase_pair")
                else 1
            )
            if self._actual_forward_view_count != expected_views:
                raise RuntimeError(
                    f"view count mismatch: {self._actual_forward_view_count} != {expected_views}."
                )
            return {
                "view_gradients": view_gradients.detach(),
                "raw_mean": raw_mean.detach(),
                "processed": processed.detach(),
            }
        finally:
            self._clear_fixed_mainline_drop_mask()
            self._gradient_replay = None

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        replay: "GradientReplay | None" = None,
        sample_ids: list[str] | None = None,
    ) -> torch.Tensor:
        if replay is not None:
            if sample_ids is None or len(sample_ids) != images.size(0):
                raise ValueError("sample_ids must match the batch when replay is enabled.")
            replay.begin_batch(sample_ids)
        self._gradient_replay = replay
        images = images.to(self.device)
        labels = labels.to(self.device)
        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone()
        momentum = torch.zeros_like(adv_pixels)
        self._clear_fixed_mainline_drop_mask()
        try:
            if self.attack_method == "original_score_postdrop_phase_pair":
                # The exact score/random/Grad-CAM mask is selected once from
                # the unperturbed input.  No iterative adversarial state and
                # no augmentation view can change it afterwards.
                if replay is not None:
                    replay.set_context(step=-1, group=-1, view=-1)
                self._initialize_fixed_mainline_drop_mask(clean_pixels, labels)

            for step_index in range(self.steps):
                if replay is not None:
                    replay.set_context(step=step_index, group=-1, view=-1)
                self._actual_forward_view_count = 0
                grad_pixels = adv_pixels.detach()
                if self.nesterov and step_index > 0:
                    grad_pixels = grad_pixels + self.decay * self.step_size * momentum.sign()
                    delta = torch.clamp(grad_pixels - clean_pixels, -self.epsilon, self.epsilon)
                    grad_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

                self._compute_generic_patch_scores(grad_pixels)
                grad_pixels = grad_pixels.detach().requires_grad_(True)
                gradient = self._attack_grad(grad_pixels, labels)
                gradient = self._apply_gaussian_residual(gradient)
                gradient = self._smooth_grad(gradient)

                expected_views = (
                    self.guide_aug_copies
                    if self.attack_method == "patch_dropout"
                    else self.input_diversity_groups * self.input_diversity_views_per_group
                    if self.attack_method in ("token_patch_dropout", "original_score_postdrop_phase_pair")
                    else 1
                )
                if self._actual_forward_view_count != expected_views:
                    raise RuntimeError(
                        f"view count mismatch: {self._actual_forward_view_count} != {expected_views}."
                    )

                # Keep the aggregated gradient's absolute scale. The update is
                # still sign-based and is projected back into the epsilon ball
                # below, so removing per-step normalization only changes the
                # relative weighting of gradients in the MI accumulator.
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

                adv_pixels = adv_pixels + self.step_size * update.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

            return self._normalize_output(adv_pixels)
        finally:
            self._clear_fixed_mainline_drop_mask()
            self._gradient_replay = None
