from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD


ATTACK_METHODS = (
    "none",
    "patch_dropout",
    "token_patch_dropout",
    "original_score_postdrop_phase_pair",
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
        feature_layer: int = 12,
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
        self.feature_layer = int(feature_layer)
        self.pixel_mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._patch_scores: torch.Tensor | None = None
        self._actual_forward_view_count = 0
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma) if self.ti_sigma > 0 else None

        if self.attack_method == "original_score_postdrop_phase_pair":
            if self.input_diversity:
                raise ValueError("the post-dropout phase-pair mainline does not combine with DIM.")
            if self.input_diversity_views_per_group != 2:
                raise ValueError("the post-dropout phase-pair mainline requires two views per group.")

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    @staticmethod
    def _normalize_grad(grad: torch.Tensor) -> torch.Tensor:
        return grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)

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
            coefficients = torch.randn(
                patch_tokens.size(0),
                patch_tokens.size(1),
                1,
                device=patch_tokens.device,
                dtype=patch_tokens.dtype,
            )
            raw = torch.bmm(centered.transpose(1, 2), coefficients).squeeze(-1)
            raw = raw / patch_tokens.size(1) ** 0.5
            return self.token_cls_noise_strength * token_rms * raw.unsqueeze(1)
        return self.token_cls_noise_strength * token_rms * torch.randn_like(patch_tokens[:, :1])

    @staticmethod
    def _opponent_channel_noise_like(pixels: torch.Tensor) -> torch.Tensor:
        if pixels.size(1) != 3:
            return torch.randn_like(pixels)
        coefficients = torch.randn_like(pixels)
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
        return self.guide_aug_strength * torch.randn_like(pixels)

    def _token_patch_dropout_noise(self, patch_tokens: torch.Tensor, base_model) -> torch.Tensor:
        token_rms = patch_tokens.detach().square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        if self.patch_dropout_noise_mode != "opponent_channel_gaussian":
            return self.guide_aug_strength * token_rms * torch.randn_like(patch_tokens)

        weight = getattr(getattr(getattr(base_model, "patch_embed", None), "proj", None), "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 4:
            return self.guide_aug_strength * token_rms * torch.randn_like(patch_tokens)
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        if in_channels != 3 or out_channels != patch_tokens.size(-1):
            return self.guide_aug_strength * token_rms * torch.randn_like(patch_tokens)

        batch, count, dimension = patch_tokens.shape
        pixel_noise = torch.empty(
            batch * count,
            3,
            kernel_h,
            kernel_w,
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        pixel_noise = self._opponent_channel_noise_like(pixel_noise).flatten(1)
        projection = weight.detach().to(patch_tokens).reshape(dimension, -1)
        token_noise = pixel_noise.matmul(projection.t()).view(batch, count, dimension)
        noise_rms = token_noise.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        return self.guide_aug_strength * token_noise * (token_rms / noise_rms)

    def _patch_score_candidate_mask(self, scores: torch.Tensor) -> torch.Tensor:
        working_scores = scores
        if self.patch_dropout_score_noise > 0:
            score_std = scores.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
            working_scores = scores + self.patch_dropout_score_noise * score_std * torch.randn_like(scores)
        if self.patch_dropout_score_quantile_jitter == 0:
            threshold = working_scores.median(dim=1, keepdim=True).values
        else:
            jitter = self.patch_dropout_score_quantile_jitter
            quantiles = 0.5 + (torch.rand(scores.size(0), device=scores.device) * 2 - 1) * jitter
            threshold = torch.stack(
                [torch.quantile(working_scores[index], quantiles[index]) for index in range(scores.size(0))]
            ).unsqueeze(1)
        if self.patch_dropout_score_mode == "high":
            return working_scores > threshold
        if self.patch_dropout_score_mode == "low":
            return working_scores < threshold
        return torch.ones_like(scores, dtype=torch.bool)

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
            count = max(1, int(round(candidates.numel() * self.patch_dropout_ratio)))
            if self.patch_dropout_sampling_mode == "bernoulli":
                selected_mask = torch.rand(candidates.numel(), device=scores.device) < self.patch_dropout_ratio
                if not selected_mask.any():
                    selected_mask[torch.randint(candidates.numel(), (1,), device=scores.device)] = True
                selected = candidates[selected_mask]
            elif self.patch_dropout_sampling_mode == "extreme":
                order = torch.argsort(
                    scores[batch_index, candidates],
                    descending=self.patch_dropout_score_mode == "high",
                )
                selected = candidates[order[:count]]
            elif self.patch_dropout_sampling_mode == "score_weighted":
                median = scores[batch_index].median()
                weights = (scores[batch_index, candidates] - median).abs().clamp_min(1e-6)
                selected = candidates[torch.multinomial(weights, count, replacement=False)]
            else:
                selected = candidates[torch.randperm(candidates.numel(), device=scores.device)[:count]]
            drop_mask[batch_index, selected] = True
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
            score_cls = torch.randn_like(cls_token)
            score_cls = score_cls / score_cls.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            score_cls = score_cls * cls_token.norm(dim=-1, keepdim=True)
        if self.token_score_cls_noise and self.token_cls_noise_strength > 0:
            score_cls = score_cls + self._make_cls_noise(score_patches)
        if self.token_score_patch_noise and self.guide_aug_strength > 0:
            score_patches = score_patches + self._token_patch_dropout_noise(score_patches, base_model)
        return F.cosine_similarity(score_patches, score_cls.expand_as(score_patches), dim=-1)

    def _compute_original_l12_drop_mask(self, pixels: torch.Tensor) -> torch.Tensor:
        base_model = self._vit_base_model(self.model)
        num_blocks = len(base_model.blocks)
        score_layer = self.feature_layer if self.feature_layer >= 0 else num_blocks + self.feature_layer
        if score_layer != num_blocks:
            raise ValueError(f"the mainline requires feature_layer={num_blocks}, got {self.feature_layer}.")
        with torch.no_grad():
            tokens = self._embed_vit_tokens(base_model, pixels)
            score_tokens = self._run_vit_blocks(base_model.blocks, tokens, 0, num_blocks)
            scores = self._score_cls_and_patches(
                score_tokens[:, :1],
                score_tokens[:, 1:],
                base_model,
            )
            candidates = self._patch_score_candidate_mask(scores)
            return self._sample_patch_dropout_mask(scores, candidates).detach()

    @staticmethod
    def _patch_drop_mask_to_image(drop_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
        count = drop_mask.size(1)
        grid = int(round(count**0.5))
        if grid * grid != count or height % grid or width % grid:
            raise ValueError("pixel patch dropout requires a square, divisible patch grid.")
        return (
            drop_mask.view(drop_mask.size(0), 1, grid, grid)
            .to(torch.float32)
            .repeat_interleave(height // grid, dim=-2)
            .repeat_interleave(width // grid, dim=-1)
        )

    @staticmethod
    def _image_mask_to_patch_drop_mask(image_mask: torch.Tensor, count: int) -> torch.Tensor:
        grid = int(round(count**0.5))
        if grid * grid != count:
            raise ValueError("expected a square patch grid.")
        height, width = image_mask.shape[-2:]
        if height % grid or width % grid:
            raise ValueError("image mask is not divisible by the patch grid.")
        pooled = F.avg_pool2d(
            image_mask,
            kernel_size=(height // grid, width // grid),
            stride=(height // grid, width // grid),
        )
        return pooled.flatten(1).gt(0.5)

    def _forward_vit_tokens(self, base_model, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tokens = self._run_vit_blocks(base_model.blocks, tokens, 0, len(base_model.blocks))
        logits = base_model.forward_head(base_model.norm(tokens))
        return F.cross_entropy(logits, labels)

    def _attack_loss_for_original_score_postdrop_phase_view(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> torch.Tensor:
        base_model = self._vit_base_model(self.model)
        tokens = self._embed_vit_tokens(base_model, pixels)
        if tokens.size(1) != drop_mask.size(1) + 1:
            raise ValueError("drop mask and token count do not match.")
        cls_token, patch_tokens = tokens[:, :1], tokens[:, 1:]
        if self.post_dropout_phase_token_noise:
            noise = self._token_patch_dropout_noise(patch_tokens, base_model)
            patch_tokens = torch.where(
                (~drop_mask).unsqueeze(-1),
                patch_tokens + noise,
                patch_tokens,
            )
        return self._forward_vit_tokens(
            base_model,
            torch.cat((cls_token, patch_tokens), dim=1),
            labels,
        )

    def _iter_original_score_postdrop_phase_pair(
        self,
        pixels: torch.Tensor,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for _group_index in range(self.input_diversity_groups):
            drop_mask = self._compute_original_l12_drop_mask(pixels)
            image_mask = self._patch_drop_mask_to_image(drop_mask, pixels.size(-2), pixels.size(-1))
            image_mask = image_mask.to(device=pixels.device, dtype=pixels.dtype)
            dropped_pixels = pixels * (1.0 - image_mask)
            phase = self._pick_input_diversity_phase()
            self._actual_forward_view_count += 1
            yield dropped_pixels, drop_mask
            # Rebuild the shared-mask pixel branch after view A autograd frees its graph.
            dropped_pixels = pixels * (1.0 - image_mask)
            shifted_pixels = self._apply_phase_shift(dropped_pixels, *phase)
            shifted_mask = self._apply_phase_shift(image_mask, *phase)
            shifted_drop_mask = self._image_mask_to_patch_drop_mask(shifted_mask, drop_mask.size(1))
            self._actual_forward_view_count += 1
            yield shifted_pixels, shifted_drop_mask

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
        image_mask = self._patch_drop_mask_to_image(drop_mask, pixels.size(-2), pixels.size(-1))
        image_mask = image_mask.to(device=pixels.device, dtype=pixels.dtype)
        noised = torch.clamp(pixels + self._pixel_noise_like(pixels), 0.0, 1.0)
        return torch.where(image_mask > 0.5, torch.zeros_like(pixels), noised)

    def _attack_loss_for_pixels(self, pixels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        model_pixels = self._input_diversity(pixels)
        logits = self.model(self._normalize(model_pixels))
        return F.cross_entropy(logits, labels)

    def _iter_attack_losses(self, pixels: torch.Tensor, labels: torch.Tensor) -> Iterator[torch.Tensor]:
        if self.attack_method == "original_score_postdrop_phase_pair":
            for view_pixels, drop_mask in self._iter_original_score_postdrop_phase_pair(pixels):
                yield self._attack_loss_for_original_score_postdrop_phase_view(view_pixels, labels, drop_mask)
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

    def _attack_grad(self, pixels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        gradients = []
        for loss in self._iter_attack_losses(pixels, labels):
            gradients.append(torch.autograd.grad(loss, pixels, retain_graph=False)[0])
        if not gradients:
            raise RuntimeError("no attack losses were generated.")
        return torch.stack(gradients).mean(dim=0)

    def attack_batch(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)
        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone()
        momentum = torch.zeros_like(adv_pixels)

        for step_index in range(self.steps):
            self._actual_forward_view_count = 0
            grad_pixels = adv_pixels.detach()
            if self.nesterov and step_index > 0:
                grad_pixels = grad_pixels + self.decay * self.step_size * momentum.sign()
                delta = torch.clamp(grad_pixels - clean_pixels, -self.epsilon, self.epsilon)
                grad_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

            self._compute_generic_patch_scores(grad_pixels)
            grad_pixels = grad_pixels.detach().requires_grad_(True)
            gradient = self._smooth_grad(self._attack_grad(grad_pixels, labels))

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

            gradient = self._normalize_grad(gradient)
            if self.use_momentum:
                momentum = self.decay * momentum + gradient
                update = momentum
            else:
                update = gradient

            adv_pixels = adv_pixels + self.step_size * update.sign()
            delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
            adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

        return self._normalize(adv_pixels)
