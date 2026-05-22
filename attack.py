import math

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD


class MIFGSMAttacker:
    """Momentum Iterative FGSM over the whole input image under an L_inf bound."""

    def __init__(
        self,
        model,
        epsilon: float = 8.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        decay: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        if step_size is not None and step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")

        self.model = model
        self.model.eval()
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size) if step_size is not None else self.epsilon / self.steps
        self.decay = float(decay)
        self.device = device if device is not None else DEVICE

        self.pixel_mean = torch.tensor(
            IMAGENET_MEAN,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(
            IMAGENET_STD,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    @staticmethod
    def _normalize_grad(grad: torch.Tensor) -> torch.Tensor:
        denom = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return grad / denom

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        for _step in range(self.steps):
            adv_pixels.requires_grad_(True)
            logits = self.model(self._normalize(adv_pixels), return_attn=False)
            loss = F.cross_entropy(logits, labels)

            grad = torch.autograd.grad(loss, adv_pixels)[0]
            grad = self._normalize_grad(grad)
            momentum = self.decay * momentum + grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

        return self._normalize(adv_pixels)


class LazyAggregationAttacker(MIFGSMAttacker):
    """
    Lazy aggregation hijack attack for ViTs.

    Uses stable attention token maps as guides for guide-based
    augmentation strategies to disrupt ViT attention.
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 20,
        decay: float = 1.0,
        layers: tuple[int, ...] = (-6, -5, -4, -3, -2, -1),
        fg_top_ratio: float = 0.25,
        grad_combine: str = "guide_aug_ce",
        spectral_transition: float = 0.04,
        ti_sigma: float = 3.0,
        input_diversity: bool = True,
        dim_resize_range: tuple[float, float] = (0.85, 1.0),
        si_scales: int = 1,
        nesterov: bool = True,
        eot_iter: int = 1,
        perturb_smooth_sigma: float = 0.0,
        lazy_spectral_delta: bool = False,
        lazy_spectral_cutoff: float = 0.25,
        attention_guide_models: tuple[torch.nn.Module, ...] | None = None,
        guide_type: str = "postsoftmax_cls",
        guide_sample_mode: str = "fixed",
        guide_entropy_temp: float = 1.0,
        attention_grad_smooth_sigma: float = 0.0,
        patch_grad_smooth_sigma: float = 0.0,
        guide_dilate_kernel: int = 1,
        guide_smooth_sigma: float = 0.0,
        guide_dynamic: bool = False,
        guide_update_interval: int = 5,
        guide_ema: float = 0.7,
        guide_aug: bool = False,
        guide_aug_copies: int = 3,
        guide_aug_mode: tuple[str, ...] = ("dropout",),
        guide_aug_strength: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            device=device,
        )
        if not layers:
            raise ValueError("layers must contain at least one layer index.")
        if not (0.0 < fg_top_ratio < 1.0):
            raise ValueError(f"fg_top_ratio must be in (0, 1), got {fg_top_ratio}.")
        valid_combine = ("guide_aug_ce",)
        if grad_combine not in valid_combine:
            raise ValueError(f"grad_combine must be one of {valid_combine}, got {grad_combine!r}.")

        self.layers = tuple(int(layer) for layer in layers)
        self.fg_top_ratio = float(fg_top_ratio)
        self.grad_combine = grad_combine
        self.spectral_transition = float(spectral_transition)
        self.ti_sigma = float(ti_sigma)
        self.input_diversity = bool(input_diversity)
        self.dim_resize_range = tuple(float(r) for r in dim_resize_range)
        self.si_scales = int(si_scales)
        if self.si_scales <= 0:
            raise ValueError(f"si_scales must be positive, got {si_scales}.")
        self.nesterov = bool(nesterov)
        self.eot_iter = int(eot_iter)
        if self.eot_iter <= 0:
            raise ValueError(f"eot_iter must be positive, got {eot_iter}.")
        self.perturb_smooth_sigma = float(perturb_smooth_sigma)
        self.lazy_spectral_delta = bool(lazy_spectral_delta)
        self.lazy_spectral_cutoff = float(lazy_spectral_cutoff)
        self.attention_guide_models = tuple(attention_guide_models or ())
        guide_types = tuple(item.strip() for item in guide_type.split(",") if item.strip())
        valid_guide_types = ("postsoftmax_cls", "qk_cls", "qk_all_queries")
        if not guide_types:
            raise ValueError("guide_type must contain at least one guide type.")
        invalid_guide_types = [item for item in guide_types if item not in valid_guide_types]
        if invalid_guide_types:
            raise ValueError(f"guide_type entries must be in {valid_guide_types}, got {invalid_guide_types}.")
        self.guide_types = guide_types
        valid_guide_sample_modes = ("fixed", "random")
        if guide_sample_mode not in valid_guide_sample_modes:
            raise ValueError(f"guide_sample_mode must be one of {valid_guide_sample_modes}, got {guide_sample_mode!r}.")
        self.guide_sample_mode = guide_sample_mode
        self.guide_entropy_temp = float(guide_entropy_temp)
        if self.guide_entropy_temp <= 0:
            raise ValueError(f"guide_entropy_temp must be positive, got {guide_entropy_temp}.")
        self.attention_grad_smooth_sigma = float(attention_grad_smooth_sigma)
        if self.attention_grad_smooth_sigma < 0:
            raise ValueError(f"attention_grad_smooth_sigma must be non-negative, got {attention_grad_smooth_sigma}.")
        self.patch_grad_smooth_sigma = float(patch_grad_smooth_sigma)
        if self.patch_grad_smooth_sigma < 0:
            raise ValueError(f"patch_grad_smooth_sigma must be non-negative, got {patch_grad_smooth_sigma}.")
        self.guide_dilate_kernel = int(guide_dilate_kernel)
        if self.guide_dilate_kernel <= 0 or self.guide_dilate_kernel % 2 == 0:
            raise ValueError(f"guide_dilate_kernel must be a positive odd integer, got {guide_dilate_kernel}.")
        self.guide_smooth_sigma = float(guide_smooth_sigma)
        if self.guide_smooth_sigma < 0:
            raise ValueError(f"guide_smooth_sigma must be non-negative, got {guide_smooth_sigma}.")
        self.guide_dynamic = bool(guide_dynamic)
        self.guide_update_interval = int(guide_update_interval)
        if self.guide_update_interval <= 0:
            raise ValueError(f"guide_update_interval must be positive, got {guide_update_interval}.")
        self.guide_ema = float(guide_ema)
        if not (0.0 <= self.guide_ema <= 1.0):
            raise ValueError(f"guide_ema must be in [0, 1], got {guide_ema}.")
        self.guide_aug = bool(guide_aug)
        self.guide_aug_copies = int(guide_aug_copies)
        if self.guide_aug_copies <= 0:
            raise ValueError(f"guide_aug_copies must be positive, got {guide_aug_copies}.")
        self.guide_aug_mode = tuple(str(mode).strip() for mode in guide_aug_mode if str(mode).strip())
        valid_guide_aug_modes = (
            "dropout",
            "mix",
            "jitter",
            "freq",
            "dropout_inner",
            "jitter_outer",
            "freq_inner",
            "dropout_all",
            "jitter_all",
            "freq_all",
        )
        if not self.guide_aug_mode:
            raise ValueError("guide_aug_mode must contain at least one mode.")
        invalid_guide_aug_modes = [mode for mode in self.guide_aug_mode if mode not in valid_guide_aug_modes]
        if invalid_guide_aug_modes:
            raise ValueError(f"guide_aug_mode entries must be in {valid_guide_aug_modes}, got {invalid_guide_aug_modes}.")
        self.guide_aug_strength = float(guide_aug_strength)
        if self.guide_aug_strength < 0:
            raise ValueError(f"guide_aug_strength must be non-negative, got {guide_aug_strength}.")
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma) if self.ti_sigma > 0 else None
        self._attention_grad_kernel = (
            self._build_ti_kernel(self.attention_grad_smooth_sigma)
            if self.attention_grad_smooth_sigma > 0
            else None
        )
        self._patch_grad_kernel = (
            self._build_ti_kernel(self.patch_grad_smooth_sigma)
            if self.patch_grad_smooth_sigma > 0
            else None
        )
        self._perturb_kernel = (
            self._build_ti_kernel(self.perturb_smooth_sigma)
            if self.perturb_smooth_sigma > 0
            else None
        )

    @staticmethod
    def _build_ti_kernel(sigma: float) -> torch.Tensor:
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g1d = torch.exp(-0.5 * (x / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] @ g1d[None, :]
        return g2d.view(1, 1, g2d.size(0), g2d.size(1))

    def _smooth_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self._ti_kernel is None or self.ti_sigma <= 0:
            return grad
        kernel = self._ti_kernel.to(grad.device, grad.dtype).repeat(grad.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(grad, (pad, pad, pad, pad), mode="reflect"), kernel, groups=grad.size(1))

    def _smooth_attention_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self._attention_grad_kernel is None or self.attention_grad_smooth_sigma <= 0:
            return grad
        kernel = self._attention_grad_kernel.to(grad.device, grad.dtype).repeat(grad.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(grad, (pad, pad, pad, pad), mode="reflect"), kernel, groups=grad.size(1))

    def _smooth_patch_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self._patch_grad_kernel is None or self.patch_grad_smooth_sigma <= 0:
            return grad
        kernel = self._patch_grad_kernel.to(grad.device, grad.dtype).repeat(grad.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(grad, (pad, pad, pad, pad), mode="reflect"), kernel, groups=grad.size(1))

    def _smooth_perturbation(self, delta: torch.Tensor) -> torch.Tensor:
        if self._perturb_kernel is None or self.perturb_smooth_sigma <= 0:
            return delta
        kernel = self._perturb_kernel.to(delta.device, delta.dtype).repeat(delta.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(delta, (pad, pad, pad, pad), mode="reflect"), kernel, groups=delta.size(1))

    def _spectral_filter_delta(self, delta: torch.Tensor) -> torch.Tensor:
        from utils import image_2d_fft_low_high_maps

        maps = image_2d_fft_low_high_maps(
            delta,
            cutoff_ratio=self.lazy_spectral_cutoff,
            transition_ratio=self.spectral_transition,
        )
        filter_weights = maps["low_ratio"].to(delta.dtype)
        if filter_weights.ndim == 2:
            filter_weights = filter_weights.unsqueeze(0)
        filter_weights = filter_weights.unsqueeze(1)
        return delta * (0.5 + 0.5 * filter_weights)

    def _input_diversity(self, images: torch.Tensor) -> torch.Tensor:
        if not self.input_diversity:
            return images
        batch_size, channels, height, width = images.shape
        lo, hi = self.dim_resize_range
        scale = lo + (hi - lo) * torch.rand(1, device=images.device)
        new_h = max(1, min(height, int(round(height * scale.item()))))
        new_w = max(1, min(width, int(round(width * scale.item()))))
        resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h = height - new_h
        pad_w = width - new_w
        top = torch.randint(0, pad_h + 1, (1,), device=images.device).item() if pad_h > 0 else 0
        left = torch.randint(0, pad_w + 1, (1,), device=images.device).item() if pad_w > 0 else 0
        bottom = pad_h - top
        right = pad_w - left
        return F.pad(resized, (left, right, top, bottom), value=0.0)

    @staticmethod
    def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
        min_vals = weights.min(dim=1, keepdim=True).values
        max_vals = weights.max(dim=1, keepdim=True).values
        return (weights - min_vals) / (max_vals - min_vals).clamp_min(1e-12)

    @staticmethod
    def _select_top_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
        num_patches = scores.size(1)
        k = max(1, min(num_patches, int(round(num_patches * ratio))))
        top_idx = torch.topk(scores, k=k, dim=1, largest=True).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        return mask.scatter(1, top_idx, True)

    @staticmethod
    def _select_bottom_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
        num_patches = scores.size(1)
        k = max(1, min(num_patches, int(round(num_patches * ratio))))
        bottom_idx = torch.topk(scores, k=k, dim=1, largest=False).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        return mask.scatter(1, bottom_idx, True)

    @staticmethod
    def _infer_num_heads_from_attn(attn_module) -> int | None:
        heads = getattr(attn_module, "num_heads", None)
        return int(heads) if heads is not None else None

    @staticmethod
    def _qkv_to_cls_attention_scores(qkv: torch.Tensor, num_heads: int | None, guide_type: str) -> torch.Tensor:
        if qkv.ndim != 3 or num_heads is None or num_heads <= 0:
            raise ValueError(f"Unsupported qkv shape/heads: {tuple(qkv.shape)}, {num_heads}.")
        bsz, num_tokens, hidden = qkv.shape
        if num_tokens < 2 or hidden % (3 * num_heads) != 0:
            raise ValueError(f"Unsupported qkv dimensions: {tuple(qkv.shape)} heads={num_heads}.")
        num_patches = num_tokens - 1
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch token count {num_patches} is not square.")
        head_dim = hidden // (3 * num_heads)
        qkv = qkv.reshape(bsz, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        qk = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        if guide_type == "postsoftmax_cls":
            return torch.softmax(qk, dim=-1)[:, :, 0, 1:].mean(dim=1)
        if guide_type == "qk_cls":
            return qk[:, :, 0, 1:].mean(dim=1)
        if guide_type == "qk_all_queries":
            return qk[:, :, :, 1:].mean(dim=(1, 2))
        raise ValueError(f"Unsupported guide_type: {guide_type}")

    @staticmethod
    def _build_qkv_from_attn_input(attn_module, attn_input: torch.Tensor) -> torch.Tensor:
        qkv_layer = getattr(attn_module, "qkv", None)
        if qkv_layer is None:
            raise ValueError("Attention module does not expose qkv.")
        q_bias = getattr(attn_module, "q_bias", None)
        if q_bias is None:
            return qkv_layer(attn_input)
        qkv_bias = torch.cat((attn_module.q_bias, attn_module.k_bias, attn_module.v_bias))
        if getattr(attn_module, "qkv_bias_separate", False):
            return qkv_layer(attn_input) + qkv_bias
        return F.linear(attn_input, weight=qkv_layer.weight, bias=qkv_bias)

    def _collect_cls_attention_scores(
        self,
        source_model,
        images: torch.Tensor,
        target_num_patches: int | None = None,
        guide_type: str | None = None,
    ) -> torch.Tensor | None:
        guide_type = self._sample_guide_type() if guide_type is None else guide_type
        module_dict = dict(source_model.model.named_modules())
        records = []
        handles = []
        try:
            for qkv_name, qkv_module in module_dict.items():
                if not qkv_name.endswith("attn.qkv"):
                    continue
                attn_name = qkv_name.rsplit(".qkv", 1)[0]
                attn_module = module_dict.get(attn_name)
                if attn_module is None:
                    continue
                record = {"attn": attn_module, "qkv": None, "input": None, "heads": self._infer_num_heads_from_attn(attn_module)}
                records.append(record)

                def qkv_hook(_module, _inputs, output, rec=record):
                    rec["qkv"] = output.detach() if isinstance(output, torch.Tensor) else None

                def pre_hook(_module, inputs, rec=record):
                    if inputs and isinstance(inputs[0], torch.Tensor):
                        rec["input"] = inputs[0].detach()

                handles.append(qkv_module.register_forward_hook(qkv_hook))
                handles.append(attn_module.register_forward_pre_hook(pre_hook))

            if not records:
                return None
            with torch.no_grad():
                _ = source_model.model(images)

            scores = []
            for record in records:
                try:
                    qkv = record["qkv"]
                    if qkv is None and record["input"] is not None:
                        qkv = self._build_qkv_from_attn_input(record["attn"], record["input"])
                    if qkv is None:
                        continue
                    score = self._qkv_to_cls_attention_scores(qkv, record["heads"], guide_type)
                    scores.append(self._normalize_weights(score.detach()))
                except (RuntimeError, ValueError):
                    continue
            if not scores:
                return None
            scores_by_size: dict[int, list[torch.Tensor]] = {}
            for score in scores:
                scores_by_size.setdefault(score.size(1), []).append(score)
            if target_num_patches is not None:
                selected = scores_by_size.get(target_num_patches)
                if not selected:
                    return None
            else:
                target_num_patches = max(
                    scores_by_size,
                    key=lambda num_patches: (len(scores_by_size[num_patches]), num_patches),
                )
                selected = scores_by_size[target_num_patches]
            max_layers = min(len(selected), len(self.layers))
            return torch.stack(selected[-max_layers:]).mean(dim=0)
        finally:
            for handle in handles:
                handle.remove()

    def _sample_guide_type(self) -> str:
        if self.guide_sample_mode == "random" and len(self.guide_types) > 1:
            idx = torch.randint(0, len(self.guide_types), (1,), device=self.pixel_mean.device).item()
            return self.guide_types[int(idx)]
        return self.guide_types[0]

    def _smooth_guide_grid(self, grid: torch.Tensor) -> torch.Tensor:
        if self.guide_smooth_sigma <= 0:
            return grid
        kernel = self._build_ti_kernel(self.guide_smooth_sigma).to(grid.device, grid.dtype)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(grid, (pad, pad, pad, pad), mode="reflect"), kernel)

    def _build_stable_attention_token_map(
        self,
        images: torch.Tensor,
        expand_shared: bool = False,
    ) -> torch.Tensor:
        guide_type = self._sample_guide_type()
        primary_score = self._collect_cls_attention_scores(self.model, images, guide_type=guide_type)
        if primary_score is None:
            raise ValueError("The white-box model did not produce compatible CLS attention scores.")
        num_patches = primary_score.size(1)
        scores = [primary_score]
        for source_model in self.attention_guide_models:
            score = self._collect_cls_attention_scores(
                source_model,
                images,
                target_num_patches=num_patches,
                guide_type=guide_type,
            )
            if score is not None:
                scores.append(score)
        stable_score = self._normalize_weights(torch.stack(scores).mean(dim=0))
        if self.guide_entropy_temp != 1.0:
            stable_score = self._normalize_weights(stable_score.clamp_min(1e-12) ** (1.0 / self.guide_entropy_temp))
        stable_mask = self._select_top_mask(stable_score, self.fg_top_ratio).to(stable_score.dtype)
        token_map = stable_score * stable_mask
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square.")
        grid = token_map.view(token_map.size(0), 1, grid_size, grid_size)
        if expand_shared:
            if self.guide_dilate_kernel > 1:
                pad = self.guide_dilate_kernel // 2
                grid = F.max_pool2d(grid, kernel_size=self.guide_dilate_kernel, stride=1, padding=pad)
            grid = self._smooth_guide_grid(grid)
            token_map = self._normalize_weights(grid.flatten(1))
        else:
            token_map = self._normalize_weights(token_map)
        return token_map.detach()

    def _token_map_to_pixel_map(self, token_map: torch.Tensor, img_size: int) -> torch.Tensor:
        num_patches = token_map.size(1)
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square.")
        grid = token_map.view(token_map.size(0), 1, grid_size, grid_size)
        pixel_map = F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False)
        flat = pixel_map.flatten(1)
        min_vals = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        return ((pixel_map - min_vals) / (max_vals - min_vals).clamp_min(1e-12)).detach()

    def _guide_augmented_pixels(self, pixels: torch.Tensor, guide_pixel_map: torch.Tensor, copy_idx: int) -> torch.Tensor:
        strength = self.guide_aug_strength
        if strength <= 0:
            return pixels
        mode = self.guide_aug_mode[copy_idx % len(self.guide_aug_mode)]
        guide = guide_pixel_map.to(pixels.device, pixels.dtype).clamp(0.0, 1.0)
        if mode == "dropout":
            noise = torch.rand_like(pixels)
            blurred = F.avg_pool2d(pixels, kernel_size=5, stride=1, padding=2)
            corrupt = 0.5 * noise + 0.5 * blurred
            background = pixels * (1.0 - strength) + corrupt * strength
            return torch.clamp(pixels * guide + background * (1.0 - guide), 0.0, 1.0)
        if mode == "dropout_all":
            noise = torch.rand_like(pixels)
            blurred = F.avg_pool2d(pixels, kernel_size=5, stride=1, padding=2)
            corrupt = 0.5 * noise + 0.5 * blurred
            return torch.clamp(pixels * (1.0 - strength) + corrupt * strength, 0.0, 1.0)
        if mode == "dropout_inner":
            noise = torch.rand_like(pixels)
            blurred = F.avg_pool2d(pixels, kernel_size=5, stride=1, padding=2)
            corrupt = 0.5 * noise + 0.5 * blurred
            foreground = pixels * (1.0 - strength) + corrupt * strength
            return torch.clamp(foreground * guide + pixels * (1.0 - guide), 0.0, 1.0)
        if mode == "mix":
            if pixels.size(0) > 1:
                mixed = pixels.roll(shifts=copy_idx + 1, dims=0)
            else:
                low_noise = torch.rand_like(pixels)
                mixed = F.avg_pool2d(low_noise, kernel_size=7, stride=1, padding=3)
            background = pixels * (1.0 - strength) + mixed * strength
            return torch.clamp(pixels * guide + background * (1.0 - guide), 0.0, 1.0)
        if mode == "jitter":
            brightness = (torch.rand(pixels.size(0), 1, 1, 1, device=pixels.device, dtype=pixels.dtype) * 2.0 - 1.0) * strength
            noise = torch.randn_like(pixels) * (strength / 2.0)
            jittered = torch.clamp(pixels * (1.0 + brightness) + noise, 0.0, 1.0)
            return torch.clamp(jittered * guide + pixels * (1.0 - guide), 0.0, 1.0)
        if mode == "jitter_all":
            brightness = (torch.rand(pixels.size(0), 1, 1, 1, device=pixels.device, dtype=pixels.dtype) * 2.0 - 1.0) * strength
            noise = torch.randn_like(pixels) * (strength / 2.0)
            return torch.clamp(pixels * (1.0 + brightness) + noise, 0.0, 1.0)
        if mode == "jitter_outer":
            brightness = (torch.rand(pixels.size(0), 1, 1, 1, device=pixels.device, dtype=pixels.dtype) * 2.0 - 1.0) * strength
            noise = torch.randn_like(pixels) * (strength / 2.0)
            jittered = torch.clamp(pixels * (1.0 + brightness) + noise, 0.0, 1.0)
            return torch.clamp(pixels * guide + jittered * (1.0 - guide), 0.0, 1.0)
        if mode == "freq":
            pooled = F.avg_pool2d(pixels, kernel_size=9, stride=1, padding=4)
            noise = F.avg_pool2d(torch.rand_like(pixels), kernel_size=9, stride=1, padding=4)
            corrupt = 0.7 * pooled + 0.3 * noise
            background = pixels * (1.0 - strength) + corrupt * strength
            return torch.clamp(pixels * guide + background * (1.0 - guide), 0.0, 1.0)
        if mode == "freq_all":
            pooled = F.avg_pool2d(pixels, kernel_size=9, stride=1, padding=4)
            noise = F.avg_pool2d(torch.rand_like(pixels), kernel_size=9, stride=1, padding=4)
            corrupt = 0.7 * pooled + 0.3 * noise
            return torch.clamp(pixels * (1.0 - strength) + corrupt * strength, 0.0, 1.0)
        if mode == "freq_inner":
            pooled = F.avg_pool2d(pixels, kernel_size=9, stride=1, padding=4)
            noise = F.avg_pool2d(torch.rand_like(pixels), kernel_size=9, stride=1, padding=4)
            corrupt = 0.7 * pooled + 0.3 * noise
            foreground = pixels * (1.0 - strength) + corrupt * strength
            return torch.clamp(foreground * guide + pixels * (1.0 - guide), 0.0, 1.0)
        raise ValueError(f"Unsupported guide augmentation mode: {mode}")

    def _guide_aug_ce_loss(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        guide_pixel_map: torch.Tensor,
    ) -> torch.Tensor:
        ce_terms = []
        for scale_idx in range(self.si_scales):
            scale = float(2 ** scale_idx)
            for _eot_idx in range(self.eot_iter):
                for copy_idx in range(self.guide_aug_copies):
                    aug_pixels = self._guide_augmented_pixels(pixels, guide_pixel_map, copy_idx)
                    logits_adv = self.model(
                        self._input_diversity(self._normalize(aug_pixels) / scale),
                        return_attn=False,
                    )
                    ce_terms.append(F.cross_entropy(logits_adv, labels))
        return torch.stack(ce_terms).mean()

    @staticmethod
    def _batch_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        a_flat = a.flatten(1)
        b_flat = b.flatten(1)
        return F.cosine_similarity(a_flat, b_flat, dim=1).mean().item()

    def _guide_entropy(self, guide: torch.Tensor) -> float:
        prob = guide / guide.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=1)
        return (entropy / math.log(prob.size(1))).mean().item()

    def _guide_topk_change_rate(self, guide_a: torch.Tensor, guide_b: torch.Tensor) -> float:
        mask_a = self._select_top_mask(guide_a, self.fg_top_ratio)
        mask_b = self._select_top_mask(guide_b, self.fg_top_ratio)
        return (mask_a != mask_b).to(torch.float32).mean(dim=1).mean().item()

    def _dynamic_guide_stats(
        self,
        step_idx: int,
        clean_guide: torch.Tensor,
        guide: torch.Tensor,
        adv_primary_score: torch.Tensor,
    ) -> dict[str, float | int]:
        adv_primary_score = self._normalize_weights(adv_primary_score.detach())
        return {
            "step": int(step_idx),
            "clean_cosine": self._batch_cosine(guide, clean_guide),
            "adv_cls_cosine": self._batch_cosine(guide, adv_primary_score),
            "entropy": self._guide_entropy(guide),
            "topk_change_rate": self._guide_topk_change_rate(clean_guide, guide),
        }

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        clean_pixels = self._denormalize(images).detach()
        guide_token_map = None
        clean_guide_token_map = None
        guide_pixel_map = None
        if self.grad_combine == "guide_aug_ce" or self.guide_aug:
            guide_token_map = self._build_stable_attention_token_map(images, expand_shared=True)
            clean_guide_token_map = guide_token_map.clone().detach()
            guide_pixel_map = self._token_map_to_pixel_map(guide_token_map, clean_pixels.size(-1))
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)
        self._last_dynamic_guide_log = []

        for step_idx in range(self.steps):
            grad_pixels = adv_pixels.detach()
            if self.nesterov and step_idx > 0:
                with torch.no_grad():
                    grad_pixels = grad_pixels + self.decay * self.step_size * momentum.sign()
                    delta = torch.clamp(grad_pixels - clean_pixels, -self.epsilon, self.epsilon)
                    grad_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)
            grad_pixels = grad_pixels.detach().requires_grad_(True)
            norm_adv = self._normalize(grad_pixels)
            if self.guide_dynamic and guide_token_map is not None:
                if step_idx % self.guide_update_interval == 0:
                    with torch.no_grad():
                        guide_adv = self._build_stable_attention_token_map(norm_adv.detach(), expand_shared=True)
                        guide_token_map = self._normalize_weights(
                            self.guide_ema * guide_token_map.to(guide_adv.device) + (1.0 - self.guide_ema) * guide_adv
                        ).detach()
                        guide_pixel_map = self._token_map_to_pixel_map(guide_token_map, clean_pixels.size(-1))
                        adv_primary_score = self._collect_cls_attention_scores(
                            self.model,
                            norm_adv.detach(),
                            target_num_patches=guide_token_map.size(1),
                        )
                        if adv_primary_score is not None and clean_guide_token_map is not None:
                            self._last_dynamic_guide_log.append(
                                self._dynamic_guide_stats(
                                    step_idx=step_idx,
                                    clean_guide=clean_guide_token_map,
                                    guide=guide_token_map,
                                    adv_primary_score=adv_primary_score,
                                )
                            )
            if self.grad_combine == "guide_aug_ce":
                ce_loss = self._guide_aug_ce_loss(grad_pixels, labels, guide_pixel_map)
            else:
                raise ValueError(f"Unknown grad_combine: {self.grad_combine!r}")

            grad = torch.autograd.grad(ce_loss, grad_pixels)[0]
            grad = self._smooth_grad(self._normalize_grad(grad))
            momentum = self.decay * momentum + grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                delta = torch.clamp(self._smooth_perturbation(delta), -self.epsilon, self.epsilon)
                if self.lazy_spectral_delta and step_idx + 1 >= self.steps // 2:
                    delta = torch.clamp(self._spectral_filter_delta(delta), -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

        return self._normalize(adv_pixels)
