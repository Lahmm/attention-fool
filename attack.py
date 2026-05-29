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
    Configurable iterative FGSM attack with optional forward augmentations,
    attention-guided augmentation, and gradient refinement modules.
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 20,
        layers: tuple[int, ...] = (-6, -5, -4, -3, -2, -1),
        ti_sigma: float = 3.0,
        input_diversity: bool = False,
        dim_resize_range: tuple[float, float] = (0.85, 1.0),
        use_si: bool = False,
        si_scales: int = 1,
        use_eot: bool = False,
        eot_iter: int = 1,
        use_momentum: bool = False,
        momentum_decay: float = 1.0,
        nesterov: bool = False,
        normalize_grad: bool = False,
        attention_guide_models: tuple[torch.nn.Module, ...] | None = None,
        attention_guide_type: str = "postsoftmax_cls",
        attention_guide_build_method: str = "pixel",
        attention_guide_patch_size: int = 16,
        guide_aug: bool = False,
        guide_aug_area: str = "background",
        guide_aug_methods: tuple[str, ...] = ("dropout",),
        guide_aug_copies: int = 3,
        guide_aug_strength: float = 0.2,
        guide_grad_norm_area: str = "none",
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=momentum_decay,
            device=device,
        )
        if not layers:
            raise ValueError("layers must contain at least one layer index.")
        if ti_sigma < 0:
            raise ValueError(f"ti_sigma must be non-negative, got {ti_sigma}.")
        if si_scales <= 0:
            raise ValueError(f"si_scales must be positive, got {si_scales}.")
        if eot_iter <= 0:
            raise ValueError(f"eot_iter must be positive, got {eot_iter}.")
        if nesterov and not use_momentum:
            raise ValueError("--ni requires --mi because Nesterov lookahead depends on momentum.")

        self.layers = tuple(int(layer) for layer in layers)
        self.ti_sigma = float(ti_sigma)
        self.input_diversity = bool(input_diversity)
        self.dim_resize_range = tuple(float(r) for r in dim_resize_range)
        self.use_si = bool(use_si)
        self.si_scales = int(si_scales)
        self.use_eot = bool(use_eot)
        self.eot_iter = int(eot_iter)
        self.use_momentum = bool(use_momentum)
        self.nesterov = bool(nesterov)
        self.normalize_grad = bool(normalize_grad)
        self.attention_guide_models = tuple(attention_guide_models or ())

        guide_types = tuple(item.strip() for item in attention_guide_type.split(",") if item.strip())
        valid_guide_types = ("postsoftmax_cls", "qk_cls", "qk_all_queries")
        if not guide_types:
            raise ValueError("attention_guide_type must contain at least one guide type.")
        invalid_guide_types = [item for item in guide_types if item not in valid_guide_types]
        if invalid_guide_types:
            raise ValueError(f"attention_guide_type entries must be in {valid_guide_types}, got {invalid_guide_types}.")
        self.attention_guide_types = guide_types

        valid_build_methods = ("pixel", "patch")
        if attention_guide_build_method not in valid_build_methods:
            raise ValueError(f"attention_guide_build_method must be one of {valid_build_methods}, got {attention_guide_build_method!r}.")
        self.attention_guide_build_method = attention_guide_build_method
        if attention_guide_patch_size <= 0:
            raise ValueError(f"attention_guide_patch_size must be positive, got {attention_guide_patch_size}.")
        self.attention_guide_patch_size = int(attention_guide_patch_size)

        self.guide_aug = bool(guide_aug)
        valid_guide_aug_areas = ("foreground", "background", "all")
        if guide_aug_area not in valid_guide_aug_areas:
            raise ValueError(f"guide_aug_area must be one of {valid_guide_aug_areas}, got {guide_aug_area!r}.")
        self.guide_aug_area = guide_aug_area
        self.guide_aug_methods = tuple(str(method).strip() for method in guide_aug_methods if str(method).strip())
        valid_guide_aug_methods = (
            "dropout",
            "jitter",
            "freq",
            "lowpass_gauss",
            "laplacian_low",
            "fft_lowboost",
            "illumination_low",
        )
        if not self.guide_aug_methods:
            raise ValueError("guide_aug_methods must contain at least one method.")
        invalid_methods = [method for method in self.guide_aug_methods if method not in valid_guide_aug_methods]
        if invalid_methods:
            raise ValueError(f"guide_aug_methods entries must be in {valid_guide_aug_methods}, got {invalid_methods}.")
        self.guide_aug_copies = int(guide_aug_copies)
        if self.guide_aug_copies <= 0:
            raise ValueError(f"guide_aug_copies must be positive, got {guide_aug_copies}.")
        self.guide_aug_strength = float(guide_aug_strength)
        if self.guide_aug_strength < 0:
            raise ValueError(f"guide_aug_strength must be non-negative, got {guide_aug_strength}.")
        valid_guide_grad_norm_areas = ("none", "foreground", "background")
        if guide_grad_norm_area not in valid_guide_grad_norm_areas:
            raise ValueError(
                f"guide_grad_norm_area must be one of {valid_guide_grad_norm_areas}, "
                f"got {guide_grad_norm_area!r}."
            )
        self.guide_grad_norm_area = guide_grad_norm_area
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma) if self.ti_sigma > 0 else None
        self._guide_lowpass_kernel = self._build_gaussian_kernel(kernel_size=15, sigma=3.0)
        self._guide_illumination_kernel = self._build_gaussian_kernel(kernel_size=31, sigma=8.0)

    @staticmethod
    def _build_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g1d = torch.exp(-0.5 * (x / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] @ g1d[None, :]
        return g2d.view(1, 1, kernel_size, kernel_size)

    @staticmethod
    def _build_ti_kernel(sigma: float) -> torch.Tensor:
        radius = int(3 * sigma)
        kernel_size = 2 * radius + 1
        return LazyAggregationAttacker._build_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

    def _apply_depthwise_kernel(self, pixels: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kernel = kernel.to(pixels.device, pixels.dtype).repeat(pixels.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(pixels, (pad, pad, pad, pad), mode="reflect"), kernel, groups=pixels.size(1))

    def _smooth_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self._ti_kernel is None or self.ti_sigma <= 0:
            return grad
        return self._apply_depthwise_kernel(grad, self._ti_kernel)

    def _input_diversity(self, images: torch.Tensor) -> torch.Tensor:
        if not self.input_diversity:
            return images
        _batch_size, _channels, height, width = images.shape
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
    def _infer_num_heads_from_attn(attn_module) -> int | None:
        heads = getattr(attn_module, "num_heads", None)
        return int(heads) if heads is not None else None

    @staticmethod
    def _qkv_to_cls_attention_scores(qkv: torch.Tensor, num_heads: int | None, attention_guide_type: str) -> torch.Tensor:
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
        if attention_guide_type == "postsoftmax_cls":
            return torch.softmax(qk, dim=-1)[:, :, 0, 1:].mean(dim=1)
        if attention_guide_type == "qk_cls":
            return qk[:, :, 0, 1:].mean(dim=1)
        if attention_guide_type == "qk_all_queries":
            return qk[:, :, :, 1:].mean(dim=(1, 2))
        raise ValueError(f"Unsupported attention_guide_type: {attention_guide_type}")

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

    def _resolve_layer_indices(self, num_layers: int) -> list[int]:
        indices = []
        for layer in self.layers:
            idx = layer if layer >= 0 else num_layers + layer
            if idx < 0 or idx >= num_layers:
                continue
            indices.append(idx)
        if not indices:
            raise ValueError(
                f"Layer indices {self.layers} are out of range for {num_layers} compatible attention layers."
            )
        return indices

    def _collect_cls_attention_scores(
        self,
        source_model,
        images: torch.Tensor,
        target_num_patches: int | None = None,
        attention_guide_type: str | None = None,
    ) -> torch.Tensor | None:
        attention_guide_type = self.attention_guide_types[0] if attention_guide_type is None else attention_guide_type
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
                    score = self._qkv_to_cls_attention_scores(qkv, record["heads"], attention_guide_type)
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
            layer_indices = self._resolve_layer_indices(len(selected))
            return torch.stack([selected[idx] for idx in layer_indices]).mean(dim=0)
        finally:
            for handle in handles:
                handle.remove()

    def _build_stable_attention_token_map(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        attention_guide_type = self.attention_guide_types[0]
        primary_score = self._collect_cls_attention_scores(self.model, images, attention_guide_type=attention_guide_type)
        if primary_score is None:
            raise ValueError("The white-box model did not produce compatible CLS attention scores.")
        num_patches = primary_score.size(1)
        scores = [primary_score]
        for source_model in self.attention_guide_models:
            score = self._collect_cls_attention_scores(
                source_model,
                images,
                target_num_patches=num_patches,
                attention_guide_type=attention_guide_type,
            )
            if score is not None:
                scores.append(score)
        token_map = self._normalize_weights(torch.stack(scores).mean(dim=0))
        return token_map.detach()

    def _token_map_to_pixel_map(self, token_map: torch.Tensor, img_size: int) -> torch.Tensor:
        num_patches = token_map.size(1)
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square.")
        grid = token_map.view(token_map.size(0), 1, grid_size, grid_size)
        pixel_map = F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False)
        return self._normalize_pixel_map(pixel_map)

    def _token_map_to_patch_pixel_map(self, token_map: torch.Tensor, img_size: int) -> torch.Tensor:
        if img_size % self.attention_guide_patch_size != 0:
            raise ValueError(
                f"attention_guide_patch_size must divide img_size, got "
                f"patch_size={self.attention_guide_patch_size}, img_size={img_size}."
            )
        num_patches = token_map.size(1)
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square.")
        grid = token_map.view(token_map.size(0), 1, grid_size, grid_size)
        target_grid_size = img_size // self.attention_guide_patch_size
        if target_grid_size != grid_size:
            grid = F.interpolate(
                grid,
                size=(target_grid_size, target_grid_size),
                mode="bilinear",
                align_corners=False,
            )
        pixel_map = F.interpolate(grid, size=(img_size, img_size), mode="nearest")
        return self._normalize_pixel_map(pixel_map)

    @staticmethod
    def _normalize_pixel_map(pixel_map: torch.Tensor) -> torch.Tensor:
        flat = pixel_map.flatten(1)
        min_vals = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        return ((pixel_map - min_vals) / (max_vals - min_vals).clamp_min(1e-12)).detach()

    def _normalize_guided_grad(
        self,
        grad: torch.Tensor,
        guide_pixel_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.guide_grad_norm_area == "none":
            return grad
        if guide_pixel_map is None:
            raise ValueError("guide_pixel_map is required when guide_grad_norm_area is enabled.")

        guide = guide_pixel_map.to(grad.device, grad.dtype).clamp(0.0, 1.0)
        if self.guide_grad_norm_area == "foreground":
            region = guide
        elif self.guide_grad_norm_area == "background":
            region = 1.0 - guide
        else:
            raise ValueError(f"Unsupported guide_grad_norm_area: {self.guide_grad_norm_area}")

        region = region.expand_as(grad)
        denom = (
            (grad.abs() * region).sum(dim=(1, 2, 3), keepdim=True)
            / region.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        ).clamp_min(1e-12)
        normalized_region_grad = grad / denom
        return normalized_region_grad * region + grad * (1.0 - region)

    def _build_guide_pixel_map(self, images: torch.Tensor, img_size: int) -> torch.Tensor:
        guide_token_map = self._build_stable_attention_token_map(images)
        if self.attention_guide_build_method == "pixel":
            return self._token_map_to_pixel_map(guide_token_map, img_size)
        if self.attention_guide_build_method == "patch":
            return self._token_map_to_patch_pixel_map(guide_token_map, img_size)
        raise ValueError(f"Unsupported attention_guide_build_method: {self.attention_guide_build_method}")

    def _lowpass_gauss_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        return self._apply_depthwise_kernel(pixels, self._guide_lowpass_kernel)

    def _laplacian_low_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        low = self._lowpass_gauss_pixels(pixels)
        high = pixels - low
        high_scale = max(0.0, 1.0 - 2.5 * self.guide_aug_strength)
        return torch.clamp(low + high * high_scale, 0.0, 1.0)

    def _fft_lowboost_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        height, width = pixels.shape[-2:]
        work = pixels.float()
        freq = torch.fft.rfft2(work, dim=(-2, -1), norm="ortho")
        fy = torch.fft.fftfreq(height, device=pixels.device, dtype=work.dtype).view(1, 1, height, 1)
        fx = torch.fft.rfftfreq(width, device=pixels.device, dtype=work.dtype).view(1, 1, 1, width // 2 + 1)
        radius = torch.sqrt(fx.square() + fy.square())
        low_weight = torch.exp(-0.5 * (radius / 0.12).square())
        boosted = freq * (1.0 + self.guide_aug_strength * low_weight)
        augmented = torch.fft.irfft2(boosted, s=(height, width), dim=(-2, -1), norm="ortho")
        return torch.clamp(augmented.to(pixels.dtype), 0.0, 1.0)

    def _illumination_low_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        illumination = self._apply_depthwise_kernel(pixels, self._guide_illumination_kernel)
        centered = illumination - illumination.mean(dim=(2, 3), keepdim=True)
        return torch.clamp(pixels + self.guide_aug_strength * centered, 0.0, 1.0)

    def _augment_full_image(self, pixels: torch.Tensor, method: str) -> torch.Tensor:
        strength = self.guide_aug_strength
        if strength <= 0:
            return pixels
        if method == "dropout":
            noise = torch.rand_like(pixels)
            blurred = F.avg_pool2d(pixels, kernel_size=5, stride=1, padding=2)
            corrupt = 0.5 * noise + 0.5 * blurred
            return torch.clamp(pixels * (1.0 - strength) + corrupt * strength, 0.0, 1.0)
        if method == "jitter":
            brightness = (torch.rand(pixels.size(0), 1, 1, 1, device=pixels.device, dtype=pixels.dtype) * 2.0 - 1.0) * strength
            noise = torch.randn_like(pixels) * (strength / 2.0)
            return torch.clamp(pixels * (1.0 + brightness) + noise, 0.0, 1.0)
        if method == "freq":
            pooled = F.avg_pool2d(pixels, kernel_size=9, stride=1, padding=4)
            noise = F.avg_pool2d(torch.rand_like(pixels), kernel_size=9, stride=1, padding=4)
            corrupt = 0.7 * pooled + 0.3 * noise
            return torch.clamp(pixels * (1.0 - strength) + corrupt * strength, 0.0, 1.0)
        if method == "lowpass_gauss":
            low = self._lowpass_gauss_pixels(pixels)
            return torch.clamp(pixels * (1.0 - strength) + low * strength, 0.0, 1.0)
        if method == "laplacian_low":
            return self._laplacian_low_pixels(pixels)
        if method == "fft_lowboost":
            return self._fft_lowboost_pixels(pixels)
        if method == "illumination_low":
            return self._illumination_low_pixels(pixels)
        raise ValueError(f"Unsupported guide augmentation method: {method}")

    def _guide_augmented_pixels(
        self,
        pixels: torch.Tensor,
        guide_pixel_map: torch.Tensor | None,
        method: str,
    ) -> torch.Tensor:
        augmented = self._augment_full_image(pixels, method)
        if self.guide_aug_area == "all":
            return augmented
        if guide_pixel_map is None:
            raise ValueError("guide_pixel_map is required unless guide_aug_area='all'.")
        guide = guide_pixel_map.to(pixels.device, pixels.dtype).clamp(0.0, 1.0)
        if self.guide_aug_area == "foreground":
            return torch.clamp(augmented * guide + pixels * (1.0 - guide), 0.0, 1.0)
        if self.guide_aug_area == "background":
            return torch.clamp(pixels * guide + augmented * (1.0 - guide), 0.0, 1.0)
        raise ValueError(f"Unsupported guide augmentation area: {self.guide_aug_area}")

    def _iter_forward_pixels(
        self,
        pixels: torch.Tensor,
        guide_pixel_map: torch.Tensor | None,
    ):
        if not self.guide_aug:
            yield pixels
            return
        for method in self.guide_aug_methods:
            for _copy_idx in range(self.guide_aug_copies):
                yield self._guide_augmented_pixels(pixels, guide_pixel_map, method)

    def _iter_attack_losses(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        guide_pixel_map: torch.Tensor | None,
    ):
        si_count = self.si_scales if self.use_si else 1
        eot_count = self.eot_iter if self.use_eot else 1
        for scale_idx in range(si_count):
            scale = float(2 ** scale_idx)
            for _eot_idx in range(eot_count):
                for forward_pixels in self._iter_forward_pixels(pixels, guide_pixel_map):
                    logits_adv = self.model(
                        self._input_diversity(self._normalize(forward_pixels) / scale),
                        return_attn=False,
                    )
                    yield F.cross_entropy(logits_adv, labels)

    def _attack_grad(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        guide_pixel_map: torch.Tensor | None,
    ) -> torch.Tensor:
        grad_sum = None
        term_count = 0
        for ce_loss in self._iter_attack_losses(pixels, labels, guide_pixel_map):
            grad_term = torch.autograd.grad(ce_loss, pixels)[0]
            grad_sum = grad_term if grad_sum is None else grad_sum + grad_term
            term_count += 1
        if grad_sum is None:
            raise RuntimeError("No attack loss terms were generated.")
        return grad_sum / float(term_count)

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        clean_pixels = self._denormalize(images).detach()
        guide_pixel_map = None
        needs_guide_map = (self.guide_aug and self.guide_aug_area != "all") or self.guide_grad_norm_area != "none"
        if needs_guide_map:
            guide_pixel_map = self._build_guide_pixel_map(images, clean_pixels.size(-1))
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        for step_idx in range(self.steps):
            grad_pixels = adv_pixels.detach()
            if self.nesterov and step_idx > 0:
                with torch.no_grad():
                    grad_pixels = grad_pixels + self.decay * self.step_size * momentum.sign()
                    delta = torch.clamp(grad_pixels - clean_pixels, -self.epsilon, self.epsilon)
                    grad_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)
            grad_pixels = grad_pixels.detach().requires_grad_(True)
            grad = self._attack_grad(grad_pixels, labels, guide_pixel_map)
            if self.normalize_grad:
                grad = self._normalize_grad(grad)
            grad = self._normalize_guided_grad(grad, guide_pixel_map)
            grad = self._smooth_grad(grad)
            if self.use_momentum:
                momentum = self.decay * momentum + grad
                update = momentum
            else:
                update = grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * update.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

        return self._normalize(adv_pixels)
