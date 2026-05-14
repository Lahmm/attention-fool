import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD, last_vit_stable_patch_frequency


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


class FFTCCAttacker(MIFGSMAttacker):
    """
    MI-FGSM with FFT-guided foreground/background contrast collapse.

    Clean-token FFT stability defines foreground-like/background-like soft
    patch groups. The attack minimizes their patch-CLS alignment gap across
    selected residual-stream layers.
    """

    def __init__(
        self,
        model,
        epsilon: float = 8.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        decay: float = 1.0,
        layers: tuple[int, ...] = (-4, -2, -1),
        lambda_contrast: float = 1.0,
        fft_topk: int = 1,
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
        if fft_topk <= 0:
            raise ValueError(f"fft_topk must be positive, got {fft_topk}.")

        self.layers = tuple(int(layer) for layer in layers)
        self.lambda_contrast = float(lambda_contrast)
        self.fft_topk = int(fft_topk)

    @staticmethod
    def _resolve_layers(layers: tuple[int, ...], num_layers: int) -> list[int]:
        resolved: list[int] = []
        for layer in layers:
            idx = layer if layer >= 0 else num_layers + layer
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Invalid layer index {layer}; model returned {num_layers} token layers.")
            if idx not in resolved:
                resolved.append(idx)
        return resolved

    def _build_clean_guides(
        self,
        images: torch.Tensor,
    ) -> tuple[list[int], dict[int, torch.Tensor]]:
        with torch.no_grad():
            _logits_clean, token_list_clean = self.model(images, return_tokens=True)

        layer_indices = self._resolve_layers(self.layers, len(token_list_clean))
        fft_weights: dict[int, torch.Tensor] = {}

        for layer_idx in layer_indices:
            tokens = token_list_clean[layer_idx].detach()
            weights = last_vit_stable_patch_frequency(
                tokens=tokens,
                topk=self.fft_topk,
                has_cls_token=True,
            ).detach()
            fft_weights[layer_idx] = self._normalize_weights(weights)

        return layer_indices, fft_weights

    @staticmethod
    def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
        min_vals = weights.min(dim=1, keepdim=True).values
        max_vals = weights.max(dim=1, keepdim=True).values
        return (weights - min_vals) / (max_vals - min_vals).clamp_min(1e-12)

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_sum = (values * weights).sum(dim=1)
        normalizer = weights.sum(dim=1).clamp_min(1e-12)
        return weighted_sum / normalizer

    def _feature_losses(
        self,
        token_list_adv: list[torch.Tensor],
        layer_indices: list[int],
        fft_weights: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        contrast_terms = []

        for layer_idx in layer_indices:
            tokens_adv = token_list_adv[layer_idx]
            cls_adv = tokens_adv[:, 0, :]
            patch_adv = tokens_adv[:, 1:, :]

            cls_for_patch = cls_adv.unsqueeze(1).expand_as(patch_adv)
            patch_cls_cos = F.cosine_similarity(patch_adv, cls_for_patch, dim=-1)
            fg_weights = fft_weights[layer_idx].to(
                device=patch_cls_cos.device,
                dtype=patch_cls_cos.dtype,
            )
            bg_weights = 1.0 - fg_weights

            fg_align = self._weighted_mean(patch_cls_cos, fg_weights)
            bg_align = self._weighted_mean(patch_cls_cos, bg_weights)
            contrast_terms.append(-torch.abs(fg_align - bg_align).mean())

        contrast_loss = torch.stack(contrast_terms).mean()
        return contrast_loss

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        layer_indices, fft_weights = self._build_clean_guides(images)
        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        for _step in range(self.steps):
            adv_pixels.requires_grad_(True)
            logits_adv, token_list_adv = self.model(
                self._normalize(adv_pixels),
                return_tokens=True,
            )
            ce_loss = F.cross_entropy(logits_adv, labels)
            contrast_loss = self._feature_losses(
                token_list_adv=token_list_adv,
                layer_indices=layer_indices,
                fft_weights=fft_weights,
            )
            loss = (
                ce_loss
                + self.lambda_contrast * contrast_loss
            )

            grad = torch.autograd.grad(loss, adv_pixels)[0]
            grad = self._normalize_grad(grad)
            momentum = self.decay * momentum + grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

        return self._normalize(adv_pixels)


class FFTCCAttackerV2(FFTCCAttacker):
    """
    Enhanced FFT-CC attacker with:
    - Attention-aware loss (target attention-FFT alignment)
    - Directional contrast loss (invert fg/bg instead of collapse)
    - Patch score inversion (invert high-FFT → low-patch-score pattern)
    - Input diversity (DIM-style random resize for transferability)
    - Dynamic FFT weight update during attack
    - Extended layer coverage with depth-aware weighting
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        decay: float = 1.0,
        layers: tuple[int, ...] = (-8, -7, -6, -5, -4, -3, -2, -1),
        lambda_contrast: float = 0.5,
        lambda_attn: float = 0.3,
        lambda_patch_score: float = 0.2,
        fft_topk: int = 2,
        fft_update_interval: int = 3,
        input_diversity: bool = True,
        dim_resize_range: tuple[float, float] = (0.9, 1.0),
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_contrast=lambda_contrast,
            fft_topk=fft_topk,
            device=device,
        )
        self.lambda_attn = float(lambda_attn)
        self.lambda_patch_score = float(lambda_patch_score)
        self.fft_update_interval = int(fft_update_interval)
        self.input_diversity = bool(input_diversity)
        self.dim_resize_range = tuple(float(r) for r in dim_resize_range)

    def _input_diversity(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random resizing for input diversity (DIM)."""
        if not self.input_diversity:
            return images
        b, c, h, w = images.shape
        lo, hi = self.dim_resize_range
        scale = lo + (hi - lo) * torch.rand(1, device=images.device)
        new_h = max(int(h * scale.item()), h - 2)
        new_w = max(int(w * scale.item()), w - 2)
        resized = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        pad_h2 = h - new_h - pad_h
        pad_w2 = w - new_w - pad_w
        if pad_h >= 0 and pad_w >= 0:
            return F.pad(resized, (pad_w, pad_w2, pad_h, pad_h2), value=0.0)
        return resized[:, :, -pad_h:new_h + pad_h, -pad_w:new_w + pad_w]

    def _attention_loss(
        self,
        attn_logits_list: list[torch.Tensor],
        layer_indices: list[int],
        fft_weights: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Attack attention distribution: flatten attention peaks, shift away from high-FFT regions."""
        terms = []
        for i, layer_idx in enumerate(layer_indices):
            attn_logits = attn_logits_list[layer_idx]  # [B, H, N, N]
            attn_weights = F.softmax(attn_logits, dim=-1)
            cls_attn = attn_weights[:, :, 0, 1:]  # [B, H, N_p]
            mean_attn = cls_attn.mean(dim=1)  # [B, N_p]

            fg_w = fft_weights[layer_idx].to(device=mean_attn.device, dtype=mean_attn.dtype)
            bg_w = 1.0 - fg_w

            log_attn = (mean_attn + 1e-12).log()
            entropy = -(mean_attn * log_attn).sum(dim=1).mean()

            attn_to_fg = (mean_attn * fg_w).sum(dim=1) / fg_w.sum(dim=1).clamp_min(1e-12)
            attn_to_bg = (mean_attn * bg_w).sum(dim=1) / bg_w.sum(dim=1).clamp_min(1e-12)
            attn_shift = attn_to_fg.mean() - attn_to_bg.mean()

            depth_weight = 0.5 + 0.5 * (i / max(len(layer_indices) - 1, 1))
            terms.append(depth_weight * (entropy + attn_shift))

        return torch.stack(terms).mean()

    def _directional_contrast_loss(
        self,
        token_list_adv: list[torch.Tensor],
        layer_indices: list[int],
        fft_weights: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Directional contrast: push foreground alignment below background alignment."""
        terms = []
        for i, layer_idx in enumerate(layer_indices):
            tokens_adv = token_list_adv[layer_idx]
            cls_adv = tokens_adv[:, 0, :]
            patch_adv = tokens_adv[:, 1:, :]

            cls_for_patch = cls_adv.unsqueeze(1).expand_as(patch_adv)
            patch_cls_cos = F.cosine_similarity(patch_adv, cls_for_patch, dim=-1)

            fg_w = fft_weights[layer_idx].to(device=patch_cls_cos.device, dtype=patch_cls_cos.dtype)
            bg_w = 1.0 - fg_w

            fg_align = self._weighted_mean(patch_cls_cos, fg_w)
            bg_align = self._weighted_mean(patch_cls_cos, bg_w)

            # fg_align - bg_align: minimize to push fg below bg
            depth_weight = 0.5 + 0.5 * (i / max(len(layer_indices) - 1, 1))
            terms.append(depth_weight * (fg_align - bg_align).mean())

        return torch.stack(terms).mean()

    def _patch_score_inversion_loss(
        self,
        token_list_adv: list[torch.Tensor],
        layer_indices: list[int],
        fft_weights: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Invert patch score pattern: make high-FFT regions have high patch scores."""
        terms = []
        for i, layer_idx in enumerate(layer_indices):
            tokens = token_list_adv[layer_idx]
            patches = tokens[:, 1:, :]
            patch_norm = patches.norm(dim=-1)

            fg_w = fft_weights[layer_idx].to(device=patch_norm.device, dtype=patch_norm.dtype)
            bg_w = 1.0 - fg_w

            fg_score = self._weighted_mean(patch_norm, fg_w)
            bg_score = self._weighted_mean(patch_norm, bg_w)

            depth_weight = 0.5 + 0.5 * (i / max(len(layer_indices) - 1, 1))
            terms.append(depth_weight * (bg_score - fg_score).mean())

        return torch.stack(terms).mean()

    def _update_fft_weights(
        self,
        token_list_adv: list[torch.Tensor],
        layer_indices: list[int],
        old_fft_weights: dict[int, torch.Tensor],
        ema: float = 0.3,
    ) -> dict[int, torch.Tensor]:
        """Update FFT weights from current adversarial tokens with EMA."""
        new_fft: dict[int, torch.Tensor] = {}
        for layer_idx in layer_indices:
            tokens = token_list_adv[layer_idx].detach()
            weights = last_vit_stable_patch_frequency(
                tokens=tokens,
                topk=self.fft_topk,
                has_cls_token=True,
            ).detach()
            weights = self._normalize_weights(weights)
            old_w = old_fft_weights[layer_idx]
            new_fft[layer_idx] = (1 - ema) * old_w + ema * weights
        return new_fft

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        layer_indices, fft_weights = self._build_clean_guides(images)
        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        for step_idx in range(self.steps):
            norm_input = self._normalize(adv_pixels)
            if self.input_diversity:
                norm_input = self._input_diversity(norm_input)

            norm_input_for_grad = norm_input.detach().clone()
            norm_input_for_grad.requires_grad_(True)

            logits_adv, attn_logits_list, token_list_adv = self.model(
                norm_input_for_grad,
                return_tokens=True,
                return_attn=True,
            )
            ce_loss = F.cross_entropy(logits_adv, labels)

            contrast_loss = self._directional_contrast_loss(
                token_list_adv=token_list_adv,
                layer_indices=layer_indices,
                fft_weights=fft_weights,
            )

            attn_loss = self._attention_loss(
                attn_logits_list=attn_logits_list,
                layer_indices=layer_indices,
                fft_weights=fft_weights,
            )

            ps_loss = self._patch_score_inversion_loss(
                token_list_adv=token_list_adv,
                layer_indices=layer_indices,
                fft_weights=fft_weights,
            )

            loss = (
                ce_loss
                + self.lambda_contrast * contrast_loss
                + self.lambda_attn * attn_loss
                + self.lambda_patch_score * ps_loss
            )

            grad = torch.autograd.grad(loss, norm_input_for_grad)[0]
            grad = self._normalize_grad(grad)
            momentum = self.decay * momentum + grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

            # Dynamic FFT weight update
            if self.fft_update_interval > 0 and (step_idx + 1) % self.fft_update_interval == 0:
                with torch.no_grad():
                    _logits, token_list_update = self.model(
                        self._normalize(adv_pixels), return_tokens=True, return_attn=False,
                    )
                fft_weights = self._update_fft_weights(
                    token_list_adv=token_list_update,
                    layer_indices=layer_indices,
                    old_fft_weights=fft_weights,
                )

        return self._normalize(adv_pixels)


class FFTCCAttackerV3(FFTCCAttacker):
    """
    Conservative FFT-CC enhancement:
    - More steps (20) for stronger attack
    - All 12 layers for full depth coverage
    - fft_topk=3 for broader FFT stability selection
    - TI-FGSM style gradient smoothing (Gaussian kernel on gradients)
    - Original collapse loss (fg/bg alignment gap minimization)
    - Dynamic FFT update
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 20,
        decay: float = 0.9,
        layers: tuple[int, ...] = tuple(range(12)),
        lambda_contrast: float = 1.0,
        fft_topk: int = 3,
        fft_update_interval: int = 5,
        ti_sigma: float = 5.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_contrast=lambda_contrast,
            fft_topk=fft_topk,
            device=device,
        )
        self.fft_update_interval = int(fft_update_interval)
        self.ti_sigma = float(ti_sigma)

        # Build TI-FGSM Gaussian kernel
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma)

    def _build_ti_kernel(self, sigma: float) -> torch.Tensor:
        """Build a 2D Gaussian kernel for gradient smoothing."""
        if sigma <= 0:
            return None
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g1d = torch.exp(-0.5 * (x / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] @ g1d[None, :]  # [K, K]
        return g2d.view(1, 1, g2d.size(0), g2d.size(1))

    def _smooth_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply TI-FGSM Gaussian smoothing to gradients."""
        if self._ti_kernel is None or self.ti_sigma <= 0:
            return grad
        kernel = self._ti_kernel.to(grad.device, grad.dtype)  # [1, 1, K, K]
        # Repeat kernel for each channel for grouped conv
        kernel = kernel.repeat(grad.size(1), 1, 1, 1)  # [C, 1, K, K]
        pad = kernel.size(2) // 2
        padded = F.pad(grad, (pad, pad, pad, pad), mode='reflect')
        smoothed = F.conv2d(padded, kernel, groups=grad.size(1))
        return smoothed

    def _update_fft_weights(
        self,
        token_list_adv: list[torch.Tensor],
        layer_indices: list[int],
        old_fft_weights: dict[int, torch.Tensor],
        ema: float = 0.3,
    ) -> dict[int, torch.Tensor]:
        """Update FFT weights from current adversarial tokens with EMA."""
        new_fft: dict[int, torch.Tensor] = {}
        for layer_idx in layer_indices:
            tokens = token_list_adv[layer_idx].detach()
            weights = last_vit_stable_patch_frequency(
                tokens=tokens,
                topk=self.fft_topk,
                has_cls_token=True,
            ).detach()
            weights = self._normalize_weights(weights)
            old_w = old_fft_weights[layer_idx]
            new_fft[layer_idx] = (1 - ema) * old_w + ema * weights
        return new_fft

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        layer_indices, fft_weights = self._build_clean_guides(images)
        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        for step_idx in range(self.steps):
            adv_pixels.requires_grad_(True)
            logits_adv, token_list_adv = self.model(
                self._normalize(adv_pixels),
                return_tokens=True,
            )
            ce_loss = F.cross_entropy(logits_adv, labels)
            contrast_loss = self._feature_losses(
                token_list_adv=token_list_adv,
                layer_indices=layer_indices,
                fft_weights=fft_weights,
            )
            loss = ce_loss + self.lambda_contrast * contrast_loss

            grad = torch.autograd.grad(loss, adv_pixels)[0]
            grad = self._smooth_grad(grad)  # TI-FGSM gradient smoothing
            grad = self._normalize_grad(grad)
            momentum = self.decay * momentum + grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

            # Dynamic FFT weight update
            if self.fft_update_interval > 0 and (step_idx + 1) % self.fft_update_interval == 0:
                with torch.no_grad():
                    _logits, token_list_update = self.model(
                        self._normalize(adv_pixels), return_tokens=True, return_attn=False,
                    )
                fft_weights = self._update_fft_weights(
                    token_list_adv=token_list_update,
                    layer_indices=layer_indices,
                    old_fft_weights=fft_weights,
                )

        return self._normalize(adv_pixels)

class FFTCCAttackerImgFFT(FFTCCAttacker):
    def __init__(self, model, epsilon=16.0/255.0, step_size=None, steps=10, decay=1.0,
                 layers=(-4,-2,-1), lambda_contrast=1.0, fft_cutoff=0.15,
                 fft_transition=0.04, device=None):
        super().__init__(model=model, epsilon=epsilon, step_size=step_size, steps=steps,
                         decay=decay, layers=layers, lambda_contrast=lambda_contrast,
                         fft_topk=1, device=device)
        self.fft_cutoff = float(fft_cutoff)
        self.fft_transition = float(fft_transition)

    def _image_fft_patch_weights(self, clean_pixels):
        from utils import image_2d_fft_low_high_maps
        fft_maps = image_2d_fft_low_high_maps(clean_pixels, cutoff_ratio=self.fft_cutoff,
                                               transition_ratio=self.fft_transition)
        low_ratio = fft_maps["low_ratio"]
        b, h, w = low_ratio.shape
        patch_h, patch_w = h // 14, w // 14
        low_reshaped = low_ratio.view(b, 14, patch_h, 14, patch_w)
        patch_low = low_reshaped.mean(dim=(2, 4))
        patch_weights = patch_low.reshape(b, -1)
        patch_weights = self._normalize_weights(patch_weights)
        weights_dict = {}
        for layer_idx in range(12):
            weights_dict[layer_idx] = patch_weights
        return weights_dict

    def attack_batch(self, images, labels):
        import torch.nn.functional as F
        images = images.to(self.device)
        labels = labels.to(self.device)
        clean_pixels = self._denormalize(images).detach()
        fft_weights = self._image_fft_patch_weights(clean_pixels)
        layer_indices = self._resolve_layers(self.layers, self.model.num_blocks)
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)
        for _step in range(self.steps):
            adv_pixels.requires_grad_(True)
            logits_adv, token_list_adv = self.model(self._normalize(adv_pixels), return_tokens=True)
            ce_loss = F.cross_entropy(logits_adv, labels)
            contrast_loss = self._feature_losses(token_list_adv=token_list_adv,
                                                  layer_indices=layer_indices,
                                                  fft_weights=fft_weights)
            loss = ce_loss + self.lambda_contrast * contrast_loss
            grad = torch.autograd.grad(loss, adv_pixels)[0]
            grad = self._normalize_grad(grad)
            momentum = self.decay * momentum + grad
            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()
        return self._normalize(adv_pixels)
