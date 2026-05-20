import math

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


class FFTCCAttackerPCGrad(FFTCCAttackerV2):
    """
    FFT-CC with PCGrad gradient surgery to resolve CE / feature loss conflicts.

    Core improvements over V2:
    - Asymmetric PCGrad: contrast gradient is projected away from conflicting CE directions
    - Warmup phase: pure CE iterations before adding feature losses
    - Per-task L2 gradient normalization before PCGrad
    - TI-FGSM gradient smoothing
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        decay: float = 1.0,
        layers: tuple[int, ...] = (-6, -5, -4, -3, -2, -1),
        lambda_contrast: float = 1.0,
        fft_topk: int = 3,
        fft_update_interval: int = 3,
        pcgrad_mode: str = "asymmetric",
        warmup_steps: int = 3,
        grad_l2_norm: float = 1.0,
        ti_sigma: float = 0.0,
        device=None,
    ):
        super().__init__(
            model=model, epsilon=epsilon, step_size=step_size, steps=steps,
            decay=decay, layers=layers, lambda_contrast=lambda_contrast,
            lambda_attn=0.0, lambda_patch_score=0.0, fft_topk=fft_topk,
            fft_update_interval=fft_update_interval, input_diversity=False,
            device=device,
        )
        valid_modes = ("asymmetric", "symmetric", "none", "orthogonalize", "adaptive", "modulate")
        if pcgrad_mode not in valid_modes:
            raise ValueError(f"pcgrad_mode must be one of {valid_modes}, got {pcgrad_mode!r}.")
        self.pcgrad_mode = pcgrad_mode
        self.warmup_steps = int(warmup_steps)
        self.grad_l2_norm = float(grad_l2_norm)
        self.ti_sigma = float(ti_sigma)
        self._ti_kernel = None
        if self.ti_sigma > 0:
            self._ti_kernel = self._build_ti_kernel(self.ti_sigma)

    @staticmethod
    def _build_ti_kernel(sigma):
        radius = int(3 * sigma)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g1d = torch.exp(-0.5 * (x / sigma) ** 2)
        g1d = g1d / g1d.sum()
        g2d = g1d[:, None] @ g1d[None, :]
        return g2d.view(1, 1, g2d.size(0), g2d.size(1))

    def _smooth_grad(self, grad):
        if self._ti_kernel is None or self.ti_sigma <= 0:
            return grad
        kernel = self._ti_kernel.to(grad.device, grad.dtype)
        kernel = kernel.repeat(grad.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(grad, (pad, pad, pad, pad), mode="reflect"), kernel, groups=grad.size(1))

    @staticmethod
    def _l2_normalize_grad(grad, target_norm=1.0):
        flat = grad.reshape(grad.size(0), -1)
        return (flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12) * target_norm).reshape_as(grad)

    def _pcgrad_combine(self, g_ce, g_contrast):
        B = g_ce.size(0)
        flat_ce = g_ce.reshape(B, -1)
        flat_ct = g_contrast.reshape(B, -1)
        dot = (flat_ce * flat_ct).sum(dim=1)
        n_ce = flat_ce.norm(dim=1).clamp_min(1e-12)
        n_ct = flat_ct.norm(dim=1).clamp_min(1e-12)
        cos_sim = dot / (n_ce * n_ct)
        avg_cos = cos_sim.mean().item()
        conflict = cos_sim < 0

        if self.pcgrad_mode in ("none",) or not conflict.any():
            return g_ce + self.lambda_contrast * g_contrast, avg_cos
        if self.pcgrad_mode == "orthogonalize":
            g_ct_res = flat_ct - (dot / (n_ce ** 2)).unsqueeze(1) * flat_ce
            return (flat_ce + self.lambda_contrast * g_ct_res).reshape_as(g_ce), avg_cos
        if self.pcgrad_mode == "adaptive":
            return (flat_ce + self.lambda_contrast * torch.clamp(cos_sim, min=0.0)[:, None] * flat_ct).reshape_as(g_ce), avg_cos
        if self.pcgrad_mode == "symmetric":
            proj_c2t = (dot / (n_ct ** 2)).unsqueeze(1) * flat_ct
            proj_t2c = (dot / (n_ce ** 2)).unsqueeze(1) * flat_ce
            return (torch.where(conflict[:, None], flat_ce - proj_c2t, flat_ce)
                    + self.lambda_contrast * torch.where(conflict[:, None], flat_ct - proj_t2c, flat_ct)).reshape_as(g_ce), avg_cos
        proj = (dot / (n_ce ** 2)).unsqueeze(1) * flat_ce
        return (flat_ce + self.lambda_contrast * torch.where(conflict[:, None], flat_ct - proj, flat_ct)).reshape_as(g_ce), avg_cos

    def _build_modulation_map(self, token_list_adv, layer_indices, fft_weights, img_size):
        avg_fg = torch.stack([fft_weights[li] for li in layer_indices]).mean(dim=0)
        B, N = avg_fg.shape
        gs = int(N ** 0.5)
        grid = avg_fg.reshape(B, 1, gs, gs)
        grid = (grid - grid.amin(dim=(1, 2, 3), keepdim=True)) / (grid.amax(dim=(1, 2, 3), keepdim=True) - grid.amin(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False).detach()

    def attack_batch(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        layer_indices, fft_weights = self._build_clean_guides(images)
        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)
        self._last_cosine_log = []

        for step_idx in range(self.steps):
            adv_pixels = adv_pixels.detach().requires_grad_(True)
            logits_adv, token_list_adv = self.model(self._normalize(adv_pixels), return_tokens=True)
            ce_loss = F.cross_entropy(logits_adv, labels)
            g_ce = torch.autograd.grad(ce_loss, adv_pixels, retain_graph=True)[0]

            if step_idx < self.warmup_steps:
                grad, avg_cos = g_ce, 1.0
            elif self.pcgrad_mode == "modulate":
                mod = self._build_modulation_map(token_list_adv, layer_indices, fft_weights, clean_pixels.size(-1))
                grad = (g_ce * (1.0 + self.lambda_contrast * mod)).detach()
                avg_cos = 0.0
            else:
                ct_loss = self._directional_contrast_loss(token_list_adv, layer_indices, fft_weights)
                g_ct = torch.autograd.grad(ct_loss, adv_pixels, retain_graph=False)[0]
                grad, avg_cos = self._pcgrad_combine(self._l2_normalize_grad(g_ce, self.grad_l2_norm), self._l2_normalize_grad(g_ct, self.grad_l2_norm))

            self._last_cosine_log.append((step_idx, avg_cos))
            grad = self._smooth_grad(self._normalize_grad(grad))
            momentum = self.decay * momentum + grad
            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)
            if self.fft_update_interval > 0 and (step_idx + 1) % self.fft_update_interval == 0:
                with torch.no_grad():
                    _, tu = self.model(self._normalize(adv_pixels), return_tokens=True)
                fft_weights = self._update_fft_weights(tu, layer_indices, fft_weights)

        cos_vals = [c for _, c in self._last_cosine_log]
        print(f"  [PCGrad] conflicts={sum(1 for c in cos_vals if c < 0)}/{len(cos_vals)} mean_cos={sum(cos_vals)/len(cos_vals):.4f}")
        return self._normalize(adv_pixels)


class LazyAggregationAttacker(MIFGSMAttacker):
    """
    Lazy aggregation hijack attack for ViTs.

    Clean attention, token norm, token FFT stability, and image low-frequency
    structure define foreground patches and background anchor patches. During
    attack, a single anchor loss pushes CLS aggregation toward anchors and away
    from foreground while CE keeps the white-box attack moving.
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 20,
        decay: float = 1.0,
        layers: tuple[int, ...] = (-6, -5, -4, -3, -2, -1),
        anchor_top_ratio: float = 0.25,
        fg_top_ratio: float = 0.25,
        lambda_anchor: float = 1.0,
        warmup_steps: int = 3,
        grad_combine: str = "pcgrad_asymmetric",
        fft_topk: int = 1,
        spectral_cutoff: float = 0.15,
        spectral_transition: float = 0.04,
        grad_l2_norm: float = 1.0,
        ti_sigma: float = 3.0,
        input_diversity: bool = True,
        dim_resize_range: tuple[float, float] = (0.85, 1.0),
        si_scales: int = 1,
        nesterov: bool = True,
        eot_iter: int = 1,
        perturb_smooth_sigma: float = 0.0,
        anchor_schedule: str = "constant",
        anchor_start_step: int | None = None,
        anchor_end_weight: float | None = None,
        lazy_spectral_delta: bool = False,
        lazy_spectral_cutoff: float = 0.25,
        anchor_mod_alpha: float = 1.0,
        fg_mod_alpha: float = 0.5,
        anchor_mod_power: float = 1.0,
        ensemble_models: tuple[torch.nn.Module, ...] | None = None,
        attention_guide_models: tuple[torch.nn.Module, ...] | None = None,
        guide_type: str = "postsoftmax_cls",
        guide_sample_mode: str = "fixed",
        guide_entropy_temp: float = 1.0,
        feature_loss_weight: float = 0.1,
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
        if not (0.0 < anchor_top_ratio < 1.0):
            raise ValueError(f"anchor_top_ratio must be in (0, 1), got {anchor_top_ratio}.")
        if not (0.0 < fg_top_ratio < 1.0):
            raise ValueError(f"fg_top_ratio must be in (0, 1), got {fg_top_ratio}.")
        if fft_topk <= 0:
            raise ValueError(f"fft_topk must be positive, got {fft_topk}.")
        valid_combine = (
            "pcgrad_asymmetric",
            "sum",
            "ce",
            "anchor_modulate",
            "stable_attention",
            "expanded_stable_attention",
            "guide_response",
            "dynamic_guide_response",
            "guide_aug_ce",
            "guide_aug_feature",
            "guide_qk_response",
        )
        if grad_combine not in valid_combine:
            raise ValueError(f"grad_combine must be one of {valid_combine}, got {grad_combine!r}.")
        valid_anchor_schedules = ("constant", "linear", "cosine")
        if anchor_schedule not in valid_anchor_schedules:
            raise ValueError(f"anchor_schedule must be one of {valid_anchor_schedules}, got {anchor_schedule!r}.")

        self.layers = tuple(int(layer) for layer in layers)
        self.anchor_top_ratio = float(anchor_top_ratio)
        self.fg_top_ratio = float(fg_top_ratio)
        self.lambda_anchor = float(lambda_anchor)
        self.warmup_steps = int(warmup_steps)
        self.grad_combine = grad_combine
        self.fft_topk = int(fft_topk)
        self.spectral_cutoff = float(spectral_cutoff)
        self.spectral_transition = float(spectral_transition)
        self.grad_l2_norm = float(grad_l2_norm)
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
        self.anchor_schedule = anchor_schedule
        self.anchor_start_step = self.warmup_steps if anchor_start_step is None else int(anchor_start_step)
        if self.anchor_start_step < 0:
            raise ValueError(f"anchor_start_step must be non-negative, got {self.anchor_start_step}.")
        self.anchor_end_weight = self.lambda_anchor if anchor_end_weight is None else float(anchor_end_weight)
        self.lazy_spectral_delta = bool(lazy_spectral_delta)
        self.lazy_spectral_cutoff = float(lazy_spectral_cutoff)
        self.anchor_mod_alpha = float(anchor_mod_alpha)
        self.fg_mod_alpha = float(fg_mod_alpha)
        self.anchor_mod_power = float(anchor_mod_power)
        if self.anchor_mod_power <= 0:
            raise ValueError(f"anchor_mod_power must be positive, got {anchor_mod_power}.")
        self.ensemble_models = tuple(ensemble_models or ())
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
        self.feature_loss_weight = float(feature_loss_weight)
        if self.feature_loss_weight < 0:
            raise ValueError(f"feature_loss_weight must be non-negative, got {feature_loss_weight}.")
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
        valid_guide_aug_modes = ("dropout", "mix", "jitter", "freq")
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

    def _anchor_weight_for_step(self, step_idx: int) -> float:
        if step_idx < self.anchor_start_step:
            return 0.0
        if self.anchor_schedule == "constant":
            return self.lambda_anchor

        ramp_end_step = max(self.anchor_start_step + 1, self.steps // 2)
        if step_idx >= ramp_end_step:
            return self.anchor_end_weight

        progress = (step_idx - self.anchor_start_step + 1) / (ramp_end_step - self.anchor_start_step + 1)
        progress = max(0.0, min(1.0, progress))
        if self.anchor_schedule == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        return self.anchor_end_weight * progress

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
    def _resolve_layers(layers: tuple[int, ...], num_layers: int) -> list[int]:
        return FFTCCAttacker._resolve_layers(layers, num_layers)

    @staticmethod
    def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
        return FFTCCAttacker._normalize_weights(weights)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(device=values.device, dtype=values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-12)

    @staticmethod
    def _select_top_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
        num_patches = scores.size(1)
        k = max(1, min(num_patches, int(round(num_patches * ratio))))
        top_idx = torch.topk(scores, k=k, dim=1, largest=True).indices
        mask = torch.zeros_like(scores, dtype=torch.bool)
        return mask.scatter(1, top_idx, True)

    @staticmethod
    def _scale_term(term: torch.Tensor) -> torch.Tensor:
        scale = term.detach().abs().mean().clamp_min(1e-6)
        return term / scale

    @staticmethod
    def _l2_normalize_grad(grad: torch.Tensor, target_norm: float = 1.0) -> torch.Tensor:
        flat = grad.reshape(grad.size(0), -1)
        normed = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (normed * target_norm).reshape_as(grad)

    def _image_low_frequency_patch_score(self, clean_pixels: torch.Tensor, num_patches: int) -> torch.Tensor:
        from utils import image_2d_fft_low_high_maps

        fft_maps = image_2d_fft_low_high_maps(
            clean_pixels,
            cutoff_ratio=self.spectral_cutoff,
            transition_ratio=self.spectral_transition,
        )
        low_ratio = fft_maps["low_ratio"]
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square.")

        bsz, height, width = low_ratio.shape
        pooled = F.adaptive_avg_pool2d(low_ratio.view(bsz, 1, height, width), (grid_size, grid_size))
        return self._normalize_weights(pooled.view(bsz, -1))

    def _build_clean_guides(
        self,
        images: torch.Tensor,
        clean_pixels: torch.Tensor,
    ) -> tuple[list[int], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        with torch.no_grad():
            _logits, attn_logits_list, token_list = self.model(
                images,
                return_attn=True,
                return_tokens=True,
            )

        layer_indices = self._resolve_layers(self.layers, len(token_list))
        first_layer = layer_indices[0]
        num_patches = token_list[first_layer].size(1) - 1
        image_low = self._image_low_frequency_patch_score(clean_pixels, num_patches)

        fg_masks: dict[int, torch.Tensor] = {}
        anchor_masks: dict[int, torch.Tensor] = {}
        for layer_idx in layer_indices:
            tokens = token_list[layer_idx].detach()
            patches = tokens[:, 1:, :]
            patch_norm = self._normalize_weights(patches.norm(dim=-1))
            fft_score = last_vit_stable_patch_frequency(
                tokens=tokens,
                topk=self.fft_topk,
                has_cls_token=True,
            ).detach()
            fft_score = self._normalize_weights(fft_score)

            attn = F.softmax(attn_logits_list[layer_idx].detach(), dim=-1)
            cls_attn = self._normalize_weights(attn[:, :, 0, 1:].mean(dim=1))

            fg_score = self._normalize_weights(cls_attn + patch_norm + fft_score)
            fg_mask = self._select_top_mask(fg_score, self.fg_top_ratio)

            anchor_score = self._normalize_weights((1.0 - fg_score) + image_low + (1.0 - cls_attn))
            anchor_score = anchor_score.masked_fill(fg_mask, -1.0)
            anchor_mask = self._select_top_mask(anchor_score, self.anchor_top_ratio)

            fg_masks[layer_idx] = fg_mask.detach()
            anchor_masks[layer_idx] = anchor_mask.detach()

        return layer_indices, fg_masks, anchor_masks

    def _anchor_loss(
        self,
        attn_logits_list: list[torch.Tensor],
        token_list: list[torch.Tensor],
        layer_indices: list[int],
        fg_masks: dict[int, torch.Tensor],
        anchor_masks: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        layer_terms = []
        for layer_idx in layer_indices:
            tokens = token_list[layer_idx]
            cls = tokens[:, 0, :]
            patches = tokens[:, 1:, :]
            fg_mask = fg_masks[layer_idx].to(device=patches.device)
            anchor_mask = anchor_masks[layer_idx].to(device=patches.device)

            attn = F.softmax(attn_logits_list[layer_idx], dim=-1)
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)
            route_score = (
                (cls_attn * anchor_mask.to(cls_attn.dtype)).sum(dim=1)
                - (cls_attn * fg_mask.to(cls_attn.dtype)).sum(dim=1)
            )

            patch_norm = patches.norm(dim=-1)
            norm_score = (
                self._masked_mean(patch_norm, anchor_mask)
                - self._masked_mean(patch_norm, fg_mask)
            )

            anchor_mean = self._masked_mean(patches, anchor_mask)
            fg_mean = self._masked_mean(patches, fg_mask)
            align_score = (
                F.cosine_similarity(cls, anchor_mean, dim=-1)
                - F.cosine_similarity(cls, fg_mean, dim=-1)
            )

            score = (
                self._scale_term(route_score)
                + self._scale_term(norm_score)
                + self._scale_term(align_score)
            )
            layer_terms.append(score.mean())

        return torch.stack(layer_terms).mean()

    def _combine_grads(
        self,
        g_ce: torch.Tensor,
        g_anchor: torch.Tensor,
        anchor_weight: float | None = None,
    ) -> tuple[torch.Tensor, float]:
        if self.grad_combine == "ce":
            return g_ce, 1.0
        if anchor_weight is None:
            anchor_weight = self.lambda_anchor

        g_ce = self._l2_normalize_grad(g_ce, self.grad_l2_norm)
        g_anchor = self._l2_normalize_grad(g_anchor, self.grad_l2_norm)
        if self.grad_combine == "sum":
            flat_ce = g_ce.reshape(g_ce.size(0), -1)
            flat_anchor = g_anchor.reshape(g_anchor.size(0), -1)
            cos = (flat_ce * flat_anchor).sum(dim=1) / (
                flat_ce.norm(dim=1).clamp_min(1e-12) * flat_anchor.norm(dim=1).clamp_min(1e-12)
            )
            return g_ce + anchor_weight * g_anchor, cos.mean().item()

        batch_size = g_ce.size(0)
        flat_ce = g_ce.reshape(batch_size, -1)
        flat_anchor = g_anchor.reshape(batch_size, -1)
        dot = (flat_ce * flat_anchor).sum(dim=1)
        ce_norm_sq = flat_ce.pow(2).sum(dim=1).clamp_min(1e-12)
        anchor_norm = flat_anchor.norm(dim=1).clamp_min(1e-12)
        ce_norm = flat_ce.norm(dim=1).clamp_min(1e-12)
        cos = dot / (ce_norm * anchor_norm)
        conflict = cos < 0
        projection = (dot / ce_norm_sq).unsqueeze(1) * flat_ce
        flat_anchor = torch.where(conflict[:, None], flat_anchor - projection, flat_anchor)
        return (flat_ce + anchor_weight * flat_anchor).reshape_as(g_ce), cos.mean().item()

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

    def _build_stable_attention_modulation_map(
        self,
        images: torch.Tensor,
        img_size: int,
        expand_shared: bool = False,
    ) -> torch.Tensor:
        token_map = self._build_stable_attention_token_map(images, expand_shared=expand_shared)
        num_patches = token_map.size(1)
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square.")
        grid = token_map.view(token_map.size(0), 1, grid_size, grid_size)
        return F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False).detach()

    def _guide_response_loss(
        self,
        attn_logits_list: list[torch.Tensor],
        guide_token_map: torch.Tensor,
        pre_softmax: bool = False,
    ) -> torch.Tensor:
        layer_indices = self._resolve_layers(self.layers, len(attn_logits_list))
        layer_responses = []
        for layer_idx in layer_indices:
            if pre_softmax:
                cls_attn = attn_logits_list[layer_idx][:, :, 0, 1:].mean(dim=1)
                cls_attn = self._normalize_weights(cls_attn)
            else:
                attn = F.softmax(attn_logits_list[layer_idx], dim=-1)
                cls_attn = attn[:, :, 0, 1:].mean(dim=1)
                cls_attn = cls_attn / cls_attn.sum(dim=1, keepdim=True).clamp_min(1e-12)
            if cls_attn.size(1) == guide_token_map.size(1):
                layer_responses.append(cls_attn)
        if not layer_responses:
            raise ValueError("No compatible adversarial CLS attention maps for guide response loss.")
        response = torch.stack(layer_responses).mean(dim=0)
        guide = guide_token_map.to(response.device, response.dtype)
        guide = guide / guide.sum(dim=1, keepdim=True).clamp_min(1e-12)
        response_loss = (response * guide).sum(dim=1).mean()
        return -response_loss if pre_softmax else response_loss

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
        if mode == "freq":
            pooled = F.avg_pool2d(pixels, kernel_size=9, stride=1, padding=4)
            noise = F.avg_pool2d(torch.rand_like(pixels), kernel_size=9, stride=1, padding=4)
            corrupt = 0.7 * pooled + 0.3 * noise
            background = pixels * (1.0 - strength) + corrupt * strength
            return torch.clamp(pixels * guide + background * (1.0 - guide), 0.0, 1.0)
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

    def _build_clean_shared_features(
        self,
        images: torch.Tensor,
        guide_token_map: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        with torch.no_grad():
            _logits, token_list = self.model(images, return_tokens=True)
        layer_indices = self._resolve_layers(self.layers, len(token_list))
        clean_features = {}
        for layer_idx in layer_indices:
            patches = token_list[layer_idx][:, 1:, :].detach()
            if patches.size(1) == guide_token_map.size(1):
                clean_features[layer_idx] = patches
        if not clean_features:
            raise ValueError("No compatible clean patch features for guide_aug_feature.")
        return clean_features

    def _guide_feature_disrupt_loss(
        self,
        token_list_adv: list[torch.Tensor],
        clean_features: dict[int, torch.Tensor],
        guide_token_map: torch.Tensor,
    ) -> torch.Tensor:
        terms = []
        guide = guide_token_map / guide_token_map.sum(dim=1, keepdim=True).clamp_min(1e-12)
        for layer_idx, clean_patches in clean_features.items():
            adv_patches = token_list_adv[layer_idx][:, 1:, :]
            if adv_patches.size(1) != guide.size(1):
                continue
            clean = clean_patches.to(adv_patches.device, adv_patches.dtype)
            weights = guide.to(adv_patches.device, adv_patches.dtype)
            cos = F.cosine_similarity(clean, adv_patches, dim=-1)
            terms.append((weights * (1.0 - cos)).sum(dim=1).mean())
        if not terms:
            raise ValueError("No compatible adversarial patch features for guide_aug_feature.")
        return torch.stack(terms).mean()

    def _guide_aug_feature_loss(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        guide_token_map: torch.Tensor,
        guide_pixel_map: torch.Tensor,
        clean_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        terms = []
        for scale_idx in range(self.si_scales):
            scale = float(2 ** scale_idx)
            for _eot_idx in range(self.eot_iter):
                for copy_idx in range(self.guide_aug_copies):
                    aug_pixels = self._guide_augmented_pixels(pixels, guide_pixel_map, copy_idx)
                    logits_adv, token_list_adv = self.model(
                        self._input_diversity(self._normalize(aug_pixels) / scale),
                        return_tokens=True,
                    )
                    ce = F.cross_entropy(logits_adv, labels)
                    feature = self._guide_feature_disrupt_loss(
                        token_list_adv=token_list_adv,
                        clean_features=clean_features,
                        guide_token_map=guide_token_map,
                    )
                    terms.append(ce + self.feature_loss_weight * feature)
        return torch.stack(terms).mean()

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

    def _build_anchor_modulation_map(
        self,
        fg_masks: dict[int, torch.Tensor],
        anchor_masks: dict[int, torch.Tensor],
        layer_indices: list[int],
        img_size: int,
    ) -> torch.Tensor:
        anchor = torch.stack([anchor_masks[layer_idx].float() for layer_idx in layer_indices]).mean(dim=0)
        foreground = torch.stack([fg_masks[layer_idx].float() for layer_idx in layer_indices]).mean(dim=0)
        token_map = self.anchor_mod_alpha * anchor + self.fg_mod_alpha * foreground
        if self.anchor_mod_power != 1.0:
            token_map = token_map ** self.anchor_mod_power
        token_map = self._normalize_weights(token_map)
        batch_size, num_patches = token_map.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise ValueError(f"Patch count {num_patches} is not a square.")
        grid = token_map.view(batch_size, 1, grid_size, grid_size)
        return F.interpolate(grid, size=(img_size, img_size), mode="bilinear", align_corners=False).detach()

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        clean_pixels = self._denormalize(images).detach()
        layer_indices = []
        fg_masks = {}
        anchor_masks = {}
        anchor_modulation = None
        guide_token_map = None
        clean_guide_token_map = None
        guide_pixel_map = None
        clean_shared_features = None
        if self.grad_combine in ("anchor_modulate", "pcgrad_asymmetric", "sum"):
            layer_indices, fg_masks, anchor_masks = self._build_clean_guides(images, clean_pixels)
        if self.grad_combine == "anchor_modulate":
            anchor_modulation = self._build_anchor_modulation_map(
                fg_masks=fg_masks,
                anchor_masks=anchor_masks,
                layer_indices=layer_indices,
                img_size=clean_pixels.size(-1),
            )
        elif self.grad_combine in ("stable_attention", "expanded_stable_attention"):
            anchor_modulation = self._build_stable_attention_modulation_map(
                images,
                clean_pixels.size(-1),
                expand_shared=self.grad_combine == "expanded_stable_attention",
            )
        elif self.grad_combine in ("guide_response", "dynamic_guide_response", "guide_aug_ce", "guide_aug_feature", "guide_qk_response") or self.guide_aug:
            guide_token_map = self._build_stable_attention_token_map(images, expand_shared=True)
            clean_guide_token_map = guide_token_map.clone().detach()
            guide_pixel_map = self._token_map_to_pixel_map(guide_token_map, clean_pixels.size(-1))
            if self.grad_combine == "guide_aug_feature":
                clean_shared_features = self._build_clean_shared_features(images, guide_token_map)
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)
        self._last_cosine_log = []
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
            if (self.guide_dynamic or self.grad_combine == "dynamic_guide_response") and guide_token_map is not None:
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
            if self.grad_combine in ("anchor_modulate", "stable_attention", "expanded_stable_attention", "guide_response", "dynamic_guide_response", "guide_aug_ce", "guide_aug_feature", "guide_qk_response"):
                if self.grad_combine == "guide_aug_ce":
                    ce_loss = self._guide_aug_ce_loss(grad_pixels, labels, guide_pixel_map)
                elif self.grad_combine == "guide_aug_feature":
                    ce_loss = self._guide_aug_feature_loss(
                        grad_pixels,
                        labels,
                        guide_token_map,
                        guide_pixel_map,
                        clean_shared_features,
                    )
                else:
                    ce_terms = []
                    source_models = (self.model, *self.ensemble_models) if self.grad_combine == "anchor_modulate" else (self.model,)
                    for scale_idx in range(self.si_scales):
                        scale = float(2 ** scale_idx)
                        for _eot_idx in range(self.eot_iter):
                            for source_model in source_models:
                                logits_adv = source_model(
                                    self._input_diversity(norm_adv / scale),
                                    return_attn=False,
                                )
                                ce_terms.append(F.cross_entropy(logits_adv, labels))
                    ce_loss = torch.stack(ce_terms).mean()
                attn_logits_adv = None
                token_list_adv = None
            else:
                logits_adv, attn_logits_adv, token_list_adv = self.model(
                    self._input_diversity(self._normalize(grad_pixels)),
                    return_attn=True,
                    return_tokens=True,
                )
                ce_loss = F.cross_entropy(logits_adv, labels)

            anchor_weight = self._anchor_weight_for_step(step_idx)
            response_score = None
            if self.grad_combine in ("guide_response", "guide_qk_response") and anchor_weight > 0.0:
                _logits_resp, attn_logits_resp = self.model(norm_adv, return_attn=True)
                response_score = self._guide_response_loss(
                    attn_logits_resp,
                    guide_token_map,
                    pre_softmax=self.grad_combine == "guide_qk_response",
                )
                ce_loss = ce_loss + anchor_weight * response_score
            g_ce = torch.autograd.grad(ce_loss, grad_pixels, retain_graph=True)[0]

            if anchor_weight <= 0.0:
                grad, avg_cos = g_ce, 1.0
            elif self.grad_combine in ("anchor_modulate", "stable_attention", "expanded_stable_attention"):
                grad = g_ce * (1.0 + anchor_weight * anchor_modulation.to(g_ce.device, g_ce.dtype))
                avg_cos = 1.0
            elif self.grad_combine in ("guide_response", "dynamic_guide_response", "guide_aug_ce", "guide_aug_feature", "guide_qk_response"):
                grad, avg_cos = g_ce, 1.0
            else:
                anchor_loss = self._anchor_loss(
                    attn_logits_list=attn_logits_adv,
                    token_list=token_list_adv,
                    layer_indices=layer_indices,
                    fg_masks=fg_masks,
                    anchor_masks=anchor_masks,
                )
                g_anchor = torch.autograd.grad(anchor_loss, grad_pixels, retain_graph=False)[0]
                grad, avg_cos = self._combine_grads(g_ce, g_anchor, anchor_weight=anchor_weight)

            self._last_cosine_log.append((step_idx, avg_cos))
            grad = self._smooth_grad(self._normalize_grad(grad))
            momentum = self.decay * momentum + grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                delta = torch.clamp(self._smooth_perturbation(delta), -self.epsilon, self.epsilon)
                if self.lazy_spectral_delta and step_idx + 1 >= max(self.steps // 2, self.anchor_start_step):
                    delta = torch.clamp(self._spectral_filter_delta(delta), -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

        cos_vals = [c for _, c in self._last_cosine_log]
        print(f"  [LazyAgg] conflicts={sum(1 for c in cos_vals if c < 0)}/{len(cos_vals)} mean_cos={sum(cos_vals)/len(cos_vals):.4f}")
        return self._normalize(adv_pixels)


class FFTForwardPassAttacker(MIFGSMAttacker):
    """
    FFT-guided forward-pass-aware attack with single CE gradient.

    Modes (mutually exclusive, set via *mode*):
      - "hardmask":  zero-out CE gradient on background (low-FFT) pixels
      - "softmask":  continuous FFT weighting of CE gradient
      - "spectral":  low-pass filter the *perturbation* each step
      - "combined":  softmask + spectral

    Core improvement over FFT-CC: no extra feature losses → no gradient interference.
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 10,
        decay: float = 1.0,
        layers: tuple[int, ...] = (-6, -5, -4, -3, -2, -1),
        fft_topk: int = 1,
        lambda_mask: float = 1.0,
        mode: str = "combined",
        spectral_cutoff: float = 0.15,
        spectral_transition: float = 0.04,
        ti_sigma: float = 0.0,
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
        valid_modes = ("hardmask", "softmask", "spectral", "combined")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}.")

        self.layers = tuple(int(l) for l in layers)
        self.fft_topk = int(fft_topk)
        self.lambda_mask = float(lambda_mask)
        self.mode = mode
        self.spectral_cutoff = float(spectral_cutoff)
        self.spectral_transition = float(spectral_transition)
        self.ti_sigma = float(ti_sigma)

        self._ti_kernel: torch.Tensor | None = None
        if self.ti_sigma > 0:
            self._ti_kernel = self._build_ti_kernel(self.ti_sigma)

    # ------------------------------------------------------------------
    # TI-FGSM helpers
    # ------------------------------------------------------------------
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
        kernel = self._ti_kernel.to(grad.device, grad.dtype)
        kernel = kernel.repeat(grad.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        padded = F.pad(grad, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(padded, kernel, groups=grad.size(1))

    # ------------------------------------------------------------------
    # Layer resolution  (shared logic from FFTCCAttacker)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_layers(layers: tuple[int, ...], num_layers: int) -> list[int]:
        resolved: list[int] = []
        for layer in layers:
            idx = layer if layer >= 0 else num_layers + layer
            if idx < 0 or idx >= num_layers:
                raise ValueError(
                    f"Invalid layer index {layer}; model has {num_layers} layers."
                )
            if idx not in resolved:
                resolved.append(idx)
        return resolved

    @staticmethod
    def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
        min_vals = weights.min(dim=1, keepdim=True).values
        max_vals = weights.max(dim=1, keepdim=True).values
        return (weights - min_vals) / (max_vals - min_vals).clamp_min(1e-12)

    # ------------------------------------------------------------------
    # FFT mask builder
    # ------------------------------------------------------------------
    def _build_fft_mask(
        self,
        images: torch.Tensor,
        mode: str,
    ) -> tuple[list[int], torch.Tensor | None]:
        """Compute per-pixel mask from FFT foreground scores.

        Returns:
            layer_indices: list of resolved layer indices
            mask: [B, 1, H, W] in [0,1] or None for spectral-only mode
        """
        if mode == "spectral":
            # No spatial mask needed
            with torch.no_grad():
                _logits, token_list = self.model(images, return_tokens=True)
            layer_indices = self._resolve_layers(self.layers, len(token_list))
            return layer_indices, None

        with torch.no_grad():
            _logits, token_list = self.model(images, return_tokens=True)

        layer_indices = self._resolve_layers(self.layers, len(token_list))

        from utils import last_vit_stable_patch_frequency

        # Average FFT foreground scores across selected layers
        all_scores: list[torch.Tensor] = []
        for layer_idx in layer_indices:
            tokens = token_list[layer_idx].detach()
            scores = last_vit_stable_patch_frequency(
                tokens=tokens, topk=self.fft_topk, has_cls_token=True,
            ).detach()  # [B, N_patches]
            all_scores.append(scores)
        avg = torch.stack(all_scores).mean(dim=0)  # [B, N_patches]
        avg = self._normalize_weights(avg)

        B, N = avg.shape
        grid_size = int(N ** 0.5)
        if grid_size * grid_size != N:
            raise ValueError(f"Patch count {N} not a square.")

        grid = avg.reshape(B, 1, grid_size, grid_size)  # [B, 1, 14, 14]

        if mode == "hardmask":
            # Top-K binary mask: keep top fraction of patches as foreground
            k = max(1, int(grid_size * grid_size * 0.3))
            flat = grid.reshape(B, -1)
            threshold = flat.kthvalue(grid_size * grid_size - k + 1, dim=1).values
            mask = (flat >= threshold.unsqueeze(1)).float()
            mask = mask.reshape(B, 1, grid_size, grid_size)
        else:
            # softmask / combined: use continuous scores
            mask = grid

        # Upsample to image space
        img_size = max(images.shape[2], images.shape[3])
        mask = F.interpolate(
            mask, size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        )  # [B, 1, H, W]
        return layer_indices, mask

    # ------------------------------------------------------------------
    # Spectral perturbation filtering
    # ------------------------------------------------------------------
    def _spectral_filter_delta(
        self, delta: torch.Tensor,
    ) -> torch.Tensor:
        """Low-pass filter the perturbation delta to remove high-freq overfitting."""
        from utils import image_2d_fft_low_high_maps

        # image_2d_fft expects [C,H,W] or [B,C,H,W]
        maps = image_2d_fft_low_high_maps(
            delta,
            cutoff_ratio=self.spectral_cutoff,
            transition_ratio=self.spectral_transition,
        )
        # low_ratio is per-pixel from grayscale conversion
        # We use it as a per-pixel filter weight
        low_ratio = maps["low_ratio"]  # [B, H, W] or [H, W]
        img_ref = delta if delta.ndim == 4 else delta.unsqueeze(0)
        filter_weights = low_ratio.to(delta.dtype)
        if filter_weights.ndim == 2:
            filter_weights = filter_weights.unsqueeze(0)
        filter_weights = filter_weights.unsqueeze(1)  # [B, 1, H, W]

        # Blend original + low-passed
        # delta_low = delta * filter_weights  (approximate low-pass)
        return delta * (0.5 + 0.5 * filter_weights)

    # ------------------------------------------------------------------
    # Attack loop
    # ------------------------------------------------------------------
    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Build FFT mask from clean forward pass
        layer_indices, fft_mask = self._build_fft_mask(images, self.mode)

        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        for _step in range(self.steps):
            adv_pixels = adv_pixels.detach().requires_grad_(True)

            logits = self.model(
                self._normalize(adv_pixels), return_attn=False,
            )
            loss = F.cross_entropy(logits, labels)

            grad = torch.autograd.grad(loss, adv_pixels)[0]

            # Apply spatial mask if applicable
            if fft_mask is not None:
                mask = fft_mask.to(grad.device, grad.dtype)
                if self.mode == "hardmask":
                    grad = grad * mask
                else:  # softmask / combined
                    grad = grad * (1.0 + self.lambda_mask * mask)

            # Normalize and momentum
            grad = self._normalize_grad(grad)
            grad = self._smooth_grad(grad)
            momentum = self.decay * momentum + grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * momentum.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)

                # Spectral filtering of perturbation
                if self.mode in ("spectral", "combined"):
                    delta = self._spectral_filter_delta(delta)

                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

        return self._normalize(adv_pixels)
