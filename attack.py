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
