import math

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD


_LOWMID_GRAD_FFT_BANDS = (0.0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 1.0)


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


class LMDSSAttacker:
    """
    LMDSS transfer attack with whole-image forward augmentations,
    DIM, TI smoothing, MI/NI updates, and low/mid-frequency DSS tuning.
    """

    def __init__(
        self,
        model,
        epsilon: float = 16.0 / 255.0,
        step_size: float | None = None,
        steps: int = 20,
        ti_sigma: float = 0.0,
        input_diversity: bool = False,
        dim_resize_range: tuple[float, float] = (0.85, 1.0),
        dim_mode: str = "full-random",
        use_momentum: bool = True,
        momentum_decay: float = 1.0,
        nesterov: bool = False,
        guide_aug: bool = False,
        guide_aug_methods: tuple[str, ...] = ("dropout",),
        guide_aug_copies: int = 3,
        guide_aug_strength: float = 0.2,
        dim_adjoint_echo: bool = False,
        lowmid_grad_tuning: bool = False,
        lowmid_grad_rotation_strength: float = 0.5,
        lowmid_grad_preserve_norm: bool = True,
        lowmid_dss_filter: bool = False,
        lowmid_dss_consistency: str = "sign",
        lowmid_dss_agreement_threshold: float = 0.67,
        attack_loss: str = "logits",
        feature_layer: int = -2,
        feature_scope: str = "block",
        device: torch.device | None = None,
    ) -> None:
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}.")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        if step_size is not None and step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        if ti_sigma < 0:
            raise ValueError(f"ti_sigma must be non-negative, got {ti_sigma}.")
        if nesterov and not use_momentum:
            raise ValueError("--ni requires MI because Nesterov lookahead depends on momentum.")
        if not 0 <= lowmid_grad_rotation_strength < 1:
            raise ValueError(
                f"lowmid_grad_rotation_strength must be in [0, 1), got {lowmid_grad_rotation_strength}."
            )
        if not isinstance(lowmid_grad_preserve_norm, bool):
            raise ValueError(
                f"lowmid_grad_preserve_norm must be bool, got {type(lowmid_grad_preserve_norm).__name__}."
            )
        if lowmid_dss_consistency not in ("sign", "cos"):
            raise ValueError(
                f"lowmid_dss_consistency must be 'sign' or 'cos', got {lowmid_dss_consistency!r}."
            )
        if not 0.0 <= lowmid_dss_agreement_threshold <= 1.0:
            raise ValueError(
                f"lowmid_dss_agreement_threshold must be in [0, 1], got {lowmid_dss_agreement_threshold}."
            )

        if attack_loss not in ("logits", "feature"):
            raise ValueError(f"attack_loss must be 'logits' or 'feature', got {attack_loss!r}.")
        if feature_scope not in ("block", "stage"):
            raise ValueError(f"feature_scope must be 'block' or 'stage', got {feature_scope!r}.")

        self.model = model
        self.model.eval()
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.step_size = float(step_size) if step_size is not None else self.epsilon / self.steps
        self.decay = float(momentum_decay)
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
        self.ti_sigma = float(ti_sigma)
        self.input_diversity = bool(input_diversity)
        self.dim_resize_range = tuple(float(r) for r in dim_resize_range)
        valid_dim_modes = ("full-random", "forward-only", "backward-only", "full-fixed", "backward-fixed")
        if dim_mode not in valid_dim_modes:
            raise ValueError(f"dim_mode must be one of {valid_dim_modes}, got {dim_mode!r}.")
        self.dim_mode = dim_mode
        self._fixed_dim_params = None
        self.use_momentum = bool(use_momentum)
        self.nesterov = bool(nesterov)
        self.dim_adjoint_echo = bool(dim_adjoint_echo)
        self.lowmid_grad_tuning = bool(lowmid_grad_tuning)
        self.lowmid_grad_rotation_strength = float(lowmid_grad_rotation_strength)
        self.lowmid_grad_preserve_norm = lowmid_grad_preserve_norm
        self.lowmid_dss_filter = bool(lowmid_dss_filter)
        self.lowmid_dss_consistency = lowmid_dss_consistency
        self.lowmid_dss_agreement_threshold = float(lowmid_dss_agreement_threshold)
        self.attack_loss = attack_loss
        self.feature_layer = int(feature_layer)
        self.feature_scope = feature_scope
        self.guide_aug = bool(guide_aug)
        self.guide_aug_methods = tuple(str(method).strip() for method in guide_aug_methods if str(method).strip())
        valid_guide_aug_methods = (
            "dropout",
            "jitter",
            "freq",
            "dim_resonance",
            "lowmid_shift",
            "white_noise",
            "antithetic_transport",
            "natural_spectrum_transport",
            "antithetic_filter_bank",
            "multiscale_adjoint_ensemble",
            "orthogonal_photometric_ensemble",
            "orthogonal_spherical_smoothing",
            "antithetic_jitter_cubature",
            "feature_trajectory_dropout",
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
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma) if self.ti_sigma > 0 else None

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    @staticmethod
    def _normalize_grad(grad: torch.Tensor) -> torch.Tensor:
        denom = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return grad / denom

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
        return LMDSSAttacker._build_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

    def _apply_depthwise_kernel(self, pixels: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        kernel = kernel.to(pixels.device, pixels.dtype).repeat(pixels.size(1), 1, 1, 1)
        pad = kernel.size(2) // 2
        return F.conv2d(F.pad(pixels, (pad, pad, pad, pad), mode="reflect"), kernel, groups=pixels.size(1))

    def _smooth_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self._ti_kernel is None or self.ti_sigma <= 0:
            return grad
        return self._apply_depthwise_kernel(grad, self._ti_kernel)

    @staticmethod
    def _fft_grad_mask(height: int, width: int, band: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
        if not 0 <= band < len(_LOWMID_GRAD_FFT_BANDS) - 1:
            raise ValueError(f"band must be in [0, {len(_LOWMID_GRAD_FFT_BANDS) - 2}], got {band}.")
        fy = torch.fft.fftfreq(height, device=device, dtype=dtype).view(height, 1)
        fx = torch.fft.fftfreq(width, device=device, dtype=dtype).view(1, width)
        radius = torch.sqrt((fy / 0.5).square() + (fx / 0.5).square()) / (2.0 ** 0.5)
        lo, hi = _LOWMID_GRAD_FFT_BANDS[band], _LOWMID_GRAD_FFT_BANDS[band + 1]
        return (radius >= lo) & (radius <= hi) if band == 0 else (radius > lo) & (radius <= hi)

    @staticmethod
    def _fft_project_grad(grad: torch.Tensor, band: int) -> torch.Tensor:
        work = grad if grad.dtype == torch.float64 else grad.float()
        mask = LMDSSAttacker._fft_grad_mask(
            grad.size(-2),
            grad.size(-1),
            band,
            device=grad.device,
            dtype=work.dtype,
        )
        freq = torch.fft.fft2(work, dim=(-2, -1), norm="ortho")
        return torch.fft.ifft2(freq * mask, dim=(-2, -1), norm="ortho").real.to(grad.dtype)

    def _lowmid_high_components(self, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lowmid = sum((self._fft_project_grad(grad, band) for band in range(6)), torch.zeros_like(grad))
        high = sum((self._fft_project_grad(grad, band) for band in range(6, 8)), torch.zeros_like(grad))
        return lowmid, high

    def _apply_lowmid_dss_filter(
        self,
        grad: torch.Tensor,
        term_grads: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor:
        if not self.lowmid_dss_filter:
            return grad
        if not term_grads:
            return grad

        lowmid, high = self._lowmid_high_components(grad)
        term_lowmid = torch.stack([self._lowmid_high_components(term)[0] for term in term_grads], dim=0)
        ref = term_lowmid.mean(dim=0)
        eps = 1e-12

        if self.lowmid_dss_consistency == "sign":
            agreement = term_lowmid.sign().eq(ref.sign().unsqueeze(0)).to(grad.dtype).mean(dim=0)
            mask = (agreement >= self.lowmid_dss_agreement_threshold).to(grad.dtype)
            filtered_lowmid = lowmid * mask
        else:
            ref_norm = ref.flatten(1).norm(p=2, dim=1).clamp_min(eps)
            term_norm = term_lowmid.flatten(2).norm(p=2, dim=2).clamp_min(eps)
            cos = (term_lowmid * ref.unsqueeze(0)).flatten(2).sum(2) / (term_norm * ref_norm.unsqueeze(0))
            gate = ((cos.mean(dim=0) + 1.0) * 0.5).clamp(0.0, 1.0).view(-1, 1, 1, 1)
            filtered_lowmid = lowmid * gate

        return filtered_lowmid + high

    def _tune_lowmid_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        if not self.lowmid_grad_tuning:
            return grad
        lowmid, high = self._lowmid_high_components(grad)

        eps = 1e-12
        lowmid_norm = lowmid.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        high_norm = high.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        valid = (lowmid_norm > eps) & (high_norm > eps)
        if not valid.any():
            return grad

        strength = torch.as_tensor(self.lowmid_grad_rotation_strength, device=grad.device, dtype=lowmid_norm.dtype)
        theta = strength * torch.atan2(high_norm, lowmid_norm)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotated_lowmid_coeff = lowmid_norm * cos_theta + high_norm * sin_theta
        rotated_high_coeff = -lowmid_norm * sin_theta + high_norm * cos_theta

        lowmid_unit = lowmid / lowmid_norm.clamp_min(eps)
        high_unit = high / high_norm.clamp_min(eps)
        rotated = rotated_lowmid_coeff * lowmid_unit + rotated_high_coeff * high_unit
        tuned = torch.where(valid, rotated, grad)

        if self.lowmid_grad_preserve_norm:
            grad_norm = grad.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
            tuned_norm = tuned.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(eps)
            tuned = tuned * (grad_norm / tuned_norm)
        return tuned

    def reset_fixed_dim(self) -> None:
        self._fixed_dim_params = None

    def _sample_dim_params(self, images: torch.Tensor) -> tuple[int, int, int, int]:
        _batch_size, _channels, height, width = images.shape
        lo, hi = self.dim_resize_range
        scale = lo + (hi - lo) * torch.rand(1, device=images.device)
        new_h = max(1, min(height, int(round(height * scale.item()))))
        new_w = max(1, min(width, int(round(width * scale.item()))))
        pad_h, pad_w = height - new_h, width - new_w
        top = torch.randint(0, pad_h + 1, (1,), device=images.device).item() if pad_h > 0 else 0
        left = torch.randint(0, pad_w + 1, (1,), device=images.device).item() if pad_w > 0 else 0
        return new_h, new_w, top, left

    @staticmethod
    def _apply_dim_transform(images: torch.Tensor, params: tuple[int, int, int, int]) -> torch.Tensor:
        _batch_size, _channels, height, width = images.shape
        new_h, new_w, top, left = params
        resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return F.pad(resized, (left, width - new_w - left, top, height - new_h - top), value=0.0)

    def _input_diversity(self, images: torch.Tensor) -> torch.Tensor:
        if not self.input_diversity:
            return images
        if self.dim_mode.endswith("fixed"):
            if self._fixed_dim_params is None:
                self._fixed_dim_params = self._sample_dim_params(images)
            params = self._fixed_dim_params
        else:
            params = self._sample_dim_params(images)
        transformed = self._apply_dim_transform(images, params)
        if self.dim_mode == "forward-only":
            return images + (transformed - images).detach()
        if self.dim_mode in ("backward-only", "backward-fixed"):
            return transformed + (images - transformed).detach()
        return transformed

    def _dim_adjoint_restore_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        _batch, _channels, height, width = pixels.shape
        new_h, new_w, top, left = self._sample_dim_params(pixels)
        transformed = self._apply_dim_transform(pixels, (new_h, new_w, top, left))
        cropped = transformed[..., top:top + new_h, left:left + new_w]
        return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)

    def _dim_resonance_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Boost the image subspace emphasized by the DIM adjoint.

        For a sampled DIM resize/pad operator J, the source gradient contains
        J^T grad L(f(Jx), y). Applying J^T J to the image and adding that
        non-DC component creates an augmentation whose local Jacobian is
        approximately I + gamma C J^T J, so the backward path amplifies the
        same low/mid-frequency subspace that random DIM preserves.
        """
        restored = self._dim_adjoint_restore_pixels(pixels)
        non_dc = restored - restored.mean(dim=(2, 3), keepdim=True)
        return torch.clamp(pixels + self.guide_aug_strength * non_dc, 0.0, 1.0)

    def _dim_adjoint_echo_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Forward-identity DIM-adjoint echo augmentation.

        The returned tensor has the same forward value as ``pixels`` but its
        backward Jacobian is approximately I + gamma J^T J for a sampled DIM
        resize/pad operator J. This directly probes whether amplifying the DIM
        adjoint low/mid subspace improves transferable gradients without
        moving the loss evaluation point in image space.
        """
        restored = self._dim_adjoint_restore_pixels(pixels)
        augmented = pixels + self.guide_aug_strength * restored
        return (augmented + (pixels - augmented).detach()).to(pixels.dtype)

    def _white_noise_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(pixels, dtype=pixels.dtype)
        return torch.clamp(pixels + self.guide_aug_strength * noise, 0.0, 1.0)

    def _lowmid_shift_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        lowmid = F.avg_pool2d(pixels, kernel_size=11, stride=1, padding=5)
        max_shift = max(1, int(round(min(pixels.size(-2), pixels.size(-1)) * 0.08)))
        shift_y = int(torch.randint(-max_shift, max_shift + 1, (1,), device=pixels.device).item())
        shift_x = int(torch.randint(-max_shift, max_shift + 1, (1,), device=pixels.device).item())
        shifted_lowmid = torch.roll(lowmid, shifts=(shift_y, shift_x), dims=(-2, -1))
        augmented = pixels + self.guide_aug_strength * (shifted_lowmid - lowmid)
        return torch.clamp(augmented, 0.0, 1.0)

    def _sample_transport_field(self, pixels: torch.Tensor, grid_size: int) -> torch.Tensor:
        """Sample a smooth, zero-mean displacement field in normalized coordinates."""
        batch, _channels, height, width = pixels.shape
        coarse = torch.randn(batch, 2, grid_size, grid_size, device=pixels.device, dtype=pixels.dtype)
        field = F.interpolate(coarse, size=(height, width), mode="bicubic", align_corners=False)
        field = field - field.mean(dim=(2, 3), keepdim=True)
        field = field / field.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
        max_pixels = self.guide_aug_strength * 0.08 * float(min(height, width))
        field[:, 0] *= 2.0 * max_pixels / float(width)
        field[:, 1] *= 2.0 * max_pixels / float(height)
        return field

    @staticmethod
    def _transport_pixels(pixels: torch.Tensor, field: torch.Tensor, direction: float) -> torch.Tensor:
        batch, _channels, height, width = pixels.shape
        yy = (torch.arange(height, device=pixels.device, dtype=pixels.dtype) + 0.5) * (2.0 / height) - 1.0
        xx = (torch.arange(width, device=pixels.device, dtype=pixels.dtype) + 0.5) * (2.0 / width) - 1.0
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        base = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        displacement = field.permute(0, 2, 3, 1) * float(direction)
        return F.grid_sample(
            pixels, base + displacement, mode="bilinear", padding_mode="reflection", align_corners=False
        ).clamp(0.0, 1.0)

    @staticmethod
    def _blend_guide_augmentation(pixels: torch.Tensor, augmented: torch.Tensor) -> torch.Tensor:
        del pixels
        return torch.clamp(augmented, 0.0, 1.0)

    def _iter_antithetic_transport_pixels(
        self,
        pixels: torch.Tensor,
        copies: int,
    ):
        """Yield +/- local transports, cancelling first-order sampling error."""
        pair_count = copies // 2
        grid_sizes = (3, 5, 7, 9)
        for pair_index in range(pair_count):
            field = self._sample_transport_field(pixels, grid_sizes[pair_index % len(grid_sizes)])
            for direction in (1.0, -1.0):
                augmented = self._transport_pixels(pixels, field, direction)
                yield self._blend_guide_augmentation(pixels, augmented)
        if copies % 2:
            yield pixels

    @staticmethod
    def _random_zero_dc_filter_response(pixels: torch.Tensor, kernel_size: int) -> torch.Tensor:
        batch, channels, height, width = pixels.shape
        kernels = torch.randn(batch, 1, kernel_size, kernel_size, device=pixels.device, dtype=pixels.dtype)
        kernels = kernels - kernels.mean(dim=(2, 3), keepdim=True)
        kernels = kernels / kernels.abs().sum(dim=(2, 3), keepdim=True).clamp_min(1e-12)
        weights = kernels.repeat_interleave(channels, dim=0)
        padding = kernel_size // 2
        flat = F.pad(pixels, (padding,) * 4, mode="reflect").reshape(1, batch * channels, height + 2 * padding, width + 2 * padding)
        response = F.conv2d(flat, weights, groups=batch * channels)
        return response.reshape(batch, channels, height, width)

    def _iter_antithetic_filter_bank_pixels(
        self,
        pixels: torch.Tensor,
        copies: int,
    ):
        """Yield paired zero-DC random filters and an identity view."""
        pair_count = copies // 2
        kernel_sizes = (3, 5, 7, 9)
        for pair_index in range(pair_count):
            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(pixels.device) if pixels.is_cuda else None
            response = self._random_zero_dc_filter_response(pixels, kernel_sizes[pair_index % 4])
            augmented = (pixels + self.guide_aug_strength * response).clamp(0.0, 1.0)
            yield self._blend_guide_augmentation(pixels, augmented)

            # Recompute after differentiating the positive view. This keeps
            # antithetic values exact without sharing a sequential loss graph.
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, pixels.device)
            response = self._random_zero_dc_filter_response(pixels, kernel_sizes[pair_index % 4])
            augmented = (pixels - self.guide_aug_strength * response).clamp(0.0, 1.0)
            yield self._blend_guide_augmentation(pixels, augmented)
        if copies % 2:
            yield pixels

    def _iter_multiscale_adjoint_pixels(
        self,
        pixels: torch.Tensor,
        copies: int,
    ):
        """Keep forward pixels fixed while sampling scale-space Jacobians."""
        kernel_sizes = (1, 3, 5, 7, 9, 11, 15, 19, 23)
        for copy_index in range(copies):
            kernel_size = kernel_sizes[copy_index % len(kernel_sizes)]
            filtered = pixels if kernel_size == 1 else F.avg_pool2d(pixels, kernel_size, 1, kernel_size // 2)
            differentiable = pixels + self.guide_aug_strength * filtered
            augmented = differentiable + (pixels - differentiable).detach()
            yield self._blend_guide_augmentation(pixels, augmented)

    def _iter_orthogonal_photometric_pixels(
        self,
        pixels: torch.Tensor,
        copies: int,
    ):
        """Yield paired exposure, contrast, saturation, and gamma views.

        The four axes alter distinct first-order image statistics. Pairing
        opposite directions cancels the ensemble's first-order photometric
        bias while retaining curvature from semantic-preserving evaluations.
        An odd view budget includes the clean image.
        """
        gray_weights = pixels.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
        pair_count = copies // 2
        axes = ("exposure", "contrast", "saturation", "gamma")
        for pair_index in range(pair_count):
            axis = axes[pair_index % len(axes)]
            for direction in (1.0, -1.0):
                factor = math.exp(direction * self.guide_aug_strength)
                if axis == "exposure":
                    augmented = pixels * factor
                elif axis == "contrast":
                    center = pixels.mean(dim=(2, 3), keepdim=True)
                    augmented = center + factor * (pixels - center)
                elif axis == "saturation":
                    gray = (pixels * gray_weights).sum(dim=1, keepdim=True)
                    augmented = gray + factor * (pixels - gray)
                else:
                    augmented = pixels.clamp_min(1e-6).pow(factor)
                augmented = augmented.clamp(0.0, 1.0)
                yield self._blend_guide_augmentation(pixels, augmented)
        if copies % 2:
            yield pixels

    def _iter_orthogonal_spherical_pixels(
        self,
        pixels: torch.Tensor,
        copies: int,
    ):
        """Yield an antithetic cubature rule on a broadband image sphere.

        Directions are zero-DC, unit-RMS, and Gram-Schmidt orthogonal per
        image. Unlike jitter, their radius is fixed and every positive
        direction has an exact negative partner.
        """
        pair_count = copies // 2
        directions: list[torch.Tensor] = []
        for _pair_index in range(pair_count):
            direction = torch.randn_like(pixels)
            direction = direction - direction.mean(dim=(2, 3), keepdim=True)
            for previous in directions:
                flat_direction = direction.flatten(1)
                flat_previous = previous.flatten(1)
                coefficient = (flat_direction * flat_previous).sum(1) / flat_previous.square().sum(1).clamp_min(1e-12)
                direction = direction - coefficient.view(-1, 1, 1, 1) * previous
            direction = direction / direction.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
            directions.append(direction)
        for direction in directions:
            for sign in (1.0, -1.0):
                augmented = (pixels + sign * (self.guide_aug_strength / 2.0) * direction).clamp(0.0, 1.0)
                yield self._blend_guide_augmentation(pixels, augmented)
        if copies % 2:
            yield pixels

    def _iter_antithetic_jitter_cubature_pixels(
        self,
        pixels: torch.Tensor,
        copies: int,
    ):
        """Yield paired brightness/noise jitter views plus identity.

        Standard jitter uses independent brightness and Gaussian-noise samples,
        so its 9-view EOT gradient carries O(1/sqrt(n)) first-order sampling
        error. This method keeps the same perturbation family and strength but
        evaluates +/- pairs. The pair mean cancels odd first-order terms, while
        the pair curvature still smooths the feature loss over photometric and
        pixel-noise directions.
        """
        pair_count = copies // 2
        for _pair_index in range(pair_count):
            brightness = (
                torch.rand(pixels.size(0), 1, 1, 1, device=pixels.device, dtype=pixels.dtype) * 2.0 - 1.0
            ) * self.guide_aug_strength
            noise = torch.randn_like(pixels) * (self.guide_aug_strength / 2.0)
            for sign in (1.0, -1.0):
                augmented = torch.clamp(pixels * (1.0 + sign * brightness) + sign * noise, 0.0, 1.0)
                yield self._blend_guide_augmentation(pixels, augmented)
        if copies % 2:
            yield pixels

    def _natural_spectrum_transport_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Transfer normalized natural-image amplitudes while preserving phase and DC."""
        batch = pixels.size(0)
        if batch > 1:
            shift = int(torch.randint(1, batch, (1,), device=pixels.device).item())
            donor = torch.roll(pixels.detach(), shifts=shift, dims=0)
        else:
            donor = torch.roll(pixels.detach(), shifts=1, dims=1)
        work = pixels if pixels.dtype == torch.float64 else pixels.float()
        donor_work = donor.to(work.dtype)
        source_fft = torch.fft.rfft2(work, dim=(-2, -1), norm="ortho")
        donor_fft = torch.fft.rfft2(donor_work, dim=(-2, -1), norm="ortho")
        source_amp, donor_amp = source_fft.abs(), donor_fft.abs()
        source_scale = source_amp.flatten(2)[:, :, 1:].square().mean(2, keepdim=True).sqrt()
        donor_scale = donor_amp.flatten(2)[:, :, 1:].square().mean(2, keepdim=True).sqrt()
        donor_amp = donor_amp * (source_scale / donor_scale.clamp_min(1e-12)).unsqueeze(-1)
        unit_phase = source_fft / source_amp.clamp_min(1e-12)
        mixed_amp = torch.lerp(source_amp, donor_amp, self.guide_aug_strength)
        mixed_amp = mixed_amp.clone()
        mixed_amp[..., 0, 0] = source_amp[..., 0, 0]
        transported = torch.fft.irfft2(unit_phase * mixed_amp, s=pixels.shape[-2:], norm="ortho")
        return transported.to(pixels.dtype).clamp(0.0, 1.0)

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
            brightness = (
                torch.rand(pixels.size(0), 1, 1, 1, device=pixels.device, dtype=pixels.dtype) * 2.0 - 1.0
            ) * strength
            noise = torch.randn_like(pixels) * (strength / 2.0)
            return torch.clamp(pixels * (1.0 + brightness) + noise, 0.0, 1.0)
        if method == "freq":
            pooled = F.avg_pool2d(pixels, kernel_size=9, stride=1, padding=4)
            noise = F.avg_pool2d(torch.rand_like(pixels), kernel_size=9, stride=1, padding=4)
            corrupt = 0.7 * pooled + 0.3 * noise
            return torch.clamp(pixels * (1.0 - strength) + corrupt * strength, 0.0, 1.0)
        if method == "dim_resonance":
            return self._dim_resonance_pixels(pixels)
        if method == "lowmid_shift":
            return self._lowmid_shift_pixels(pixels)
        if method == "white_noise":
            return self._white_noise_pixels(pixels)
        if method == "natural_spectrum_transport":
            return self._natural_spectrum_transport_pixels(pixels)
        raise ValueError(f"Unsupported guide augmentation method: {method}")

    def _guide_augmented_pixels(
        self,
        pixels: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        return self._augment_full_image(pixels, method)

    def _iter_forward_pixels(
        self,
        pixels: torch.Tensor,
    ):
        if not self.guide_aug:
            yield pixels
            return

        for method in self.guide_aug_methods:
            if method == "antithetic_transport":
                yield from self._iter_antithetic_transport_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
            if method == "antithetic_filter_bank":
                yield from self._iter_antithetic_filter_bank_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
            if method == "multiscale_adjoint_ensemble":
                yield from self._iter_multiscale_adjoint_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
            if method == "orthogonal_photometric_ensemble":
                yield from self._iter_orthogonal_photometric_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
            if method == "orthogonal_spherical_smoothing":
                yield from self._iter_orthogonal_spherical_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
            if method == "antithetic_jitter_cubature":
                yield from self._iter_antithetic_jitter_cubature_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
            for _copy_idx in range(self.guide_aug_copies):
                yield self._guide_augmented_pixels(pixels, method)


    def _iter_feature_trajectory_dropout_pixels(
        self,
        pixels: torch.Tensor,
        pilot_grad: torch.Tensor,
        copies: int,
    ):
        """Yield feature-trajectory views under paired dropout corruption.

        A pilot feature-loss gradient estimates the local transferable attack
        direction. The EOT views integrate along that sign trajectory while
        applying paired dropout/blur corruption at each point, favoring
        gradients that remain stable under structured evidence removal instead
        of photometric jitter.
        """
        direction = pilot_grad.detach().sign()
        corruption_strength = min(1.0, self.guide_aug_strength * 1.25)
        for index in range(copies):
            radius = self.step_size * float(index + 1)
            lookahead = torch.clamp(pixels + radius * direction, 0.0, 1.0)
            if index % 2 == 1:
                noise = 1.0 - torch.rand_like(lookahead)
            else:
                noise = torch.rand_like(lookahead)
            blurred = F.avg_pool2d(lookahead, kernel_size=5, stride=1, padding=2)
            corrupt = 0.5 * noise + 0.5 * blurred
            augmented = torch.clamp(
                lookahead * (1.0 - corruption_strength) + corrupt * corruption_strength,
                0.0,
                1.0,
            )
            yield self._blend_guide_augmentation(pixels, augmented)

    def _attack_loss_for_pixels(
        self,
        forward_pixels: torch.Tensor,
        labels: torch.Tensor,
        clean_feature_target: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.dim_adjoint_echo:
            forward_pixels = self._dim_adjoint_echo_pixels(forward_pixels)
        model_pixels = self._input_diversity(forward_pixels)
        if self.attack_loss == "logits":
            logits_adv = self.model(
                self._normalize(model_pixels),
                return_attn=False,
            )
            return F.cross_entropy(logits_adv, labels)

        if clean_feature_target is None:
            raise RuntimeError("clean_feature_target is required for feature loss.")
        adv_features = self._extract_layer_patch_features(model_pixels)
        cosine = F.cosine_similarity(adv_features, clean_feature_target, dim=-1)
        return 1.0 - cosine.mean()

    def _iter_attack_losses(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        clean_feature_target: torch.Tensor | None = None,
    ):
        if self.guide_aug and "feature_trajectory_dropout" in self.guide_aug_methods:
            pilot_loss = self._attack_loss_for_pixels(pixels, labels, clean_feature_target)
            pilot_grad = torch.autograd.grad(pilot_loss, pixels, retain_graph=False)[0].detach()
            for method in self.guide_aug_methods:
                if method == "feature_trajectory_dropout":
                    for forward_pixels in self._iter_feature_trajectory_dropout_pixels(
                        pixels, pilot_grad, self.guide_aug_copies
                    ):
                        yield self._attack_loss_for_pixels(forward_pixels, labels, clean_feature_target)
                else:
                    for _copy_idx in range(self.guide_aug_copies):
                        forward_pixels = self._guide_augmented_pixels(pixels, method)
                        yield self._attack_loss_for_pixels(forward_pixels, labels, clean_feature_target)
            return

        for forward_pixels in self._iter_forward_pixels(pixels):
            yield self._attack_loss_for_pixels(forward_pixels, labels, clean_feature_target)

    def _extract_layer_patch_features(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.feature_scope == "stage":
            outputs = self.model(
                self._normalize(pixels),
                return_attn=False,
                return_stage_tokens=True,
            )
            token_name = "stage outputs"
            type_hint = "(logits, stage_tokens) with return_stage_tokens=True"
        else:
            outputs = self.model(
                self._normalize(pixels),
                return_attn=False,
                return_tokens=True,
            )
            token_name = "feature layers"
            type_hint = "(logits, block_tokens) with return_tokens=True"

        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise TypeError(f"Feature loss requires a model that returns {type_hint}.")
        _logits, feature_outputs = outputs
        num_layers = len(feature_outputs)
        layer_idx = self.feature_layer if self.feature_layer >= 0 else num_layers + self.feature_layer
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"feature_layer {self.feature_layer} is out of range for {num_layers} {token_name}.")
        features = feature_outputs[layer_idx]
        preparer = getattr(self.model, "prepare_feature_tokens", None)
        if preparer is not None:
            return preparer(features)
        if features.ndim != 3 or features.size(1) < 2:
            raise ValueError(
                f"Feature layer {layer_idx} must return [B,N,D] tokens or be handled by "
                f"model.prepare_feature_tokens(), got {tuple(features.shape)}."
            )
        return features[:, 1:, :]

    def _attack_grad_terms(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        clean_feature_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        grad_sum = None
        term_grads = []
        for attack_loss in self._iter_attack_losses(pixels, labels, clean_feature_target):
            grad_term = torch.autograd.grad(attack_loss, pixels, retain_graph=False)[0]
            term_grads.append(grad_term)
            grad_sum = grad_term if grad_sum is None else grad_sum + grad_term
        if grad_sum is None:
            raise RuntimeError("No attack loss terms were generated.")
        return grad_sum / float(len(term_grads)), tuple(term_grads)

    def _attack_grad(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        clean_feature_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grad, _term_grads = self._attack_grad_terms(
            pixels, labels, clean_feature_target
        )
        return grad

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)

        clean_pixels = self._denormalize(images).detach()
        clean_feature_target = None
        if self.attack_loss == "feature":
            with torch.no_grad():
                clean_feature_target = self._extract_layer_patch_features(clean_pixels).detach()

        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        for step_idx in range(self.steps):
            grad_pixels = adv_pixels.detach()
            if self.nesterov and step_idx > 0:
                with torch.no_grad():
                    grad_pixels = grad_pixels + self.decay * self.step_size * momentum.sign()
                    delta = grad_pixels - clean_pixels
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    grad_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

            grad_pixels = grad_pixels.detach().requires_grad_(True)
            if self.lowmid_dss_filter:
                grad, term_grads = self._attack_grad_terms(
                    grad_pixels, labels, clean_feature_target
                )
            else:
                grad = self._attack_grad(grad_pixels, labels, clean_feature_target)
                term_grads = None
            grad = self._smooth_grad(grad)
            if term_grads is not None:
                term_grads = tuple(self._smooth_grad(term) for term in term_grads)
            grad = self._apply_lowmid_dss_filter(grad, term_grads)
            grad = self._tune_lowmid_gradient(grad)
            grad = self._normalize_grad(grad)

            if self.use_momentum:
                momentum = self.decay * momentum + grad
                update = momentum
            else:
                update = grad

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * update.sign()
                delta = adv_pixels - clean_pixels
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

        return self._normalize(adv_pixels)
