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
        dim_padding_mode: str = "zero",
        dim_padding_blur_kernel: int = 5,
        use_momentum: bool = True,
        momentum_decay: float = 1.0,
        nesterov: bool = False,
        guide_aug: bool = False,
        guide_aug_methods: tuple[str, ...] = ("dropout",),
        guide_aug_copies: int = 3,
        guide_aug_strength: float = 0.2,
        patch_dropout_ratio: float = 0.3,
        patch_dropout_score_mode: str = "high",
        patch_dropout_fill_mode: str = "zero_noise",
        patch_dropout_noise_mode: str = "gaussian",
        dim_adjoint_echo: bool = False,
        lowmid_grad_tuning: bool = False,
        lowmid_grad_rotation_strength: float = 0.5,
        lowmid_grad_preserve_norm: bool = True,
        lowmid_dss_filter: bool = False,
        lowmid_dss_consistency: str = "sign",
        lowmid_dss_agreement_threshold: float = 0.67,
        spatial_sign_reinforcement: bool = False,
        spatial_sign_reinforcement_sigma: float = 1.0,
        spatial_sign_reinforcement_strength: float = 0.2,
        grad_momentum_agreement: bool = False,
        grad_momentum_agreement_strength: float = 0.2,
        grad_momentum_agreement_sigma: float = 0.0,
        grad_momentum_conflict_suppression_strength: float = 0.0,
        cross_step_sign_vote: bool = False,
        cross_step_sign_vote_window: int = 5,
        cross_step_sign_vote_strength: float = 0.2,
        view_consistent_agreement: bool = False,
        view_consistent_agreement_strength: float = 0.3,
        view_consistent_agreement_threshold: float = 0.0,
        fft_sign_regularization: bool = False,
        fft_sign_regularization_cutoff: float = 0.25,
        fft_sign_regularization_strength: float = 0.5,
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
        if spatial_sign_reinforcement_sigma <= 0:
            raise ValueError(
                f"spatial_sign_reinforcement_sigma must be positive, got {spatial_sign_reinforcement_sigma}."
            )
        if spatial_sign_reinforcement_strength < 0:
            raise ValueError(
                f"spatial_sign_reinforcement_strength must be non-negative, got {spatial_sign_reinforcement_strength}."
            )
        if grad_momentum_agreement_strength < 0:
            raise ValueError(
                f"grad_momentum_agreement_strength must be non-negative, got {grad_momentum_agreement_strength}."
            )
        if grad_momentum_agreement_sigma < 0:
            raise ValueError(
                f"grad_momentum_agreement_sigma must be non-negative, got {grad_momentum_agreement_sigma}."
            )
        if grad_momentum_conflict_suppression_strength < 0:
            raise ValueError(
                "grad_momentum_conflict_suppression_strength must be non-negative, "
                f"got {grad_momentum_conflict_suppression_strength}."
            )
        if cross_step_sign_vote_window <= 0:
            raise ValueError(
                f"cross_step_sign_vote_window must be positive, got {cross_step_sign_vote_window}."
            )
        if cross_step_sign_vote_strength < 0:
            raise ValueError(
                f"cross_step_sign_vote_strength must be non-negative, got {cross_step_sign_vote_strength}."
            )
        if view_consistent_agreement_strength < 0:
            raise ValueError(
                "view_consistent_agreement_strength must be non-negative, "
                f"got {view_consistent_agreement_strength}."
            )
        if not 0.0 <= view_consistent_agreement_threshold <= 1.0:
            raise ValueError(
                "view_consistent_agreement_threshold must be in [0, 1], "
                f"got {view_consistent_agreement_threshold}."
            )
        if fft_sign_regularization_cutoff <= 0 or fft_sign_regularization_cutoff > 1.0:
            raise ValueError(
                f"fft_sign_regularization_cutoff must be in (0, 1], got {fft_sign_regularization_cutoff}."
            )
        if not 0.0 <= fft_sign_regularization_strength <= 1.0:
            raise ValueError(
                f"fft_sign_regularization_strength must be in [0, 1], got {fft_sign_regularization_strength}."
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
        valid_dim_padding_modes = ("zero", "detach-blur", "grad-blur")
        if dim_padding_mode not in valid_dim_padding_modes:
            raise ValueError(
                f"dim_padding_mode must be one of {valid_dim_padding_modes}, got {dim_padding_mode!r}."
            )
        if dim_padding_blur_kernel <= 0 or dim_padding_blur_kernel % 2 == 0:
            raise ValueError(
                f"dim_padding_blur_kernel must be a positive odd integer, got {dim_padding_blur_kernel}."
            )
        self.dim_padding_mode = dim_padding_mode
        self.dim_padding_blur_kernel = int(dim_padding_blur_kernel)
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
        self.spatial_sign_reinforcement = bool(spatial_sign_reinforcement)
        self.spatial_sign_reinforcement_sigma = float(spatial_sign_reinforcement_sigma)
        self.spatial_sign_reinforcement_strength = float(spatial_sign_reinforcement_strength)
        self.grad_momentum_agreement = bool(grad_momentum_agreement)
        self.grad_momentum_agreement_strength = float(grad_momentum_agreement_strength)
        self.grad_momentum_agreement_sigma = float(grad_momentum_agreement_sigma)
        self.grad_momentum_conflict_suppression_strength = float(
            grad_momentum_conflict_suppression_strength
        )
        self.cross_step_sign_vote = bool(cross_step_sign_vote)
        self.cross_step_sign_vote_window = int(cross_step_sign_vote_window)
        self.cross_step_sign_vote_strength = float(cross_step_sign_vote_strength)
        self.view_consistent_agreement = bool(view_consistent_agreement)
        self.view_consistent_agreement_strength = float(view_consistent_agreement_strength)
        self.view_consistent_agreement_threshold = float(view_consistent_agreement_threshold)
        self.fft_sign_regularization = bool(fft_sign_regularization)
        self.fft_sign_regularization_cutoff = float(fft_sign_regularization_cutoff)
        self.fft_sign_regularization_strength = float(fft_sign_regularization_strength)
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
            "dim_stable_edge",
            "dim_stable_edge_mix",
            "dim_consensus_trajectory",
            "dim_consensus_evidence_trajectory",
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
            "patch_dropout",
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
        self.patch_dropout_ratio = float(patch_dropout_ratio)
        if not 0.0 <= self.patch_dropout_ratio <= 1.0:
            raise ValueError(f"patch_dropout_ratio must be in [0, 1], got {patch_dropout_ratio}.")
        self.patch_dropout_score_mode = str(patch_dropout_score_mode)
        if self.patch_dropout_score_mode not in ("high", "low"):
            raise ValueError(
                "patch_dropout_score_mode must be 'high' or 'low', "
                f"got {patch_dropout_score_mode!r}."
            )
        self.patch_dropout_fill_mode = str(patch_dropout_fill_mode)
        valid_patch_dropout_fill_modes = (
            "zero_noise",
            "random_high_score_inpaint",
            "context_high_score_blend",
            "nearest_high_score_inpaint",
        )
        if self.patch_dropout_fill_mode not in valid_patch_dropout_fill_modes:
            raise ValueError(
                "patch_dropout_fill_mode must be one of "
                f"{valid_patch_dropout_fill_modes}, got {patch_dropout_fill_mode!r}."
            )
        self.patch_dropout_noise_mode = str(patch_dropout_noise_mode)
        valid_patch_dropout_noise_modes = (
            "gaussian",
            "antithetic_gaussian",
            "rademacher_cubature",
            "patch_cov_gaussian",
            "score_weighted_gaussian",
            "inverse_score_weighted_gaussian",
            "opponent_channel_gaussian",
            "patch_embed_rowspace",
            "opponent_smooth_patch",
            "hybrid_dct_midfreq",
        )
        if self.patch_dropout_noise_mode not in valid_patch_dropout_noise_modes:
            raise ValueError(
                "patch_dropout_noise_mode must be one of "
                f"{valid_patch_dropout_noise_modes}, got {patch_dropout_noise_mode!r}."
            )
        self._patch_scores: torch.Tensor | None = None
        self._patch_dropout_antithetic_noise: torch.Tensor | None = None
        self._ti_kernel = self._build_ti_kernel(self.ti_sigma) if self.ti_sigma > 0 else None

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    @staticmethod
    def _normalize_grad(grad: torch.Tensor) -> torch.Tensor:
        denom = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return grad / denom

    def _apply_spatial_sign_reinforcement(
        self,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.spatial_sign_reinforcement
            or self.spatial_sign_reinforcement_strength <= 0
        ):
            return update
        radius = max(1, int(math.ceil(3.0 * self.spatial_sign_reinforcement_sigma)))
        kernel = self._build_gaussian_kernel(
            kernel_size=2 * radius + 1,
            sigma=self.spatial_sign_reinforcement_sigma,
        ).to(update.device, update.dtype)
        smooth_update = self._apply_depthwise_kernel(update, kernel)
        smooth_abs = self._apply_depthwise_kernel(update.abs(), kernel).clamp_min(1e-12)
        confidence = (smooth_update.abs() / smooth_abs).clamp(0.0, 1.0)
        reinforce = confidence * smooth_update.sign()
        return update + self.spatial_sign_reinforcement_strength * reinforce

    def _apply_grad_momentum_agreement(
        self,
        update: torch.Tensor,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.grad_momentum_agreement
            or (
                self.grad_momentum_agreement_strength <= 0
                and self.grad_momentum_conflict_suppression_strength <= 0
            )
        ):
            return update
        update_direction = update.sign()
        grad_direction = grad.sign()
        agreement = (update_direction != 0) & update_direction.eq(grad_direction)
        conflict = (update_direction != 0) & (grad_direction != 0) & update_direction.ne(grad_direction)
        agreement_weight = agreement.to(update.dtype)
        if self.grad_momentum_agreement_sigma > 0:
            radius = max(1, int(math.ceil(3.0 * self.grad_momentum_agreement_sigma)))
            kernel = self._build_gaussian_kernel(
                kernel_size=2 * radius + 1,
                sigma=self.grad_momentum_agreement_sigma,
            ).to(update.device, update.dtype)
            agreement_weight = self._apply_depthwise_kernel(agreement_weight, kernel).clamp(0.0, 1.0)
        reinforce = agreement_weight * update_direction
        suppress = conflict.to(update.dtype) * update_direction
        return (
            update
            + self.grad_momentum_agreement_strength * reinforce
            - self.grad_momentum_conflict_suppression_strength * suppress
        )

    def _apply_view_consistent_agreement(
        self,
        update: torch.Tensor,
        term_grads: tuple[torch.Tensor, ...] | None,
    ) -> torch.Tensor:
        if (
            not self.view_consistent_agreement
            or self.view_consistent_agreement_strength <= 0
            or not term_grads
        ):
            return update
        update_direction = update.sign()
        view_signs = torch.stack([term.sign() for term in term_grads], dim=0)
        support = view_signs.eq(update_direction.unsqueeze(0)).to(update.dtype).mean(dim=0)
        support = support * update_direction.ne(0).to(update.dtype)
        if self.view_consistent_agreement_threshold > 0:
            support = support * (support >= self.view_consistent_agreement_threshold).to(update.dtype)
        reinforce = support * update_direction
        return update + self.view_consistent_agreement_strength * reinforce

    def _apply_cross_step_sign_vote(
        self,
        update: torch.Tensor,
        sign_history: list[torch.Tensor],
    ) -> torch.Tensor:
        if not self.cross_step_sign_vote or self.cross_step_sign_vote_strength <= 0:
            return update
        sign_history.append(update.sign().detach())
        if len(sign_history) > self.cross_step_sign_vote_window:
            del sign_history[:-self.cross_step_sign_vote_window]
        vote = torch.stack(sign_history, dim=0).to(update.dtype).mean(dim=0)
        reinforce = vote.abs() * vote.sign()
        return update + self.cross_step_sign_vote_strength * reinforce

    def _apply_fft_sign_regularization(
        self,
        update: torch.Tensor,
    ) -> torch.Tensor:
        """Suppress high-frequency noise in update (momentum) before sign().

        The sign field after sign(momentum) is highly fragmented (only ~41%
        adjacent pixel agreement) despite the gradient being 96% low-frequency.
        The nonlinear sign() amplifies small high-frequency fluctuations.

        We low-pass filter the update in the frequency domain to preserve the
        low/mid-frequency structure while suppressing sign-flipping noise.
        """
        if (
            not self.fft_sign_regularization
            or self.fft_sign_regularization_strength <= 0
        ):
            return update
        bsz, ch, h, w = update.shape
        fy = torch.fft.fftfreq(h, device=update.device, dtype=torch.float32).view(h, 1)
        fx = torch.fft.fftfreq(w, device=update.device, dtype=torch.float32).view(1, w)
        radius = (fx.square() + fy.square()).sqrt().view(1, 1, h, w)  # [1, 1, H, W]
        mask = (radius < self.fft_sign_regularization_cutoff).to(torch.float32)

        work = update.float()
        freq = torch.fft.fft2(work, dim=(-2, -1), norm="ortho")
        filtered = torch.fft.ifft2(freq * mask, dim=(-2, -1), norm="ortho").real.to(update.dtype)

        strength = self.fft_sign_regularization_strength
        return (1.0 - strength) * update + strength * filtered

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

    def _apply_dim_transform(self, images: torch.Tensor, params: tuple[int, int, int, int]) -> torch.Tensor:
        _batch_size, _channels, height, width = images.shape
        new_h, new_w, top, left = params
        resized = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=False)
        zero_padded = F.pad(resized, (left, width - new_w - left, top, height - new_h - top), value=0.0)
        if self.dim_padding_mode == "zero":
            return zero_padded

        kernel_size = self.dim_padding_blur_kernel
        padding = kernel_size // 2
        fill = F.avg_pool2d(
            F.pad(images, (padding, padding, padding, padding), mode="reflect"),
            kernel_size=kernel_size,
            stride=1,
        )
        if self.dim_padding_mode == "detach-blur":
            fill = fill.detach()
        mask = torch.zeros_like(images[:, :1])
        mask[..., top:top + new_h, left:left + new_w] = 1.0
        return zero_padded * mask + fill * (1.0 - mask)

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

    @staticmethod
    def _edge_center_gate(pixels: torch.Tensor) -> torch.Tensor:
        gray_weights = pixels.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
        gray = (pixels * gray_weights).sum(dim=1, keepdim=True)
        kx = pixels.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
        ky = pixels.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
        padded = F.pad(gray, (1, 1, 1, 1), mode="reflect")
        edge = (F.conv2d(padded, kx).square() + F.conv2d(padded, ky).square()).sqrt()
        edge = edge / edge.flatten(1).quantile(0.90, dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
        edge_gate = (0.35 + 0.65 * edge.clamp(0.0, 1.0)).detach()

        height, width = pixels.shape[-2:]
        yy = torch.linspace(-1.0, 1.0, height, device=pixels.device, dtype=pixels.dtype).view(height, 1)
        xx = torch.linspace(-1.0, 1.0, width, device=pixels.device, dtype=pixels.dtype).view(1, width)
        radius = (xx.square() + yy.square()).sqrt().clamp(0.0, 1.0)
        center_gate = (0.55 + 0.45 * (1.0 - radius)).view(1, 1, height, width)
        return edge_gate * center_gate

    def _dim_stable_edge_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Forward-identity DIM-adjoint edge/low-mid gradient echo.

        High-transfer AE analysis showed stronger transfer when perturbations
        shift energy from high frequency into mid frequency and align with
        image edges while avoiding border-heavy energy. This view keeps the
        forward image fixed, but its backward Jacobian emphasizes the sampled
        DIM adjoint residual after low/mid smoothing and edge/center gating.
        """
        restored = self._dim_adjoint_restore_pixels(pixels)
        residual = restored - pixels
        residual = F.avg_pool2d(residual, kernel_size=5, stride=1, padding=2)
        residual = residual - residual.mean(dim=(2, 3), keepdim=True)
        gate = self._edge_center_gate(pixels)
        direction = residual * gate
        direction_scale = residual.abs().mean(dim=(1, 2, 3), keepdim=True).detach()
        current_scale = direction.abs().mean(dim=(1, 2, 3), keepdim=True).detach().clamp_min(1e-12)
        direction = direction * (direction_scale / current_scale)
        differentiable = pixels + self.guide_aug_strength * direction
        return (differentiable + (pixels - differentiable).detach()).to(pixels.dtype)

    def _iter_dim_stable_edge_pixels(self, pixels: torch.Tensor, copies: int):
        for _copy_idx in range(copies):
            yield self._blend_guide_augmentation(pixels, self._dim_stable_edge_pixels(pixels))

    def _dim_stable_edge_mix_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        restored = self._dim_adjoint_restore_pixels(pixels)
        residual = restored - pixels
        residual = F.avg_pool2d(residual, kernel_size=5, stride=1, padding=2)
        gate = self._edge_center_gate(pixels)
        augmented = pixels + self.guide_aug_strength * gate * residual
        return torch.clamp(augmented, 0.0, 1.0)

    def _iter_dim_stable_edge_mix_pixels(self, pixels: torch.Tensor, copies: int):
        for _copy_idx in range(copies):
            yield self._blend_guide_augmentation(pixels, self._dim_stable_edge_mix_pixels(pixels))

    def _iter_dim_consensus_trajectory_pixels(
        self,
        pixels: torch.Tensor,
        consensus_grad: torch.Tensor,
        copies: int,
    ):
        smooth_grad = F.avg_pool2d(consensus_grad.detach(), kernel_size=5, stride=1, padding=2)
        direction = smooth_grad.sign()
        for index in range(copies):
            radius = self.step_size * float(index + 1)
            augmented = torch.clamp(pixels + radius * direction, 0.0, 1.0)
            yield self._blend_guide_augmentation(pixels, augmented)

    def _iter_dim_consensus_evidence_trajectory_pixels(
        self,
        pixels: torch.Tensor,
        consensus_grad: torch.Tensor,
        copies: int,
    ):
        smooth_grad = F.avg_pool2d(consensus_grad.detach(), kernel_size=5, stride=1, padding=2)
        direction = smooth_grad.sign()
        corruption_strength = min(1.0, self.guide_aug_strength * 1.25)
        batch, channels, height, width = pixels.shape
        for index in range(copies):
            radius = self.step_size * float(index + 1)
            lookahead = torch.clamp(pixels + radius * direction, 0.0, 1.0)
            coarse_h = max(4, height // 8)
            coarse_w = max(4, width // 8)
            coarse_noise = torch.rand(batch, channels, coarse_h, coarse_w, device=pixels.device, dtype=pixels.dtype)
            noise = F.interpolate(coarse_noise, size=(height, width), mode="bilinear", align_corners=False)
            if index % 2 == 1:
                noise = 1.0 - noise
            blurred = F.avg_pool2d(lookahead, kernel_size=5, stride=1, padding=2)
            evidence_mix = 0.5 * noise + 0.5 * blurred
            augmented = torch.clamp(
                lookahead * (1.0 - corruption_strength) + evidence_mix * corruption_strength,
                0.0,
                1.0,
            )
            yield self._blend_guide_augmentation(pixels, augmented)

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
        if method == "dim_stable_edge":
            return self._dim_stable_edge_pixels(pixels)
        if method == "dim_stable_edge_mix":
            return self._dim_stable_edge_mix_pixels(pixels)
        if method == "lowmid_shift":
            return self._lowmid_shift_pixels(pixels)
        if method == "white_noise":
            return self._white_noise_pixels(pixels)
        if method == "natural_spectrum_transport":
            return self._natural_spectrum_transport_pixels(pixels)
        if method == "patch_dropout":
            return self._patch_dropout_pixels(pixels)
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
            if method == "dim_stable_edge":
                yield from self._iter_dim_stable_edge_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
            if method == "dim_stable_edge_mix":
                yield from self._iter_dim_stable_edge_mix_pixels(
                    pixels, self.guide_aug_copies
                )
                continue
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
            if method == "patch_dropout":
                self._patch_dropout_antithetic_noise = None
            for _copy_idx in range(self.guide_aug_copies):
                yield self._guide_augmented_pixels(pixels, method)
            if method == "patch_dropout":
                self._patch_dropout_antithetic_noise = None


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
        special_methods = {
            "feature_trajectory_dropout",
            "dim_consensus_trajectory",
            "dim_consensus_evidence_trajectory",
        }
        if self.guide_aug and any(method in self.guide_aug_methods for method in special_methods):
            pilot_grad = None
            consensus_grad = None
            if "feature_trajectory_dropout" in self.guide_aug_methods:
                pilot_loss = self._attack_loss_for_pixels(pixels, labels, clean_feature_target)
                pilot_grad = torch.autograd.grad(pilot_loss, pixels, retain_graph=False)[0].detach()
            if (
                "dim_consensus_trajectory" in self.guide_aug_methods
                or "dim_consensus_evidence_trajectory" in self.guide_aug_methods
            ):
                pilot_count = max(1, min(4, self.guide_aug_copies))
                pilot_grads = []
                for _pilot_idx in range(pilot_count):
                    pilot_loss = self._attack_loss_for_pixels(pixels, labels, clean_feature_target)
                    pilot_grads.append(torch.autograd.grad(pilot_loss, pixels, retain_graph=False)[0].detach())
                consensus_grad = torch.stack(pilot_grads, dim=0).mean(dim=0)

            for method in self.guide_aug_methods:
                if method == "feature_trajectory_dropout":
                    for forward_pixels in self._iter_feature_trajectory_dropout_pixels(
                        pixels, pilot_grad, self.guide_aug_copies
                    ):
                        yield self._attack_loss_for_pixels(forward_pixels, labels, clean_feature_target)
                elif method == "dim_consensus_trajectory":
                    for forward_pixels in self._iter_dim_consensus_trajectory_pixels(
                        pixels, consensus_grad, self.guide_aug_copies
                    ):
                        yield self._attack_loss_for_pixels(forward_pixels, labels, clean_feature_target)
                elif method == "dim_consensus_evidence_trajectory":
                    for forward_pixels in self._iter_dim_consensus_evidence_trajectory_pixels(
                        pixels, consensus_grad, self.guide_aug_copies
                    ):
                        yield self._attack_loss_for_pixels(forward_pixels, labels, clean_feature_target)
                else:
                    if method == "patch_dropout":
                        self._patch_dropout_antithetic_noise = None
                    for _copy_idx in range(self.guide_aug_copies):
                        forward_pixels = self._guide_augmented_pixels(pixels, method)
                        yield self._attack_loss_for_pixels(forward_pixels, labels, clean_feature_target)
                    if method == "patch_dropout":
                        self._patch_dropout_antithetic_noise = None
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

    def _compute_patch_scores(self, pixels: torch.Tensor) -> None:
        """Compute CLS-cosine patch importance scores from the feature layer.

        Stores scores in ``self._patch_scores`` as a [B, N_patch] tensor.
        Called once per step using the current adversarial image.
        """
        if not self._patch_dropout_active():
            self._patch_scores = None
            return
        with torch.no_grad():
            normalized = self._normalize(pixels.detach())
            if self.feature_scope == "stage":
                outputs = self.model(normalized, return_attn=False, return_stage_tokens=True)
            else:
                outputs = self.model(normalized, return_attn=False, return_tokens=True)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                self._patch_scores = None
                return
            _logits, feature_outputs = outputs
            num_layers = len(feature_outputs)
            layer_idx = self.feature_layer if self.feature_layer >= 0 else num_layers + self.feature_layer
            if layer_idx < 0 or layer_idx >= num_layers:
                self._patch_scores = None
                return
            features = feature_outputs[layer_idx]  # [B, N_patch+1, D]
            if features.ndim != 3 or features.size(1) < 2:
                self._patch_scores = None
                return
            cls_token = features[:, 0, :]          # [B, D]
            patch_tokens = features[:, 1:, :]      # [B, N_patch, D]
            self._patch_scores = F.cosine_similarity(
                patch_tokens, cls_token.unsqueeze(1).expand_as(patch_tokens), dim=-1,
            )

    def _patch_dropout_active(self) -> bool:
        return "patch_dropout" in self.guide_aug_methods

    def _patch_embed_projection_weight(self, patch_h: int, patch_w: int) -> torch.Tensor | None:
        base_model = getattr(self.model, "model", self.model)
        patch_embed = getattr(base_model, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        weight = getattr(proj, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 4:
            return None
        out_channels, in_channels, kernel_h, kernel_w = weight.shape
        if in_channels != 3 or kernel_h != patch_h or kernel_w != patch_w:
            return None
        return weight.detach().to(device=self.device, dtype=torch.float32).reshape(out_channels, -1)

    def _patch_dropout_noise(self, pixels: torch.Tensor, grid_size: int) -> torch.Tensor:
        if self.patch_dropout_noise_mode == "gaussian":
            return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)
        if self.patch_dropout_noise_mode == "rademacher_cubature":
            signs = torch.empty_like(pixels).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            return self.guide_aug_strength * signs
        if self.patch_dropout_noise_mode == "opponent_channel_gaussian":
            if pixels.size(1) != 3:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)
            coeff = torch.randn_like(pixels, dtype=pixels.dtype)
            luma = (0.5 ** 0.5) * coeff[:, 0:1]
            red_green = (1.25 ** 0.5) * coeff[:, 1:2]
            yellow_blue = (1.25 ** 0.5) * coeff[:, 2:3]
            inv_sqrt2 = 2.0 ** -0.5
            inv_sqrt3 = 3.0 ** -0.5
            inv_sqrt6 = 6.0 ** -0.5
            noise = torch.cat(
                (
                    inv_sqrt3 * luma + inv_sqrt2 * red_green + inv_sqrt6 * yellow_blue,
                    inv_sqrt3 * luma - inv_sqrt2 * red_green + inv_sqrt6 * yellow_blue,
                    inv_sqrt3 * luma - 2.0 * inv_sqrt6 * yellow_blue,
                ),
                dim=1,
            )
            return self.guide_aug_strength * noise
        if self.patch_dropout_noise_mode == "opponent_smooth_patch":
            # Opponent-channel noise with cross-patch spatial correlation.
            # 1. Generate noise at patch-grid resolution (grid×grid)
            # 2. Gaussian-smooth for cross-patch correlation (σ=1.5 patches)
            # 3. Bilinear upsample to full resolution
            # 4. Apply opponent color transform
            B, C, H, W = pixels.shape
            if C != 3:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)
            patch_h = H // grid_size
            patch_w = W // grid_size
            if patch_h * grid_size != H or patch_w * grid_size != W:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)

            # Generate independent Gaussian noise per channel at patch-grid resolution
            z_grid = torch.randn(B, C, grid_size, grid_size, device=pixels.device, dtype=pixels.dtype)

            # Gaussian smoothing for cross-patch correlation (sigma=1.5 at grid scale)
            kernel_size = 7
            sigma = 1.5
            ax = torch.arange(kernel_size, device=pixels.device, dtype=pixels.dtype) - kernel_size // 2
            gauss_1d = torch.exp(-0.5 * (ax / sigma).square())
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d[:, None] @ gauss_1d[None, :]
            kernel = gauss_2d.view(1, 1, kernel_size, kernel_size).expand(C, 1, kernel_size, kernel_size)

            z_smooth = F.conv2d(
                F.pad(z_grid, (kernel_size // 2,) * 4, mode="reflect"),
                kernel, groups=C,
            )

            # Bilinear upsample to full resolution (smooth within-patch variation)
            noise_pixel = F.interpolate(z_smooth, size=(H, W), mode="bilinear", align_corners=False)

            # Normalize to unit RMS per image
            noise_pixel = noise_pixel - noise_pixel.mean(dim=(2, 3), keepdim=True)
            rms = noise_pixel.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
            noise_pixel = noise_pixel / rms

            # Apply opponent color transform to the pixel-space noise
            coeff_type = noise_pixel  # [B, 3, H, W] — already unit-RMS per channel
            luma = (0.5 ** 0.5) * coeff_type[:, 0:1]
            rg = (1.25 ** 0.5) * coeff_type[:, 1:2]
            yb = (1.25 ** 0.5) * coeff_type[:, 2:3]
            inv_sqrt2 = 2.0 ** -0.5
            inv_sqrt3 = 3.0 ** -0.5
            inv_sqrt6 = 6.0 ** -0.5
            noise = torch.cat(
                (
                    inv_sqrt3 * luma + inv_sqrt2 * rg + inv_sqrt6 * yb,
                    inv_sqrt3 * luma - inv_sqrt2 * rg + inv_sqrt6 * yb,
                    inv_sqrt3 * luma - 2.0 * inv_sqrt6 * yb,
                ),
                dim=1,
            )
            return self.guide_aug_strength * noise
        if self.patch_dropout_noise_mode == "hybrid_dct_midfreq":
            # Hybrid: opponent-channel pixel noise + DCT mid-frequency patch-grid noise.
            # Alpha=0.7: 70% opponent pixel (CNN benefit), 30% DCT mid-freq (ViT diversity).
            B, C, H, W = pixels.shape
            if C != 3:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)
            patch_h = H // grid_size
            patch_w = W // grid_size
            if patch_h * grid_size != H or patch_w * grid_size != W:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)

            alpha = 0.7  # Weight of opponent pixel noise

            # Component 1: Per-pixel opponent-channel noise
            coeff_pixel = torch.randn_like(pixels, dtype=pixels.dtype)
            luma_p = (0.5 ** 0.5) * coeff_pixel[:, 0:1]
            rg_p = (1.25 ** 0.5) * coeff_pixel[:, 1:2]
            yb_p = (1.25 ** 0.5) * coeff_pixel[:, 2:3]
            inv_sqrt2 = 2.0 ** -0.5
            inv_sqrt3 = 3.0 ** -0.5
            inv_sqrt6 = 6.0 ** -0.5
            noise_opponent = torch.cat(
                (
                    inv_sqrt3 * luma_p + inv_sqrt2 * rg_p + inv_sqrt6 * yb_p,
                    inv_sqrt3 * luma_p - inv_sqrt2 * rg_p + inv_sqrt6 * yb_p,
                    inv_sqrt3 * luma_p - 2.0 * inv_sqrt6 * yb_p,
                ),
                dim=1,
            )
            # Normalize to unit RMS
            noise_opponent = noise_opponent - noise_opponent.mean(dim=(2, 3), keepdim=True)
            noise_opponent = noise_opponent / noise_opponent.square().mean(
                dim=(1, 2, 3), keepdim=True
            ).sqrt().clamp_min(1e-12)

            # Component 2: DCT mid-frequency at patch-grid resolution
            # Manual DCT: x[u,v] = sum_i sum_j pixel[i,j] * cos(pi*u*(i+0.5)/N) * cos(pi*v*(j+0.5)/N)
            # We do inverse: generate random coeffs, then IDCT to get spatial pattern
            N = grid_size
            # Precompute DCT basis: basis[u,v,i,j] = cos(pi*u*(i+0.5)/N) * cos(pi*v*(j+0.5)/N)
            i_idx = torch.arange(N, device=pixels.device, dtype=torch.float64)
            j_idx = torch.arange(N, device=pixels.device, dtype=torch.float64)
            u_idx = torch.arange(N, device=pixels.device, dtype=torch.float64)
            v_idx = torch.arange(N, device=pixels.device, dtype=torch.float64)

            # Basis vectors: cos(pi * u * (i+0.5) / N) — shape [N, N] where entry [u,i]
            basis_u = torch.cos(torch.pi * u_idx[:, None] * (i_idx[None, :] + 0.5) / N)  # [N, N]
            basis_v = torch.cos(torch.pi * v_idx[:, None] * (j_idx[None, :] + 0.5) / N)  # [N, N]

            # Generate random DCT coefficients, zero out outside mid-frequency band
            z_dct = torch.randn(B, C, N, N, device=pixels.device, dtype=torch.float64)
            # Mid-frequency mask: keep frequencies 2-7
            freq_u = u_idx[:, None].expand(N, N)  # [N, N]
            freq_v = v_idx[None, :].expand(N, N)  # [N, N]
            freq_r = (freq_u ** 2 + freq_v ** 2).sqrt()
            mid_mask = ((freq_r >= 2) & (freq_r < 7)).to(torch.float64)  # [N, N]
            z_dct = z_dct * mid_mask[None, None, :, :]

            # IDCT: spatial[i,j] = (2/N) * sum_u sum_v coeff[u,v] * basis[u,i] * basis[v,j]
            # = (2/N) * basis_u^T @ coeff @ basis_v
            # With orthonormal normalization: include c(u), c(v) factors
            c = torch.ones(N, device=pixels.device, dtype=torch.float64)
            c[0] = 1.0 / (2.0 ** 0.5)
            norm_factor = (2.0 / N) ** 0.5
            # Weighted basis: c[u] * cos(...) * sqrt(2/N)
            weighted_u = basis_u * c[:, None] * norm_factor  # [N, N]
            weighted_v = basis_v * c[:, None] * norm_factor  # [N, N]

            # IDCT for each batch/channel: spatial = weighted_u^T @ coeff @ weighted_v
            # batched: [B, C, N, N]
            z_spatial = torch.einsum("ui,bcuv,vj->bcij", weighted_u.to(dtype=z_dct.dtype), z_dct, weighted_v.to(dtype=z_dct.dtype))

            # Bilinear upsample to full resolution
            noise_dct = F.interpolate(
                z_spatial.to(dtype=pixels.dtype), size=(H, W), mode="bilinear", align_corners=False
            )

            # Normalize
            noise_dct = noise_dct - noise_dct.mean(dim=(2, 3), keepdim=True)
            noise_dct = noise_dct / noise_dct.square().mean(
                dim=(1, 2, 3), keepdim=True
            ).sqrt().clamp_min(1e-12)

            # Combine
            noise = (alpha ** 0.5) * noise_opponent + ((1.0 - alpha) ** 0.5) * noise_dct
            return self.guide_aug_strength * noise
        if self.patch_dropout_noise_mode == "patch_cov_gaussian":
            B, C, H, W = pixels.shape
            patch_h = H // grid_size
            patch_w = W // grid_size
            if patch_h * grid_size != H or patch_w * grid_size != W:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)
            rho = 0.2
            iid = torch.randn_like(pixels, dtype=pixels.dtype)
            patch_latent = torch.randn(B, C, grid_size, grid_size, device=pixels.device, dtype=pixels.dtype)
            patch_noise = F.interpolate(patch_latent, size=(H, W), mode="nearest")
            noise = ((1.0 - rho) ** 0.5) * iid + (rho ** 0.5) * patch_noise
            return self.guide_aug_strength * noise
        if self.patch_dropout_noise_mode == "score_weighted_gaussian":
            B, C, H, W = pixels.shape
            if self._patch_scores is None or self._patch_scores.size(1) != grid_size * grid_size:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)
            scores = self._patch_scores.to(device=pixels.device, dtype=pixels.dtype)
            centered = scores - scores.mean(dim=1, keepdim=True)
            scaled = centered / centered.std(dim=1, keepdim=True).clamp_min(1e-6)
            patch_weights = torch.exp(0.5 * scaled).clamp(0.5, 2.0)
            patch_weights = patch_weights / patch_weights.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            weights = F.interpolate(
                patch_weights.view(B, 1, grid_size, grid_size),
                size=(H, W),
                mode="nearest",
            ).expand(-1, C, -1, -1)
            return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype) * weights
        if self.patch_dropout_noise_mode == "inverse_score_weighted_gaussian":
            B, C, H, W = pixels.shape
            if self._patch_scores is None or self._patch_scores.size(1) != grid_size * grid_size:
                return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)
            scores = self._patch_scores.to(device=pixels.device, dtype=pixels.dtype)
            centered = scores - scores.mean(dim=1, keepdim=True)
            scaled = centered / centered.std(dim=1, keepdim=True).clamp_min(1e-6)
            patch_weights = torch.exp(-0.5 * scaled).clamp(0.5, 2.0)
            patch_weights = patch_weights / patch_weights.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
            weights = F.interpolate(
                patch_weights.view(B, 1, grid_size, grid_size),
                size=(H, W),
                mode="nearest",
            ).expand(-1, C, -1, -1)
            return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype) * weights
        if self.patch_dropout_noise_mode == "antithetic_gaussian":
            if (
                self._patch_dropout_antithetic_noise is not None
                and self._patch_dropout_antithetic_noise.shape == pixels.shape
                and self._patch_dropout_antithetic_noise.device == pixels.device
                and self._patch_dropout_antithetic_noise.dtype == pixels.dtype
            ):
                noise = self._patch_dropout_antithetic_noise
                self._patch_dropout_antithetic_noise = None
                return self.guide_aug_strength * noise
            noise = torch.randn_like(pixels, dtype=pixels.dtype)
            self._patch_dropout_antithetic_noise = -noise
            return self.guide_aug_strength * noise

        B, C, H, W = pixels.shape
        patch_h = H // grid_size
        patch_w = W // grid_size
        if patch_h * grid_size != H or patch_w * grid_size != W:
            return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)

        weight = self._patch_embed_projection_weight(patch_h, patch_w)
        if weight is None:
            return self.guide_aug_strength * torch.randn_like(pixels, dtype=pixels.dtype)

        num_patches = grid_size * grid_size
        coeff = torch.randn(B, num_patches, weight.size(0), device=pixels.device, dtype=weight.dtype)
        patch_noise = torch.einsum("bnd,dl->bnl", coeff, weight.to(pixels.device))
        patch_noise = patch_noise.reshape(B, grid_size, grid_size, C, patch_h, patch_w)
        noise = patch_noise.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W).to(dtype=pixels.dtype)
        noise = noise - noise.mean(dim=(2, 3), keepdim=True)
        noise = noise / noise.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-12)
        return self.guide_aug_strength * noise

    def _patch_dropout_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """Randomly corrupt selected patch-score regions per guide augmentation copy.

        Marks patches above or below the per-image median CLS cosine score
        according to ``patch_dropout_score_mode``, then randomly selects
        ``patch_dropout_ratio`` of them and applies the configured fill mode
        to the corresponding image regions.
        """
        if self._patch_scores is None:
            return pixels

        B, C, H, W = pixels.shape
        N = self._patch_scores.size(1)  # 196 for ViT-B/16
        gh = int(round(N ** 0.5))
        if gh * gh != N:
            return pixels  # non-square grid; fall back to identity

        # Mark the configured score subset around the per-image median.
        median = self._patch_scores.median(dim=1, keepdim=True).values  # [B, 1]
        if self.patch_dropout_score_mode == "high":
            candidate_mask = self._patch_scores > median
        else:
            candidate_mask = self._patch_scores < median
        candidate_mask = candidate_mask.to(torch.float32)               # [B, N]

        # Per batch item: randomly select patch_dropout_ratio of candidate patches.
        pixel_mask: torch.Tensor | None = None
        if self.patch_dropout_ratio > 0:
            drop_mask = torch.zeros_like(candidate_mask)  # [B, N]
            for b in range(B):
                candidate_idx = candidate_mask[b].nonzero(as_tuple=True)[0]
                n_candidates = candidate_idx.numel()
                if n_candidates == 0:
                    continue
                n_drop = max(1, int(round(n_candidates * self.patch_dropout_ratio)))
                perm = torch.randperm(n_candidates, device=pixels.device)[:n_drop]
                drop_mask[b, candidate_idx[perm]] = 1.0
            pixel_mask = F.interpolate(
                drop_mask.view(B, 1, gh, gh), size=(H, W), mode="nearest",
            )  # [B, 1, H, W]
            pixel_mask = pixel_mask.expand(-1, C, -1, -1)  # [B, C, H, W]

        # Add noise to non-dropped patches; mode controls its spatial/token structure.
        noised = torch.clamp(pixels + self._patch_dropout_noise(pixels, gh), 0.0, 1.0)
        if pixel_mask is None:
            return noised
        if self.patch_dropout_fill_mode == "zero_noise":
            return torch.where(pixel_mask > 0.5, torch.zeros_like(pixels), noised)

        local_context = F.avg_pool2d(pixels, kernel_size=5, stride=1, padding=2)
        fill_pixels = local_context
        patch_h = H // gh
        patch_w = W // gh
        if patch_h * gh == H and patch_w * gh == W:
            source_patches = (
                pixels.view(B, C, gh, patch_h, gh, patch_w)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(B, N, C, patch_h, patch_w)
            )
            local_patches = (
                local_context.view(B, C, gh, patch_h, gh, patch_w)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(B, N, C, patch_h, patch_w)
            )
            fill_patches = local_patches.clone()
            high_score_mask = self._patch_scores > median
            for b in range(B):
                target_idx = drop_mask[b].nonzero(as_tuple=True)[0]
                donor_pool = high_score_mask[b].nonzero(as_tuple=True)[0]
                if target_idx.numel() == 0 or donor_pool.numel() == 0:
                    continue
                if self.patch_dropout_fill_mode in ("context_high_score_blend", "random_high_score_inpaint"):
                    donor_idx = donor_pool[
                        torch.randint(donor_pool.numel(), (target_idx.numel(),), device=pixels.device)
                    ]
                    donor_patches = source_patches[b, donor_idx]
                    if self.patch_dropout_fill_mode == "context_high_score_blend":
                        fill_patches[b, target_idx] = 0.5 * local_patches[b, target_idx] + 0.5 * donor_patches
                    else:
                        fill_patches[b, target_idx] = donor_patches
                else:
                    target_y = target_idx.div(gh, rounding_mode="floor")
                    target_x = target_idx.remainder(gh)
                    donor_y = donor_pool.div(gh, rounding_mode="floor")
                    donor_x = donor_pool.remainder(gh)
                    distance = (
                        (target_y[:, None] - donor_y[None, :]).square()
                        + (target_x[:, None] - donor_x[None, :]).square()
                    )
                    nearest = distance.argmin(dim=1)
                    fill_patches[b, target_idx] = source_patches[b, donor_pool[nearest]]
            fill_pixels = (
                fill_patches.view(B, gh, gh, C, patch_h, patch_w)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(B, C, H, W)
            )
        return torch.where(pixel_mask > 0.5, fill_pixels.clamp(0.0, 1.0), noised)

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
        self._patch_scores = None
        if self.attack_loss == "feature":
            with torch.no_grad():
                clean_feature_target = self._extract_layer_patch_features(clean_pixels).detach()

        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)
        sign_history: list[torch.Tensor] = []

        for step_idx in range(self.steps):
            grad_pixels = adv_pixels.detach()
            if self.nesterov and step_idx > 0:
                with torch.no_grad():
                    grad_pixels = grad_pixels + self.decay * self.step_size * momentum.sign()
                    delta = grad_pixels - clean_pixels
                    delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                    grad_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)

            grad_pixels = grad_pixels.detach()
            self._compute_patch_scores(grad_pixels)
            grad_pixels = grad_pixels.requires_grad_(True)
            if self.lowmid_dss_filter or self.view_consistent_agreement:
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
            agreement_term_grads = None
            if term_grads is not None:
                agreement_term_grads = tuple(
                    self._normalize_grad(self._tune_lowmid_gradient(term)) for term in term_grads
                )

            if self.use_momentum:
                momentum = self.decay * momentum + grad
                update = momentum
            else:
                update = grad
            update = self._apply_grad_momentum_agreement(update, grad)
            update = self._apply_view_consistent_agreement(update, agreement_term_grads)
            update = self._apply_cross_step_sign_vote(update, sign_history)
            update = self._apply_spatial_sign_reinforcement(update)
            update = self._apply_fft_sign_regularization(update)

            with torch.no_grad():
                adv_pixels = adv_pixels + self.step_size * update.sign()
                delta = adv_pixels - clean_pixels
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

        return self._normalize(adv_pixels)
