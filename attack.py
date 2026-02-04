# attack.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD

try:
    import lightly.transforms as _lt
    RandomResize = getattr(_lt, "RandomResize", None)
    RandomCrop = getattr(_lt, "RandomCrop", None)
    HorizontalFlip = getattr(_lt, "HorizontalFlip", None)
    GaussianBlur = getattr(_lt, "GaussianBlur", None)
    ColorJitter = getattr(_lt, "ColorJitter", None)
    _LIGHTLY_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    RandomResize = RandomCrop = HorizontalFlip = GaussianBlur = ColorJitter = None
    _LIGHTLY_IMPORT_ERROR = exc


class AttentionFoolImageAttacker:
    """
    Two-stage common-evidence attack with ensemble support.
    Stage1: amplify common evidence focus.
    Stage2: mislead classification while preserving evidence focus and stability.
    """
    def __init__(
        self,
        model,
        models: List | None = None,
        img_size: int = 224,
        steps: int = 100,
        stage1_steps: int = 0,
        step_size: float = 1.0 / 255.0,
        eps: float = 8.0 / 255.0,
        k_common: int = 8,
        num_views: int = 8,
        noise_eps: float = 4.0 / 255.0,
        importance_method: str = "grad_token",
        tau: float = 0.07,
        lambda_focus: float = 1.0,
        lambda_stab: float = 1.0,
        lambda_preserve: float = 1.0,
        lambda_focus2: float | None = None,
        lambda_stab2: float | None = None,
        lambda_ce1: float = 0.0,
        lambda_var_model: float = 1.0,
        lambda_var_aug: float = 1.0,
        norm_type: str = "linf",
        use_momentum: bool = True,
        momentum_mu: float = 0.9,
        log_every: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self._use_lightly = all([RandomResize, RandomCrop, HorizontalFlip, GaussianBlur, ColorJitter])
        if not self._use_lightly:
            if _LIGHTLY_IMPORT_ERROR is not None:
                reason = f"{_LIGHTLY_IMPORT_ERROR}"
            else:
                missing = [name for name, obj in [
                    ("RandomResize", RandomResize),
                    ("RandomCrop", RandomCrop),
                    ("HorizontalFlip", HorizontalFlip),
                    ("GaussianBlur", GaussianBlur),
                    ("ColorJitter", ColorJitter),
                ] if obj is None]
                reason = f"missing {', '.join(missing)}"
            print(
                "Warning: lightly transforms not available; falling back to torchvision transforms "
                f"({reason})."
            )

        self.model = model
        if models is None or len(models) == 0:
            self.models = [model]
        else:
            self.models = models

        for m in self.models:
            m.eval()

        self.img_size = img_size
        self.steps = steps
        self.stage1_steps = stage1_steps
        self.step_size = step_size
        self.eps = eps
        self.k_common = k_common
        self.num_views = num_views
        self.noise_eps = noise_eps
        self.importance_method = importance_method
        self.tau = tau
        self.lambda_focus = lambda_focus
        self.lambda_stab = lambda_stab
        self.lambda_preserve = lambda_preserve
        self.lambda_focus2 = lambda_focus2 if lambda_focus2 is not None else lambda_focus
        self.lambda_stab2 = lambda_stab2 if lambda_stab2 is not None else lambda_stab
        self.lambda_ce1 = lambda_ce1
        self.lambda_var_model = lambda_var_model
        self.lambda_var_aug = lambda_var_aug
        self.norm_type = norm_type
        self.use_momentum = use_momentum
        self.momentum_mu = momentum_mu
        self.log_every = log_every

        self.device = device if device is not None else DEVICE

        patch_size = 16
        patch_embed = getattr(getattr(model, "model", None), "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            patch_size = patch_embed.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)

        self.pixel_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        self._to_pil = transforms.ToPILImage()
        self._to_tensor = transforms.ToTensor()
        self._pil_aug = self._build_pil_aug()

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _softmax_normalize(self, scores: torch.Tensor, tau: float) -> torch.Tensor:
        if tau <= 0:
            scores = torch.relu(scores)
            return scores / (scores.sum(dim=-1, keepdim=True) + 1e-12)
        return torch.softmax(scores / tau, dim=-1)

    def _safe_instantiate(self, cls, kwargs_list: List[dict]) -> object:
        for kwargs in kwargs_list:
            try:
                return cls(**kwargs)
            except TypeError:
                continue
        return cls()

    def _build_pil_aug(self) -> transforms.Compose:
        if self._use_lightly:
            resize = self._safe_instantiate(
                RandomResize,
                [
                    {"min_size": self.img_size, "max_size": int(self.img_size * 1.1)},
                    {"size": int(self.img_size * 1.1)},
                    {"size": self.img_size},
                ],
            )
            crop = self._safe_instantiate(
                RandomCrop,
                [
                    {"size": self.img_size},
                    {"size": (self.img_size, self.img_size)},
                ],
            )
            flip = self._safe_instantiate(HorizontalFlip, [{"p": 0.5}, {}])
            blur = self._safe_instantiate(
                GaussianBlur,
                [
                    {"kernel_size": 3, "sigma": (0.1, 0.5)},
                    {"sigma": (0.1, 0.5)},
                    {},
                ],
            )
            jitter = self._safe_instantiate(
                ColorJitter,
                [
                    {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02},
                    {},
                ],
            )
            return transforms.Compose(
                [
                    resize,
                    crop,
                    flip,
                    blur,
                    jitter,
                    transforms.Resize((self.img_size, self.img_size)),
                ]
            )

        # Fallback to torchvision transforms
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.img_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.Resize((self.img_size, self.img_size)),
            ]
        )

    def _attention_rollout(self, attn_list: List[torch.Tensor]) -> torch.Tensor:
        if not attn_list:
            raise ValueError("Attention list is empty. Ensure model is called with return_attn=True.")

        num_tokens = attn_list[0].shape[-1]
        joint = None
        for attn in attn_list:
            attn_heads = attn.mean(dim=1)  # [B, N, N]
            eye = torch.eye(num_tokens, device=attn_heads.device).unsqueeze(0)
            attn_heads = attn_heads + eye
            attn_heads = attn_heads / (attn_heads.sum(dim=-1, keepdim=True) + 1e-12)
            if joint is None:
                joint = attn_heads
            else:
                joint = attn_heads @ joint

        cls_to_patches = joint[:, 0, 1:]
        return cls_to_patches

    def _apply_pil_aug(self, image: Image.Image) -> torch.Tensor:
        aug = self._pil_aug(image)
        if isinstance(aug, torch.Tensor):
            tensor = aug
        else:
            tensor = self._to_tensor(aug)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        tensor = tensor.clamp(0.0, 1.0)
        return self._normalize(tensor)

    def _build_selection_views(self, image_norm: torch.Tensor) -> List[torch.Tensor]:
        views: List[torch.Tensor] = []
        image_pixels = self._denormalize(image_norm).clamp(0.0, 1.0)
        base_pil = self._to_pil(image_pixels.squeeze(0).cpu())

        num_aug = max(1, self.num_views // 2)
        num_noise = max(0, self.num_views - num_aug)

        for _ in range(num_aug):
            views.append(self._apply_pil_aug(base_pil))

        for _ in range(num_noise):
            noise = torch.randn_like(image_pixels) * self.noise_eps
            noisy_pixels = (image_pixels + noise).clamp(0.0, 1.0)
            views.append(self._normalize(noisy_pixels))

        return views

    def _diff_augment(self, image_pixels: torch.Tensor) -> torch.Tensor:
        # Differentiable light augmentations on tensor inputs.
        _, _, h, w = image_pixels.shape
        scale = float(torch.empty(1).uniform_(0.9, 1.0))
        ratio = float(torch.empty(1).uniform_(0.9, 1.1))
        crop_h = max(1, int(h * scale))
        crop_w = max(1, int(w * scale * ratio))
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        i = int(torch.empty(1).uniform_(0, h - crop_h + 1))
        j = int(torch.empty(1).uniform_(0, w - crop_w + 1))
        cropped = image_pixels[:, :, i:i + crop_h, j:j + crop_w]
        resized = F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)

        if float(torch.rand(1)) < 0.5:
            resized = torch.flip(resized, dims=[3])

        brightness = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
        contrast = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
        saturation = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))

        jittered = resized * brightness
        mean = jittered.mean(dim=(2, 3), keepdim=True)
        jittered = (jittered - mean) * contrast + mean
        gray = jittered.mean(dim=1, keepdim=True)
        jittered = jittered * saturation + gray * (1.0 - saturation)

        sigma = float(torch.empty(1).uniform_(0.1, 0.5))
        jittered = TF.gaussian_blur(jittered, kernel_size=[3, 3], sigma=[sigma, sigma])

        jittered = jittered.clamp(0.0, 1.0)
        return self._normalize(jittered)

    def _build_diff_views(self, image_pixels: torch.Tensor) -> List[torch.Tensor]:
        views: List[torch.Tensor] = []
        num_aug = max(1, self.num_views // 2)
        num_noise = max(0, self.num_views - num_aug)

        for _ in range(num_aug):
            views.append(self._diff_augment(image_pixels))

        for _ in range(num_noise):
            noise = torch.randn_like(image_pixels) * self.noise_eps
            noisy_pixels = (image_pixels + noise).clamp(0.0, 1.0)
            views.append(self._normalize(noisy_pixels))

        return views

    def _compute_common_evidence(
        self,
        image_norm: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        views = self._build_selection_views(image_norm)

        per_model_view: List[torch.Tensor] = []
        with torch.no_grad():
            for model in self.models:
                view_scores: List[torch.Tensor] = []
                for view in views:
                    _logits, attn_list = model(view, return_attn=True)
                    rollout = self._attention_rollout(attn_list)
                    rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-12)
                    view_scores.append(rollout.squeeze(0))
                per_model_view.append(torch.stack(view_scores, dim=0))

        scores = torch.stack(per_model_view, dim=0)  # [M, V, P]
        mean = scores.mean(dim=(0, 1))
        mean_per_model = scores.mean(dim=1)
        var_model = mean_per_model.var(dim=0, unbiased=False) if scores.size(0) > 1 else torch.zeros_like(mean)
        var_aug = scores.var(dim=1, unbiased=False).mean(dim=0)

        common_score = mean - self.lambda_var_model * var_model - self.lambda_var_aug * var_aug
        k = min(self.k_common, common_score.numel())
        topk = torch.topk(common_score, k=k, dim=0).indices
        W = self._softmax_normalize(common_score, self.tau)

        stats = {
            "mean": mean.detach().cpu(),
            "var_model": var_model.detach().cpu(),
            "var_aug": var_aug.detach().cpu(),
            "common_score": common_score.detach().cpu(),
        }
        return topk, W, stats

    def _build_patch_mask(self, patch_indices: torch.Tensor, img_size: int) -> torch.Tensor:
        num_patches_total = (img_size // self.patch_size) ** 2
        grid = int(np.sqrt(num_patches_total))
        if grid * grid != num_patches_total:
            raise ValueError("Unexpected patch grid size. Check img_size and patch_size.")

        mask = torch.zeros((1, 1, img_size, img_size), device=self.device)
        grid_w = img_size // self.patch_size
        for idx in patch_indices.tolist():
            row = idx // grid_w
            col = idx % grid_w
            r0 = row * self.patch_size
            r1 = r0 + self.patch_size
            c0 = col * self.patch_size
            c1 = c0 + self.patch_size
            mask[:, :, r0:r1, c0:c1] = 1.0
        return mask

    def _compute_token_importance(
        self,
        model,
        images_norm: torch.Tensor,
        labels: torch.Tensor,
        method: str,
        tau: float,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if method not in {"grad_token", "legrad", "attn_rollout"}:
            raise ValueError(f"Unknown importance method: {method}")

        if method in {"grad_token", "legrad"}:
            if not torch.is_grad_enabled():
                with torch.enable_grad():
                    logits, tokens = model(images_norm, return_tokens=True)
                    if tokens is None:
                        raise RuntimeError("Token embeddings not captured; ensure model supports return_tokens.")
                    ce_loss = F.cross_entropy(logits, labels)
                    grads = torch.autograd.grad(
                        ce_loss,
                        tokens,
                        retain_graph=True,
                        create_graph=create_graph,
                    )[0]
                    scores = (grads * tokens).abs().sum(dim=-1)
                    scores = scores[:, 1:]
                    importance = self._softmax_normalize(scores, tau)
                    return importance, logits, ce_loss
            else:
                logits, tokens = model(images_norm, return_tokens=True)
                if tokens is None:
                    raise RuntimeError("Token embeddings not captured; ensure model supports return_tokens.")
                ce_loss = F.cross_entropy(logits, labels)
                grads = torch.autograd.grad(
                    ce_loss,
                    tokens,
                    retain_graph=True,
                    create_graph=create_graph,
                )[0]
                scores = (grads * tokens).abs().sum(dim=-1)
                scores = scores[:, 1:]
                importance = self._softmax_normalize(scores, tau)
                return importance, logits, ce_loss

        logits, attn_list = model(images_norm, return_attn=True)
        rollout = self._attention_rollout(attn_list)
        importance = self._softmax_normalize(rollout, tau)
        ce_loss = F.cross_entropy(logits, labels)
        return importance, logits, ce_loss

    def _compute_importance_stack(
        self,
        image_pixels: torch.Tensor,
        label: torch.Tensor,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        views = self._build_diff_views(image_pixels)
        per_model_view: List[torch.Tensor] = []
        ce_losses: List[torch.Tensor] = []

        for model in self.models:
            view_scores: List[torch.Tensor] = []
            for view in views:
                importance, _logits, ce_loss = self._compute_token_importance(
                    model=model,
                    images_norm=view,
                    labels=label,
                    method=self.importance_method,
                    tau=self.tau,
                    create_graph=create_graph,
                )
                view_scores.append(importance.squeeze(0))
                ce_losses.append(ce_loss)
            per_model_view.append(torch.stack(view_scores, dim=0))

        stack = torch.stack(per_model_view, dim=0)  # [M, V, P]
        ce_mean = torch.stack(ce_losses).mean()
        return stack, ce_mean

    def _compute_focus_loss(
        self,
        stack: torch.Tensor,
        W: torch.Tensor,
        S_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if W is not None:
            weights = W.to(stack.device)
        else:
            weights = S_mask.to(stack.device)

        mass = (stack * weights).sum(dim=-1)
        focus_loss = -torch.log(mass + 1e-6).mean()
        return focus_loss, mass.mean()

    def _compute_stability_loss(self, stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # stack: [M, V, P]
        mean_view = stack.mean(dim=1, keepdim=True)
        var_aug = ((stack - mean_view) ** 2).mean()
        if stack.size(0) > 1:
            mean_model = stack.mean(dim=0, keepdim=True)
            var_model = ((stack - mean_model) ** 2).mean()
        else:
            var_model = torch.zeros_like(var_aug)
        return var_aug, var_model

    def _project_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "linf":
            return delta.clamp(-self.eps, self.eps)
        if self.norm_type == "l2":
            flat = delta.view(delta.size(0), -1)
            norm = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
            factor = torch.clamp(self.eps / norm, max=1.0)
            return (flat * factor).view_as(delta)
        raise ValueError(f"Unknown norm type: {self.norm_type}")

    def _pgd_update(self, delta: torch.Tensor, grad: torch.Tensor, momentum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_momentum:
            grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
            momentum = self.momentum_mu * momentum + grad / grad_norm
            update = momentum
        else:
            update = grad
        delta = delta + self.step_size * update.sign()
        delta = self._project_delta(delta)
        return delta, momentum

    def _pgd_two_stage(
        self,
        image_norm: torch.Tensor,
        label: torch.Tensor,
        S_indices: torch.Tensor,
        W: torch.Tensor,
        init: str = "zero",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        image_norm = image_norm.to(self.device)
        label = label.to(self.device)

        image_pixels = self._denormalize(image_norm).clamp(0.0, 1.0)
        grid_w = self.img_size // self.patch_size
        num_patches = grid_w * grid_w
        S_mask = torch.zeros((1, num_patches), device=self.device)
        S_mask[:, S_indices] = 1.0
        W = W.view(1, -1)
        mask = self._build_patch_mask(S_indices, image_pixels.shape[-1])

        if init == "rand":
            delta = torch.empty_like(image_pixels).uniform_(-self.eps, self.eps)
            delta = self._project_delta(delta)
        elif init == "zero":
            delta = torch.zeros_like(image_pixels)
        else:
            raise ValueError(f"Unknown init type: {init}")
        momentum = torch.zeros_like(delta)

        logs: Dict[str, List[float]] = {"stage1": [], "stage2": []}

        for step in range(self.stage1_steps):
            delta.requires_grad_(True)
            adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
            stack, ce_mean = self._compute_importance_stack(adv_pixels, label, create_graph=True)
            focus_loss, mass_mean = self._compute_focus_loss(stack, W, S_mask)
            var_aug, var_model = self._compute_stability_loss(stack)
            stab_loss = var_aug + var_model
            total = self.lambda_focus * focus_loss + self.lambda_stab * stab_loss + self.lambda_ce1 * ce_mean

            grad = torch.autograd.grad(total, delta, retain_graph=False)[0]
            delta, momentum = self._pgd_update(delta, grad, momentum)
            delta = delta.detach()

            if self.log_every > 0 and (step + 1) % self.log_every == 0:
                logs["stage1"].append(
                    float(total.detach().cpu())
                )
                print(
                    f"Stage1[{step + 1}/{self.stage1_steps}] "
                    f"L_focus={focus_loss.item():.4f} "
                    f"L_stab={stab_loss.item():.4f} "
                    f"L_ce={ce_mean.item():.4f} "
                    f"mass={mass_mean.item():.4f}"
                )

        if self.stage1_steps == 0:
            A_ref = W.detach()
        else:
            with torch.no_grad():
                adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
                stack, _ = self._compute_importance_stack(adv_pixels, label, create_graph=False)
                A_ref = stack.mean(dim=(0, 1)).detach()

        for step in range(self.steps):
            delta.requires_grad_(True)
            adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
            stack, ce_mean = self._compute_importance_stack(adv_pixels, label, create_graph=True)
            focus_loss, mass_mean = self._compute_focus_loss(stack, W, S_mask)
            var_aug, var_model = self._compute_stability_loss(stack)
            stab_loss = var_aug + var_model
            preserve_loss = ((stack - A_ref) ** 2).mean()

            total = (
                ce_mean
                + self.lambda_focus2 * focus_loss
                + self.lambda_stab2 * stab_loss
                + self.lambda_preserve * preserve_loss
            )

            grad = torch.autograd.grad(total, delta, retain_graph=False)[0]
            delta, momentum = self._pgd_update(delta, grad, momentum)
            delta = delta.detach()

            if self.log_every > 0 and (step + 1) % self.log_every == 0:
                logs["stage2"].append(float(total.detach().cpu()))
                print(
                    f"Stage2[{step + 1}/{self.steps}] "
                    f"L_ce={ce_mean.item():.4f} "
                    f"L_focus={focus_loss.item():.4f} "
                    f"L_stab={stab_loss.item():.4f} "
                    f"L_preserve={preserve_loss.item():.4f} "
                    f"mass={mass_mean.item():.4f}"
                )

        adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
        adv_norm = self._normalize(adv_pixels)

        with torch.no_grad():
            stack, _ = self._compute_importance_stack(adv_pixels, label, create_graph=False)
            A_stage2 = stack.mean(dim=(0, 1)).detach()
            per_model = stack.mean(dim=1).detach()

        extra = {
            "A_ref": A_ref.cpu(),
            "A_stage2": A_stage2.cpu(),
            "per_model": per_model.cpu(),
        }
        return adv_norm, delta, mask.detach(), extra

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        init: str = "zero",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
        images = images.to(self.device)
        labels = labels.to(self.device)

        adv_list: List[torch.Tensor] = []
        delta_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        patch_indices_list: List[torch.Tensor] = []
        extra_list: List[Dict[str, torch.Tensor]] = []

        for idx in range(images.size(0)):
            image = images[idx:idx + 1]
            label = labels[idx:idx + 1]
            S_indices, W, stats = self._compute_common_evidence(image, label)
            adv, delta, mask, extra = self._pgd_two_stage(image, label, S_indices, W, init=init)
            adv_list.append(adv)
            delta_list.append(delta)
            mask_list.append(mask)
            patch_indices_list.append(S_indices)
            extra.update(stats)
            extra_list.append(extra)

        x_adv = torch.cat(adv_list, dim=0)
        delta = torch.cat(delta_list, dim=0)
        masks = torch.cat(mask_list, dim=0)
        return x_adv, delta, masks, patch_indices_list, extra_list

    def _build_vis_grid(
        self,
        clean_norm: torch.Tensor,
        adv_norm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        clean = self._denormalize(clean_norm).clamp(0.0, 1.0)
        adv = self._denormalize(adv_norm).clamp(0.0, 1.0)
        diff = (adv - clean).abs()
        diff_vis = (diff / (self.eps + 1e-12)).clamp(0.0, 1.0)

        mask_1c = mask.to(clean.device)[:, :1, :, :]
        overlay = clean.clone()
        red = torch.tensor([1.0, 0.0, 0.0], device=overlay.device).view(1, 3, 1, 1)
        alpha = 0.4
        overlay = overlay * (1.0 - alpha * mask_1c) + red * (alpha * mask_1c)
        overlay = overlay.clamp(0.0, 1.0)

        grid = make_grid(
            torch.cat([clean, adv, diff_vis, overlay], dim=0),
            nrow=2,
            padding=2,
        )
        return grid

    def _map_to_image(self, values: torch.Tensor) -> torch.Tensor:
        grid_w = self.img_size // self.patch_size
        map_2d = values.view(1, 1, grid_w, grid_w)
        map_2d = map_2d - map_2d.min()
        map_2d = map_2d / (map_2d.max() + 1e-12)
        up = F.interpolate(map_2d, size=(self.img_size, self.img_size), mode="nearest")
        return up.repeat(1, 3, 1, 1)

    def save_visualizations(
        self,
        clean_images: torch.Tensor,
        adv_images: torch.Tensor,
        masks: torch.Tensor,
        output_dir: str,
        filenames: List[str] | None = None,
        extras: List[Dict[str, torch.Tensor]] | None = None,
    ) -> List[Path]:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        for idx in range(clean_images.size(0)):
            grid = self._build_vis_grid(
                clean_images[idx:idx + 1],
                adv_images[idx:idx + 1],
                masks[idx:idx + 1],
            )
            if filenames is not None:
                stem = Path(filenames[idx]).stem
                filename = f"vis_{stem}.png"
            else:
                stem = f"{idx:05d}"
                filename = f"vis_{stem}.png"
            path = output_dir_path / filename
            save_image(grid, str(path))
            saved.append(path)

            if extras is None:
                continue

            extra = extras[idx]
            mean_map = self._map_to_image(extra["mean"].to(self.device))
            var_model = self._map_to_image(extra["var_model"].to(self.device))
            var_aug = self._map_to_image(extra["var_aug"].to(self.device))
            common = self._map_to_image(extra["common_score"].to(self.device))
            mask_1c = masks[idx:idx + 1].to(self.device)[:, :1, :, :]
            overlay = mean_map.clone()
            red = torch.tensor([1.0, 0.0, 0.0], device=overlay.device).view(1, 3, 1, 1)
            overlay = overlay * (1.0 - 0.4 * mask_1c) + red * (0.4 * mask_1c)
            overlay = overlay.clamp(0.0, 1.0)
            evidence_grid = make_grid(
                torch.cat([mean_map, var_model, var_aug, common, overlay], dim=0),
                nrow=3,
                padding=2,
            )
            ev_path = output_dir_path / f"evidence_{stem}.png"
            save_image(evidence_grid, str(ev_path))
            saved.append(ev_path)

            A_ref = self._map_to_image(extra["A_ref"].to(self.device))
            A_stage2 = self._map_to_image(extra["A_stage2"].to(self.device))
            diff = (A_stage2 - A_ref).abs()
            stage_grid = make_grid(torch.cat([A_ref, A_stage2, diff], dim=0), nrow=3, padding=2)
            st_path = output_dir_path / f"stage_{stem}.png"
            save_image(stage_grid, str(st_path))
            saved.append(st_path)

            per_model = extra.get("per_model")
            if per_model is not None and per_model.size(0) > 1:
                maps = [self._map_to_image(per_model[i].to(self.device)) for i in range(per_model.size(0))]
                model_grid = make_grid(torch.cat(maps, dim=0), nrow=min(4, len(maps)), padding=2)
                md_path = output_dir_path / f"ensemble_{stem}.png"
                save_image(model_grid, str(md_path))
                saved.append(md_path)

        return saved
