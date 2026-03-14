from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD


class AttentionFoolImageAttacker:
    """
    Single-model, multi-view attribution concentration attack.

    The perturbation is optimized so that different augmented views of the same
    adversarial sample:
    1. are misclassified,
    2. rely on a shared attribution pattern,
    3. compress that attribution into a compact region,
    4. use that compact region to support the wrong decision.
    """

    def __init__(
        self,
        model,
        img_size: int = 224,
        steps: int = 100,
        step_size: float = 1.0 / 255.0,
        eps: float = 8.0 / 255.0,
        region_topk: int = 8,
        num_views: int = 8,
        noise_eps: float = 4.0 / 255.0,
        tau: float = 0.07,
        lambda_cls: float = 1.0,
        lambda_align: float = 1.0,
        lambda_compact: float = 1.0,
        lambda_couple: float = 1.0,
        norm_type: str = "linf",
        use_momentum: bool = True,
        momentum_mu: float = 0.9,
        log_every: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.model.eval()

        self.img_size = img_size
        self.steps = steps
        self.step_size = step_size
        self.eps = eps
        self.region_topk = region_topk
        self.num_views = num_views
        self.noise_eps = noise_eps
        self.tau = tau
        self.lambda_cls = lambda_cls
        self.lambda_align = lambda_align
        self.lambda_compact = lambda_compact
        self.lambda_couple = lambda_couple
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

        self.pixel_mean = torch.tensor(
            IMAGENET_MEAN, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(
            IMAGENET_STD, dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _softmax_normalize(self, scores: torch.Tensor, tau: float) -> torch.Tensor:
        if tau <= 0:
            scores = torch.relu(scores)
            return scores / (scores.sum(dim=-1, keepdim=True) + 1e-12)
        return torch.softmax(scores / tau, dim=-1)

    def _normalize_distribution(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.clamp(values, min=0.0)
        return values / (values.sum(dim=-1, keepdim=True) + 1e-12)

    def _diff_augment(self, image_pixels: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image_pixels.shape
        scale = float(torch.empty(1).uniform_(0.9, 1.0))
        ratio = float(torch.empty(1).uniform_(0.9, 1.1))
        crop_h = min(h, max(1, int(h * scale)))
        crop_w = min(w, max(1, int(w * scale * ratio)))

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
        views: List[torch.Tensor] = [self._normalize(image_pixels)]
        if self.num_views <= 1:
            return views

        remaining = self.num_views - 1
        num_aug = max(1, remaining // 2)
        num_noise = max(0, remaining - num_aug)

        for _ in range(num_aug):
            views.append(self._diff_augment(image_pixels))

        for _ in range(num_noise):
            noise = torch.randn_like(image_pixels) * self.noise_eps
            noisy_pixels = (image_pixels + noise).clamp(0.0, 1.0)
            views.append(self._normalize(noisy_pixels))

        return views

    def _class_token_attribution_from_logits(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        class_indices: torch.Tensor,
        create_graph: bool,
    ) -> torch.Tensor:
        scores = logits.gather(1, class_indices.view(-1, 1)).sum()
        grads = torch.autograd.grad(
            scores,
            tokens,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        token_scores = (grads * tokens).abs().sum(dim=-1)
        token_scores = token_scores[:, 1:]
        return self._softmax_normalize(token_scores, self.tau)

    def _compute_view_attribution_stack(
        self,
        image_pixels: torch.Tensor,
        label: torch.Tensor,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def _inner() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            views = self._build_diff_views(image_pixels)
            logits_list: List[torch.Tensor] = []
            tokens_list: List[torch.Tensor] = []
            ce_losses: List[torch.Tensor] = []

            for view in views:
                logits, tokens = self.model(view, return_tokens=True)
                if tokens is None:
                    raise RuntimeError("Token embeddings not captured; ensure model supports return_tokens.")
                logits_list.append(logits)
                tokens_list.append(tokens)
                ce_losses.append(F.cross_entropy(logits, label))

            mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)
            masked_logits = mean_logits.clone()
            masked_logits.scatter_(1, label.view(-1, 1), float("-inf"))
            wrong_class = masked_logits.argmax(dim=1).detach()

            wrong_attrs: List[torch.Tensor] = []
            true_attrs: List[torch.Tensor] = []
            for logits, tokens in zip(logits_list, tokens_list):
                wrong_attrs.append(
                    self._class_token_attribution_from_logits(
                        logits=logits,
                        tokens=tokens,
                        class_indices=wrong_class,
                        create_graph=create_graph,
                    ).squeeze(0)
                )
                true_attrs.append(
                    self._class_token_attribution_from_logits(
                        logits=logits,
                        tokens=tokens,
                        class_indices=label,
                        create_graph=create_graph,
                    ).squeeze(0)
                )

            wrong_stack = torch.stack(wrong_attrs, dim=0)
            true_stack = torch.stack(true_attrs, dim=0)
            ce_mean = torch.stack(ce_losses).mean()
            return wrong_stack, true_stack, ce_mean, wrong_class

        if torch.is_grad_enabled():
            return _inner()
        with torch.enable_grad():
            return _inner()

    def _compute_shared_prototype(self, stack: torch.Tensor) -> torch.Tensor:
        return self._normalize_distribution(stack.mean(dim=0))

    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = self._normalize_distribution(p)
        q = self._normalize_distribution(q)
        m = 0.5 * (p + q)
        kl_pm = (p * ((p + 1e-12).log() - (m + 1e-12).log())).sum(dim=-1)
        kl_qm = (q * ((q + 1e-12).log() - (m + 1e-12).log())).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)

    def _compute_align_loss(self, stack: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        prototype_expanded = prototype.unsqueeze(0).expand_as(stack)
        return self._js_divergence(stack, prototype_expanded).mean()

    def _compute_compact_loss(self, prototype: torch.Tensor) -> torch.Tensor:
        return -(prototype * (prototype + 1e-12).log()).sum()

    def _compute_region_weights(self, prototype: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = min(self.region_topk, prototype.numel())
        indices = torch.topk(prototype, k=k, dim=0).indices
        region_weights = torch.zeros_like(prototype)
        region_weights[indices] = prototype[indices]
        region_weights = self._normalize_distribution(region_weights)
        return indices, region_weights

    def _compute_coupled_loss(
        self,
        wrong_stack: torch.Tensor,
        true_stack: torch.Tensor,
        region_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        region = region_weights.unsqueeze(0)
        wrong_mass = (wrong_stack * region).sum(dim=-1).mean()
        true_mass = (true_stack * region).sum(dim=-1).mean()
        return true_mass - wrong_mass, wrong_mass, true_mass

    def _build_patch_mask(self, patch_indices: torch.Tensor, img_size: int) -> torch.Tensor:
        grid_w = img_size // self.patch_size
        mask = torch.zeros((1, 1, img_size, img_size), device=self.device)
        for idx in patch_indices.tolist():
            row = idx // grid_w
            col = idx % grid_w
            r0 = row * self.patch_size
            r1 = r0 + self.patch_size
            c0 = col * self.patch_size
            c1 = c0 + self.patch_size
            mask[:, :, r0:r1, c0:c1] = 1.0
        return mask

    def _project_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "linf":
            return delta.clamp(-self.eps, self.eps)
        if self.norm_type == "l2":
            flat = delta.view(delta.size(0), -1)
            norm = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
            factor = torch.clamp(self.eps / norm, max=1.0)
            return (flat * factor).view_as(delta)
        raise ValueError(f"Unknown norm type: {self.norm_type}")

    def _pgd_update(
        self,
        delta: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_momentum:
            grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
            momentum = self.momentum_mu * momentum + grad / grad_norm
            update = momentum
        else:
            update = grad
        delta = delta + self.step_size * update.sign()
        delta = self._project_delta(delta)
        return delta, momentum

    def _pgd_single_stage(
        self,
        image_norm: torch.Tensor,
        label: torch.Tensor,
        init: str = "zero",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        image_norm = image_norm.to(self.device)
        label = label.to(self.device)
        image_pixels = self._denormalize(image_norm).clamp(0.0, 1.0)

        if init == "rand":
            delta = torch.empty_like(image_pixels).uniform_(-self.eps, self.eps)
            delta = self._project_delta(delta)
        elif init == "zero":
            delta = torch.zeros_like(image_pixels)
        else:
            raise ValueError(f"Unknown init type: {init}")

        momentum = torch.zeros_like(delta)

        for step in range(self.steps):
            delta.requires_grad_(True)
            adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
            wrong_stack, true_stack, ce_mean, wrong_class = self._compute_view_attribution_stack(
                image_pixels=adv_pixels,
                label=label,
                create_graph=True,
            )
            wrong_prototype = self._compute_shared_prototype(wrong_stack)
            align_loss = self._compute_align_loss(wrong_stack, wrong_prototype)
            compact_loss = self._compute_compact_loss(wrong_prototype)
            region_indices, region_weights = self._compute_region_weights(wrong_prototype)
            coupled_loss, wrong_mass, true_mass = self._compute_coupled_loss(
                wrong_stack=wrong_stack,
                true_stack=true_stack,
                region_weights=region_weights,
            )

            objective = (
                self.lambda_cls * ce_mean
                - self.lambda_align * align_loss
                - self.lambda_compact * compact_loss
                - self.lambda_couple * coupled_loss
            )

            grad = torch.autograd.grad(objective, delta, retain_graph=False)[0]
            delta, momentum = self._pgd_update(delta, grad, momentum)
            delta = delta.detach()

            if self.log_every > 0 and (step + 1) % self.log_every == 0:
                print(
                    f"Attack[{step + 1}/{self.steps}] "
                    f"L_cls={ce_mean.item():.4f} "
                    f"L_align={align_loss.item():.4f} "
                    f"L_compact={compact_loss.item():.4f} "
                    f"L_couple={coupled_loss.item():.4f} "
                    f"wrong_mass={wrong_mass.item():.4f} "
                    f"true_mass={true_mass.item():.4f}"
                )

        adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
        adv_norm = self._normalize(adv_pixels)
        wrong_stack, true_stack, _ce_mean, wrong_class = self._compute_view_attribution_stack(
            image_pixels=adv_pixels,
            label=label,
            create_graph=False,
        )
        wrong_prototype = self._compute_shared_prototype(wrong_stack).detach()
        true_prototype = self._compute_shared_prototype(true_stack).detach()
        region_indices, region_weights = self._compute_region_weights(wrong_prototype)
        mask = self._build_patch_mask(region_indices, adv_pixels.shape[-1])

        extra = {
            "wrong_prototype": wrong_prototype.cpu(),
            "true_prototype": true_prototype.cpu(),
            "region_weights": region_weights.detach().cpu(),
            "wrong_class": wrong_class.detach().cpu(),
        }
        return adv_norm, delta.detach(), mask.detach(), region_indices.detach(), extra

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
            adv, delta, mask, region_indices, extra = self._pgd_single_stage(image, label, init=init)
            adv_list.append(adv)
            delta_list.append(delta)
            mask_list.append(mask)
            patch_indices_list.append(region_indices)
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
        values = values.view(-1)
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
            wrong_proto = self._map_to_image(extra["wrong_prototype"].to(self.device))
            true_proto = self._map_to_image(extra["true_prototype"].to(self.device))
            region_map = self._map_to_image(extra["region_weights"].to(self.device))
            diff_map = (wrong_proto - true_proto).abs()
            mask_1c = masks[idx:idx + 1].to(self.device)[:, :1, :, :]
            overlay = wrong_proto.clone()
            red = torch.tensor([1.0, 0.0, 0.0], device=overlay.device).view(1, 3, 1, 1)
            overlay = overlay * (1.0 - 0.4 * mask_1c) + red * (0.4 * mask_1c)
            overlay = overlay.clamp(0.0, 1.0)
            evidence_grid = make_grid(
                torch.cat([wrong_proto, true_proto, region_map, diff_map, overlay], dim=0),
                nrow=3,
                padding=2,
            )
            ev_path = output_dir_path / f"evidence_{stem}.png"
            save_image(evidence_grid, str(ev_path))
            saved.append(ev_path)

        return saved
