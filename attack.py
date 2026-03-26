from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD


class AttentionFoolImageAttacker:
    """
    Single-model, multi-view clean-prototype suppression attack.

    The perturbation is optimized so that different augmented views of the same
    adversarial sample:
    1. are misclassified,
    2. identify the clean sample's shared true-class attribution prototype,
    3. suppress that prototype evidence under perturbation across multiple views.
    """

    def __init__(
        self,
        model,
        img_size: int = 224,
        steps: int = 100,
        step_size: float = 1.0 / 255.0,
        eps: float = 8.0 / 255.0,
        num_views: int = 8,
        noise_eps: float = 4.0 / 255.0,
        tau: float = 0.07,
        lambda_cls: float = 1.0,
        lambda_support: float = 1.0,
        norm_type: str = "linf",
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
        self.num_views = num_views
        self.noise_eps = noise_eps
        self.tau = tau
        self.lambda_cls = lambda_cls
        self.lambda_support = lambda_support
        self.norm_type = norm_type
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
        brightness = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
        contrast = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))
        saturation = 1.0 + float(torch.empty(1).uniform_(-0.1, 0.1))

        jittered = image_pixels * brightness
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

    def _class_token_support_from_logits(
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
        # Keep only positive token evidence for the selected class so the
        # suppression loss directly targets supportive evidence.
        token_support = torch.relu(grads * tokens).sum(dim=-1)
        return token_support[:, 1:]

    def _class_token_attribution_from_logits(
        self,
        logits: torch.Tensor,
        tokens: torch.Tensor,
        class_indices: torch.Tensor,
        create_graph: bool,
    ) -> torch.Tensor:
        token_support = self._class_token_support_from_logits(
            logits=logits,
            tokens=tokens,
            class_indices=class_indices,
            create_graph=create_graph,
        )
        return self._softmax_normalize(token_support, self.tau)

    def _compute_true_class_view_data(
        self,
        image_pixels: torch.Tensor,
        label: torch.Tensor,
        create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _inner() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            views = self._build_diff_views(image_pixels)
            attr_list: List[torch.Tensor] = []
            support_list: List[torch.Tensor] = []
            ce_losses: List[torch.Tensor] = []

            for view in views:
                logits, tokens = self.model(view, return_tokens=True)
                if tokens is None:
                    raise RuntimeError("Token embeddings not captured; ensure model supports return_tokens.")
                ce_losses.append(F.cross_entropy(logits, label))
                support_list.append(
                    self._class_token_support_from_logits(
                        logits=logits,
                        tokens=tokens,
                        class_indices=label,
                        create_graph=create_graph,
                    ).squeeze(0)
                )
                attr_list.append(
                    self._class_token_attribution_from_logits(
                        logits=logits,
                        tokens=tokens,
                        class_indices=label,
                        create_graph=create_graph,
                    ).squeeze(0)
                )

            attr_stack = torch.stack(attr_list, dim=0)
            support_stack = torch.stack(support_list, dim=0)
            ce_mean = torch.stack(ce_losses).mean()
            return attr_stack, support_stack, ce_mean

        if torch.is_grad_enabled():
            return _inner()
        with torch.enable_grad():
            return _inner()

    def _compute_shared_prototype(self, stack: torch.Tensor) -> torch.Tensor:
        # Linearly aggregate multi-view clean attributions and normalize only
        # when converting to a weighting vector.
        return self._normalize_distribution(stack.sum(dim=0))

    def _compute_prototype_suppression_loss(
        self,
        support_stack: torch.Tensor,
        prototype_weights: torch.Tensor,
        clean_support_baseline: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        per_view_support = (support_stack * prototype_weights.unsqueeze(0)).sum(dim=-1)
        normalized_support = per_view_support / (clean_support_baseline + 1e-12)
        return normalized_support.mean(), per_view_support

    def _prototype_to_dense_map(self, values: torch.Tensor) -> torch.Tensor:
        values = values.view(-1)
        grid_w = self.img_size // self.patch_size
        map_2d = values.view(1, 1, grid_w, grid_w)
        map_2d = map_2d - map_2d.min()
        map_2d = map_2d / (map_2d.max() + 1e-12)
        return F.interpolate(map_2d, size=(self.img_size, self.img_size), mode="nearest")

    def _compute_entropy(self, prototype: torch.Tensor) -> torch.Tensor:
        prototype = self._normalize_distribution(prototype)
        return -(prototype * (prototype + 1e-12).log()).sum()

    def _project_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "linf":
            return delta.clamp(-self.eps, self.eps)
        if self.norm_type == "l2":
            flat = delta.view(delta.size(0), -1)
            norm = flat.norm(p=2, dim=1, keepdim=True) + 1e-12
            factor = torch.clamp(self.eps / norm, max=1.0)
            return (flat * factor).view_as(delta)
        raise ValueError(f"Unknown norm type: {self.norm_type}")

    def _mifgsm_update(
        self,
        delta: torch.Tensor,
        grad: torch.Tensor,
        momentum: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_norm = grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
        normalized_grad = grad / grad_norm
        momentum = self.momentum_mu * momentum + normalized_grad
        delta = delta + self.step_size * momentum.sign()
        delta = self._project_delta(delta)
        return delta, momentum

    def _mifgsm_single_stage(
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
        clean_attr_stack, clean_support_stack, _ = self._compute_true_class_view_data(
            image_pixels=image_pixels,
            label=label,
            create_graph=False,
        )
        clean_prototype = self._compute_shared_prototype(clean_attr_stack).detach()
        prototype_raw = clean_attr_stack.sum(dim=0).detach()
        prototype_weights = self._normalize_distribution(prototype_raw).detach()
        prototype_indices = torch.arange(
            prototype_weights.numel(),
            device=prototype_weights.device,
            dtype=torch.long,
        )
        clean_support_mean = clean_support_stack.mean(dim=0).detach()
        clean_support_baseline = (
            (clean_support_stack.detach() * prototype_weights.unsqueeze(0))
            .sum(dim=-1)
            .mean()
            .detach()
            .clamp_min(1e-12)
        )

        for step in range(self.steps):
            delta.requires_grad_(True)
            adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
            true_attr_stack, true_support_stack, ce_mean = self._compute_true_class_view_data(
                image_pixels=adv_pixels,
                label=label,
                create_graph=True,
            )
            support_loss, per_view_support = self._compute_prototype_suppression_loss(
                support_stack=true_support_stack,
                prototype_weights=prototype_weights,
                clean_support_baseline=clean_support_baseline,
            )
            normalized_support = per_view_support / (clean_support_baseline + 1e-12)

            objective = (
                self.lambda_cls * ce_mean
                - self.lambda_support * support_loss
            )

            grad = torch.autograd.grad(objective, delta, retain_graph=False)[0]
            delta, momentum = self._mifgsm_update(delta, grad, momentum)
            delta = delta.detach()

            if self.log_every > 0 and (step + 1) % self.log_every == 0:
                print(
                    f"Attack[{step + 1}/{self.steps}] "
                    f"L_cls={ce_mean.item():.4f} "
                    f"L_proto={support_loss.item():.4f} "
                    f"support_ratio={normalized_support.mean().item():.4f} "
                    f"H_clean={self._compute_entropy(clean_prototype).item():.4f}"
                )

        adv_pixels = (image_pixels + delta).clamp(0.0, 1.0)
        adv_norm = self._normalize(adv_pixels)
        adv_attr_stack, adv_support_stack, _ce_mean = self._compute_true_class_view_data(
            image_pixels=adv_pixels,
            label=label,
            create_graph=False,
        )
        adv_true_prototype = self._compute_shared_prototype(adv_attr_stack).detach()
        adv_support_mean = adv_support_stack.mean(dim=0).detach()
        dense_map = self._prototype_to_dense_map(prototype_weights)
        support_drop = torch.clamp(clean_support_mean - adv_support_mean, min=0.0)

        extra = {
            "clean_true_prototype": clean_prototype.cpu(),
            "adv_true_prototype": adv_true_prototype.cpu(),
            "prototype_raw": prototype_raw.detach().cpu(),
            "prototype_weights": prototype_weights.detach().cpu(),
            # Kept for backward compatibility with old visualization keys.
            "stable_token_weights": prototype_weights.detach().cpu(),
            "stable_token_mask": torch.ones_like(prototype_weights).detach().cpu(),
            "stable_token_score": prototype_raw.detach().cpu(),
            "stable_token_indices": prototype_indices.detach().cpu(),
            "clean_true_support": clean_support_mean.cpu(),
            "adv_true_support": adv_support_mean.cpu(),
            "support_drop": support_drop.cpu(),
        }
        return adv_norm, delta.detach(), dense_map.detach(), prototype_indices.detach(), extra

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
        dense_map_list: List[torch.Tensor] = []
        stable_indices_list: List[torch.Tensor] = []
        extra_list: List[Dict[str, torch.Tensor]] = []

        for idx in range(images.size(0)):
            image = images[idx:idx + 1]
            label = labels[idx:idx + 1]
            adv, delta, dense_map, stable_indices, extra = self._mifgsm_single_stage(image, label, init=init)
            adv_list.append(adv)
            delta_list.append(delta)
            dense_map_list.append(dense_map)
            stable_indices_list.append(stable_indices)
            extra_list.append(extra)

        x_adv = torch.cat(adv_list, dim=0)
        delta = torch.cat(delta_list, dim=0)
        dense_maps = torch.cat(dense_map_list, dim=0)
        return x_adv, delta, dense_maps, stable_indices_list, extra_list

    def _build_vis_grid(
        self,
        clean_norm: torch.Tensor,
        adv_norm: torch.Tensor,
        dense_map: torch.Tensor,
    ) -> torch.Tensor:
        clean = self._denormalize(clean_norm).clamp(0.0, 1.0)
        adv = self._denormalize(adv_norm).clamp(0.0, 1.0)
        diff = (adv - clean).abs()
        diff_vis = (diff / (self.eps + 1e-12)).clamp(0.0, 1.0)

        map_1c = dense_map.to(clean.device)[:, :1, :, :]
        overlay = clean.clone()
        red = torch.tensor([1.0, 0.0, 0.0], device=overlay.device).view(1, 3, 1, 1)
        alpha = 0.4
        overlay = overlay * (1.0 - alpha * map_1c) + red * (alpha * map_1c)
        overlay = overlay.clamp(0.0, 1.0)

        grid = make_grid(
            torch.cat([clean, adv, diff_vis, overlay], dim=0),
            nrow=2,
            padding=2,
        )
        return grid

    def _map_to_image(self, values: torch.Tensor) -> torch.Tensor:
        return self._prototype_to_dense_map(values).repeat(1, 3, 1, 1)

    def save_visualizations(
        self,
        clean_images: torch.Tensor,
        adv_images: torch.Tensor,
        dense_maps: torch.Tensor,
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
                dense_maps[idx:idx + 1],
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
            clean_proto = self._map_to_image(extra["clean_true_prototype"].to(self.device))
            adv_proto = self._map_to_image(extra["adv_true_prototype"].to(self.device))
            stable_map = self._map_to_image(extra["stable_token_weights"].to(self.device))
            support_drop_map = self._map_to_image(extra["support_drop"].to(self.device))
            map_1c = dense_maps[idx:idx + 1].to(self.device)[:, :1, :, :]
            overlay = clean_proto.clone()
            red = torch.tensor([1.0, 0.0, 0.0], device=overlay.device).view(1, 3, 1, 1)
            overlay = overlay * (1.0 - 0.4 * map_1c) + red * (0.4 * map_1c)
            overlay = overlay.clamp(0.0, 1.0)
            evidence_grid = make_grid(
                torch.cat([clean_proto, adv_proto, stable_map, support_drop_map, overlay], dim=0),
                nrow=3,
                padding=2,
            )
            ev_path = output_dir_path / f"evidence_{stem}.png"
            save_image(evidence_grid, str(ev_path))
            saved.append(ev_path)

        return saved
