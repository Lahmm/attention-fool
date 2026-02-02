# attack.py
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD


def _normalize_dir(grad: torch.Tensor) -> torch.Tensor:
    if grad.numel() == 0:
        return grad
    grad_flat = grad.flatten(1)
    norm = grad_flat.norm(p=2, dim=1, keepdim=True)
    return (grad_flat / (norm + 1e-12)).view_as(grad)


class AttentionFoolImageAttacker:
    """
    Full-image PGD attacker with gradient direction consistency over QK logits.
    loss_type: "ce" or "ce+qk_dir"
    """
    def __init__(
        self,
        model,
        img_size: int = 224,
        steps: int = 250,
        step_size: float = 8.0 / 255.0,
        lambda_attn: float = 1.0,
        loss_type: str = "ce+qk_dir",
        use_momentum: bool = False,
        momentum_mu: float = 0.9,
        device: torch.device | None = None,
        attn_layer_set: set[int] | None = None,
        eps: float = 8.0 / 255.0,
    ) -> None:

        self.model = model
        self.model.eval()

        self.img_size = img_size

        self.steps = steps
        self.step_size = step_size
        self.lambda_attn = lambda_attn
        self.loss_type = loss_type
        self.use_momentum = use_momentum
        self.momentum_mu = momentum_mu
        self.eps = eps
        self.attn_layer_set = attn_layer_set

        self.device = device if device is not None else DEVICE

        self.pixel_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.pixel_std + self.pixel_mean

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.pixel_mean) / self.pixel_std

    def _cosine_step_size(self, iteration: int) -> float:
        if self.steps <= 1:
            return self.step_size
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * iteration / (self.steps - 1)))
        return self.step_size * cos_decay

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        init: str = "rand",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x_adv, delta)
        x_adv: normalized adversarial images [B, 3, H, W]
        delta: pixel-space perturbation [B, 3, H, W]
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        images_pixels = self._denormalize(images)  # [B,3,H,W]
        B = images_pixels.size(0)

        if init == "rand":
            delta = torch.empty_like(images_pixels).uniform_(-self.eps, self.eps)
        elif init == "zero":
            delta = torch.zeros_like(images_pixels)
        else:
            raise ValueError(f"Unknown init type: {init}")

        delta.requires_grad_(True)
        momentum = torch.zeros_like(delta)

        for iter_idx in range(self.steps):
            step = self._cosine_step_size(iter_idx)

            for b in range(B):
                if not delta.requires_grad:
                    delta.requires_grad_(True)

                adv_pixels_b = (images_pixels[b:b+1] + delta[b:b+1]).clamp(0.0, 1.0)
                x_adv_b = self._normalize(adv_pixels_b)

                logits_b, attn_logits_list_b = self.model(x_adv_b, return_attn=True)
                ce_loss = F.cross_entropy(logits_b, labels[b:b+1])

                if self.loss_type == "ce":
                    ce_grad = torch.autograd.grad(ce_loss, delta, retain_graph=False)[0]
                    final_dir = _normalize_dir(ce_grad[b:b+1])
                elif self.loss_type == "ce+qk_dir":
                    num_layers = len(attn_logits_list_b)
                    if self.attn_layer_set:
                        invalid = [idx for idx in self.attn_layer_set if idx < 1 or idx > num_layers]
                        if invalid:
                            raise ValueError(
                                f"attn_layer_set contains invalid layers {sorted(invalid)} for {num_layers} layers."
                            )
                        layer_indices = [idx - 1 for idx in sorted(self.attn_layer_set)]
                    else:
                        layer_indices = list(range(num_layers))

                    total_heads = sum(attn_logits_list_b[idx].shape[1] for idx in layer_indices)
                    retain_for_qk = total_heads > 0
                    ce_grad = torch.autograd.grad(ce_loss, delta, retain_graph=retain_for_qk)[0]
                    ce_dir = _normalize_dir(ce_grad[b:b+1])

                    if total_heads == 0:
                        qk_dir = torch.zeros_like(ce_dir)
                    else:
                        u_sum = torch.zeros_like(ce_dir)
                        head_idx = 0
                        for layer_idx in layer_indices:
                            attn_logits = attn_logits_list_b[layer_idx]
                            num_heads = attn_logits.shape[1]
                            for h in range(num_heads):
                                b_lh = attn_logits[:, h, :, :]
                                qk_loss = b_lh[:, 0, 1:].mean()
                                retain = head_idx < (total_heads - 1)
                                g_lh = torch.autograd.grad(qk_loss, delta, retain_graph=retain)[0]
                                u_lh = _normalize_dir(g_lh[b:b+1])
                                u_sum = u_sum + u_lh
                                head_idx += 1
                        qk_dir = _normalize_dir(u_sum)

                    final_dir = _normalize_dir(ce_dir + self.lambda_attn * qk_dir)
                else:
                    raise ValueError(f"Unknown loss_type: {self.loss_type}")

                with torch.no_grad():
                    if self.use_momentum:
                        momentum[b:b+1] = self.momentum_mu * momentum[b:b+1] + final_dir
                        update_dir = momentum[b:b+1]
                    else:
                        update_dir = final_dir

                    delta[b:b+1] = delta[b:b+1] + step * update_dir.sign()
                    delta[b:b+1].clamp_(-self.eps, self.eps)

                delta = delta.detach()
                delta.requires_grad_(True)

        final_pixels = (images_pixels + delta.detach()).clamp(0.0, 1.0)
        x_adv = self._normalize(final_pixels)

        return x_adv, delta.detach()
