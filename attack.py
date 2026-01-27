# attack.py
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD



def compute_attention_variance_loss(
    attn_logits_list: List[torch.Tensor],
    cls_only: bool = True,
    attn_layer_set: set[int] | None = None,
    standardize: str = "center",  # "center" or "zscore"
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    计算选定层注意力“logits 空间”的方差均值(不做 softmax),并在计算方差前做一次标准化
    与原版区别：不再对 batch 维做最终平均；返回每个样本一个值，形状为 [B]

    参数：
    - attn_logits_list: 每层一个 Tensor, 形状 [B, H, N, N]
    - attn_layer_set: 1-based layer indices set; empty set uses all layers
    - cls_only: 是否仅使用 CLS query (query index=0)
    - standardize: "center" or "zscore"
    - eps: 数值稳定项

    返回：
    - per-sample 方差张量，形状 [B](对层、head、query 聚合后；不对 batch 聚合)
    """
    if not attn_logits_list:
        raise ValueError("attn_logits_list is empty")

    if attn_layer_set is None:
        attn_layer_set = set()

    # 选择参与计算的层
    if len(attn_layer_set) == 0:
        selected_attn = attn_logits_list
    else:
        num_layers = len(attn_logits_list)
        invalid = [idx for idx in attn_layer_set if idx < 1 or idx > num_layers]
        if invalid:
            raise ValueError(
                f"需要{sorted(invalid)}层的注意力对于{num_layers}层的模型不合法"
            )
        layer_indices = sorted(attn_layer_set)
        selected_attn = [attn_logits_list[idx - 1] for idx in layer_indices]

    per_layer_vars: List[torch.Tensor] = []

    for attn_logits in selected_attn:
        # attn_logits: [B, H, N, N]，最后一维是 key
        scores = attn_logits

        # 一次标准化（不做 softmax）
        if standardize == "center":
            scores = scores - scores.mean(dim=-1, keepdim=True)
        elif standardize == "zscore":
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True, unbiased=False)
            scores = (scores - mean) / (std + eps)
        else:
            raise ValueError(f"Unknown standardize mode: {standardize}")

        # 在 key 维计算方差，并对 head/query 聚合，保留 batch
        if cls_only:
            # scores[:, :, 0, :] -> [B, H, N] -> var over key => [B, H] -> mean over head => [B]
            var = scores[:, :, 0, :].var(dim=-1, unbiased=False).mean(dim=1)  # [B]
        else:
            # scores.var(dim=-1) -> [B, H, N(query)] -> mean over head/query => [B]
            var = scores.var(dim=-1, unbiased=False).mean(dim=(1, 2))  # [B]

        per_layer_vars.append(var)

    layer_vars = torch.stack(per_layer_vars, dim=0)  # [L', B]
    return layer_vars.mean(dim=0)  # [B]


class AttentionFoolImageAttacker:
    """
    - 基于注意力方差损失的整图 PGD 攻击器
    - loss_type: "ce", "attn", "ce+attn", "ce+attn_cls"
    - 改动:attack_batch 在每个 step 内遍历 batch 中每张图，逐样本单独反传与更新(不做 batch 平均)
    """
    def __init__(
        self,
        model,
        img_size: int = 224,
        patch_size: int = 16,
        patch_row: int = 0,
        patch_col: int = 0,
        steps: int = 250,
        step_size: float = 8.0 / 255.0,
        lambda_attn: float = 1.0,
        loss_type: str = "ce+attn",
        use_momentum: bool = False,
        momentum_mu: float = 0.9,
        device: torch.device | None = None,
        attn_layer_set: set[int] | None = None,
        eps: float = 8.0 / 255.0,
    ) -> None:
        self.model = model
        self.model.eval()

        self.img_size = img_size
        # patch_* 为兼容保留，整图攻击不使用
        self.patch_size = patch_size
        self.patch_row = patch_row
        self.patch_col = patch_col

        self.steps = steps
        self.step_size = step_size
        self.lambda_attn = lambda_attn
        self.loss_type = loss_type
        self.use_momentum = use_momentum
        self.momentum_mu = momentum_mu
        self.eps = eps
        self.attn_layer_set = attn_layer_set

        self.device = device if device is not None else DEVICE  # 依赖你工程里的 DEVICE

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

    def _compute_total_loss_single(
        self,
        logits: torch.Tensor,                     # [1, C]
        attn_logits_list: List[torch.Tensor],     # list of [1, H, N, N]
        labels: torch.Tensor,                     # [1]
    ) -> torch.Tensor:
        """
        返回单样本标量 loss(不涉及 batch 平均)
        注意:attn_term = -Var,配合 PGD 上升(delta += step*sign(grad)会倾向于最小化 Var
        """
        ce_loss = F.cross_entropy(logits, labels)  # 标量（batch=1 时）

        if self.loss_type == "ce":
            return ce_loss

        cls_only = (self.loss_type == "ce+attn_cls")
        attn_var_vec = compute_attention_variance_loss(
            attn_logits_list=attn_logits_list,
            cls_only=cls_only,
            attn_layer_set=self.attn_layer_set,
        )  # [1]
        attn_var = attn_var_vec[0]  # 标量

        attn_term = -attn_var

        if self.loss_type == "attn":
            return self.lambda_attn * attn_term
        elif self.loss_type in ("ce+attn", "ce+attn_cls"):
            return ce_loss + self.lambda_attn * attn_term

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        init: str = "rand",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (x_adv, delta)
        x_adv: 归一化空间中的对抗图像 [B, 3, H, W]
        delta: 像素空间中的逐图扰动 [B, 3, H, W]

        改动点：每个 step 内，对 b=0..B-1 逐样本计算标量 loss 并更新 delta[b]。
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

            # 逐样本遍历：每次只对一个样本反传与更新
            for b in range(B):
                adv_pixels_b = (images_pixels[b:b+1] + delta[b:b+1]).clamp(0.0, 1.0)  # [1,3,H,W]
                x_adv_b = self._normalize(adv_pixels_b)                                # [1,3,H,W]

                logits_b, attn_logits_list_b = self.model(x_adv_b, return_attn=True)

                loss_b = self._compute_total_loss_single(
                    logits=logits_b,
                    attn_logits_list=attn_logits_list_b,
                    labels=labels[b:b+1],
                )

                loss_b.backward()

                with torch.no_grad():
                    grad_b = delta.grad[b:b+1]  # [1,3,H,W]

                    if self.use_momentum:
                        g_flat = grad_b.view(1, -1)
                        g_norm = g_flat.norm(p=2, dim=1, keepdim=True) + 1e-12
                        g_normed = (g_flat / g_norm).view_as(grad_b)

                        momentum[b:b+1] = self.momentum_mu * momentum[b:b+1] + g_normed
                        delta[b:b+1] = delta[b:b+1] + step * momentum[b:b+1].sign()
                    else:
                        delta[b:b+1] = delta[b:b+1] + step * grad_b.sign()

                    # 投影到 L_inf ball
                    delta[b:b+1].clamp_(-self.eps, self.eps)

                    # 清梯度（全量清，避免累积；由于逐样本反传，这里清一次最安全）
                    if delta.grad is not None:
                        delta.grad.zero_()

        final_pixels = (images_pixels + delta.detach()).clamp(0.0, 1.0)
        x_adv = self._normalize(final_pixels)

        return x_adv, delta.detach()
