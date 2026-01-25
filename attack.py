# attack.py
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD


def compute_attention_variance_loss(
    attn_logits_list: List[torch.Tensor],
    cls_only: bool = False,
    attn_layer_index: int = -1,
    k_last: int | None = None,
    standardize: str = "center",  # "center" or "zscore"
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    计算选定层注意力“logits 空间”的方差均值（不做 softmax），并在计算方差前做一次标准化。

    标准化选项：
    - standardize="center": 仅对 key 维做去均值（中心化），避免整体偏置影响方差（推荐）。
    - standardize="zscore": 对 key 维做 z-score（减均值除标准差）；注意此时 key 维方差被规范到 ~1，
      如果你随后又对 key 维求方差，数值会接近常数，优化信号会明显变弱/失去区分度（一般不推荐用于“最小化方差”）。

    参数：
    - attn_logits_list: 每层一个 Tensor，形状 [B, H, N, N]
    - k_last: 若提供，则使用最后 k 层；否则使用 attn_layer_index
    - cls_only: 是否仅使用 CLS query (query index=0)
    - attn_layer_index: 使用的层索引（支持负索引）
    - eps: 数值稳定项

    返回：
    - 标量（对层、batch 均值后的方差）
    """
    if not attn_logits_list:
        raise ValueError("attn_logits_list is empty")

    # 选择参与计算的层
    if k_last is not None and k_last > 0:
        selected_attn = attn_logits_list if k_last >= len(attn_logits_list) else attn_logits_list[-k_last:]
    else:
        idx = attn_layer_index
        if idx < 0:
            idx = len(attn_logits_list) + idx
        if idx < 0 or idx >= len(attn_logits_list):
            raise ValueError(
                f"attn_layer_index {attn_layer_index} out of range for {len(attn_logits_list)} layers"
            )
        selected_attn = [attn_logits_list[idx]]

    per_layer_vars: List[torch.Tensor] = []

    for attn_logits in selected_attn:
        # attn_logits: [B, H, N, N]，最后一维是 key
        scores = attn_logits

        # 一次标准化（不做 softmax）
        if standardize == "center":
            # key 维中心化：scores - mean_key
            scores = scores - scores.mean(dim=-1, keepdim=True)
        elif standardize == "zscore":
            # key 维 z-score： (scores - mean_key) / std_key
            mean = scores.mean(dim=-1, keepdim=True)
            std = scores.std(dim=-1, keepdim=True, unbiased=False)
            scores = (scores - mean) / (std + eps)
        else:
            raise ValueError(f"Unknown standardize mode: {standardize}")

        # 在 key 维计算方差，并聚合
        if cls_only:
            # 仅 CLS query 的 key 分布：scores[:, :, 0, :] -> [B, H, N]
            var = scores[:, :, 0, :].var(dim=-1, unbiased=False).mean(dim=1)  # [B]
        else:
            # 所有 query：scores.var(dim=-1) -> [B, H, N(query)]
            var = scores.var(dim=-1, unbiased=False).mean(dim=(1, 2))  # [B]

        per_layer_vars.append(var)

    layer_vars = torch.stack(per_layer_vars, dim=0)  # [L', B]
    return layer_vars.mean(dim=0).mean()



class AttentionFoolImageAttacker:
    """
    基于注意力方差损失的整图 PGD 攻击器。
    - delta 在整张图像上优化。
    - loss_type: "ce", "attn", "ce+attn", "ce+attn_cls"。
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
        k_last: int | None = None,
        eps: float = 8.0 / 255.0,
        attn_layer_index: int = -1,
    ) -> None:
        """
        :param steps:         PGD 迭代次数
        :param step_size:     像素空间步长 [0, 1]
        :param lambda_attn:   注意力损失权重
        :param loss_type:     "ce" / "attn" / "ce+attn" / "ce+attn_cls"
        :param use_momentum:  是否使用 momentum-PGD
        :param momentum_mu:   动量衰减系数
        :param device:        运行设备
        :param k_last:        若设置则使用最后 k 层的注意力方差
        :param eps:           delta 的 L_inf 上限
        :param attn_layer_index: 注意力方差使用的层索引（默认最后一层）
        """
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
        self.attn_layer_index = attn_layer_index
        self.k_last = k_last

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

    def _compute_total_loss(
        self,
        logits: torch.Tensor,
        attn_logits_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        损失组合：
        - "ce"          -> L_ce
        - "attn"        -> lambda_attn * (-Var)
        - "ce+attn"     -> L_ce + lambda_attn * (-Var)
        - "ce+attn_cls" -> L_ce + lambda_attn * (-Var)，仅使用 CLS query
        """
        ce_loss = F.cross_entropy(logits, labels)

        if self.loss_type == "ce":
            return ce_loss

        cls_only = (self.loss_type == "ce+attn_cls")
        attn_var = compute_attention_variance_loss(
            attn_logits_list=attn_logits_list,
            cls_only=cls_only,
            attn_layer_index=self.attn_layer_index,
            k_last=self.k_last,
        )
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
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        images_pixels = self._denormalize(images)

        if init == "rand":
            delta = torch.empty_like(images_pixels).uniform_(-self.eps, self.eps)
        elif init == "zero":
            delta = torch.zeros_like(images_pixels)
        else:
            raise ValueError(f"Unknown init type: {init}")

        delta.requires_grad_(True)

        momentum = torch.zeros_like(delta)

        for iter_idx in range(self.steps):
            adv_pixels = (images_pixels + delta).clamp(0.0, 1.0)
            x_adv = self._normalize(adv_pixels)

            logits, attn_logits_list = self.model(x_adv, return_attn=True)

            total_loss = self._compute_total_loss(
                logits=logits,
                attn_logits_list=attn_logits_list,
                labels=labels,
            )

            total_loss.backward()

            with torch.no_grad():
                grad = delta.grad
                step = self._cosine_step_size(iter_idx)

                if self.use_momentum:
                    g_flat = grad.view(grad.size(0), -1)
                    g_norm = g_flat.norm(p=2, dim=1, keepdim=True) + 1e-12
                    g_normed = (g_flat / g_norm).view_as(grad)

                    momentum = self.momentum_mu * momentum + g_normed
                    delta.data = delta.data + step * momentum.sign()
                else:
                    delta.data = delta.data + step * grad.sign()

                delta.data.clamp_(-self.eps, self.eps)

                if delta.grad is not None:
                    delta.grad.zero_()

        final_pixels = (images_pixels + delta.detach()).clamp(0.0, 1.0)
        x_adv = self._normalize(final_pixels)

        return x_adv, delta.detach()

