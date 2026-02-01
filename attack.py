# attack.py
import math
from typing import List, Tuple
import numpy as np

import torch
import torch.nn.functional as F

from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD

# 计算注意力权重相关函数
def attention_weights_from_logits(attn_logits: torch.Tensor) -> np.ndarray:
    # attn_logits: [B, H, N, N] -> weights: [B, N, N] 多头平均取注意力权重
    attn_weights = torch.softmax(attn_logits, dim=-1)
    attn_weights = attn_weights.mean(dim=1)
    return attn_weights.detach().cpu().numpy()


def attention_weights_per_head_from_logits(attn_logits: torch.Tensor) -> np.ndarray:
    """
    将输入的注意力logits(单样本) [B,H,N.N] 转化为每个head的注意力权重矩阵
    """
    if attn_logits.dim() != 4:
        raise ValueError("attn_logits 必须是一个 4D tensor [B, H, N, N].")
    if attn_logits.size(0) != 1:
        raise ValueError("每次只能输入一个样本.")

    attn_weights = torch.softmax(attn_logits, dim=-1)  # [1, H, N, N]
    return attn_weights[0].detach().cpu().numpy()

# 把矩阵线性归一化到 [0,1]
def _minmax_normalize(a: np.ndarray) -> np.ndarray:
    a_min = float(a.min())
    a_max = float(a.max())
    return (a - a_min) / (a_max - a_min + 1e-8)


# 计算结构相似性指数（SSIM）
def _ssim(a: np.ndarray, b: np.ndarray, c1: float, c2: float) -> float:
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / (den + 1e-8))


def _compute_ssim_matrix(attn_mats: List[np.ndarray], c1: float, c2: float) -> np.ndarray:
    """
    对每一层的注意力矩阵做 min-max 归一化
    计算任意两层之间的 SSIM,形成 num_layers x num_layers 的矩阵
    对角线填 1.0(层与自身相似度)
    """
    num_layers = len(attn_mats)
    mats = [_minmax_normalize(mat) for mat in attn_mats]
    ssim_mat = np.zeros((num_layers, num_layers), dtype=np.float32)
    for i in range(num_layers):
        for j in range(num_layers):
            ssim_mat[i, j] = _ssim(mats[i], mats[j], c1=c1, c2=c2)
    np.fill_diagonal(ssim_mat, 1.0)
    return ssim_mat


def _select_layers_by_threshold(ssim_mat: np.ndarray, threshold: float) -> List[int]:
    """
    对每一层，查看它与其他层的相似度是否超过阈值
    只要该层与任一其他层相似度 > threshold,就选中该层
    如果最终没有任何层被选中，就回退到选“平均相似度最高”的那一层
    """
    num_layers = ssim_mat.shape[0]
    selected: List[int] = []
    for i in range(num_layers):
        row = np.delete(ssim_mat[i], i)
        if np.any(row > threshold):
            selected.append(i)
    if not selected:
        mean_scores = (ssim_mat.sum(axis=1) - 1.0) / max(num_layers - 1, 1)
        selected = [int(np.argmax(mean_scores))]
    return selected


def compute_fused_attention_weights(
    attn_logits_list: List[torch.Tensor],
    c1: float = 1e-4,
    c2: float = 9e-4,
    ssim_threshold: float = 0.8,
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """
    对单个样本,计算经过SSIM选择的融合注意力
    返回 (combined_attention, selected_layer_indices, ssim_matrix).
    """
    if not attn_logits_list:
        raise ValueError("注意力logits为空")

    attn_mats = []
    for attn_logits in attn_logits_list:
        weights = attention_weights_from_logits(attn_logits)
        if weights.ndim != 3 or weights.shape[0] != 1:
            raise ValueError("期望输入单个样本的注意力logits(批量大小=1)")
        attn_mats.append(weights[0])

    ssim_mat = _compute_ssim_matrix(attn_mats, c1=c1, c2=c2)
    selected_layers = _select_layers_by_threshold(ssim_mat, ssim_threshold)

    weights = np.ones(len(selected_layers), dtype=np.float32) / max(len(selected_layers), 1)
    combined = np.zeros_like(attn_mats[0], dtype=np.float32)
    for idx, w in zip(selected_layers, weights):
        combined += w * attn_mats[idx]

    return combined, selected_layers, ssim_mat


def select_closest_heads_by_jsd(
    fused_attention: np.ndarray,
    per_head_attention: np.ndarray,
    top_k: int = 2,
    eps: float = 1e-12,
) -> List[int]:
    """
    计算融合注意力与每个头的注意力之间的 JSD,并返回
    JSD 最小的 top_k 个头的索引
    """
    if fused_attention.ndim != 2:
        raise ValueError("fused_attention must be a 2D [N, N] matrix.")

    heads = per_head_attention
    if heads.ndim == 4:
        if heads.shape[0] != 1:
            raise ValueError("per_head_attention with 4 dims must have batch size 1.")
        heads = heads[0]
    if heads.ndim != 3:
        raise ValueError("per_head_attention must be [H, N, N] or [1, H, N, N].")

    if heads.shape[1:] != fused_attention.shape:
        raise ValueError("Head attention shape must match fused_attention shape.")

    num_heads = heads.shape[0]
    if num_heads == 0:
        raise ValueError("per_head_attention has zero heads.")

    k = min(top_k, num_heads)
    fused = fused_attention.astype(np.float64, copy=False)
    fused = np.clip(fused, 0.0, None)
    fused = fused.reshape(-1)
    fused = fused / (fused.sum() + eps)

    jsd_scores = np.zeros((num_heads,), dtype=np.float64)
    for h in range(num_heads):
        head = heads[h].astype(np.float64, copy=False)
        head = np.clip(head, 0.0, None)
        head = head.reshape(-1)
        head = head / (head.sum() + eps)

        m = 0.5 * (fused + head)
        kl_fm = np.sum(fused * np.log((fused + eps) / (m + eps)))
        kl_hm = np.sum(head * np.log((head + eps) / (m + eps)))
        jsd_scores[h] = 0.5 * (kl_fm + kl_hm)

    closest = np.argsort(jsd_scores)[:k]
    return [int(i) for i in closest]


def build_target_distribution_from_fused(
    fused_attention: np.ndarray,
    cls_only: bool = True,
    map_to_patch: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    从融合的注意力矩阵中构建目标分布
    如果 cls_only 为 True,则只使用 CLS token 的注意力分布
    如果 map_to_patch 为 False,则返回完整的注意力分布 (所有token之间的注意力关系)
    如果 map_to_patch 为 True,则返回 patch token 的注意力分布 (patch 落在那些注意力块上)
    """
    if fused_attention.ndim != 2:
        raise ValueError("fused_attention must be a 2D [N, N] matrix.")

    if not map_to_patch:
        target = fused_attention.astype(np.float64, copy=False)
        target = np.clip(target, 0.0, None)
        target = target / (target.sum() + eps)
        return target

    if cls_only:
        target = fused_attention[0, 1:]
    else:
        target = fused_attention[:, 1:].mean(axis=0)

    target = target.astype(np.float64, copy=False)
    target = np.clip(target, 0.0, None)
    target = target / (target.sum() + eps)
    return target


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
        steps: int = 250,
        step_size: float = 8.0 / 255.0,
        lambda_attn: float = 1.0,
        loss_type: str = "ce+attn_target",
        use_momentum: bool = False,
        momentum_mu: float = 0.9,
        device: torch.device | None = None,
        attn_layer_set: set[int] | None = None,
        eps: float = 8.0 / 255.0,
        attn_target_mode: str = "cls",  # "cls" or "avg"
        attn_map_to_patch: bool = True,
        ssim_c1: float = 1e-4,
        ssim_c2: float = 9e-4,
        ssim_threshold: float = 0.8,
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
        self.attn_target_mode = attn_target_mode
        self.attn_map_to_patch = attn_map_to_patch

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

        if self.loss_type in ("attn_target", "ce+attn_target"):
            # Fused attention from SSIM (layer selection for fusion only; head selection uses attn_layer_set)
            fused_attention, _selected_layers, _ssim_mat = compute_fused_attention_weights(
                attn_logits_list=attn_logits_list,
            )

            if self.attn_target_mode not in ("cls", "avg"):
                raise ValueError("attn_target_mode must be 'cls' or 'avg'.")

            target_np = build_target_distribution_from_fused(
                fused_attention=fused_attention,
                cls_only=(self.attn_target_mode == "cls"),
                map_to_patch=self.attn_map_to_patch,
            )
            target = torch.from_numpy(target_np).to(self.device).float()

            if self.attn_layer_set is None or len(self.attn_layer_set) == 0:
                layer_indices = list(range(len(attn_logits_list)))
            else:
                num_layers = len(attn_logits_list)
                invalid = [idx for idx in self.attn_layer_set if idx < 1 or idx > num_layers]
                if invalid:
                    raise ValueError(
                        f"attn_layer_set contains invalid layers {sorted(invalid)} for {num_layers} layers."
                    )
                layer_indices = [idx - 1 for idx in sorted(self.attn_layer_set)]

            selected_heads: dict[int, list[int]] = {}
            for layer_idx in layer_indices:
                per_head_np = attention_weights_per_head_from_logits(attn_logits_list[layer_idx])
                head_indices = select_closest_heads_by_jsd(
                    fused_attention=fused_attention,
                    per_head_attention=per_head_np,
                )
                selected_heads[layer_idx] = head_indices

            sim_terms: List[torch.Tensor] = []
            for layer_idx in layer_indices:
                attn_logits = attn_logits_list[layer_idx]
                attn_weights = torch.softmax(attn_logits, dim=-1)  # [1, H, N, N]
                for h in selected_heads[layer_idx]:
                    head_attn = attn_weights[:, h, :, :]  # [1, N, N]
                    if self.attn_map_to_patch:
                        if self.attn_target_mode == "cls":
                            vec = head_attn[:, 0, 1:]  # [1, P]
                        else:
                            vec = head_attn[:, :, 1:].mean(dim=1)  # [1, P]
                        vec = vec / (vec.sum(dim=1, keepdim=True) + 1e-12)
                        sim = (vec * target).sum(dim=1)  # [1]
                    else:
                        mat = head_attn
                        mat = mat / (mat.sum(dim=(1, 2), keepdim=True) + 1e-12)
                        sim = (mat * target).sum(dim=(1, 2))  # [1]
                    sim_terms.append(sim)

            if not sim_terms:
                raise ValueError("No selected heads found for attention target loss.")

            sim_mean = torch.stack(sim_terms, dim=0).mean()
            attn_loss = -sim_mean

            if self.loss_type == "attn_target":
                return self.lambda_attn * attn_loss
            return ce_loss + self.lambda_attn * attn_loss      

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
