# attack.py
from typing import List, Tuple
import torch
import torch.nn.functional as F

from utils import DEVICE

# 工具函数：贴 patch、计算 patch 对应的 token index
def apply_patch(images: torch.Tensor,patch: torch.Tensor,top: int,left: int,) -> torch.Tensor:
    """
    把 patch 贴到一批图像的固定位置。
    :param top:    patch 左上角在图像中的行坐标（像素）
    :param left:   patch 左上角在图像中的列坐标（像素）
    """
    B, C, H, W = images.shape
    _, _, ph, pw = patch.shape

    x_adv = images.clone()

    if patch.size(0) == 1 and B > 1:
        # 共享一个 universal patch
        patch_b = patch.expand(B, -1, -1, -1)
    else:
        patch_b = patch

    x_adv[:, :, top:top + ph, left:left + pw] = patch_b
    return x_adv


def get_patch_token_index(img_size: int,patch_size: int,patch_row: int,patch_col: int,) -> int:
    """
    给定 patch 在 patch 网格中的 (row, col)，计算对应的 token index。
    """
    num_patches_per_dim = img_size // patch_size
    patch_idx = patch_row * num_patches_per_dim + patch_col
    token_idx = 1 + patch_idx  # +1 是因为 index 0 通常是 cls token
    return token_idx

# Attention-Fool 损失：L_kq / L_kq*
def compute_attention_fool_loss(attn_logits_list: List[torch.Tensor],key_token_idx: int,cls_only: bool = False,) -> torch.Tensor:

    layer_losses = []

    for attn_logits in attn_logits_list:
        # attn_logits: [B, H, N, N]
        B, H, N, _ = attn_logits.shape
        # 对应 key 的那一列: [B, H, N]
        col = attn_logits[:, :, :, key_token_idx]

        if cls_only:
            L_lh = col[:, :, 0]
        else:
            L_lh = torch.logsumexp(col, dim=-1)

        # head 维 log-sum-exp: [B, H] -> [B]
        L_l = torch.logsumexp(L_lh, dim=1)
        layer_losses.append(L_l)

    # layer 维 log-sum-exp: [L, B] -> [B]
    layer_losses = torch.stack(layer_losses, dim=0)  
    L = torch.logsumexp(layer_losses, dim=0)      

    return L.mean()

# Attention-Fool Patch 攻击器
class AttentionFoolPatchAttacker:
    """
    - patch 大小和位置由 img_size / patch_size / (patch_row, patch_col) 控制；
    - 与类别数量完全解耦，只要 labels 是合法的类别 id 即可；
    - 优化目标可以是："ce" "attn" "ce+attn" "ce+attn_cls" 
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
    ) -> None:
        """
        :param steps:         PGD 迭代步数
        :param step_size:     每步更新步长（像素 0~1 空间）
        :param lambda_attn:   Attention loss 权重 λ
        :param loss_type:     "ce" / "attn" / "ce+attn" / "ce+attn_cls"
        :param use_momentum:  是否使用 momentum-PGD
        :param momentum_mu:   动量衰减系数 μ
        :param device:        设备；若为 None,则使用全局 DEVICE
        """
        self.model = model
        self.model.eval()

        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_row = patch_row
        self.patch_col = patch_col

        self.steps = steps
        self.step_size = step_size
        self.lambda_attn = lambda_attn
        self.loss_type = loss_type
        self.use_momentum = use_momentum
        self.momentum_mu = momentum_mu

        self.device = device if device is not None else DEVICE

        # 目标 key 的 token index（i*）
        self.key_token_idx = get_patch_token_index(
            img_size=img_size,
            patch_size=patch_size,
            patch_row=patch_row,
            patch_col=patch_col,
        )

    # 损失函数
    def _compute_total_loss(
        self,
        logits: torch.Tensor,
        attn_logits_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        根据 loss_type 组合：
        - "ce"          -> L_ce
        - "attn"        -> λ·L_kq 或 λ·L_kq*
        - "ce+attn"     -> L_ce + λ·L_kq
        - "ce+attn_cls" -> L_ce + λ·L_kq*
        """
        ce_loss = F.cross_entropy(logits, labels)

        if self.loss_type == "ce":
            return ce_loss

        cls_only = (self.loss_type == "ce+attn_cls")
        attn_loss = compute_attention_fool_loss(
            attn_logits_list=attn_logits_list,
            key_token_idx=self.key_token_idx,
            cls_only=cls_only,
        )

        if self.loss_type == "attn":
            return self.lambda_attn * attn_loss
        elif self.loss_type in ("ce+attn", "ce+attn_cls"):
            return ce_loss + self.lambda_attn * attn_loss
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    # 攻击
    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        init: str = "rand",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """return (x_adv, patch) x_adv: 对抗图像 [B, 3, H, W] patch: 最终 patch [1, 3, ph, pw]
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        #初始化patch #todo 可以考虑制定target pic
        if init == "rand":
            patch = torch.rand(
                1, 3, self.patch_size, self.patch_size,
                device=self.device,
            )
        elif init == "zero":
            patch = torch.zeros(
                1, 3, self.patch_size, self.patch_size,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown init type: {init}")

        patch.requires_grad_(True)

        # 动量缓存（若使用 momentum-PGD）
        momentum = torch.zeros_like(patch)

        for _ in range(self.steps):
            # 1) 贴 patch
            x_adv = apply_patch(
                images=images,
                patch=patch,
                top=self.patch_row * self.patch_size,
                left=self.patch_col * self.patch_size,
            )

            # 2) 前向：得到 logits + 各层 attn logits
            logits, attn_logits_list = self.model(x_adv, return_attn=True)

            # 3) 按论文损失组合 CE 和 Attention-Fool
            total_loss = self._compute_total_loss(
                logits=logits,
                attn_logits_list=attn_logits_list,
                labels=labels,
            )

            # 4) 对 patch 梯度上升
            total_loss.backward()

            with torch.no_grad():
                grad = patch.grad

                if self.use_momentum:
                    # 带 L2 归一化的 momentum-PGD
                    g_flat = grad.view(grad.size(0), -1)
                    g_norm = g_flat.norm(p=2, dim=1, keepdim=True) + 1e-12
                    g_normed = (g_flat / g_norm).view_as(grad)

                    momentum = self.momentum_mu * momentum + g_normed
                    patch.data = patch.data + self.step_size * momentum.sign()
                else:
                    # 普通 PGD
                    patch.data = patch.data + self.step_size * grad.sign()

                # 投影到 [0,1] 像素范围
                patch.data.clamp_(0.0, 1.0)

                # 清空梯度
                patch.grad.zero_()

        # 最终对抗样本
        x_adv = apply_patch(
            images=images,
            patch=patch.detach(),
            top=self.patch_row * self.patch_size,
            left=self.patch_col * self.patch_size,
        )

        return x_adv, patch.detach()