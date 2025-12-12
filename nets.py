# nets.py
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import timm

from utils import DEVICE  

class ViTWithAttn(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int | None = None,
        pretrained: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device if device is not None else DEVICE

        create_kwargs = {}
        if num_classes is not None:
            # 如果传了 num_classes，就只改分类头的输出维度；
            create_kwargs["num_classes"] = num_classes

        self.model: nn.Module = timm.create_model(
            model_name,
            pretrained=pretrained,
            **create_kwargs,
        )

        # 存放每一层的 attn logits（B, H, N, N）
        self.attn_logits: List[torch.Tensor] = []
        self._capture_attn: bool = False

        self._norm_eps = 1e-6

        # 注册 hook，在每个 blocks.*.attn.qkv 上算 qk^T / sqrt(d_k)
        self._register_qkv_hooks()

        # 搬到设备
        self.to(self.device)

    # 注册 hook：在每个 blocks.*.attn.qkv 上挂一个 forward_hook
    def _register_qkv_hooks(self) -> None:
        """
        在所有 "attn.qkv" 线性层上注册 forward hook。
        hook 会拿到 qkv 的输出，并在这里自己算出 dot-product attention logits
        """
        module_dict: Dict[str, nn.Module] = dict(self.model.named_modules())

        # 保存每个 qkv 模块对应的 (num_heads, head_dim)
        self._qkv_meta: Dict[nn.Module, Tuple[int, int]] = {}

        for name, module in module_dict.items():
            # 典型名字类似：blocks.0.attn.qkv
            if not name.endswith("attn.qkv"):
                continue

            parent_name = name.rsplit(".", 1)[0]  # "blocks.0.attn"
            attn_mod = module_dict.get(parent_name, None)
            if attn_mod is None:
                continue

            num_heads = getattr(attn_mod, "num_heads", None)
            if num_heads is None:
                continue

            if not hasattr(module, "out_features"):
                continue
            out_features: int = module.out_features  # 3 * dim
            head_dim = (out_features // 3) // num_heads

            self._qkv_meta[module] = (num_heads, head_dim)
            module.register_forward_hook(self._make_qkv_hook(module))

    def _normalize_head_tokens(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        按 head 内 token 的平均 L2 范数进行缩放，确保每个 head 的 Query/Key 平均范数为 1。
        """
        norms = tensor.norm(p=2, dim=-1, keepdim=True)  # [B, H, N, 1]
        mean_norm = norms.mean(dim=-2, keepdim=True)    # [B, H, 1, 1]
        return tensor / (mean_norm + self._norm_eps)

    def _make_qkv_hook(self, qkv_module: nn.Module):
        """
        为指定的 qkv_module 生成一个 forward_hook。
        在 hook 里，用 qkv 输出计算：
            q, k -> attn_logits = q @ k^T / sqrt(d_k)
        """

        def hook(module: nn.Module, inputs, output):
            # output: [B, N, 3*dim]
            if module not in self._qkv_meta:
                return
            if not self._capture_attn:
                return

            num_heads, head_dim = self._qkv_meta[module]
            qkv = output  # [B, N, 3*dim]
            B, N, _ = qkv.shape
            # [B, N, 3, H, d] -> [3, B, H, N, d]
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _v = qkv[0], qkv[1], qkv[2]  # [B, H, N, d]

            # 对每个 head 的 query/key 做 L_{1,2} 归一化
            q = self._normalize_head_tokens(q)
            k = self._normalize_head_tokens(k)

            # dot-product attention logits: [B, H, N, N]
            attn_logits = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)

            self.attn_logits.append(attn_logits)

        return hook

    # forward
    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        :param x: [B, 3, H, W]
        :param return_attn: 是否返回 attn logits 列表
        """
        x = x.to(self.device)

        # 清空上一次 forward 的 attention 缓存
        self.attn_logits = []
        self._capture_attn = return_attn

        logits = self.model(x)  # [B, num_classes]
        self._capture_attn = False

        if return_attn:
            attn_list = [t for t in self.attn_logits]
            return logits, attn_list
        else:
            return logits
