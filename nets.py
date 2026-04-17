from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn

from utils import DEVICE

DEFAULT_MODEL_NAME = "vit_base_patch16_224"
DEFAULT_PRETRAINED = True


class ViTWithHook(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int | None = None,
        pretrained: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else DEVICE

        create_kwargs = {}
        if num_classes is not None:
            create_kwargs["num_classes"] = num_classes

        self.model: nn.Module = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **create_kwargs,
        )
        self.num_blocks = self._infer_num_blocks()

        self.attn_logits: List[Optional[torch.Tensor]] = []
        self.values: List[Optional[torch.Tensor]] = []
        self.block_tokens: List[Optional[torch.Tensor]] = []

        self._capture_attn = False
        self._capture_values = False
        self._capture_tokens = False

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._qkv_meta: Dict[nn.Module, Tuple[int, int, int]] = {}
        self._block_meta: Dict[nn.Module, int] = {}

        self._reset_caches()
        self._register_qkv_hooks()
        self._register_block_token_hooks()

        self.to(self.device)

    @staticmethod
    def _extract_block_index(module_name: str) -> int | None:
        parts = module_name.split(".")
        if len(parts) < 2 or parts[0] != "blocks" or not parts[1].isdigit():
            return None
        return int(parts[1])

    def _infer_num_blocks(self) -> int:
        max_block_idx = -1
        for name, _module in self.model.named_modules():
            block_idx = self._extract_block_index(name)
            if block_idx is not None:
                max_block_idx = max(max_block_idx, block_idx)
        return max_block_idx + 1

    def _reset_caches(self) -> None:
        self.attn_logits = [None] * self.num_blocks
        self.values = [None] * self.num_blocks
        self.block_tokens = [None] * self.num_blocks

    def _register_qkv_hooks(self) -> None:
        module_dict: Dict[str, nn.Module] = dict(self.model.named_modules())

        for name, module in module_dict.items():
            if not name.endswith("attn.qkv"):
                continue

            block_idx = self._extract_block_index(name)
            if block_idx is None:
                continue

            parent_name = name.rsplit(".", 1)[0]
            attn_mod = module_dict.get(parent_name, None)
            if attn_mod is None:
                continue

            num_heads = getattr(attn_mod, "num_heads", None)
            if num_heads is None or not hasattr(module, "out_features"):
                continue

            out_features = int(module.out_features)
            if out_features % 3 != 0:
                continue

            head_dim = (out_features // 3) // int(num_heads)
            if head_dim <= 0:
                continue

            self._qkv_meta[module] = (int(num_heads), int(head_dim), int(block_idx))
            handle = module.register_forward_hook(self._make_qkv_hook(module))
            self._hook_handles.append(handle)

    def _register_block_token_hooks(self) -> None:
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            return

        for block_idx, block in enumerate(blocks):
            self._block_meta[block] = int(block_idx)
            handle = block.register_forward_hook(self._make_block_token_hook(block))
            self._hook_handles.append(handle)

    def _make_qkv_hook(self, qkv_module: nn.Module):
        def hook(module: nn.Module, _inputs, output):
            if module is not qkv_module:
                return
            if module not in self._qkv_meta:
                return
            if not (self._capture_attn or self._capture_values):
                return
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                return

            num_heads, head_dim, block_idx = self._qkv_meta[module]
            bsz, num_tokens, hidden = output.shape
            expected_hidden = 3 * num_heads * head_dim
            if hidden != expected_hidden:
                return

            qkv = output.reshape(bsz, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)

            if self._capture_attn:
                q, k = qkv[0], qkv[1]
                attn_logits = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                self.attn_logits[block_idx] = attn_logits

            if self._capture_values:
                self.values[block_idx] = qkv[2]

        return hook

    def _make_block_token_hook(self, block_module: nn.Module):
        def hook(module: nn.Module, _inputs, output):
            if module is not block_module:
                return
            if not self._capture_tokens:
                return

            block_idx = self._block_meta.get(module, None)
            if block_idx is None:
                return

            if isinstance(output, torch.Tensor):
                self.block_tokens[block_idx] = output
                return

            if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                self.block_tokens[block_idx] = output[0]

        return hook

    @staticmethod
    def _finalize_cache(
        cache: List[Optional[torch.Tensor]],
        cache_name: str,
    ) -> List[torch.Tensor]:
        missing = [idx for idx, value in enumerate(cache) if value is None]
        if missing:
            raise RuntimeError(f"Failed to capture {cache_name} for blocks: {missing}")
        return [value for value in cache if value is not None]

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        return_values: bool = False,
        return_tokens: bool = False,
    ):
        x = x.to(self.device)

        self._reset_caches()
        self._capture_attn = return_attn
        self._capture_values = return_values
        self._capture_tokens = return_tokens

        logits = self.model(x)

        self._capture_attn = False
        self._capture_values = False
        self._capture_tokens = False

        outputs: List[object] = [logits]
        if return_attn:
            outputs.append(self._finalize_cache(self.attn_logits, "attn logits"))
        if return_values:
            outputs.append(self._finalize_cache(self.values, "value tensors"))
        if return_tokens:
            outputs.append(self._finalize_cache(self.block_tokens, "block token outputs"))

        if len(outputs) == 1:
            return logits
        return tuple(outputs)


def build_vit_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device: torch.device = DEVICE,
) -> ViTWithHook:
    model = ViTWithHook(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        device=device,
    )
    return model


ViTWithAttn = ViTWithHook
