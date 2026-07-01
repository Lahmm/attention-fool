from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from utils import DEVICE


DEFAULT_PRETRAINED = True


def create_timm_model(model_name: str, *, num_classes: int | None, pretrained: bool) -> nn.Module:
    import timm

    create_kwargs = {}
    if num_classes is not None:
        create_kwargs["num_classes"] = num_classes
    return timm.create_model(model_name, pretrained=pretrained, **create_kwargs)


class WhiteBoxWithHook(nn.Module):
    default_model_name: str = ""

    def __init__(
        self,
        model_name: str | None = None,
        num_classes: int | None = None,
        pretrained: bool = DEFAULT_PRETRAINED,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else DEVICE
        self.model_name = model_name or self.default_model_name
        self.model = create_timm_model(self.model_name, num_classes=num_classes, pretrained=pretrained)

        self.feature_modules: list[nn.Module] = list(self._feature_modules())
        if not self.feature_modules:
            raise ValueError(f"No compatible feature layers found for {self.model_name}.")
        self.stage_modules: list[nn.Module] = list(self._stage_modules())

        self.feature_tokens: List[Optional[torch.Tensor]] = []
        self.stage_tokens: List[Optional[torch.Tensor]] = []
        self.attn_logits: List[Optional[torch.Tensor]] = []
        self.values: List[Optional[torch.Tensor]] = []
        self._capture_tokens = False
        self._capture_stage_tokens = False
        self._capture_attn = False
        self._capture_values = False
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._qkv_meta: Dict[nn.Module, Tuple[int, int, int]] = {}

        self._reset_caches()
        self._register_feature_hooks()
        self._register_stage_hooks()
        self._register_qkv_hooks()
        self.to(self.device)

    @property
    def num_blocks(self) -> int:
        return len(self.feature_modules)

    @property
    def num_stages(self) -> int:
        return len(self.stage_modules)

    def _feature_modules(self) -> Iterable[nn.Module]:
        raise NotImplementedError

    def _stage_modules(self) -> Iterable[nn.Module]:
        return ()

    def _reset_caches(self) -> None:
        self.feature_tokens = [None] * len(self.feature_modules)
        self.stage_tokens = [None] * len(self.stage_modules)
        self.attn_logits = [None] * len(self._qkv_meta)
        self.values = [None] * len(self._qkv_meta)

    def _register_feature_hooks(self) -> None:
        for layer_idx, module in enumerate(self.feature_modules):
            handle = module.register_forward_hook(self._make_feature_hook(layer_idx))
            self._hook_handles.append(handle)

    def _register_stage_hooks(self) -> None:
        for stage_idx, module in enumerate(self.stage_modules):
            handle = module.register_forward_hook(self._make_stage_hook(stage_idx))
            self._hook_handles.append(handle)

    def _register_qkv_hooks(self) -> None:
        module_dict = dict(self.model.named_modules())
        for name, module in module_dict.items():
            if not name.endswith("attn.qkv"):
                continue
            attn_name = name.rsplit(".qkv", 1)[0]
            attn_mod = module_dict.get(attn_name)
            num_heads = getattr(attn_mod, "num_heads", None) if attn_mod is not None else None
            if num_heads is None or not hasattr(module, "out_features"):
                continue
            out_features = int(module.out_features)
            if out_features % (3 * int(num_heads)) != 0:
                continue
            head_dim = out_features // (3 * int(num_heads))
            if head_dim <= 0:
                continue
            qkv_idx = len(self._qkv_meta)
            self._qkv_meta[module] = (int(num_heads), int(head_dim), qkv_idx)
            handle = module.register_forward_hook(self._make_qkv_hook(module))
            self._hook_handles.append(handle)

    def _make_qkv_hook(self, qkv_module: nn.Module):
        def hook(module: nn.Module, _inputs, output):
            if module is not qkv_module or module not in self._qkv_meta:
                return
            if not (self._capture_attn or self._capture_values):
                return
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                return
            num_heads, head_dim, qkv_idx = self._qkv_meta[module]
            bsz, num_tokens, hidden = output.shape
            expected_hidden = 3 * num_heads * head_dim
            if hidden != expected_hidden:
                return
            qkv = output.reshape(bsz, num_tokens, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            if self._capture_attn:
                q, k = qkv[0], qkv[1]
                self.attn_logits[qkv_idx] = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
            if self._capture_values:
                self.values[qkv_idx] = qkv[2]

        return hook

    def _make_feature_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs, output):
            if not self._capture_tokens:
                return
            self.feature_tokens[layer_idx] = self._extract_tensor_output(output)

        return hook

    def _make_stage_hook(self, stage_idx: int):
        def hook(_module: nn.Module, _inputs, output):
            if not self._capture_stage_tokens:
                return
            self.stage_tokens[stage_idx] = self._extract_tensor_output(output)

        return hook

    @staticmethod
    def _extract_tensor_output(output) -> torch.Tensor | None:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        return None

    @staticmethod
    def _finalize_cache(cache: List[Optional[torch.Tensor]], cache_name: str) -> List[torch.Tensor]:
        missing = [idx for idx, value in enumerate(cache) if value is None]
        if missing:
            raise RuntimeError(f"Failed to capture {cache_name} for layers: {missing}")
        return [value for value in cache if value is not None]

    @staticmethod
    def prepare_feature_tokens(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            return features
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        raise ValueError(f"Unsupported feature tensor shape: {tuple(features.shape)}.")

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        return_values: bool = False,
        return_tokens: bool = False,
        return_stage_tokens: bool = False,
    ):
        x = x.to(self.device)
        self._reset_caches()
        self._capture_attn = return_attn
        self._capture_values = return_values
        self._capture_tokens = return_tokens
        self._capture_stage_tokens = return_stage_tokens

        logits = self.model(x)

        self._capture_attn = False
        self._capture_values = False
        self._capture_tokens = False
        self._capture_stage_tokens = False
        outputs: list[object] = [logits]
        if return_attn:
            outputs.append(self._finalize_cache(self.attn_logits, "attn logits") if self._qkv_meta else [])
        if return_values:
            outputs.append(self._finalize_cache(self.values, "value tensors") if self._qkv_meta else [])
        if return_tokens:
            outputs.append(self._finalize_cache(self.feature_tokens, "feature layer outputs"))
        if return_stage_tokens:
            outputs.append(self._finalize_cache(self.stage_tokens, "stage outputs"))

        if len(outputs) == 1:
            return logits
        return tuple(outputs)


class ClsTokenMixin:
    @staticmethod
    def prepare_feature_tokens(features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.size(1) < 2:
            raise ValueError(f"Expected CLS + patch tokens with shape [B,N,D], got {tuple(features.shape)}.")
        return features[:, 1:, :]


class OptionalClsTokenMixin:
    @staticmethod
    def prepare_feature_tokens(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim != 3:
            raise ValueError(f"Unsupported feature tensor shape: {tuple(features.shape)}.")
        num_tokens = features.size(1)
        patch_count = num_tokens - 1
        grid = int(patch_count ** 0.5)
        if patch_count > 0 and grid * grid == patch_count:
            return features[:, 1:, :]
        return features


def sequential_modules(container) -> list[nn.Module]:
    if container is None:
        return []
    return [module for module in container]


def nested_stage_blocks(stages: Iterable[nn.Module]) -> list[nn.Module]:
    modules: list[nn.Module] = []
    for stage in stages:
        blocks = getattr(stage, "blocks", None)
        if blocks is None:
            continue
        modules.extend(sequential_modules(blocks))
    return modules
