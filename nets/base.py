from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from utils import DEVICE

LOCAL_HF_CACHE = Path(__file__).resolve().parents[1] / "data" / "huggingface"
# Model construction always resolves weights from the repository-local cache.
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"


DEFAULT_PRETRAINED = True


@dataclass
class PatchScoreFeatures:
    """Architecture-neutral local/global features used for patch scoring.

    ``layer_id`` is the stable public identifier accepted by the attack and
    experiment CLIs.  ``source_name`` remains a human-readable description of
    the concrete module(s), while ``global_mode`` makes it explicit whether
    the global representation came from a learned token, CaiT class
    attention, or global average pooling.
    """

    local_tokens: torch.Tensor
    global_token: torch.Tensor
    grid_size: tuple[int, int]
    source_name: str
    layer_id: str = "final"
    global_mode: str = "cls"

    def validate(self) -> None:
        if self.local_tokens.ndim != 3:
            raise ValueError(
                f"local_tokens must have shape [B,N,D], got {tuple(self.local_tokens.shape)}."
            )
        if self.global_token.ndim != 3 or self.global_token.size(1) != 1:
            raise ValueError(
                f"global_token must have shape [B,1,D], got {tuple(self.global_token.shape)}."
            )
        if self.local_tokens.size(0) != self.global_token.size(0):
            raise ValueError("local and global feature batch sizes do not match.")
        if self.local_tokens.size(2) != self.global_token.size(2):
            raise ValueError("local and global feature dimensions do not match.")
        if self.local_tokens.size(1) != self.grid_size[0] * self.grid_size[1]:
            raise ValueError("local token count does not match grid_size.")
        if not self.layer_id:
            raise ValueError("layer_id must be non-empty.")
        if self.global_mode not in {"cls", "class_attention_cls", "gap"}:
            raise ValueError(
                "global_mode must be cls, class_attention_cls, or gap, got "
                f"{self.global_mode!r}."
            )


@dataclass
class AttackFeatureState:
    """Opaque resumable state at a model's first RGB-projected feature map."""

    local_tokens: torch.Tensor
    grid_size: tuple[int, int]
    context: object
    rgb_projection_weight: torch.Tensor
    projection_kernel: tuple[int, int]
    projection_stride: tuple[int, int]
    projection_padding: tuple[int, int]
    projection_dilation: tuple[int, int] = (1, 1)

    def validate(self) -> None:
        if self.local_tokens.ndim != 3:
            raise ValueError(
                f"local_tokens must have shape [B,N,D], got {tuple(self.local_tokens.shape)}."
            )
        if self.local_tokens.size(1) != self.grid_size[0] * self.grid_size[1]:
            raise ValueError("attack token count does not match grid_size.")
        weight = self.rgb_projection_weight
        if weight.ndim != 4 or weight.size(1) != 3:
            raise ValueError("the mainline requires a first-layer RGB Conv2d projection.")
        if weight.size(0) != self.local_tokens.size(2):
            raise ValueError("RGB projection output channels do not match local feature channels.")


def conv2d_attack_metadata(module: nn.Module) -> dict[str, object]:
    """Return strict RGB projection metadata for an attack feature state."""
    if not isinstance(module, nn.Conv2d) or module.in_channels != 3:
        raise ValueError("the mainline requires an RGB Conv2d projection module.")
    return {
        "rgb_projection_weight": module.weight,
        "projection_kernel": tuple(int(value) for value in module.kernel_size),
        "projection_stride": tuple(int(value) for value in module.stride),
        "projection_padding": tuple(int(value) for value in module.padding),
        "projection_dilation": tuple(int(value) for value in module.dilation),
    }


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
        config = getattr(self.model, "pretrained_cfg", {})
        self.model_mean = tuple(float(value) for value in config.get("mean", (0.5, 0.5, 0.5)))
        self.model_std = tuple(float(value) for value in config.get("std", (0.5, 0.5, 0.5)))

        self.feature_modules: list[nn.Module] = list(self._feature_modules())
        if not self.feature_modules:
            raise ValueError(f"No compatible feature layers found for {self.model_name}.")
        self.feature_tokens: List[Optional[torch.Tensor]] = []
        self._capture_tokens = False
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._reset_caches()
        self._register_feature_hooks()
        self.to(self.device)

    @property
    def num_blocks(self) -> int:
        return len(self.feature_modules)

    def _feature_modules(self) -> Iterable[nn.Module]:
        raise NotImplementedError

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        raise NotImplementedError(f"patch-score extraction is not implemented for {self.model_name}.")

    def patch_score_layer_candidates(self) -> tuple[str, ...]:
        """Return pre-registered, canonical routing checkpoints.

        ``final`` is intentionally an alias rather than a candidate: callers
        that scan layers should not accidentally evaluate the same endpoint
        twice.
        """
        return ("final",)

    def prepare_attack_feature_state(self, x: torch.Tensor) -> AttackFeatureState:
        raise NotImplementedError(f"attack feature preparation is not implemented for {self.model_name}.")

    def forward_from_attack_feature_state(
        self,
        state: AttackFeatureState,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError(f"resumable attack forward is not implemented for {self.model_name}.")

    def _reset_caches(self) -> None:
        self.feature_tokens = [None] * len(self.feature_modules)

    def _register_feature_hooks(self) -> None:
        for layer_idx, module in enumerate(self.feature_modules):
            handle = module.register_forward_hook(self._make_feature_hook(layer_idx))
            self._hook_handles.append(handle)

    def _make_feature_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs, output):
            if not self._capture_tokens:
                return
            self.feature_tokens[layer_idx] = self._extract_tensor_output(output)

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

    def forward(
        self,
        x: torch.Tensor,
        return_tokens: bool = False,
    ):
        x = x.to(self.device)
        self._reset_caches()
        self._capture_tokens = return_tokens

        logits = self.model(x)

        self._capture_tokens = False
        outputs: list[object] = [logits]
        if return_tokens:
            outputs.append(self._finalize_cache(self.feature_tokens, "feature layer outputs"))

        if len(outputs) == 1:
            return logits
        return tuple(outputs)


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
