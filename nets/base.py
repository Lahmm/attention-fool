from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

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
    """Architecture-neutral local/global features at a progressive checkpoint."""

    local_tokens: torch.Tensor
    global_token: torch.Tensor
    grid_size: tuple[int, int]
    source_name: str
    layer_id: str
    global_mode: str

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
        if self.global_mode not in {"cls", "gap"}:
            raise ValueError(
                "global_mode must be cls or gap, got "
                f"{self.global_mode!r}."
            )


@dataclass
class ProgressiveInputState:
    """Initial progressive representation and its RGB projection geometry."""

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
            raise ValueError("progressive input token count does not match grid_size.")
        weight = self.rgb_projection_weight
        if weight.ndim != 4 or weight.size(1) != 3:
            raise ValueError("the progressive attack requires an RGB Conv2d projection.")
        if weight.size(0) != self.local_tokens.size(2):
            raise ValueError("RGB projection output channels do not match local feature channels.")


@dataclass
class ProgressiveAttackState:
    """Opaque architecture state used during progressive checkpoint traversal."""

    local_tokens: torch.Tensor
    grid_size: tuple[int, int]
    context: object

    def validate(self) -> None:
        if self.local_tokens.ndim != 3:
            raise ValueError(
                "progressive local_tokens must have shape [B,N,D], got "
                f"{tuple(self.local_tokens.shape)}."
            )
        if self.local_tokens.size(1) != self.grid_size[0] * self.grid_size[1]:
            raise ValueError("progressive local token count does not match grid_size.")


def conv2d_progressive_metadata(module: nn.Module) -> dict[str, object]:
    """Return RGB projection metadata required by progressive opponent noise."""
    if not isinstance(module, nn.Conv2d) or module.in_channels != 3:
        raise ValueError("the progressive attack requires an RGB Conv2d projection module.")
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


class ProgressiveAdapter(nn.Module):
    """Base contract implemented by every progressive source-model adapter."""

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
        self.model = create_timm_model(
            self.model_name, num_classes=num_classes, pretrained=pretrained
        )
        config = getattr(self.model, "pretrained_cfg", {})
        self.model_mean = tuple(
            float(value) for value in config.get("mean", (0.5, 0.5, 0.5))
        )
        self.model_std = tuple(
            float(value) for value in config.get("std", (0.5, 0.5, 0.5))
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.to(self.device))

    def progressive_checkpoint_candidates(self) -> tuple[str, ...]:
        raise NotImplementedError(
            f"progressive checkpoints are not implemented for {self.model_name}."
        )

    def default_progressive_checkpoints(self) -> tuple[str, ...]:
        raise NotImplementedError(
            f"progressive checkpoint defaults are not implemented for {self.model_name}."
        )

    def default_progressive_drop_ratios(self) -> tuple[float, ...]:
        raise NotImplementedError(
            f"progressive drop-ratio defaults are not implemented for {self.model_name}."
        )

    def default_progressive_score_mode(self) -> str:
        return "cosine"

    def default_progressive_opponent_noise_strength(self) -> float:
        return 0.2

    def prepare_progressive_input(self, x: torch.Tensor) -> ProgressiveInputState:
        raise NotImplementedError(
            f"progressive input preparation is not implemented for {self.model_name}."
        )

    def begin_progressive_forward(self, x: torch.Tensor) -> ProgressiveAttackState:
        raise NotImplementedError(
            f"progressive forward preparation is not implemented for {self.model_name}."
        )

    def replace_progressive_local_tokens(
        self,
        state: ProgressiveAttackState,
        local_tokens: torch.Tensor,
    ) -> ProgressiveAttackState:
        raise NotImplementedError(
            f"progressive token replacement is not implemented for {self.model_name}."
        )

    def advance_progressive_state(
        self,
        state: ProgressiveAttackState,
        checkpoint_id: str,
    ) -> ProgressiveAttackState:
        raise NotImplementedError(
            f"progressive traversal is not implemented for {self.model_name}."
        )

    def progressive_score_features(
        self,
        state: ProgressiveAttackState,
        checkpoint_id: str,
    ) -> PatchScoreFeatures:
        raise NotImplementedError(
            f"progressive scoring is not implemented for {self.model_name}."
        )

    def apply_progressive_mask(
        self,
        state: ProgressiveAttackState,
        mask: torch.Tensor,
    ) -> ProgressiveAttackState:
        raise NotImplementedError(
            f"progressive masking is not implemented for {self.model_name}."
        )

    def finish_progressive_forward(self, state: ProgressiveAttackState) -> torch.Tensor:
        raise NotImplementedError(
            f"progressive forward completion is not implemented for {self.model_name}."
        )
