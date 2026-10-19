from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

import torch
from torch import nn

from utils import DEVICE

LOCAL_HF_CACHE = Path(__file__).resolve().parents[1] / "data" / "huggingface"
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"


@dataclass
class ProgressiveState:
    """Local features and architecture context at a forward boundary."""

    local_tokens: torch.Tensor
    grid_size: tuple[int, int]
    context: dict
    rgb_projection_weight: torch.Tensor | None = None

    def validate(self) -> None:
        if self.local_tokens.ndim != 3:
            raise ValueError("local_tokens must have shape [B,N,D].")
        if self.local_tokens.size(1) != self.grid_size[0] * self.grid_size[1]:
            raise ValueError("local token count does not match grid_size.")
        if self.rgb_projection_weight is not None:
            weight = self.rgb_projection_weight
            if weight.ndim != 4 or weight.size(1) != 3:
                raise ValueError("opponent noise requires an RGB Conv2d projection.")
            if weight.size(0) != self.local_tokens.size(2):
                raise ValueError("RGB projection and local feature dimensions differ.")


class ProgressiveAdapter(nn.Module):
    default_model_name = ""
    checkpoints: tuple[str, ...] = ()
    drop_ratios: tuple[float, ...] = ()
    opponent_noise_strength = 0.2

    def __init__(self, model_name=None, *, pretrained=True, device=None):
        super().__init__()
        import timm

        self.device = device if device is not None else DEVICE
        self.model_name = model_name or self.default_model_name
        self.model = timm.create_model(self.model_name, pretrained=pretrained)
        config = self.model.pretrained_cfg
        self.model_mean = tuple(config.get("mean", (0.5, 0.5, 0.5)))
        self.model_std = tuple(config.get("std", (0.5, 0.5, 0.5)))
        self.to(self.device)
        self.eval()

    def forward(self, x):
        return self.model(x.to(self.device))

    def begin_progressive_forward(self, x) -> ProgressiveState:
        raise NotImplementedError

    def advance_progressive_state(self, state, checkpoint_id) -> ProgressiveState:
        raise NotImplementedError

    def finish_progressive_forward(self, state):
        raise NotImplementedError

    @staticmethod
    def apply_progressive_mask(state, mask) -> ProgressiveState:
        if mask.shape != state.local_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("progressive mask must be boolean and match local tokens.")
        local = state.local_tokens.masked_fill(mask.unsqueeze(-1), 0)
        return replace(state, local_tokens=local)
