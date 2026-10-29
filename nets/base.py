from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from utils import DEVICE

LOCAL_HF_CACHE = Path(__file__).resolve().parents[1] / "data" / "huggingface"
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"


@dataclass(frozen=True)
class SampleCheckpoint:
    checkpoint: str
    peak_layer: str
    variance: float
    grid_size: tuple[int, int]


def select_variance_peaks(observations):
    if not observations:
        raise ValueError("at least one block observation is required")
    variances = []
    finite = []
    nonzero = []
    for layer, checkpoint, grid, local, reference in observations:
        local, reference = local.float(), reference.float()
        if local.ndim != 3 or reference.shape != (local.size(0), local.size(2)):
            raise ValueError("expected local [B,N,D] and CLS [B,D]")
        if local.size(1) != grid[0] * grid[1]:
            raise ValueError("observation grid does not match the local tokens")
        finite.append(torch.isfinite(local).all() & torch.isfinite(reference).all())
        nonzero.append((local.norm(dim=-1) > 0).all() & (reference.norm(dim=-1) > 0).all())
        scores = F.cosine_similarity(local, reference[:, None], dim=-1, eps=1e-8).clamp(-1, 1)
        variances.append(scores.double().var(dim=1, unbiased=False))
    values = torch.stack(variances)
    checks = torch.stack((torch.stack(finite).all(), torch.stack(nonzero).all()))
    packed = torch.cat((values.flatten(), checks.to(values))).cpu()
    if not bool(packed[-2]):
        raise ValueError("non-finite checkpoint observation")
    if not bool(packed[-1]):
        raise ValueError("cosine similarity requires nonzero representations")
    values = packed[:-2].reshape(values.shape)
    peaks = values.argmax(dim=0)
    return tuple(
        SampleCheckpoint(observations[index][1], observations[index][0],
                         float(values[index, sample]), observations[index][2])
        for sample, index in enumerate(peaks.tolist())
    )


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
    dynamic_checkpoint = False

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

    def select_variance_checkpoints(self, normalized_inputs):
        raise NotImplementedError

    def checkpoint_order(self, checkpoint_id):
        raise NotImplementedError

    def checkpoint_grid_size(self, checkpoint_id):
        raise NotImplementedError

    @staticmethod
    def apply_progressive_mask(state, mask) -> ProgressiveState:
        if mask.shape != state.local_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("progressive mask must be boolean and match local tokens.")
        local = state.local_tokens.masked_fill(mask.unsqueeze(-1), 0)
        return replace(state, local_tokens=local)
