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
    direction_variance_ratio: float
    contraction_score: float
    contraction_fraction: float
    input_total_variance: float
    output_total_variance: float
    grid_size: tuple[int, int]


def feature_statistics(local, reference):
    if local.ndim != 3 or reference.shape != (local.size(0), local.size(2)):
        raise ValueError('expected local [B,N,D] and reference [B,D]')
    if min(local.shape) == 0:
        raise ValueError('empty local feature tensor')
    local, reference = local.float(), reference.float()
    if not bool(torch.isfinite(local).all() & torch.isfinite(reference).all()):
        raise ValueError('non-finite features')
    local64, reference64 = local.double(), reference.double()
    local_norm = local64.norm(dim=-1, keepdim=True)
    reference_norm = reference64.norm(dim=-1, keepdim=True)
    if bool((local_norm == 0).any() | (reference_norm == 0).any()):
        raise ValueError('nonzero local features and reference required')
    unit = local64 / local_norm
    direction = reference64 / reference_norm
    center = unit.mean(dim=1, keepdim=True)
    centered = unit - center
    identical = (unit == unit[:, :1]).all(dim=2).all(dim=1)
    centered = torch.where(identical[:, None, None], 0.0, centered)
    total_variance = centered.square().sum(dim=2).mean(dim=1)
    average_direction_variance = total_variance / local.size(2)
    projected_variance = (centered * direction[:, None]).sum(dim=2).square().mean(dim=1)
    denominator = torch.where(average_direction_variance > 0, average_direction_variance, 1.0)
    ratio = torch.where(average_direction_variance > 0, projected_variance / denominator, 0.0)
    raw = F.cosine_similarity(local, reference[:, None], dim=-1, eps=1e-8).clamp(-1, 1).double()
    output = {
        'score_mean_float64': (unit * direction[:, None]).sum(dim=2).mean(dim=1),
        'global_direction_population_variance': projected_variance,
        'total_local_population_variance': total_variance,
        'average_direction_population_variance': average_direction_variance,
        'direction_variance_ratio': ratio,
        'mainline_score_mean': raw.mean(dim=1),
        'mainline_score_population_variance': raw.var(dim=1, unbiased=False),
        'degenerate_local_variance': total_variance == 0,
    }
    if not all(bool(torch.isfinite(value).all()) for value in output.values()):
        raise ValueError('non-finite statistics')
    return output


def contraction_statistics(local, reference, block_input):
    if block_input.shape != local.shape:
        raise ValueError("block input/output local shapes must match")
    stats = feature_statistics(local, reference)
    before = feature_statistics(block_input, reference)['total_local_population_variance']
    after = stats['total_local_population_variance']
    denominator = torch.where(before > 0, before, 1.0)
    fraction = torch.where(before > 0, (1 - after / denominator).clamp_min(0), 0.0)
    stats.update(input_total_variance=before, output_total_variance=after,
                 contraction_fraction=fraction,
                 contraction_score=stats['direction_variance_ratio'] * fraction)
    return stats


def select_contraction_peaks(observations):
    if not observations:
        raise ValueError("at least one block observation is required")
    all_stats = []
    for layer, checkpoint, grid, local, reference, block_input in observations:
        if local.size(1) != grid[0] * grid[1]:
            raise ValueError("observation grid does not match local tokens")
        all_stats.append({key: value.cpu() for key, value in contraction_statistics(local, reference, block_input).items()})
    values = torch.stack([s['contraction_score'] for s in all_stats])
    peaks = values.argmax(dim=0)
    return tuple(
        SampleCheckpoint(observations[index][1], observations[index][0],
                         float(all_stats[index]['mainline_score_population_variance'][sample]),
                         float(all_stats[index]['direction_variance_ratio'][sample]),
                         float(all_stats[index]['contraction_score'][sample]),
                         float(all_stats[index]['contraction_fraction'][sample]),
                         float(all_stats[index]['input_total_variance'][sample]),
                         float(all_stats[index]['output_total_variance'][sample]),
                         observations[index][2])
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
                raise ValueError("Gaussian augmentation requires an RGB Conv2d projection.")
            if weight.size(0) != self.local_tokens.size(2):
                raise ValueError("RGB projection and local feature dimensions differ.")


class ProgressiveAdapter(nn.Module):
    default_model_name = ""
    drop_ratio: float
    noise_strength = 0.2

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

    def select_contraction_checkpoints(self, normalized_inputs):
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
