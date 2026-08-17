"""Shared, forward-only utilities for the retained semantic experiments.

This module intentionally contains no attack optimization or transfer-gradient
logic.  It keeps the semantic evidence chain reproducible without depending on
the archived selector/calibration experiments.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor


def _find_image(image_dir: Path, image_name: str) -> Path | None:
    direct = image_dir / image_name
    if direct.is_file():
        return direct
    stem = Path(image_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_samples(
    image_dir: Path,
    annotations_path: Path,
    offset: int,
    limit: int,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Load an exact deterministic slice of annotated RGB images."""
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    available: list[tuple[str, Path, int]] = []
    for image_name in sorted(annotations):
        path = _find_image(image_dir, image_name)
        if path is not None:
            available.append(
                (image_name, path, int(annotations[image_name]["class_id"]))
            )
    selected = available[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"requested {limit} samples at offset {offset}, found {len(selected)}."
        )
    to_tensor = ToTensor()
    pixels = torch.stack(
        [to_tensor(Image.open(path).convert("RGB")) for _, path, _ in selected]
    )
    labels = torch.tensor([label for _, _, label in selected], dtype=torch.long)
    return [name for name, _, _ in selected], pixels, labels


def normalize(model, pixels: torch.Tensor) -> torch.Tensor:
    mean = torch.as_tensor(
        model.model_mean, device=pixels.device, dtype=pixels.dtype
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        model.model_std, device=pixels.device, dtype=pixels.dtype
    ).view(1, 3, 1, 1)
    return (pixels - mean) / std


def rank_rows(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    rank_values = torch.arange(
        values.size(1), device=values.device, dtype=torch.float32
    )
    return ranks.scatter(1, order, rank_values.expand_as(order))


def row_spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_rank = rank_rows(left)
    right_rank = rank_rows(right)
    left_rank -= left_rank.mean(dim=1, keepdim=True)
    right_rank -= right_rank.mean(dim=1, keepdim=True)
    return F.cosine_similarity(left_rank, right_rank, dim=1)


def common_map(
    scores: torch.Tensor,
    grid: tuple[int, int],
    common_grid: int,
) -> torch.Tensor:
    maps = scores.reshape(scores.size(0), 1, *grid)
    return F.interpolate(
        maps, size=(common_grid, common_grid), mode="area"
    ).flatten(1)


def rank_norm(values: torch.Tensor) -> torch.Tensor:
    return rank_rows(values) / max(1, values.size(1) - 1)


def top_mask(values: torch.Tensor, ratio: float) -> torch.Tensor:
    count = max(1, int(round(values.size(1) * ratio)))
    indices = values.topk(count, dim=1).indices
    return torch.zeros_like(values, dtype=torch.bool).scatter(1, indices, True)


def bootstrap_ci(
    values: torch.Tensor,
    *,
    seed: int,
    repeats: int = 10000,
) -> list[float]:
    values = values.float().cpu()
    if values.numel() == 0:
        raise ValueError("bootstrap_ci requires at least one value")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        values.numel(), (repeats, values.numel()), generator=generator
    )
    means = values[indices].mean(dim=1)
    return [
        float(value)
        for value in torch.quantile(means, torch.tensor([0.025, 0.975]))
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _first_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"hook value contains no tensor: {type(value)!r}")


def capture_patch_score_activation(
    model,
    normalized: torch.Tensor,
    *,
    score_layer: str = "final",
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return logits and the adapter-declared logit-connected activation."""
    specification = model.patch_score_activation_capture(score_layer)
    specification.validate()
    captured: dict[str, torch.Tensor] = {}

    if specification.hook_type == "input":
        def hook(_module, inputs):
            captured["activation"] = _first_tensor(inputs)

        handle = specification.module.register_forward_pre_hook(hook)
    else:
        def hook(_module, _inputs, output):
            captured["activation"] = _first_tensor(output)

        handle = specification.module.register_forward_hook(hook)
    try:
        logits = model(normalized)
    finally:
        handle.remove()
    if "activation" not in captured:
        raise RuntimeError(
            f"failed to capture patch-score activation: {specification.source_name}"
        )
    return logits, captured["activation"], specification.source_name
