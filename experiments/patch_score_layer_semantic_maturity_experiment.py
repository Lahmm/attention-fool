"""Describe semantic maturity at every registered patch-score checkpoint.

This experiment is deliberately descriptive: none of its representation
metrics select the production routing layer.  Attack-layer decisions require
matched transferable-gradient and transfer-ASR evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.semantic_forward_utils import (
    load_samples,
    normalize,
    row_spearman,
    write_csv,
    write_json,
)
from nets import PATCH_SCORE_LAYER_CANDIDATES, WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all", help="all or a comma-separated model list")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--phase-shift", default="8,8", help="dx,dy in image pixels")
    parser.add_argument("--score-temperature", type=float, default=0.1)
    parser.add_argument("--top-ratio", type=float, default=0.15)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/patch_score_layer_semantic_maturity"),
    )
    return parser.parse_args()


def cosine_scores(features) -> torch.Tensor:
    return F.cosine_similarity(
        features.local_tokens,
        features.global_token.expand_as(features.local_tokens),
        dim=-1,
    )


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    """Linear CKA between sample-by-feature matrices of arbitrary widths."""
    if left.ndim != 2 or right.ndim != 2 or left.size(0) != right.size(0):
        raise ValueError("linear CKA inputs must be [samples,features] with matching samples.")
    if left.size(0) < 2:
        raise ValueError("linear CKA requires at least two samples.")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    cross = left.T @ right
    numerator = cross.square().sum()
    denominator = torch.sqrt(
        (left.T @ left).square().sum() * (right.T @ right).square().sum()
    )
    if denominator <= 0:
        return float("nan")
    return float((numerator / denominator).clamp(0, 1).item())


def class_geometry(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    """Return class separation plus a deterministic split 1-NN diagnostic."""
    normalized = F.normalize(features.float(), dim=1)
    similarities = normalized @ normalized.T
    pair_mask = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
    same = labels[:, None].eq(labels[None, :]) & pair_mask
    different = labels[:, None].ne(labels[None, :]) & pair_mask
    within = float(similarities[same].mean().item()) if same.any() else float("nan")
    between = float(similarities[different].mean().item()) if different.any() else float("nan")

    # Alternate occurrences within each class.  Only test examples whose class
    # is represented in the training half contribute to the reported accuracy.
    occurrences: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        occurrences.setdefault(int(label), []).append(index)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in occurrences.values():
        train_indices.extend(indices[::2])
        test_indices.extend(indices[1::2])
    eligible = [
        index
        for index in test_indices
        if any(labels[train].item() == labels[index].item() for train in train_indices)
    ]
    if eligible and train_indices:
        nearest = similarities[eligible][:, train_indices].argmax(dim=1)
        predictions = labels[torch.tensor(train_indices)[nearest]]
        accuracy = float(predictions.eq(labels[eligible]).float().mean().item())
    else:
        accuracy = float("nan")
    return {
        "within_class_cosine": within,
        "between_class_cosine": between,
        "class_cosine_margin": within - between,
        "split_1nn_accuracy": accuracy,
        "split_1nn_evaluated": len(eligible),
    }


def score_distribution_metrics(
    scores: torch.Tensor,
    *,
    temperature: float,
    top_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = F.softmax(scores / temperature, dim=1)
    normalized_entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=1) / math.log(scores.size(1))
    count = max(1, int(round(scores.size(1) * top_ratio)))
    top_mass = probabilities.topk(count, dim=1).values.sum(dim=1)
    score_std = scores.std(dim=1, unbiased=False)
    return normalized_entropy, top_mass, score_std


def phase_shift(tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Match the mainline reflect-padded translation convention."""
    if dx == 0 and dy == 0:
        return tensor
    padded = F.pad(
        tensor,
        (max(0, dx), max(0, -dx), max(0, dy), max(0, -dy)),
        mode="reflect",
    )
    start_y = max(0, -dy)
    start_x = max(0, -dx)
    return padded[..., start_y : start_y + tensor.size(-2), start_x : start_x + tensor.size(-1)]


def align_phase_scores(
    scores: torch.Tensor,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    dx: int,
    dy: int,
) -> torch.Tensor:
    maps = scores.reshape(scores.size(0), 1, *grid_size)
    pixel_maps = F.interpolate(maps, size=image_size, mode="bilinear", align_corners=False)
    aligned = phase_shift(pixel_maps, -dx, -dy)
    return F.interpolate(aligned, size=grid_size, mode="area").flatten(1)


def optional_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def mean(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def run_model(model_name: str, names, pixels_cpu, labels_cpu, args, device):
    model = build_whitebox_model(1000, model_name, pretrained=True, device=device).eval()
    layer_rows: dict[str, list[dict[str, object]]] = {}
    global_features: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict[str, object]] = {}
    dx, dy = (int(value) for value in args.phase_shift.split(","))

    for layer in PATCH_SCORE_LAYER_CANDIDATES[model_name]:
        rows: list[dict[str, object]] = []
        globals_for_layer = []
        for start in range(0, len(names), args.batch_size):
            end = min(len(names), start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            with torch.inference_mode():
                clean = model.extract_patch_score_features(normalize(model, pixels), score_layer=layer)
                flipped = model.extract_patch_score_features(
                    normalize(model, torch.flip(pixels, dims=(-1,))), score_layer=layer
                )
                shifted = model.extract_patch_score_features(
                    normalize(model, phase_shift(pixels, dx, dy)), score_layer=layer
                )
                clean_scores = cosine_scores(clean)
                flip_scores = cosine_scores(flipped).reshape(
                    pixels.size(0), *flipped.grid_size
                ).flip(-1).flatten(1)
                shifted_scores = align_phase_scores(
                    cosine_scores(shifted),
                    shifted.grid_size,
                    tuple(pixels.shape[-2:]),
                    dx,
                    dy,
                )
                entropy, top_mass, score_std = score_distribution_metrics(
                    clean_scores,
                    temperature=args.score_temperature,
                    top_ratio=args.top_ratio,
                )
                flip_global = F.cosine_similarity(
                    clean.global_token.flatten(1), flipped.global_token.flatten(1), dim=1
                )
                phase_global = F.cosine_similarity(
                    clean.global_token.flatten(1), shifted.global_token.flatten(1), dim=1
                )
                flip_score = row_spearman(clean_scores, flip_scores)
                phase_score = row_spearman(clean_scores, shifted_scores)

            globals_for_layer.append(clean.global_token[:, 0].float().cpu())
            metadata[layer] = {
                "global_mode": clean.global_mode,
                "source_name": clean.source_name,
                "grid_size": list(clean.grid_size),
                "native_token_count": clean.local_tokens.size(1),
            }
            for local_index, image_index in enumerate(range(start, end)):
                rows.append(
                    {
                        "model": model_name,
                        "layer": layer,
                        "global_mode": clean.global_mode,
                        "image_name": names[image_index],
                        "label": int(labels_cpu[image_index]),
                        "global_flip_cosine": float(flip_global[local_index].cpu()),
                        "global_phase_cosine": float(phase_global[local_index].cpu()),
                        "score_flip_spearman": float(flip_score[local_index].cpu()),
                        "score_phase_spearman": float(phase_score[local_index].cpu()),
                        "score_normalized_entropy": float(entropy[local_index].cpu()),
                        "score_top_probability_mass": float(top_mass[local_index].cpu()),
                        "score_std": float(score_std[local_index].cpu()),
                    }
                )
        layer_rows[layer] = rows
        global_features[layer] = torch.cat(globals_for_layer)

    final_layer = PATCH_SCORE_LAYER_CANDIDATES[model_name][-1]
    summaries = []
    for layer in PATCH_SCORE_LAYER_CANDIDATES[model_name]:
        geometry = class_geometry(global_features[layer], labels_cpu)
        rows = layer_rows[layer]
        summaries.append(
            {
                "model": model_name,
                "layer": layer,
                **metadata[layer],
                "linear_cka_to_final_global": linear_cka(
                    global_features[layer], global_features[final_layer]
                ),
                **geometry,
                **{
                    key: mean(rows, key)
                    for key in (
                        "global_flip_cosine",
                        "global_phase_cosine",
                        "score_flip_spearman",
                        "score_phase_spearman",
                        "score_normalized_entropy",
                        "score_top_probability_mass",
                        "score_std",
                    )
                },
            }
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [row for rows in layer_rows.values() for row in rows], summaries


def main() -> None:
    args = parse_args()
    if args.samples < 2 or args.batch_size <= 0:
        raise ValueError("samples must be at least 2 and batch-size must be positive.")
    if args.score_temperature <= 0 or not 0 < args.top_ratio <= 1:
        raise ValueError("score-temperature must be positive and top-ratio must be in (0,1].")
    try:
        shift_values = tuple(int(value) for value in args.phase_shift.split(","))
    except ValueError as exc:
        raise ValueError("phase-shift must be two comma-separated integers.") from exc
    if len(shift_values) != 2:
        raise ValueError("phase-shift must be dx,dy.")
    models = (
        list(WHITEBOX_MODEL_CHOICES)
        if args.models == "all"
        else [item.strip() for item in args.models.split(",") if item.strip()]
    )
    invalid = set(models) - set(WHITEBOX_MODEL_CHOICES)
    if invalid or not models:
        raise ValueError(f"invalid models: {sorted(invalid)}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, pixels, labels = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for model_name in models:
        rows, model_summaries = run_model(model_name, names, pixels, labels, args, device)
        all_rows.extend(rows)
        summaries.extend(model_summaries)

    write_csv(args.output_dir / "per_image.csv", all_rows)
    write_csv(args.output_dir / "layer_summary.csv", summaries)
    payload = {
        "protocol": {
            "purpose": "descriptive_only_not_layer_selection",
            "models": models,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "seed": args.seed,
            "phase_shift": list(shift_values),
            "score_temperature": args.score_temperature,
            "top_ratio": args.top_ratio,
            "final_reference_layer": {
                model: PATCH_SCORE_LAYER_CANDIDATES[model][-1] for model in models
            },
        },
        "layers": [
            {
                key: optional_number(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
            for row in summaries
        ],
    }
    write_json(args.output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
