"""E1/E2: cross-layer patch-score promotion and cross-model agreement.

Patch scores from every registered layer are converted to within-image ranks
on a common spatial grid.  The primary promotion map is the late-layer rank
minus the early-layer rank.  The experiment reports whether low-to-high
promotion is image-specific, shared across architectures, and stable to a
horizontal flip.
"""

from __future__ import annotations

import argparse
import itertools
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
    bootstrap_ci,
    common_map,
    load_samples,
    normalize,
    rank_norm,
    row_spearman,
    top_mask,
    write_csv,
    write_json,
)
from nets import PATCH_SCORE_LAYER_CANDIDATES, WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
PRIMARY_LAYERS = {
    "vit_base_patch16_224": ("block3", "block12"),
    "cait_s24_224": ("block6_gap", "block24_gap"),
    "pit_b_224": ("stage1_block3", "stage3_block4"),
    "visformer_small": ("stage1_block4", "stage3_block4"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=628)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--common-grid", type=int, default=7)
    parser.add_argument("--top-ratio", type=float, default=0.15)
    parser.add_argument("--early-low-quantile", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/patch_score_promotion_e1_e2"),
    )
    return parser.parse_args()


def cosine_scores(features) -> torch.Tensor:
    return F.cosine_similarity(
        features.local_tokens,
        features.global_token.expand_as(features.local_tokens),
        dim=-1,
    )


def row_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection = (left & right).sum(dim=1).float()
    union = (left | right).sum(dim=1).float()
    return intersection / union.clamp_min(1)


def row_pearson(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float() - left.float().mean(dim=1, keepdim=True)
    right = right.float() - right.float().mean(dim=1, keepdim=True)
    return F.cosine_similarity(left, right, dim=1)


def mean(value: torch.Tensor) -> float:
    return float(value.float().mean().item())


def extract_model_maps(model_name, pixels_cpu, args, device):
    model = build_whitebox_model(1000, model_name, pretrained=True, device=device).eval()
    maps: dict[str, list[torch.Tensor]] = {
        layer: [] for layer in PATCH_SCORE_LAYER_CANDIDATES[model_name]
    }
    flip_maps: dict[str, list[torch.Tensor]] = {
        layer: [] for layer in PATCH_SCORE_LAYER_CANDIDATES[model_name]
    }
    metadata: dict[str, dict[str, object]] = {}
    for layer in PATCH_SCORE_LAYER_CANDIDATES[model_name]:
        for start in range(0, pixels_cpu.size(0), args.batch_size):
            pixels = pixels_cpu[start : start + args.batch_size].to(device)
            with torch.inference_mode():
                clean = model.extract_patch_score_features(normalize(model, pixels), score_layer=layer)
                flipped = model.extract_patch_score_features(
                    normalize(model, torch.flip(pixels, dims=(-1,))), score_layer=layer
                )
                clean_map = common_map(cosine_scores(clean), clean.grid_size, args.common_grid)
                flip_native = cosine_scores(flipped).reshape(pixels.size(0), *flipped.grid_size)
                flip_native = flip_native.flip(-1).flatten(1)
                flip_map = common_map(flip_native, flipped.grid_size, args.common_grid)
            maps[layer].append(clean_map.float().cpu())
            flip_maps[layer].append(flip_map.float().cpu())
            metadata[layer] = {
                "global_mode": clean.global_mode,
                "source_name": clean.source_name,
                "native_grid": list(clean.grid_size),
            }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        {layer: torch.cat(chunks) for layer, chunks in maps.items()},
        {layer: torch.cat(chunks) for layer, chunks in flip_maps.items()},
        metadata,
    )


def main() -> None:
    args = parse_args()
    if args.samples <= 1 or args.batch_size <= 0 or args.common_grid <= 1:
        raise ValueError("samples, batch-size, and common-grid are invalid.")
    if not 0 < args.top_ratio < 0.5 or not 0 < args.early_low_quantile < 1:
        raise ValueError("top-ratio and early-low-quantile are invalid.")
    models = (
        list(WHITEBOX_MODEL_CHOICES)
        if args.models == "all"
        else [item.strip() for item in args.models.split(",") if item.strip()]
    )
    if not models or set(models) - set(WHITEBOX_MODEL_CHOICES):
        raise ValueError(f"invalid models: {models}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, pixels, labels = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )

    raw_maps: dict[str, dict[str, torch.Tensor]] = {}
    raw_flip_maps: dict[str, dict[str, torch.Tensor]] = {}
    metadata: dict[str, dict[str, dict[str, object]]] = {}
    model_summary: dict[str, dict[str, object]] = {}
    per_image_rows: list[dict[str, object]] = []

    for model_name in models:
        maps, flip_maps, model_metadata = extract_model_maps(
            model_name, pixels, args, device
        )
        raw_maps[model_name] = maps
        raw_flip_maps[model_name] = flip_maps
        metadata[model_name] = model_metadata
        early, late = PRIMARY_LAYERS[model_name]
        early_rank = rank_norm(maps[early])
        late_rank = rank_norm(maps[late])
        flip_early_rank = rank_norm(flip_maps[early])
        flip_late_rank = rank_norm(flip_maps[late])
        promotion = late_rank - early_rank
        flip_promotion = flip_late_rank - flip_early_rank
        promoted = early_rank.lt(args.early_low_quantile) & late_rank.ge(1 - args.top_ratio)
        flip_promoted = (
            flip_early_rank.lt(args.early_low_quantile)
            & flip_late_rank.ge(1 - args.top_ratio)
        )
        promotion_top = top_mask(promotion, args.top_ratio)
        flip_promotion_top = top_mask(flip_promotion, args.top_ratio)
        final_top = top_mask(late_rank, args.top_ratio)
        flip_final_top = top_mask(flip_late_rank, args.top_ratio)
        adjacent = {}
        layers = list(PATCH_SCORE_LAYER_CANDIDATES[model_name])
        for left, right in zip(layers, layers[1:]):
            adjacent[f"{left}->{right}"] = mean(row_spearman(maps[left], maps[right]))
        model_summary[model_name] = {
            "primary_early": early,
            "primary_late": late,
            "early_late_spearman": mean(row_spearman(maps[early], maps[late])),
            "strict_promotion_fraction": mean(promoted.float()),
            "images_with_strict_promotion": mean(promoted.any(dim=1).float()),
            "promotion_top_mean_gain": mean(
                promotion[promotion_top].reshape(args.samples, -1)
            ),
            "promotion_flip_spearman": mean(row_spearman(promotion, flip_promotion)),
            "promotion_flip_top_iou": mean(row_iou(promotion_top, flip_promotion_top)),
            "strict_promotion_flip_iou": mean(row_iou(promoted, flip_promoted)),
            "final_high_flip_top_iou": mean(row_iou(final_top, flip_final_top)),
            "adjacent_score_spearman": adjacent,
        }
        for index, image_name in enumerate(names):
            per_image_rows.append(
                {
                    "model": model_name,
                    "image_name": image_name,
                    "label": int(labels[index]),
                    "early_late_spearman": float(
                        row_spearman(maps[early][index:index+1], maps[late][index:index+1])[0]
                    ),
                    "strict_promotion_fraction": float(promoted[index].float().mean()),
                    "promotion_flip_spearman": float(
                        row_spearman(promotion[index:index+1], flip_promotion[index:index+1])[0]
                    ),
                    "promotion_flip_top_iou": float(
                        row_iou(promotion_top[index:index+1], flip_promotion_top[index:index+1])[0]
                    ),
                }
            )

    promotions = {}
    promotion_tops = {}
    strict_promotions = {}
    final_tops = {}
    for model_name in models:
        early, late = PRIMARY_LAYERS[model_name]
        early_rank = rank_norm(raw_maps[model_name][early])
        late_rank = rank_norm(raw_maps[model_name][late])
        promotions[model_name] = late_rank - early_rank
        promotion_tops[model_name] = top_mask(promotions[model_name], args.top_ratio)
        strict_promotions[model_name] = (
            early_rank.lt(args.early_low_quantile) & late_rank.ge(1 - args.top_ratio)
        )
        final_tops[model_name] = top_mask(late_rank, args.top_ratio)

    pair_summary: dict[str, dict[str, object]] = {}
    same_minus_mismatch_spearman = []
    same_minus_mismatch_iou = []
    generator = torch.Generator().manual_seed(args.seed + 101)
    spatial_permutation = torch.randperm(args.common_grid**2, generator=generator)
    for left, right in itertools.combinations(models, 2):
        key = f"{left}__{right}"
        same_spearman = row_spearman(promotions[left], promotions[right])
        same_iou = row_iou(promotion_tops[left], promotion_tops[right])
        mismatch_right = promotions[right].roll(1, dims=0)
        mismatch_top = promotion_tops[right].roll(1, dims=0)
        mismatch_spearman = row_spearman(promotions[left], mismatch_right)
        mismatch_iou = row_iou(promotion_tops[left], mismatch_top)
        permuted_spearman = row_spearman(
            promotions[left], promotions[right][:, spatial_permutation]
        )
        permuted_iou = row_iou(
            promotion_tops[left], promotion_tops[right][:, spatial_permutation]
        )
        strict_iou = row_iou(strict_promotions[left], strict_promotions[right])
        final_iou = row_iou(final_tops[left], final_tops[right])
        spearman_gap = same_spearman - mismatch_spearman
        iou_gap = same_iou - mismatch_iou
        same_minus_mismatch_spearman.append(spearman_gap)
        same_minus_mismatch_iou.append(iou_gap)
        pair_summary[key] = {
            "same_image_promotion_spearman": mean(same_spearman),
            "mismatched_image_promotion_spearman": mean(mismatch_spearman),
            "spatial_permutation_promotion_spearman": mean(permuted_spearman),
            "same_minus_mismatched_spearman": mean(spearman_gap),
            "same_image_promotion_top_iou": mean(same_iou),
            "mismatched_image_promotion_top_iou": mean(mismatch_iou),
            "spatial_permutation_promotion_top_iou": mean(permuted_iou),
            "same_minus_mismatched_top_iou": mean(iou_gap),
            "strict_promotion_iou": mean(strict_iou),
            "final_high_top_iou": mean(final_iou),
        }

    top_votes = torch.stack([promotion_tops[model].int() for model in models]).sum(dim=0)
    cross_model_summary: dict[str, object] = {"pairwise": pair_summary}
    if pair_summary:
        spearman_gaps = torch.cat(same_minus_mismatch_spearman)
        iou_gaps = torch.cat(same_minus_mismatch_iou)
        cross_model_summary.update({
            "macro_same_image_promotion_spearman": sum(
                item["same_image_promotion_spearman"] for item in pair_summary.values()
            ) / len(pair_summary),
            "macro_mismatched_image_promotion_spearman": sum(
                item["mismatched_image_promotion_spearman"] for item in pair_summary.values()
            ) / len(pair_summary),
            "macro_same_image_promotion_top_iou": sum(
                item["same_image_promotion_top_iou"] for item in pair_summary.values()
            ) / len(pair_summary),
            "macro_mismatched_image_promotion_top_iou": sum(
                item["mismatched_image_promotion_top_iou"] for item in pair_summary.values()
            ) / len(pair_summary),
            "macro_final_high_top_iou": sum(
                item["final_high_top_iou"] for item in pair_summary.values()
            ) / len(pair_summary),
            "same_minus_mismatched_spearman_ci95": bootstrap_ci(
                spearman_gaps, seed=args.seed + 201
            ),
            "same_minus_mismatched_top_iou_ci95": bootstrap_ci(
                iou_gaps, seed=args.seed + 202
            ),
            "consensus_top_fraction_at_least_3_of_4": mean(top_votes.ge(3).float()),
            "consensus_top_fraction_4_of_4": mean(top_votes.eq(4).float()),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_image.csv", per_image_rows)
    torch.save(
        {
            "names": names,
            "labels": labels,
            "models": models,
            "common_grid": args.common_grid,
            "primary_layers": {model: PRIMARY_LAYERS[model] for model in models},
            "raw_maps": raw_maps,
            "raw_flip_maps": raw_flip_maps,
            "promotions": promotions,
            "promotion_top_masks": promotion_tops,
            "strict_promotion_masks": strict_promotions,
            "final_top_masks": final_tops,
        },
        args.output_dir / "maps.pt",
    )
    payload = {
        "protocol": {
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "models": models,
            "common_grid": [args.common_grid, args.common_grid],
            "top_ratio": args.top_ratio,
            "top_count": max(1, int(round(args.common_grid**2 * args.top_ratio))),
            "early_low_quantile": args.early_low_quantile,
            "primary_layers": {model: list(PRIMARY_LAYERS[model]) for model in models},
            "definition": "within-image late patch-score rank minus early patch-score rank",
            "patch_score_noise": False,
        },
        "model_summary": model_summary,
        "cross_model_summary": cross_model_summary,
        "metadata": metadata,
    }
    write_json(args.output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
