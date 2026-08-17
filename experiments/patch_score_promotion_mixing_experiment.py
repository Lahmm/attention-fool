"""E3: decompose promoted-patch alignment gains inside the late mixing block."""

from __future__ import annotations

import argparse
import json
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
    top_mask,
    write_csv,
    write_json,
)
from nets import build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
TARGET_BLOCKS = {
    "vit_base_patch16_224": lambda model: model.model.blocks[11],
    "cait_s24_224": lambda model: model.model.blocks[23],
    "pit_b_224": lambda model: model.model.transformers[2].blocks[3],
    "visformer_small": lambda model: model.model.stage3[3],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--maps",
        type=Path,
        default=Path("outputs/research/patch_score_promotion_e1_e2/maps.pt"),
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sample-offset", type=int, default=628)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/patch_score_promotion_e3_mixing"),
    )
    return parser.parse_args()


def tensor_from(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"cannot capture tensor from {type(value)!r}")


def local_and_global(model_name: str, tensor: torch.Tensor):
    if tensor.ndim == 4:
        local = tensor.flatten(2).transpose(1, 2)
        return local, local.mean(dim=1, keepdim=True), tuple(tensor.shape[-2:])
    if tensor.ndim != 3:
        raise ValueError(f"unexpected block tensor shape: {tuple(tensor.shape)}")
    if model_name in {"vit_base_patch16_224", "pit_b_224"}:
        local, global_token = tensor[:, 1:], tensor[:, :1]
    else:
        local = tensor
        global_token = local.mean(dim=1, keepdim=True)
    count = local.size(1)
    side = int(round(count**0.5))
    if side * side != count:
        raise ValueError(f"local token count {count} is not square")
    return local, global_token, (side, side)


def score_with_reference(local: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(local, reference.expand_as(local), dim=-1)


def mask_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1).float()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.maps, map_location="cpu", weights_only=False)
    models = list(payload["models"])
    if args.samples <= 1 or args.samples > len(payload["names"]):
        raise ValueError("samples must be between 2 and the E1 sample count")
    names, pixels_cpu, labels = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    if names != payload["names"][: args.samples]:
        raise ValueError("E3 image identities do not match the E1 map artifact")
    common_grid_size = int(payload["common_grid"])
    top_ratio = payload["promotion_top_masks"][models[0]].sum(dim=1)[0].item() / (
        common_grid_size**2
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}

    for model_index, model_name in enumerate(models):
        model = build_whitebox_model(1000, model_name, pretrained=True, device=device).eval()
        block = TARGET_BLOCKS[model_name](model)
        if not hasattr(block, "norm2"):
            raise ValueError(f"target block for {model_name} has no norm2 mixing boundary")
        captures: dict[str, torch.Tensor] = {}

        def pre_hook(_module, inputs):
            captures["pre"] = tensor_from(inputs)

        def mix_hook(_module, inputs):
            captures["post_mix"] = tensor_from(inputs)

        def post_hook(_module, _inputs, output):
            captures["post_block"] = tensor_from(output)

        handles = [
            block.register_forward_pre_hook(pre_hook),
            block.norm2.register_forward_pre_hook(mix_hook),
            block.register_forward_hook(post_hook),
        ]
        model_rows: list[dict[str, object]] = []
        early, late = payload["primary_layers"][model_name]
        early_rank = rank_norm(payload["raw_maps"][model_name][early][: args.samples])
        late_rank = rank_norm(payload["raw_maps"][model_name][late][: args.samples])
        stable_high = top_mask(torch.minimum(early_rank, late_rank), top_ratio)
        stable_low = top_mask(-torch.maximum(early_rank, late_rank), top_ratio)
        promotion_mask = payload["promotion_top_masks"][model_name][: args.samples].bool()
        generator = torch.Generator().manual_seed(args.seed + 1000 + model_index)
        random_masks = []
        count = int(promotion_mask[0].sum())
        for _ in range(args.samples):
            indices = torch.randperm(common_grid_size**2, generator=generator)[:count]
            random_masks.append(
                torch.zeros(common_grid_size**2, dtype=torch.bool).scatter(0, indices, True)
            )
        masks = {
            "promotion": promotion_mask,
            "stable_high": stable_high,
            "stable_low": stable_low,
            "random": torch.stack(random_masks),
        }

        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            captures.clear()
            with torch.inference_mode():
                model(normalize(model, pixels))
            if set(captures) != {"pre", "post_mix", "post_block"}:
                raise RuntimeError(f"incomplete block captures for {model_name}: {captures.keys()}")
            pre_local, _, grid = local_and_global(model_name, captures["pre"])
            mix_local, _, mix_grid = local_and_global(model_name, captures["post_mix"])
            post_local, post_global, post_grid = local_and_global(model_name, captures["post_block"])
            if grid != mix_grid or grid != post_grid:
                raise ValueError("mixing block changed spatial grid unexpectedly")
            pre_score = common_map(
                score_with_reference(pre_local, post_global), grid, common_grid_size
            )
            mix_score = common_map(
                score_with_reference(mix_local, post_global), grid, common_grid_size
            )
            post_score = common_map(
                score_with_reference(post_local, post_global), grid, common_grid_size
            )
            mix_gain = (mix_score - pre_score).cpu()
            mlp_gain = (post_score - mix_score).cpu()
            total_gain = (post_score - pre_score).cpu()
            for condition, full_mask in masks.items():
                batch_mask = full_mask[start:end]
                values = {
                    "mix_gain": mask_mean(mix_gain, batch_mask),
                    "mlp_gain": mask_mean(mlp_gain, batch_mask),
                    "total_gain": mask_mean(total_gain, batch_mask),
                }
                for local_index, image_index in enumerate(range(start, end)):
                    model_rows.append(
                        {
                            "model": model_name,
                            "image_name": names[image_index],
                            "condition": condition,
                            **{key: float(value[local_index]) for key, value in values.items()},
                        }
                    )
        for handle in handles:
            handle.remove()
        rows.extend(model_rows)
        by_condition = {
            condition: [row for row in model_rows if row["condition"] == condition]
            for condition in masks
        }
        condition_summary = {}
        for condition, condition_rows in by_condition.items():
            condition_summary[condition] = {
                metric: sum(float(row[metric]) for row in condition_rows) / len(condition_rows)
                for metric in ("mix_gain", "mlp_gain", "total_gain")
            }
        promotion_mix = torch.tensor([row["mix_gain"] for row in by_condition["promotion"]])
        summaries[model_name] = {
            "target_block": {
                "vit_base_patch16_224": "blocks[11]",
                "cait_s24_224": "blocks[23]",
                "pit_b_224": "transformers[2].blocks[3]",
                "visformer_small": "stage3[3]",
            }[model_name],
            "conditions": condition_summary,
            "promotion_minus_stable_low_mix_gain": float(
                promotion_mix.mean()
                - torch.tensor([row["mix_gain"] for row in by_condition["stable_low"]]).mean()
            ),
            "promotion_minus_random_mix_gain": float(
                promotion_mix.mean()
                - torch.tensor([row["mix_gain"] for row in by_condition["random"]]).mean()
            ),
            "promotion_mix_gain_ci95": bootstrap_ci(
                promotion_mix, seed=args.seed + 2000 + model_index
            ),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(args.output_dir / "per_image.csv", rows)
    output = {
        "protocol": {
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "models": models,
            "reference": "post-block global representation held fixed across pre/mix/post",
            "mix_boundary": "input to block.norm2 (post token-mixing residual)",
        },
        "results": summaries,
    }
    write_json(args.output_dir / "summary.json", output)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
