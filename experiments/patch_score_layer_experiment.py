"""Compare patch-score routing at early, middle, and final feature layers.

This is a routing-only diagnostic.  It keeps the masked pixel-space
evaluation, drop budget, random seeds, and route candidates fixed across
layers; it does not run the iterative attack or tune a layer toward ASR.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from nets.base import PatchScoreFeatures
from patch_score_mechanism_experiment import (
    build_route_masks,
    evaluate_masked_logits,
    find_image,
    model_normalize,
    patch_masks,
    write_rows,
)


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WHITEBOX_MODEL_CHOICES, default="vit_base_patch16_224")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--route-repeats", type=int, default=4)
    parser.add_argument("--drop-ratio", type=float, default=0.15)
    parser.add_argument("--layers", type=str, default="early,mid,final")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/patch_score_layers"))
    return parser.parse_args()


def load_samples(image_dir: Path, annotations_path: Path, limit: int):
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    selected = []
    for image_name in sorted(annotations):
        path = find_image(image_dir, image_name)
        if path is not None:
            selected.append((image_name, path, int(annotations[image_name]["class_id"])))
        if len(selected) >= limit:
            break
    if not selected:
        raise RuntimeError("no annotated images were found")
    to_tensor = ToTensor()
    pixels = torch.stack(
        [to_tensor(Image.open(path).convert("RGB")) for _, path, _ in selected], dim=0
    )
    labels = torch.tensor([label for _, _, label in selected], dtype=torch.long)
    return [name for name, _, _ in selected], pixels, labels


def cosine_scores(features: PatchScoreFeatures) -> torch.Tensor:
    return F.cosine_similarity(
        features.local_tokens,
        features.global_token.expand_as(features.local_tokens),
        dim=-1,
    )


def resolve_layers(spec: str, count: int) -> list[tuple[str, int]]:
    aliases = {"early": 0, "mid": count // 2, "final": count - 1}
    result: list[tuple[str, int]] = []
    for raw in (item.strip() for item in spec.split(",")):
        if not raw:
            continue
        if raw in aliases:
            index = aliases[raw]
            label = raw
        else:
            index = int(raw)
            if index < 0:
                index += count
            label = f"layer_{index}"
        if not 0 <= index < count:
            raise ValueError(f"layer index {index} is outside [0, {count})")
        item = (label, index)
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("layers must contain at least one layer")
    return result


def generic_features(model, tokens: torch.Tensor, layer_index: int, final_index: int) -> PatchScoreFeatures:
    """Convert captured architecture-specific activations to local/global tokens."""
    if layer_index == final_index:
        # Preserve the production definition, especially CaiT's token-only CLS.
        raise RuntimeError("final layer must be supplied by extract_patch_score_features")
    if tokens.ndim == 4:
        batch, channels, height, width = tokens.shape
        local = tokens.flatten(2).transpose(1, 2)
        return PatchScoreFeatures(
            local_tokens=local,
            global_token=local.mean(dim=1, keepdim=True),
            grid_size=(int(height), int(width)),
            source_name=f"feature_modules[{layer_index}]",
        )
    if tokens.ndim != 3:
        raise ValueError(f"unsupported captured activation shape: {tuple(tokens.shape)}")
    token_count = tokens.size(1)
    # ViT/PiT expose a prefix token; CaiT's patch blocks expose only locals.
    has_prefix = model.model_name.startswith(("vit_", "pit_"))
    if has_prefix:
        local = tokens[:, 1:]
        global_token = tokens[:, :1]
    else:
        local = tokens
        global_token = local.mean(dim=1, keepdim=True)
    patch_count = local.size(1)
    side = int(round(patch_count ** 0.5))
    if side * side != patch_count:
        raise ValueError(f"cannot infer square patch grid from {token_count} tokens")
    return PatchScoreFeatures(
        local_tokens=local,
        global_token=global_token,
        grid_size=(side, side),
        source_name=f"feature_modules[{layer_index}]",
    )


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.sample_batch <= 0 or args.route_repeats <= 0:
        raise ValueError("samples, sample-batch, and route-repeats must be positive")
    if not 0.0 < args.drop_ratio <= 1.0:
        raise ValueError("drop-ratio must be in (0, 1]")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, pixels_cpu, labels_cpu = load_samples(args.image_dir, args.annotations, args.samples)
    model = build_whitebox_model(num_classes=1000, model_name=args.model, pretrained=True, device=device)
    model.eval()
    layers = resolve_layers(args.layers, model.num_blocks)
    strategies = ("high_score_extreme", "low_score_extreme", "random_uniform", "score_deviation_extreme")
    rows: list[dict[str, object]] = []
    stability: dict[str, list[torch.Tensor]] = {}
    for batch_start in range(0, len(names), args.sample_batch):
        batch_end = min(len(names), batch_start + args.sample_batch)
        pixels = pixels_cpu[batch_start:batch_end].to(device)
        labels = labels_cpu[batch_start:batch_end].to(device)
        normalized = model_normalize(model, pixels)
        with torch.no_grad():
            clean_logits, captured = model(normalized, return_tokens=True)
        image_masks_by_layer: dict[str, torch.Tensor] = {}
        for layer_label, layer_index in layers:
            if layer_index == model.num_blocks - 1:
                with torch.no_grad():
                    features = model.extract_patch_score_features(normalized, score_layer="final")
            else:
                features = generic_features(model, captured[layer_index], layer_index, model.num_blocks - 1)
            features.validate()
            scores = cosine_scores(features)
            image_masks = patch_masks(features.grid_size, pixels.size(-2), pixels.size(-1), device)
            image_masks_by_layer[layer_label] = image_masks
            for strategy_index, strategy in enumerate(strategies):
                key = f"{layer_label}:{strategy}"
                stability.setdefault(key, [])
                masks = []
                for repeat in range(args.route_repeats):
                    generator = torch.Generator().manual_seed(
                        args.seed + 100000 * batch_start + 1000 * layer_index + 100 * strategy_index + repeat
                    )
                    mask = build_route_masks(scores.cpu(), strategy, args.drop_ratio, generator).to(device)
                    masks.append(mask.cpu())
                    logit_drop, loss_increase, prediction_changed = evaluate_masked_logits(
                        model, pixels, labels, mask, image_masks
                    )
                    selected_scores = scores.masked_select(mask).reshape(scores.size(0), -1)
                    for local_index in range(batch_end - batch_start):
                        rows.append({
                            "model": args.model,
                            "sample_index": batch_start + local_index,
                            "image_name": names[batch_start + local_index],
                            "layer": layer_label,
                            "layer_index": layer_index,
                            "feature_source": features.source_name,
                            "grid_h": features.grid_size[0],
                            "grid_w": features.grid_size[1],
                            "strategy": strategy,
                            "drop_ratio": args.drop_ratio,
                            "repeat": repeat,
                            "clean_correct": bool(clean_logits.argmax(dim=1)[local_index].eq(labels[local_index]).cpu()),
                            "route_logit_drop": float(logit_drop[local_index].cpu()),
                            "route_loss_increase": float(loss_increase[local_index].cpu()),
                            "route_prediction_changed": bool(prediction_changed[local_index].cpu()),
                            "selected_score_mean": float(selected_scores[local_index].mean().cpu()),
                        })
                for left_index in range(len(masks)):
                    for right_index in range(left_index):
                        left, right = masks[left_index], masks[right_index]
                        intersection = (left & right).sum(dim=1).float()
                        union = (left | right).sum(dim=1).float().clamp_min(1.0)
                        stability[key].append(intersection / union)
        del pixels, labels, normalized, clean_logits, captured
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "config": {
            "model": args.model,
            "samples": len(names),
            "sample_batch": args.sample_batch,
            "route_repeats": args.route_repeats,
            "drop_ratio": args.drop_ratio,
            "layers": [{"label": label, "index": index} for label, index in layers],
            "seed": args.seed,
        },
        "mask_pairwise_iou": {
            key: float(torch.cat(values).mean()) if values else 1.0
            for key, values in stability.items()
        },
    }
    for layer_label, _ in layers:
        summary[layer_label] = {}
        for strategy in strategies:
            selected = [
                row for row in rows if row["layer"] == layer_label and row["strategy"] == strategy
            ]
            summary[layer_label][strategy] = {
                "logit_drop_mean": float(sum(float(row["route_logit_drop"]) for row in selected) / len(selected)),
                "loss_increase_mean": float(sum(float(row["route_loss_increase"]) for row in selected) / len(selected)),
                "prediction_changed_rate": float(sum(bool(row["route_prediction_changed"]) for row in selected) / len(selected)),
            }
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "route_per_image_raw.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
