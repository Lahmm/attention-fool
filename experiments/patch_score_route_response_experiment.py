"""Measure feature and gradient responses to final-layer patch-score routes."""

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
from patch_score_mechanism_experiment import (
    build_route_masks,
    cosine_scores,
    evaluate_masked_logits,
    extract_features,
    find_image,
    model_normalize,
    patch_masks,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WHITEBOX_MODEL_CHOICES, default="vit_base_patch16_224")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--route-repeats", type=int, default=4)
    parser.add_argument("--drop-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "clean_resized_images")
    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "data" / "image_name_to_class_id_and_name.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/patch_score_route_response"))
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
    pixels = torch.stack([to_tensor(Image.open(path).convert("RGB")) for _, path, _ in selected])
    labels = torch.tensor([label for _, _, label in selected], dtype=torch.long)
    return [name for name, _, _ in selected], pixels, labels


def loss_gradient(model, pixels: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    variable = pixels.detach().clone().requires_grad_(True)
    logits = model.model(model_normalize(model, variable))
    loss = F.cross_entropy(logits, labels)
    gradient = torch.autograd.grad(loss, variable, retain_graph=False, create_graph=False)[0]
    return gradient.detach()


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.flatten(1)
    right = right.flatten(1)
    return F.cosine_similarity(left, right, dim=1)


def summarize_rows(rows: list[dict[str, object]], strategy: str) -> dict[str, float]:
    selected = [row for row in rows if row["strategy"] == strategy]
    if not selected:
        raise ValueError(f"no rows for strategy {strategy}")
    metrics = (
        "route_logit_drop",
        "route_loss_increase",
        "global_feature_shift",
        "kept_local_feature_shift",
        "kept_score_abs_change",
        "gradient_cosine",
        "gradient_norm_ratio",
    )
    return {
        f"{metric}_mean": sum(float(row[metric]) for row in selected) / len(selected)
        for metric in metrics
    }


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
    strategies = (
        "high_score_extreme",
        "low_score_extreme",
        "random_uniform",
        "score_deviation_extreme",
    )
    rows: list[dict[str, object]] = []
    stability: dict[str, list[torch.Tensor]] = {strategy: [] for strategy in strategies}
    for batch_start in range(0, len(names), args.sample_batch):
        batch_end = min(len(names), batch_start + args.sample_batch)
        pixels = pixels_cpu[batch_start:batch_end].to(device)
        labels = labels_cpu[batch_start:batch_end].to(device)
        with torch.no_grad():
            clean_features = extract_features(model, pixels)
            clean_scores = cosine_scores(clean_features)
            clean_logits = model.model(model_normalize(model, pixels))
        clean_gradient = loss_gradient(model, pixels, labels)
        image_masks = patch_masks(clean_features.grid_size, pixels.size(-2), pixels.size(-1), device)
        for strategy_index, strategy in enumerate(strategies):
            masks = []
            for repeat in range(args.route_repeats):
                generator = torch.Generator().manual_seed(args.seed + 100000 * batch_start + 1000 * strategy_index + repeat)
                mask = build_route_masks(clean_scores.cpu(), strategy, args.drop_ratio, generator).to(device)
                masks.append(mask.cpu())
                image_mask = torch.einsum("bn,nchw->bchw", mask.float(), image_masks.float()).clamp_max(1.0)
                masked_pixels = pixels * (1.0 - image_mask)
                with torch.no_grad():
                    masked_features = extract_features(model, masked_pixels)
                    route_logit_drop, route_loss_increase, prediction_changed = evaluate_masked_logits(
                        model, pixels, labels, mask, image_masks
                    )
                global_similarity = cosine_rows(clean_features.global_token, masked_features.global_token)
                global_shift = 1.0 - global_similarity
                kept = ~mask
                clean_local = clean_features.local_tokens
                masked_local = masked_features.local_tokens
                local_similarity = F.cosine_similarity(clean_local, masked_local, dim=-1)
                kept_local_shift = (1.0 - local_similarity).masked_select(kept).reshape(pixels.size(0), -1).mean(dim=1)
                score_change = (clean_scores - cosine_scores(masked_features)).abs()
                kept_score_abs_change = score_change.masked_select(kept).reshape(pixels.size(0), -1).mean(dim=1)
                masked_gradient = loss_gradient(model, masked_pixels, labels)
                gradient_cosine = cosine_rows(clean_gradient, masked_gradient)
                gradient_norm_ratio = masked_gradient.flatten(1).norm(dim=1) / clean_gradient.flatten(1).norm(dim=1).clamp_min(1e-12)
                for local_index in range(batch_end - batch_start):
                    rows.append({
                        "model": args.model,
                        "sample_index": batch_start + local_index,
                        "image_name": names[batch_start + local_index],
                        "strategy": strategy,
                        "drop_ratio": args.drop_ratio,
                        "repeat": repeat,
                        "clean_correct": bool(clean_logits.argmax(dim=1)[local_index].eq(labels[local_index]).cpu()),
                        "route_logit_drop": float(route_logit_drop[local_index].cpu()),
                        "route_loss_increase": float(route_loss_increase[local_index].cpu()),
                        "route_prediction_changed": bool(prediction_changed[local_index].cpu()),
                        "global_feature_shift": float(global_shift[local_index].cpu()),
                        "kept_local_feature_shift": float(kept_local_shift[local_index].cpu()),
                        "kept_score_abs_change": float(kept_score_abs_change[local_index].cpu()),
                        "gradient_cosine": float(gradient_cosine[local_index].cpu()),
                        "gradient_norm_ratio": float(gradient_norm_ratio[local_index].cpu()),
                    })
            for left_index in range(len(masks)):
                for right_index in range(left_index):
                    left, right = masks[left_index], masks[right_index]
                    intersection = (left & right).sum(dim=1).float()
                    union = (left | right).sum(dim=1).float().clamp_min(1.0)
                    stability[strategy].append(intersection / union)
        del pixels, labels, clean_features, clean_scores, clean_logits, clean_gradient, image_masks
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "config": {
            "model": args.model,
            "samples": len(names),
            "sample_batch": args.sample_batch,
            "route_repeats": args.route_repeats,
            "drop_ratio": args.drop_ratio,
            "seed": args.seed,
        },
        "mask_pairwise_iou": {
            strategy: float(torch.cat(values).mean()) if values else 1.0
            for strategy, values in stability.items()
        },
        "strategies": {strategy: summarize_rows(rows, strategy) for strategy in strategies},
    }
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "route_response_raw.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
