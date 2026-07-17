"""Fair final-token comparison between patch-score routing and token Grad-CAM."""

from __future__ import annotations

import argparse
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
    cosine_scores,
    evaluate_masked_logits,
    extract_features,
    find_image,
    model_normalize,
    patch_masks,
    rank_tensor,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WHITEBOX_MODEL_CHOICES, default="vit_base_patch16_224")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--route-repeats", type=int, default=4)
    parser.add_argument("--drop-ratio", type=float, default=0.15)
    parser.add_argument("--target-mode", choices=("predicted", "true", "non_predicted"), default="predicted")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "clean_resized_images")
    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "data" / "image_name_to_class_id_and_name.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/patch_score_gradcam"))
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


def capture_gradcam_activation(model, normalized: torch.Tensor):
    """Capture the input to the final feature block, which remains logit-connected."""
    captured_input: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        captured_input["value"] = inputs[0]

    handle = model.feature_modules[-1].register_forward_pre_hook(pre_hook)
    try:
        logits, captured = model(normalized, return_tokens=True)
    finally:
        handle.remove()
    if "value" not in captured_input:
        raise RuntimeError("failed to capture the final block input for Grad-CAM")
    return logits, captured, captured_input["value"]


def row_spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_rank = rank_tensor(left)
    right_rank = rank_tensor(right)
    left_rank = left_rank - left_rank.mean(dim=1, keepdim=True)
    right_rank = right_rank - right_rank.mean(dim=1, keepdim=True)
    denominator = left_rank.norm(dim=1) * right_rank.norm(dim=1)
    return (left_rank * right_rank).sum(dim=1) / denominator.clamp_min(1e-12)


def local_from_captured(model, captured: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if captured.ndim == 4:
        local = captured.flatten(2).transpose(1, 2)
        gradient_local = None
    elif model.model_name.startswith(("vit_", "pit_")):
        local = captured[:, 1:]
        gradient_local = None
    else:
        local = captured
        gradient_local = None
    return local, gradient_local


def select_mask(values: torch.Tensor, selector: str, count: int, generator: torch.Generator) -> torch.Tensor:
    batch, patches = values.shape
    masks = torch.zeros(batch, patches, dtype=torch.bool)
    for index in range(batch):
        if selector == "patch_score":
            candidates = values[index].topk(max(1, patches // 2)).indices
        elif selector.startswith("gradcam"):
            candidates = values[index].topk(max(1, patches // 2)).indices
        elif selector == "random_uniform":
            candidates = torch.arange(patches)
        else:
            raise ValueError(selector)
        chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:count]]
        masks[index, chosen] = True
    return masks


def summary_for(rows: list[dict[str, object]], selector: str) -> dict[str, float]:
    selected = [row for row in rows if row["selector"] == selector]
    metrics = (
        "route_logit_drop",
        "route_loss_increase",
        "route_prediction_changed",
        "map_rank_spearman",
        "map_top_half_iou",
        "candidate_zero_gradcam",
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
    selectors = ("patch_score", "gradcam_relu", "gradcam_signed", "gradcam_abs", "random_uniform")
    map_variants = ("patch_score", "random_uniform", "gradcam_relu", "gradcam_signed", "gradcam_abs")
    rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    for batch_start in range(0, len(names), args.sample_batch):
        batch_end = min(len(names), batch_start + args.sample_batch)
        pixels = pixels_cpu[batch_start:batch_end].to(device)
        labels = labels_cpu[batch_start:batch_end].to(device)
        normalized = model_normalize(model, pixels)
        logits_graph, captured, gradcam_activation = capture_gradcam_activation(model, normalized)
        predicted = logits_graph.argmax(dim=1)
        if args.target_mode == "predicted":
            target = predicted
        elif args.target_mode == "true":
            target = labels
        else:
            target = (predicted + 1) % logits_graph.size(1)
        target_logits = logits_graph.gather(1, target[:, None]).sum()
        full_gradient = torch.autograd.grad(target_logits, gradcam_activation, retain_graph=False)[0].detach()
        with torch.no_grad():
            clean_features = extract_features(model, normalized)
            patch_score_map = cosine_scores(clean_features)
            clean_logits = logits_graph.detach()
        if gradcam_activation.ndim == 4:
            local = gradcam_activation.flatten(2).transpose(1, 2).detach()
            gradient_local = full_gradient.flatten(2).transpose(1, 2)
            grid_size = (gradcam_activation.size(-2), gradcam_activation.size(-1))
        elif model.model_name.startswith(("vit_", "pit_")):
            local = gradcam_activation[:, 1:].detach()
            gradient_local = full_gradient[:, 1:]
            side = int(round(local.size(1) ** 0.5))
            grid_size = (side, side)
        else:
            local = gradcam_activation.detach()
            gradient_local = full_gradient
            side = int(round(local.size(1) ** 0.5))
            grid_size = (side, side)
        alpha = gradient_local.mean(dim=1)
        gradcam_signed = (local * alpha[:, None]).sum(dim=2)
        gradcam_maps = {
            "gradcam_relu": F.relu(gradcam_signed),
            "gradcam_signed": gradcam_signed,
            "gradcam_abs": gradcam_signed.abs(),
        }
        patch_top = patch_score_map.topk(max(1, patch_score_map.size(1) // 2), dim=1).indices
        patch_top_mask = torch.zeros_like(patch_score_map, dtype=torch.bool).scatter_(1, patch_top, True)
        map_metrics = {}
        for map_name, map_value in gradcam_maps.items():
            map_top = map_value.topk(max(1, map_value.size(1) // 2), dim=1).indices
            map_top_mask = torch.zeros_like(map_value, dtype=torch.bool).scatter_(1, map_top, True)
            map_metrics[map_name] = (
                row_spearman(patch_score_map, map_value),
                (patch_top_mask & map_top_mask).sum(dim=1).float()
                / (patch_top_mask | map_top_mask).sum(dim=1).float().clamp_min(1.0),
            )
        map_values = {"patch_score": patch_score_map, "random_uniform": patch_score_map}
        map_values.update(gradcam_maps)
        map_metric_values = {
            "patch_score": (torch.ones_like(patch_score_map[:, 0]), torch.ones_like(patch_score_map[:, 0])),
            "random_uniform": (torch.ones_like(patch_score_map[:, 0]), torch.ones_like(patch_score_map[:, 0])),
        }
        map_metric_values.update(map_metrics)
        zero_gradcam = gradcam_maps["gradcam_relu"].abs().sum(dim=1).eq(0)
        image_masks = patch_masks(grid_size, pixels.size(-2), pixels.size(-1), device)
        values = map_values
        for selector_index, selector in enumerate(selectors):
            masks = []
            for repeat in range(args.route_repeats):
                generator = torch.Generator().manual_seed(args.seed + 100000 * batch_start + 1000 * selector_index + repeat)
                drop_count = max(1, int(round(values[selector].size(1) * args.drop_ratio)))
                mask = select_mask(values[selector].cpu(), selector, drop_count, generator).to(device)
                masks.append(mask.cpu())
                route_logit_drop, route_loss_increase, prediction_changed = evaluate_masked_logits(
                    model, pixels, labels, mask, image_masks
                )
                for local_index in range(batch_end - batch_start):
                    rows.append({
                        "model": args.model,
                        "sample_index": batch_start + local_index,
                        "image_name": names[batch_start + local_index],
                        "target_mode": args.target_mode,
                        "selector": selector,
                        "drop_ratio": args.drop_ratio,
                        "repeat": repeat,
                        "clean_correct": bool(clean_logits.argmax(dim=1)[local_index].eq(labels[local_index]).cpu()),
                        "route_logit_drop": float(route_logit_drop[local_index].cpu()),
                        "route_loss_increase": float(route_loss_increase[local_index].cpu()),
                        "route_prediction_changed": bool(prediction_changed[local_index].cpu()),
                        "map_rank_spearman": float(map_metric_values[selector][0][local_index].cpu()),
                        "map_top_half_iou": float(map_metric_values[selector][1][local_index].cpu()),
                        "candidate_zero_gradcam": float(zero_gradcam[local_index].cpu()),
                    })
        map_rows.extend([
            {
                "model": args.model,
                "sample_index": batch_start + local_index,
                "image_name": names[batch_start + local_index],
                "target_mode": args.target_mode,
                "map_variant": map_name,
                "map_rank_spearman": float(map_metric_values[map_name][0][local_index].cpu()),
                "map_top_half_iou": float(map_metric_values[map_name][1][local_index].cpu()),
                "gradcam_zero": bool(zero_gradcam[local_index].cpu()),
            }
            for map_name in map_variants
            for local_index in range(batch_end - batch_start)
        ])
        del pixels, labels, normalized, logits_graph, captured, clean_features, patch_score_map, gradcam_signed
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "config": {
            "model": args.model,
            "samples": len(names),
            "sample_batch": args.sample_batch,
            "route_repeats": args.route_repeats,
            "drop_ratio": args.drop_ratio,
            "target_mode": args.target_mode,
            "score_layer": "final",
            "gradcam": "token Grad-CAM ReLU on logit-connected final-block input activation",
            "seed": args.seed,
        },
        "map": {
            map_name: {
                "rank_spearman_mean": sum(float(row["map_rank_spearman"]) for row in map_rows if row["map_variant"] == map_name) / len([row for row in map_rows if row["map_variant"] == map_name]),
                "top_half_iou_mean": sum(float(row["map_top_half_iou"]) for row in map_rows if row["map_variant"] == map_name) / len([row for row in map_rows if row["map_variant"] == map_name]),
            }
            for map_name in map_variants
        },
        "gradcam_relu_zero_fraction": sum(bool(row["gradcam_zero"]) for row in map_rows if row["map_variant"] == "gradcam_relu") / len([row for row in map_rows if row["map_variant"] == "gradcam_relu"]),
        "selectors": {selector: summary_for(rows, selector) for selector in selectors},
    }
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "route_per_image_raw.csv", rows)
    write_rows(output_dir / "map_per_image_raw.csv", map_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
