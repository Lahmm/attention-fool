"""Protocol A at frozen routing layers: patch-score versus true-class Grad-CAM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack import PatchScoreAttacker
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from patch_score_routing_gradient_experiment import load_samples, normalize, rank_rows
from routing_config import FrozenRoutingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-config", type=Path, required=True)
    parser.add_argument("--models", default=",".join(WHITEBOX_MODEL_CHOICES))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sample-offset", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data/clean_resized_images")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT / "data/image_name_to_class_id_and_name.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/patch_score_gradcam_selected_layer"),
    )
    return parser.parse_args()


def row_spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_rank = rank_rows(left)
    right_rank = rank_rows(right)
    left_rank -= left_rank.mean(dim=1, keepdim=True)
    right_rank -= right_rank.mean(dim=1, keepdim=True)
    return F.cosine_similarity(left_rank, right_rank, dim=1)


def top_iou(left: torch.Tensor, right: torch.Tensor, count: int) -> torch.Tensor:
    count = max(1, min(count, left.size(1)))
    left_mask = torch.zeros_like(left, dtype=torch.bool).scatter(
        1, left.topk(count, dim=1).indices, True
    )
    right_mask = torch.zeros_like(right, dtype=torch.bool).scatter(
        1, right.topk(count, dim=1).indices, True
    )
    intersection = (left_mask & right_mask).sum(dim=1).float()
    union = (left_mask | right_mask).sum(dim=1).float().clamp_min(1)
    return intersection / union


def activation_local_global(activation: torch.Tensor, grid_size: tuple[int, int]):
    local = PatchScoreAttacker._local_activation_for_grid(activation, grid_size)
    patch_count = grid_size[0] * grid_size[1]
    if activation.ndim == 3 and activation.size(1) > patch_count:
        global_token = activation[:, :1]
    else:
        global_token = local.mean(dim=1, keepdim=True)
    return local, global_token


def capture_forward(model, normalized: torch.Tensor, layer: str):
    capture = model.patch_score_activation_capture(layer)
    capture.validate()
    captured: dict[str, torch.Tensor] = {}

    def input_hook(_module, inputs):
        captured["activation"] = inputs[0]

    def output_hook(_module, _inputs, output):
        captured["activation"] = output[0] if isinstance(output, (tuple, list)) else output

    handle = (
        capture.module.register_forward_pre_hook(input_hook)
        if capture.hook_type == "input"
        else capture.module.register_forward_hook(output_hook)
    )
    try:
        logits = model(normalized)
    finally:
        handle.remove()
    if "activation" not in captured:
        raise RuntimeError("failed to capture selected-layer activation.")
    return logits, captured["activation"], capture


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive.")
    config = FrozenRoutingConfig.load(args.routing_config)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    invalid = set(models) - set(WHITEBOX_MODEL_CHOICES)
    if invalid:
        raise ValueError(f"unknown models: {sorted(invalid)}")
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, object]] = []
    for model_name in models:
        model = build_whitebox_model(1000, model_name, pretrained=True, device=device).eval()
        layer = config.layer_for(model_name)
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            normalized = normalize(model, pixels)
            with torch.no_grad():
                production = model.extract_patch_score_features(
                    normalized, score_layer=layer
                )
                production_score = F.cosine_similarity(
                    production.local_tokens,
                    production.global_token.expand_as(production.local_tokens),
                    dim=2,
                )
            logits, activation, capture = capture_forward(model, normalized, layer)
            local, global_token = activation_local_global(activation, production.grid_size)
            same_activation_score = F.cosine_similarity(
                local, global_token.expand_as(local), dim=2
            )
            true_logit = logits.gather(1, labels[:, None]).sum()
            alternate = (labels + 1) % logits.size(1)
            alternate_logit = logits.gather(1, alternate[:, None]).sum()
            true_gradient = torch.autograd.grad(
                true_logit, activation, retain_graph=True
            )[0]
            alternate_gradient = torch.autograd.grad(
                alternate_logit, activation, retain_graph=False
            )[0]
            true_local_gradient = PatchScoreAttacker._local_activation_for_grid(
                true_gradient, production.grid_size
            )
            alternate_local_gradient = PatchScoreAttacker._local_activation_for_grid(
                alternate_gradient, production.grid_size
            )
            true_alpha = true_local_gradient.mean(dim=1)
            alternate_alpha = alternate_local_gradient.mean(dim=1)
            gradcam = F.relu((local * true_alpha[:, None]).sum(dim=2)).detach()
            alternate_gradcam = F.relu(
                (local * alternate_alpha[:, None]).sum(dim=2)
            ).detach()
            same_activation_score = same_activation_score.detach()
            patch_count = gradcam.size(1)
            half_count = max(1, patch_count // 2)
            drop_count = max(1, round(0.15 * patch_count))
            metrics = {
                "patch_gradcam_spearman": row_spearman(same_activation_score, gradcam),
                "patch_gradcam_top_half_iou": top_iou(
                    same_activation_score, gradcam, half_count
                ),
                "patch_gradcam_top_drop_iou": top_iou(
                    same_activation_score, gradcam, drop_count
                ),
                "production_same_activation_spearman": row_spearman(
                    production_score, same_activation_score
                ),
                "true_alternate_gradcam_spearman": row_spearman(
                    gradcam, alternate_gradcam
                ),
                "true_alternate_gradcam_top_drop_iou": top_iou(
                    gradcam, alternate_gradcam, drop_count
                ),
            }
            for index in range(end - start):
                all_rows.append(
                    {
                        "model": model_name,
                        "sample_index": start + index,
                        "image_name": names[start + index],
                        "layer": layer,
                        "production_global_mode": production.global_mode,
                        "activation_source": capture.source_name,
                        "grid_h": production.grid_size[0],
                        "grid_w": production.grid_size[1],
                        "drop_count": drop_count,
                        "gradcam_true_class": int(labels[index].cpu()),
                        "gradcam_alternate_class": int(alternate[index].cpu()),
                        "gradcam_zero": bool(gradcam[index].abs().sum().eq(0).cpu()),
                        "alternate_gradcam_zero": bool(
                            alternate_gradcam[index].abs().sum().eq(0).cpu()
                        ),
                        **{
                            name: float(value[index].cpu())
                            for name, value in metrics.items()
                        },
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_rows(args.output_dir / "per_image.csv", all_rows)
    summary = {
        "config": {
            "models": models,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "target_mode": "true_class",
            "alternate_control": "(true_class + 1) modulo class_count",
            "layers": {model: config.layer_for(model) for model in models},
        },
        "results": {},
    }
    metric_names = (
        "patch_gradcam_spearman",
        "patch_gradcam_top_half_iou",
        "patch_gradcam_top_drop_iou",
        "production_same_activation_spearman",
        "true_alternate_gradcam_spearman",
        "true_alternate_gradcam_top_drop_iou",
    )
    for model_name in models:
        selected = [row for row in all_rows if row["model"] == model_name]
        summary["results"][model_name] = {
            **{
                name: sum(float(row[name]) for row in selected) / len(selected)
                for name in metric_names
            },
            "gradcam_zero_fraction": sum(bool(row["gradcam_zero"]) for row in selected)
            / len(selected),
            "activation_source": selected[0]["activation_source"],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
