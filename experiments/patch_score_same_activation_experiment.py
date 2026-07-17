"""Protocol A: compare patch-score and Grad-CAM on exactly the same activation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from patch_score_gradcam_experiment import capture_gradcam_activation, load_samples
from patch_score_mechanism_experiment import rank_tensor


def row_spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = rank_tensor(left).float()
    right = rank_tensor(right).float()
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    return (left * right).sum(dim=1) / (left.norm(dim=1) * right.norm(dim=1)).clamp_min(1e-12)


def common_local_global(model, activation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    if activation.ndim == 4:
        local = activation.flatten(2).transpose(1, 2)
        grid = (int(activation.size(-2)), int(activation.size(-1)))
        global_token = local.mean(dim=1, keepdim=True)
    elif model.model_name.startswith(("vit_", "pit_")):
        local = activation[:, 1:]
        side = int(round(local.size(1) ** 0.5))
        grid = (side, side)
        global_token = activation[:, :1]
    else:
        local = activation
        side = int(round(local.size(1) ** 0.5))
        grid = (side, side)
        # CaiT final-block input contains local tokens but no CLS token.
        global_token = local.mean(dim=1, keepdim=True)
    return local, global_token, grid


def top_half_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    count = max(1, left.size(1) // 2)
    left_mask = torch.zeros_like(left, dtype=torch.bool).scatter_(1, left.topk(count, dim=1).indices, True)
    right_mask = torch.zeros_like(right, dtype=torch.bool).scatter_(1, right.topk(count, dim=1).indices, True)
    return (left_mask & right_mask).sum(dim=1).float() / (left_mask | right_mask).sum(dim=1).float().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(WHITEBOX_MODEL_CHOICES))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "clean_resized_images")
    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "data" / "image_name_to_class_id_and_name.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/patch_score_same_activation"))
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    names, pixels_cpu, labels_cpu = load_samples(args.image_dir, args.annotations, args.samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for model_name in models:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = build_whitebox_model(num_classes=1000, model_name=model_name, pretrained=True, device=device)
        model.eval()
        map_values = {name: [] for name in ("patch_score", "gradcam_relu", "gradcam_signed", "gradcam_abs")}
        zero_values = {name: [] for name in ("gradcam_relu", "gradcam_signed", "gradcam_abs")}
        for start in range(0, len(names), args.sample_batch):
            end = min(len(names), start + args.sample_batch)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            normalized = (pixels - torch.tensor(model.model_mean, device=device).view(1, 3, 1, 1)) / torch.tensor(model.model_std, device=device).view(1, 3, 1, 1)
            logits, _captured, activation = capture_gradcam_activation(model, normalized)
            predicted = logits.argmax(dim=1)
            gradient = torch.autograd.grad(logits.gather(1, predicted[:, None]).sum(), activation)[0].detach()
            local, global_token, _grid = common_local_global(model, activation.detach())
            patch_score = F.cosine_similarity(local, global_token.expand_as(local), dim=-1)
            local_gradient = gradient.flatten(2).transpose(1, 2) if gradient.ndim == 4 else (gradient[:, 1:] if model.model_name.startswith(("vit_", "pit_")) else gradient)
            local_for_gradcam = local
            alpha = local_gradient.mean(dim=1)
            signed = (local_for_gradcam * alpha[:, None]).sum(dim=2)
            maps = {"patch_score": patch_score, "gradcam_relu": F.relu(signed), "gradcam_signed": signed, "gradcam_abs": signed.abs()}
            for name, value in maps.items():
                if name == "patch_score":
                    continue
                map_values[name].append(row_spearman(patch_score, value).cpu())
                map_values.setdefault(f"{name}:iou", []).append(top_half_iou(patch_score, value).cpu())
                zero_values[name].append(value.abs().sum(dim=1).eq(0).cpu())
            del pixels, labels, normalized, logits, _captured, activation, gradient, local, global_token, patch_score, signed
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[model_name] = {
            name: {
                "rank_spearman_mean": float(torch.cat(map_values[name]).mean()),
                "top_half_iou_mean": float(torch.cat(map_values[f"{name}:iou"]).mean()),
                "zero_fraction": float(torch.cat(zero_values[name]).float().mean()),
            }
            for name in ("gradcam_relu", "gradcam_signed", "gradcam_abs")
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "config": {
            "models": models,
            "samples": len(names),
            "sample_batch": args.sample_batch,
            "protocol": "same logit-connected local activation; patch-score uses its global token or local mean, Grad-CAM uses target predicted-class gradient",
            "seed": args.seed,
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
