"""Compare patch-score and token Grad-CAM map stability under a horizontal flip."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from patch_score_gradcam_experiment import load_samples
from patch_score_gradcam_transfer_experiment import SELECTORS, source_maps
from patch_score_mechanism_experiment import rank_tensor


def row_spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = rank_tensor(left).float()
    right = rank_tensor(right).float()
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    return (left * right).sum(dim=1) / (left.norm(dim=1) * right.norm(dim=1)).clamp_min(1e-12)


def unflip_patch_map(values: torch.Tensor) -> torch.Tensor:
    side = int(round(values.size(1) ** 0.5))
    return values.reshape(values.size(0), side, side).flip(-1).reshape(values.size(0), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=",".join(WHITEBOX_MODEL_CHOICES))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "clean_resized_images")
    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "data" / "image_name_to_class_id_and_name.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/patch_score_gradcam_stability"))
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    names, pixels_cpu, labels_cpu = load_samples(args.image_dir, args.annotations, args.samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries = {}
    for model_name in models:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        model = build_whitebox_model(num_classes=1000, model_name=model_name, pretrained=True, device=device)
        model.eval()
        values = {selector: [] for selector in SELECTORS}
        zero_values = {selector: [] for selector in SELECTORS}
        for start in range(0, len(names), args.sample_batch):
            end = min(len(names), start + args.sample_batch)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            clean_maps, _, _ = source_maps(model, pixels, labels)
            flipped_maps, _, _ = source_maps(model, torch.flip(pixels, dims=[3]), labels)
            for selector in SELECTORS:
                clean = clean_maps[selector]
                flipped_back = unflip_patch_map(flipped_maps[selector])
                values[selector].append(row_spearman(clean, flipped_back).cpu())
                top_count = max(1, clean.size(1) // 2)
                clean_top = torch.zeros_like(clean, dtype=torch.bool).scatter_(1, clean.topk(top_count, dim=1).indices, True)
                flipped_top = torch.zeros_like(flipped_back, dtype=torch.bool).scatter_(1, flipped_back.topk(top_count, dim=1).indices, True)
                zero_values[selector].append(flipped_back.abs().sum(dim=1).eq(0).cpu())
                values.setdefault(f"{selector}:iou", []).append(
                    ((clean_top & flipped_top).sum(dim=1).float() / (clean_top | flipped_top).sum(dim=1).float().clamp_min(1.0)).cpu()
                )
                # Keep the map values separate from the IoU values in summary construction.
            del pixels, labels, clean_maps, flipped_maps
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summaries[model_name] = {
            selector: {
                "rank_spearman_mean": float(torch.cat(values[selector]).mean()),
                "top_half_iou_mean": float(torch.cat(values[f"{selector}:iou"]).mean()),
                "flipped_zero_fraction": float(torch.cat(zero_values[selector]).float().mean()),
            }
            for selector in SELECTORS
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {
        "config": {"models": models, "samples": len(names), "sample_batch": args.sample_batch, "view": "horizontal flip, map unflipped before comparison", "seed": args.seed},
        "results": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
