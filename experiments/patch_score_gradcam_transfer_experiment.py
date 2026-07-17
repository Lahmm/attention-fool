"""Cross-architecture transfer test for patch-score and token Grad-CAM masks."""

from __future__ import annotations

import argparse
import csv
import gc
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
    extract_features,
    find_image,
    model_normalize,
    patch_masks,
    write_rows,
)


SELECTORS = ("patch_score", "gradcam_relu", "gradcam_signed", "gradcam_abs", "random_uniform")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=str, default=",".join(WHITEBOX_MODEL_CHOICES))
    parser.add_argument("--targets", type=str, default=",".join(WHITEBOX_MODEL_CHOICES))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--route-repeats", type=int, default=4)
    parser.add_argument("--drop-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--image-dir", type=Path, default=REPO_ROOT / "data" / "clean_resized_images")
    parser.add_argument("--annotations", type=Path, default=REPO_ROOT / "data" / "image_name_to_class_id_and_name.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research/patch_score_gradcam_transfer"))
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


def source_maps(model, pixels: torch.Tensor, labels: torch.Tensor):
    normalized = model_normalize(model, pixels)
    logits_graph, captured = model(normalized, return_tokens=True)
    target = logits_graph.argmax(dim=1)
    target_logits = logits_graph.gather(1, target[:, None]).sum()
    full_gradient = torch.autograd.grad(target_logits, captured[-1], retain_graph=False)[0].detach()
    with torch.no_grad():
        features = extract_features(model, normalized)
        patch_score = cosine_scores(features)
    if captured[-1].ndim == 4:
        local = captured[-1].flatten(2).transpose(1, 2).detach()
        gradient_local = full_gradient.flatten(2).transpose(1, 2)
        grid_size = (captured[-1].size(-2), captured[-1].size(-1))
    elif model.model_name.startswith(("vit_", "pit_")):
        local = captured[-1][:, 1:].detach()
        gradient_local = full_gradient[:, 1:]
        side = int(round(local.size(1) ** 0.5))
        grid_size = (side, side)
    else:
        local = captured[-1].detach()
        gradient_local = full_gradient
        side = int(round(local.size(1) ** 0.5))
        grid_size = (side, side)
    alpha = gradient_local.mean(dim=1)
    signed = (local * alpha[:, None]).sum(dim=2)
    maps = {
        "patch_score": patch_score.detach(),
        "gradcam_relu": F.relu(signed).detach(),
        "gradcam_signed": signed.detach(),
        "gradcam_abs": signed.abs().detach(),
        "random_uniform": patch_score.detach(),
    }
    image_patch_masks = patch_masks(grid_size, pixels.size(-2), pixels.size(-1), pixels.device)
    del normalized, logits_graph, captured, features, patch_score, local, gradient_local, alpha, signed
    return maps, image_patch_masks.cpu(), grid_size


def masks_from_map(values: torch.Tensor, selector: str, drop_count: int, repeats: int, seed: int, image_patch_masks: torch.Tensor, batch_start: int):
    batch, patch_count = values.shape
    output = []
    for repeat in range(repeats):
        generator = torch.Generator().manual_seed(seed + 100000 * batch_start + 1000 * SELECTORS.index(selector) + repeat)
        selected = torch.zeros(batch, patch_count, dtype=torch.bool)
        for index in range(batch):
            if selector == "random_uniform":
                candidates = torch.arange(patch_count)
            else:
                candidates = values[index].topk(max(1, patch_count // 2)).indices
            chosen = candidates[torch.randperm(candidates.numel(), generator=generator)[:drop_count]]
            selected[index, chosen] = True
        image_mask = torch.einsum("bn,nchw->bchw", selected.float(), image_patch_masks.float()).clamp_max(1.0)
        output.append(image_mask)
    return output


def target_logits(model, pixels: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.model(model_normalize(model, pixels))


def run_target(model, pixels_cpu: torch.Tensor, labels_cpu: torch.Tensor, masks_by_selector: dict[str, torch.Tensor], batch_size: int, source: str, target: str):
    rows = []
    for start in range(0, pixels_cpu.size(0), batch_size):
        end = min(pixels_cpu.size(0), start + batch_size)
        pixels = pixels_cpu[start:end].to(model.device)
        labels = labels_cpu[start:end].to(model.device)
        clean_logits = target_logits(model, pixels)
        clean_true = clean_logits.gather(1, labels[:, None]).squeeze(1)
        clean_loss = F.cross_entropy(clean_logits, labels, reduction="none")
        for selector in SELECTORS:
            masks = masks_by_selector[selector][start:end]
            for repeat in range(len(masks[0])):
                image_mask = torch.stack([masks[index][repeat] for index in range(end - start)]).to(model.device)
                masked_logits = target_logits(model, pixels * (1.0 - image_mask))
                masked_true = masked_logits.gather(1, labels[:, None]).squeeze(1)
                masked_loss = F.cross_entropy(masked_logits, labels, reduction="none")
                for index in range(end - start):
                    rows.append({
                        "source_model": source,
                        "target_model": target,
                        "sample_index": start + index,
                        "selector": selector,
                        "repeat": repeat,
                        "target_clean_correct": bool(clean_logits.argmax(dim=1)[index].eq(labels[index]).cpu()),
                        "target_logit_drop": float((clean_true[index] - masked_true[index]).cpu()),
                        "target_loss_increase": float((masked_loss[index] - clean_loss[index]).cpu()),
                        "target_prediction_changed": bool(masked_logits.argmax(dim=1)[index].ne(clean_logits.argmax(dim=1)[index]).cpu()),
                    })
        del pixels, labels, clean_logits
        if model.device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    if not 0.0 < args.drop_ratio <= 1.0 or args.samples <= 0:
        raise ValueError("invalid drop ratio or sample count")
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    invalid = set(sources + targets) - set(WHITEBOX_MODEL_CHOICES)
    if invalid:
        raise ValueError(f"unknown models: {sorted(invalid)}")
    names, pixels_cpu, labels_cpu = load_samples(args.image_dir, args.annotations, args.samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for source in sources:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        source_model = build_whitebox_model(num_classes=1000, model_name=source, pretrained=True, device=device)
        source_model.eval()
        maps_by_selector = {selector: [] for selector in SELECTORS}
        masks_by_selector = {selector: [] for selector in SELECTORS}
        for start in range(0, len(names), args.sample_batch):
            end = min(len(names), start + args.sample_batch)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            maps, image_patch_masks, grid_size = source_maps(source_model, pixels, labels)
            drop_count = max(1, int(round(grid_size[0] * grid_size[1] * args.drop_ratio)))
            for selector in SELECTORS:
                selector_masks = masks_from_map(
                    maps[selector], selector, drop_count, args.route_repeats, args.seed,
                    image_patch_masks, start,
                )
                masks_by_selector[selector].extend([
                    [selector_masks[repeat][index].clone() for repeat in range(args.route_repeats)]
                    for index in range(end - start)
                ])
            del pixels, labels, maps, image_patch_masks
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del source_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for target in targets:
            target_model = build_whitebox_model(num_classes=1000, model_name=target, pretrained=True, device=device)
            target_model.eval()
            all_rows.extend(run_target(target_model, pixels_cpu, labels_cpu, masks_by_selector, args.sample_batch, source, target))
            del target_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = {
        "config": {
            "sources": sources,
            "targets": targets,
            "samples": len(names),
            "sample_batch": args.sample_batch,
            "route_repeats": args.route_repeats,
            "drop_ratio": args.drop_ratio,
            "mask_source": "source-model final local-token map",
            "gradcam_variants": ["relu", "signed", "abs"],
            "seed": args.seed,
        },
        "results": {},
    }
    for source in sources:
        summary["results"][source] = {}
        for target in targets:
            summary["results"][source][target] = {}
            for selector in SELECTORS:
                selected = [row for row in all_rows if row["source_model"] == source and row["target_model"] == target and row["selector"] == selector]
                summary["results"][source][target][selector] = {
                    "target_logit_drop_mean": sum(float(row["target_logit_drop"]) for row in selected) / len(selected),
                    "target_loss_increase_mean": sum(float(row["target_loss_increase"]) for row in selected) / len(selected),
                    "target_prediction_changed_rate": sum(bool(row["target_prediction_changed"]) for row in selected) / len(selected),
                }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "transfer_raw.csv", all_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
