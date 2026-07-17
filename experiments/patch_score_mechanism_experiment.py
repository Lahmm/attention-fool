"""Mechanism-first experiments for patch-score-guided patch dropping.

This script deliberately evaluates the selector before running a full attack:

1. single-patch occlusion importance versus final-layer patch-score;
2. matched-budget high-score, low-score, random, and extreme routing;
3. mask score enrichment and repeated-mask stability.

The script does not tune toward a favorable result. It writes raw per-sample
and per-patch observations together with aggregate summaries so that the
motivation can be revised if the evidence does not support it.
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

# Allow direct execution as `python experiments/patch_score_mechanism_experiment.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "research" / "patch_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=WHITEBOX_MODEL_CHOICES, default="vit_base_patch16_224")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--route-repeats", type=int, default=8)
    parser.add_argument("--occlusion-chunk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def find_image(image_dir: Path, image_name: str) -> Path | None:
    direct = image_dir / image_name
    if direct.is_file():
        return direct
    stem = Path(image_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def load_samples(image_dir: Path, annotations_path: Path, limit: int) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    selected: list[tuple[str, Path, int]] = []
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
        [to_tensor(Image.open(path).convert("RGB")) for _, path, _ in selected],
        dim=0,
    )
    labels = torch.tensor([label for _, _, label in selected], dtype=torch.long)
    return [name for name, _, _ in selected], pixels, labels


def model_normalize(model, pixels: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(model.model_mean, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor(model.model_std, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels - mean) / std


@torch.no_grad()
def extract_features(model, pixels: torch.Tensor):
    return model.extract_patch_score_features(model_normalize(model, pixels), score_layer="final")


@torch.no_grad()
def get_logits(model, pixels: torch.Tensor) -> torch.Tensor:
    return model.model(model_normalize(model, pixels))


def cosine_scores(features) -> torch.Tensor:
    return F.cosine_similarity(
        features.local_tokens,
        features.global_token.expand_as(features.local_tokens),
        dim=-1,
    )


def patch_masks(grid_size: tuple[int, int], height: int, width: int, device: torch.device) -> torch.Tensor:
    grid_h, grid_w = grid_size
    count = grid_h * grid_w
    coarse = torch.zeros(count, 1, grid_h, grid_w, device=device)
    indices = torch.arange(count, device=device)
    coarse[indices, 0, indices // grid_w, indices % grid_w] = 1.0
    return F.interpolate(coarse, size=(height, width), mode="nearest").bool()


def rank_tensor(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    rank_values = torch.arange(values.size(1), device=values.device, dtype=torch.float32)
    ranks.scatter_(1, order, rank_values.expand(values.size(0), -1))
    return ranks


def row_correlation(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a - a.mean(dim=1, keepdim=True)
        b = b - b.mean(dim=1, keepdim=True)
        denominator = a.square().sum(dim=1).sqrt() * b.square().sum(dim=1).sqrt()
        return (a * b).sum(dim=1) / denominator.clamp_min(1e-12)

    pearson = corr(left, right)
    spearman = corr(rank_tensor(left), rank_tensor(right))
    return pearson, spearman


def random_selection(candidates: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]]


def build_route_masks(
    scores: torch.Tensor,
    strategy: str,
    ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    batch, count = scores.shape
    drop_count = max(1, int(round(count * ratio)))
    masks = torch.zeros(batch, count, dtype=torch.bool)
    for index in range(batch):
        if strategy == "high_score_stochastic":
            candidates = scores[index].topk(max(1, count // 2)).indices
            selected = random_selection(candidates, drop_count, generator)
        elif strategy == "low_score_stochastic":
            candidates = scores[index].topk(max(1, count // 2), largest=False).indices
            selected = random_selection(candidates, drop_count, generator)
        elif strategy == "random_uniform":
            selected = random_selection(torch.arange(count), drop_count, generator)
        elif strategy == "score_deviation_stochastic":
            deviation = (scores[index] - scores[index].median()).abs()
            candidates = deviation.topk(max(1, count // 2)).indices
            selected = random_selection(candidates, drop_count, generator)
        elif strategy == "high_score_extreme":
            selected = scores[index].topk(drop_count).indices
        elif strategy == "low_score_extreme":
            selected = scores[index].topk(drop_count, largest=False).indices
        elif strategy == "score_deviation_extreme":
            deviation = (scores[index] - scores[index].median()).abs()
            selected = deviation.topk(drop_count).indices
        else:
            raise ValueError(f"unknown routing strategy: {strategy}")
        masks[index, selected] = True
    return masks


@torch.no_grad()
def evaluate_masked_logits(
    model,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    image_patch_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_mask = torch.einsum("bn,nchw->bchw", mask.float(), image_patch_masks.float()).clamp_max(1.0)
    masked_pixels = pixels * (1.0 - image_mask)
    logits = get_logits(model, masked_pixels)
    clean_logits = get_logits(model, pixels)
    clean_true = clean_logits.gather(1, labels[:, None]).squeeze(1)
    masked_true = logits.gather(1, labels[:, None]).squeeze(1)
    clean_loss = F.cross_entropy(clean_logits, labels, reduction="none")
    masked_loss = F.cross_entropy(logits, labels, reduction="none")
    return clean_true - masked_true, masked_loss - clean_loss, (logits.argmax(dim=1) != clean_logits.argmax(dim=1))


@torch.no_grad()
def single_patch_occlusion(
    model,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    image_patch_masks: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = pixels.shape
    patch_count = image_patch_masks.size(0)
    clean_logits = get_logits(model, pixels)
    clean_true = clean_logits.gather(1, labels[:, None]).squeeze(1)
    clean_loss = F.cross_entropy(clean_logits, labels, reduction="none")
    logit_drop = torch.empty(batch, patch_count, device=pixels.device)
    loss_increase = torch.empty_like(logit_drop)
    for start in range(0, patch_count, chunk_size):
        end = min(patch_count, start + chunk_size)
        masks = image_patch_masks[start:end].float()
        occluded = pixels[:, None] * (1.0 - masks[None])
        occluded = occluded.reshape(batch * (end - start), 3, height, width)
        logits = get_logits(model, occluded).reshape(batch, end - start, -1)
        true_logits = logits.gather(2, labels[:, None, None].expand(-1, end - start, 1)).squeeze(-1)
        repeated_labels = labels[:, None].expand(-1, end - start).reshape(-1)
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), repeated_labels, reduction="none")
        logit_drop[:, start:end] = clean_true[:, None] - true_logits
        loss_increase[:, start:end] = losses.reshape(batch, end - start) - clean_loss[:, None]
    return logit_drop, loss_increase


def summarize(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten()
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "median": float(flat.median().cpu()),
        "q10": float(torch.quantile(flat, 0.1).cpu()),
        "q90": float(torch.quantile(flat, 0.9).cpu()),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.route_repeats <= 0 or args.sample_batch <= 0:
        raise ValueError("samples, sample-batch, and route-repeats must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, pixels_cpu, labels_cpu = load_samples(args.image_dir, args.annotations, args.samples)

    model = build_whitebox_model(num_classes=1000, model_name=args.model, pretrained=True, device=device)
    model.eval()
    score_batches: list[torch.Tensor] = []
    correct_batches: list[torch.Tensor] = []
    logit_drop_batches: list[torch.Tensor] = []
    loss_increase_batches: list[torch.Tensor] = []
    grid_size: tuple[int, int] | None = None
    score_source = ""
    route_rows: list[dict[str, object]] = []
    stability_values: dict[str, list[torch.Tensor]] = {}
    strategies = (
        "high_score_stochastic",
        "low_score_stochastic",
        "random_uniform",
        "score_deviation_stochastic",
        "high_score_extreme",
        "low_score_extreme",
        "score_deviation_extreme",
    )

    for batch_start in range(0, len(names), args.sample_batch):
        batch_end = min(len(names), batch_start + args.sample_batch)
        pixels = pixels_cpu[batch_start:batch_end].to(device)
        labels = labels_cpu[batch_start:batch_end].to(device)
        with torch.no_grad():
            features = extract_features(model, pixels)
            scores = cosine_scores(features)
            clean_logits = get_logits(model, pixels)
        if grid_size is None:
            grid_size = features.grid_size
            score_source = features.source_name
        elif grid_size != features.grid_size:
            raise RuntimeError("patch grid changed across sample batches")
        image_masks = patch_masks(grid_size, pixels.size(-2), pixels.size(-1), device)
        correct = clean_logits.argmax(dim=1).eq(labels)
        logit_drop, loss_increase = single_patch_occlusion(
            model,
            pixels,
            labels,
            image_masks,
            args.occlusion_chunk,
        )
        score_batches.append(scores.cpu())
        correct_batches.append(correct.cpu())
        logit_drop_batches.append(logit_drop.cpu())
        loss_increase_batches.append(loss_increase.cpu())

        masks_by_repeat: dict[str, list[torch.Tensor]] = {strategy: [] for strategy in strategies}
        for strategy_index, strategy in enumerate(strategies):
            stability_values.setdefault(strategy, [])
            for repeat in range(args.route_repeats):
                generator = torch.Generator().manual_seed(
                    args.seed + 100000 * batch_start + 1000 * strategy_index + repeat
                )
                mask = build_route_masks(scores.detach().cpu(), strategy, 0.15, generator).to(device)
                masks_by_repeat[strategy].append(mask.detach().cpu())
                route_logit_drop, route_loss_increase, prediction_changed = evaluate_masked_logits(
                    model,
                    pixels,
                    labels,
                    mask,
                    image_masks,
                )
                selected_scores = scores.masked_select(mask).reshape(scores.size(0), -1)
                route_rows.append(
                    {
                        "model": args.model,
                        "sample_start": batch_start,
                        "sample_count": batch_end - batch_start,
                        "strategy": strategy,
                        "repeat": repeat,
                        "clean_correct_rate": float(correct.float().mean().cpu()),
                        "route_logit_drop_mean": float(route_logit_drop.mean().cpu()),
                        "route_loss_increase_mean": float(route_loss_increase.mean().cpu()),
                        "route_prediction_change_rate": float(prediction_changed.float().mean().cpu()),
                        "selected_score_mean": float(selected_scores.mean().cpu()),
                        "selected_score_std": float(selected_scores.std(unbiased=False).cpu()),
                        "all_score_mean": float(scores.mean().cpu()),
                        "all_score_std": float(scores.std(unbiased=False).cpu()),
                    }
                )
        for strategy, masks in masks_by_repeat.items():
            for left_index in range(len(masks)):
                for right_index in range(left_index):
                    left = masks[left_index]
                    right = masks[right_index]
                    intersection = (left & right).sum(dim=1).float()
                    union = (left | right).sum(dim=1).float().clamp_min(1.0)
                    stability_values[strategy].append(intersection / union)
        del pixels, labels, features, scores, clean_logits, image_masks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = torch.cat(score_batches, dim=0)
    correct = torch.cat(correct_batches, dim=0)
    logit_drop = torch.cat(logit_drop_batches, dim=0)
    loss_increase = torch.cat(loss_increase_batches, dim=0)
    if grid_size is None:
        raise RuntimeError("no samples were processed")
    pearson_logit, spearman_logit = row_correlation(scores, logit_drop)
    pearson_loss, spearman_loss = row_correlation(scores, loss_increase)

    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, object]] = []
    for sample_index, name in enumerate(names):
        for patch_index in range(scores.size(1)):
            raw_rows.append(
                {
                    "model": args.model,
                    "sample_index": sample_index,
                    "image_name": name,
                    "label": int(labels_cpu[sample_index].cpu()),
                    "clean_correct": bool(correct[sample_index].cpu()),
                    "patch_index": patch_index,
                    "patch_score": float(scores[sample_index, patch_index].cpu()),
                    "single_patch_logit_drop": float(logit_drop[sample_index, patch_index].cpu()),
                    "single_patch_loss_increase": float(loss_increase[sample_index, patch_index].cpu()),
                }
            )
    write_rows(output_dir / "single_patch_raw.csv", raw_rows)

    write_rows(output_dir / "route_raw.csv", route_rows)

    stability = {
        strategy: float(torch.cat(values).mean()) if values else 1.0
        for strategy, values in stability_values.items()
    }

    summary = {
        "config": {
            "model": args.model,
            "samples": len(names),
            "sample_batch": args.sample_batch,
            "route_repeats": args.route_repeats,
            "occlusion_chunk": args.occlusion_chunk,
            "seed": args.seed,
            "device": str(device),
            "score_source": score_source,
            "score_grid": list(grid_size),
            "drop_ratio": 0.15,
            "drop_count": max(1, round(scores.size(1) * 0.15)),
        },
        "clean_accuracy": float(correct.float().mean().cpu()),
        "score_vs_single_patch_occlusion": {
            "logit_drop_pearson": summarize(pearson_logit),
            "logit_drop_spearman": summarize(spearman_logit),
            "loss_increase_pearson": summarize(pearson_loss),
            "loss_increase_spearman": summarize(spearman_loss),
        },
        "single_patch_logit_drop": summarize(logit_drop),
        "single_patch_loss_increase": summarize(loss_increase),
        "mask_pairwise_iou": stability,
        "route_rows": route_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
