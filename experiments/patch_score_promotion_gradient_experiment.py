"""E4: test whether promoted semantic routes improve transferable gradients.

This is a minimal exploitation experiment rather than a full attack search.  A
clean image determines one fixed 7x7 route mask.  The mask is reused by every
original/phase view and gates opponent-channel noise at the model's first RGB
projection.  Pixels and tokens are never dropped.  All conditions receive the
same sample-keyed noise draws and the same total feature-space RMS.
"""

from __future__ import annotations

import argparse
import csv
import gc
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

from attack import PatchScoreAttacker
from experiments.patch_score_promotion_observation import bootstrap_ci
from experiments.patch_score_routing_gradient_experiment import (
    clean_gradient,
    cosine_rows,
    load_samples,
    normalize,
    row_spearman,
    sign_agreement_rows,
    view_metrics,
)
from gradient_replay import GradientReplay
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
CONDITIONS = ("promotion", "final_high", "random", "uniform")
PHASES = ((4, 4), (8, 8), (12, 12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=WHITEBOX_MODEL_CHOICES, required=True)
    parser.add_argument("--targets", default="auto")
    parser.add_argument(
        "--maps",
        type=Path,
        default=Path("outputs/research/patch_score_promotion_e1_e2/maps.pt"),
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=628)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--noise-strength", type=float, default=0.2)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=0.75)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/patch_score_promotion_e4_gradients"),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def mask_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    intersection = (left & right).sum(dim=1).float()
    union = (left | right).sum(dim=1).clamp_min(1).float()
    return intersection / union


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.norm() * right.norm()
    if denominator <= 0:
        return float("nan")
    return float((left @ right / denominator).item())


def make_random_masks(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    masks = []
    for sample_mask in reference:
        count = int(sample_mask.sum())
        indices = torch.randperm(sample_mask.numel(), generator=generator)[:count]
        masks.append(torch.zeros_like(sample_mask).scatter(0, indices, True))
    return torch.stack(masks)


def route_image_mask(route_mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    side = int(round(math.sqrt(route_mask.size(1))))
    if side * side != route_mask.size(1):
        raise ValueError("route mask must have a square common grid")
    return F.interpolate(
        route_mask.reshape(route_mask.size(0), 1, side, side).float(),
        size=(height, width),
        mode="nearest",
    )


def build_attacker(model, args: argparse.Namespace) -> PatchScoreAttacker:
    return PatchScoreAttacker(
        model,
        epsilon=args.epsilon,
        steps=args.steps,
        attack_method="none",
        input_diversity_groups=args.groups,
        input_diversity_views_per_group=2,
        post_dropout_feature_noise_strength=args.noise_strength,
        post_dropout_feature_noise_type="opponent_projected",
        gaussian_sigma=args.gaussian_sigma,
        gaussian_alpha=args.gaussian_alpha,
        device=model.device,
    )


def routed_gradient_probe(
    model,
    attacker: PatchScoreAttacker,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    route_mask: torch.Tensor,
    condition: str,
    sample_ids: list[str],
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return 20 matched-view gradients for one fixed clean route mask."""
    replay = GradientReplay(seed)
    replay.begin_batch(sample_ids)
    probe = pixels.detach().requires_grad_(True)
    base_mask = route_image_mask(route_mask, pixels.size(-2), pixels.size(-1)).to(pixels)
    gradients = []
    attacker._gradient_replay = replay
    try:
        for group in range(attacker.input_diversity_groups):
            phase = PHASES[group % len(PHASES)]
            for view in range(2):
                replay.set_context(step=0, group=group, view=view)
                if view == 0:
                    view_pixels = probe
                    view_mask = base_mask
                else:
                    view_pixels = attacker._apply_phase_shift(probe, *phase)
                    view_mask = attacker._apply_phase_shift(base_mask, *phase)
                state = model.prepare_attack_feature_state(normalize(model, view_pixels))
                state.validate()
                raw_noise = attacker._strict_opponent_feature_noise(state)
                if condition == "uniform":
                    selected = torch.ones(
                        state.local_tokens.shape[:2],
                        device=state.local_tokens.device,
                        dtype=torch.bool,
                    )
                else:
                    selected = attacker._image_mask_to_projection_drop_mask(view_mask, state)
                gated = torch.where(
                    selected.unsqueeze(-1), raw_noise, torch.zeros_like(raw_noise)
                )
                matched = attacker._match_feature_noise_rms(
                    state.local_tokens, gated, f"promotion_route_{condition}"
                )
                logits = model.forward_from_attack_feature_state(
                    state, state.local_tokens + matched
                )
                loss = F.cross_entropy(logits, labels)
                gradients.append(torch.autograd.grad(loss, probe, retain_graph=False)[0])
    finally:
        attacker._gradient_replay = None
    views = torch.stack(gradients)
    raw_mean = views.mean(dim=0)
    processed = attacker._apply_gaussian_residual(raw_mean)
    return {
        "view_gradients": views.detach(),
        "raw_mean": raw_mean.detach(),
        "processed": processed.detach(),
    }


def paired_summary(
    cross_rows: list[dict[str, object]],
    condition: str,
    baseline: str,
    metric: str,
    seed: int,
) -> dict[str, object]:
    selected = {
        (row["target_model"], row["sample_index"]): float(row[metric])
        for row in cross_rows
        if row["condition"] == condition
    }
    reference = {
        (row["target_model"], row["sample_index"]): float(row[metric])
        for row in cross_rows
        if row["condition"] == baseline
    }
    keys = sorted(set(selected) & set(reference))
    delta = torch.tensor([selected[key] - reference[key] for key in keys])
    return {
        "mean": float(delta.mean()),
        "ci95": bootstrap_ci(delta, seed=seed),
        "positive_fraction": float(delta.gt(0).float().mean()),
        "comparisons": len(keys),
    }


def main() -> None:
    args = parse_args()
    if args.samples <= 1 or args.batch_size <= 0 or args.groups <= 0:
        raise ValueError("samples must exceed one; batch-size and groups must be positive")
    if args.groups * 2 != 20:
        raise ValueError("E4 uses the production 10x2=20-view protocol")
    payload = torch.load(args.maps, map_location="cpu", weights_only=False)
    if args.source not in payload["models"]:
        raise ValueError("source is absent from the E1 map artifact")
    targets = (
        [name for name in payload["models"] if name != args.source]
        if args.targets == "auto"
        else [item.strip() for item in args.targets.split(",") if item.strip()]
    )
    if args.source in targets or set(targets) - set(payload["models"]):
        raise ValueError("targets must be E1 models other than the source")
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    if names != payload["names"][: args.samples]:
        raise ValueError("E4 image identities do not match the E1 map artifact")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    promotion = payload["promotion_top_masks"][args.source][: args.samples].bool()
    final_high = payload["final_top_masks"][args.source][: args.samples].bool()
    masks = {
        "promotion": promotion,
        "final_high": final_high,
        "random": make_random_masks(promotion, args.seed + 1000),
        "uniform": torch.ones_like(promotion),
    }

    source_model = build_whitebox_model(
        1000, args.source, pretrained=True, device=device
    ).eval()
    attacker = build_attacker(source_model, args)
    source_clean_parts = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        clean, _, _, _ = clean_gradient(
            source_model,
            pixels_cpu[start:end].to(device),
            labels_cpu[start:end].to(device),
        )
        source_clean_parts.append(clean.cpu())
    source_clean = torch.cat(source_clean_parts)

    route_rows: list[dict[str, object]] = []
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    for condition in CONDITIONS:
        raw_parts, processed_parts = [], []
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            result = routed_gradient_probe(
                source_model,
                attacker,
                pixels,
                labels,
                masks[condition][start:end].to(device),
                condition,
                names[start:end],
                args.seed,
            )
            raw, processed = result["raw_mean"], result["processed"]
            raw_parts.append(raw.cpu())
            processed_parts.append(processed.cpu())
            view_cosine, view_sign, effective_rank = view_metrics(
                result["view_gradients"], raw
            )
            clean_cosine = cosine_rows(raw, source_clean[start:end].to(device))
            for local_index, sample_index in enumerate(range(start, end)):
                route_rows.append(
                    {
                        "source_model": args.source,
                        "sample_index": sample_index,
                        "image_name": names[sample_index],
                        "condition": condition,
                        "selected_common_patches": int(
                            masks[condition][sample_index].sum()
                        ),
                        "clean_route_gradient_cosine": float(clean_cosine[local_index]),
                        "view_cosine_to_raw_mean": float(view_cosine[local_index]),
                        "view_sign_agreement": float(view_sign[local_index]),
                        "view_effective_rank": float(effective_rank[local_index]),
                    }
                )
        gradients[condition] = {
            "raw": torch.cat(raw_parts),
            "processed": torch.cat(processed_parts),
        }

    cross_rows: list[dict[str, object]] = []
    step_size = args.epsilon / args.steps
    for target_name in targets:
        target_model = build_whitebox_model(
            1000, target_name, pretrained=True, device=device
        ).eval()
        overlap = mask_iou(
            promotion,
            payload["promotion_top_masks"][target_name][: args.samples].bool(),
        )
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            target_gradient, clean_true, clean_loss, clean_pred = clean_gradient(
                target_model, pixels, labels
            )
            for condition in CONDITIONS:
                raw = gradients[condition]["raw"][start:end].to(device)
                processed = gradients[condition]["processed"][start:end].to(device)
                adversarial = torch.clamp(
                    pixels + step_size * processed.sign(), 0.0, 1.0
                )
                with torch.no_grad():
                    adv_logits = target_model(normalize(target_model, adversarial))
                    adv_true = adv_logits.gather(1, labels[:, None]).squeeze(1)
                    adv_loss = F.cross_entropy(adv_logits, labels, reduction="none")
                raw_cosine = cosine_rows(raw, target_gradient)
                processed_cosine = cosine_rows(processed, target_gradient)
                raw_sign = sign_agreement_rows(raw, target_gradient)
                processed_sign = sign_agreement_rows(processed, target_gradient)
                for local_index, sample_index in enumerate(range(start, end)):
                    cross_rows.append(
                        {
                            "source_model": args.source,
                            "target_model": target_name,
                            "sample_index": sample_index,
                            "image_name": names[sample_index],
                            "condition": condition,
                            "source_target_promotion_iou": float(overlap[sample_index]),
                            "target_clean_correct": bool(
                                clean_pred[local_index].eq(labels[local_index]).cpu()
                            ),
                            "raw_target_gradient_cosine": float(raw_cosine[local_index]),
                            "processed_target_gradient_cosine": float(
                                processed_cosine[local_index]
                            ),
                            "raw_target_sign_agreement": float(raw_sign[local_index]),
                            "processed_target_sign_agreement": float(
                                processed_sign[local_index]
                            ),
                            "one_step_target_logit_drop": float(
                                clean_true[local_index] - adv_true[local_index]
                            ),
                            "one_step_target_loss_increase": float(
                                adv_loss[local_index] - clean_loss[local_index]
                            ),
                            "one_step_target_prediction_changed": bool(
                                adv_logits.argmax(1)[local_index]
                                .ne(clean_pred[local_index])
                                .cpu()
                            ),
                        }
                    )
        del target_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = (
        "raw_target_gradient_cosine",
        "processed_target_gradient_cosine",
        "raw_target_sign_agreement",
        "processed_target_sign_agreement",
        "one_step_target_logit_drop",
        "one_step_target_loss_increase",
    )
    summary: dict[str, object] = {
        "protocol": {
            "source": args.source,
            "targets": targets,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "seed": args.seed,
            "conditions": list(CONDITIONS),
            "views": args.groups * 2,
            "mask_policy": "one clean fixed mask reused across all views",
            "patch_operation": "route-gated noise only; no pixel or token drop",
            "noise": "opponent-channel projected through initial RGB convolution",
            "rms_policy": "post-gating total feature RMS matched across conditions",
            "noise_strength": args.noise_strength,
            "one_step_size": step_size,
            "gaussian_residual": {
                "sigma": args.gaussian_sigma,
                "alpha": args.gaussian_alpha,
            },
        },
        "route_mean": {},
        "cross_model_mean": {},
        "promotion_paired_deltas": {},
        "overlap_response_correlation": {},
    }
    for condition in CONDITIONS:
        route_selected = [row for row in route_rows if row["condition"] == condition]
        cross_selected = [row for row in cross_rows if row["condition"] == condition]
        summary["route_mean"][condition] = {
            key: mean(route_selected, key)
            for key in (
                "clean_route_gradient_cosine",
                "view_cosine_to_raw_mean",
                "view_sign_agreement",
                "view_effective_rank",
            )
        }
        summary["cross_model_mean"][condition] = {
            key: mean(cross_selected, key) for key in metrics
        }
    for baseline_index, baseline in enumerate(("random", "final_high", "uniform")):
        summary["promotion_paired_deltas"][baseline] = {
            metric: paired_summary(
                cross_rows,
                "promotion",
                baseline,
                metric,
                args.seed + 3000 + 100 * baseline_index + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }

    for target_name in targets:
        target_rows = [row for row in cross_rows if row["target_model"] == target_name]
        promotion_response = {
            int(row["sample_index"]): float(row["one_step_target_logit_drop"])
            for row in target_rows
            if row["condition"] == "promotion"
        }
        random_response = {
            int(row["sample_index"]): float(row["one_step_target_logit_drop"])
            for row in target_rows
            if row["condition"] == "random"
        }
        ordered = sorted(set(promotion_response) & set(random_response))
        advantage = torch.tensor(
            [promotion_response[index] - random_response[index] for index in ordered]
        )
        overlap = torch.tensor(
            [
                float(
                    next(
                        row["source_target_promotion_iou"]
                        for row in target_rows
                        if int(row["sample_index"]) == index
                    )
                )
                for index in ordered
            ]
        )
        summary["overlap_response_correlation"][target_name] = {
            "pearson": pearson(overlap, advantage),
            "spearman": float(row_spearman(overlap[None], advantage[None])[0]),
            "mean_promotion_iou": float(overlap.mean()),
        }

    output_dir = args.output_dir / args.source
    write_csv(output_dir / "route_metrics.csv", route_rows)
    write_csv(output_dir / "cross_model_metrics.csv", cross_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
