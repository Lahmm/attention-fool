"""Production-route representation and cross-model gradient diagnostics."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack import PatchScoreAttacker
from gradient_replay import GradientReplay
from nets import PATCH_SCORE_LAYER_CANDIDATES, WHITEBOX_MODEL_CHOICES, build_whitebox_model
from routing_config import FrozenRoutingConfig


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
CONDITIONS = ("selected", "opposite", "deviation", "random", "no_drop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=WHITEBOX_MODEL_CHOICES, required=True)
    parser.add_argument("--targets", default="auto")
    parser.add_argument("--routing-config", type=Path, required=True)
    parser.add_argument("--layers", default="frozen", help="frozen, all, or comma-separated IDs")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/patch_score_routing_gradients"),
    )
    return parser.parse_args()


def load_samples(image_dir: Path, annotations_path: Path, offset: int, limit: int):
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    available = []
    for image_name in sorted(annotations):
        direct = image_dir / image_name
        path = direct if direct.is_file() else None
        if path is None:
            for suffix in (".png", ".jpg", ".jpeg"):
                candidate = image_dir / f"{Path(image_name).stem}{suffix}"
                if candidate.is_file():
                    path = candidate
                    break
        if path is not None:
            available.append((image_name, path, int(annotations[image_name]["class_id"])))
    selected = available[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(
            f"requested {limit} samples at offset {offset}, found {len(selected)}."
        )
    to_tensor = ToTensor()
    pixels = torch.stack(
        [to_tensor(Image.open(path).convert("RGB")) for _, path, _ in selected]
    )
    labels = torch.tensor([label for _, _, label in selected], dtype=torch.long)
    return [name for name, _, _ in selected], pixels, labels


def normalize(model, pixels: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(model.model_mean, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor(model.model_std, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels - mean) / std


def cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1)


def sign_agreement_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_sign = left.flatten(1).sign()
    right_sign = right.flatten(1).sign()
    valid = left_sign.ne(0) | right_sign.ne(0)
    numerator = (left_sign.eq(right_sign) & valid).sum(dim=1).float()
    return numerator / valid.sum(dim=1).clamp_min(1).float()


def rank_rows(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=1)
    ranks = torch.empty_like(order, dtype=torch.float32)
    rank_values = torch.arange(values.size(1), device=values.device, dtype=torch.float32)
    return ranks.scatter(1, order, rank_values.expand_as(order))


def row_spearman(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_rank = rank_rows(left)
    right_rank = rank_rows(right)
    left_rank -= left_rank.mean(dim=1, keepdim=True)
    right_rank -= right_rank.mean(dim=1, keepdim=True)
    return F.cosine_similarity(left_rank, right_rank, dim=1)


def view_metrics(view_gradients: torch.Tensor, raw_mean: torch.Tensor):
    # [V,B,C,H,W] -> [B,V,P]
    views = view_gradients.flatten(2).transpose(0, 1)
    mean = raw_mean.flatten(1)
    view_cosine = F.cosine_similarity(views, mean[:, None], dim=2).mean(dim=1)
    mean_sign = mean.sign()[:, None]
    view_sign = views.sign()
    valid = mean_sign.ne(0).expand_as(view_sign)
    sign_agreement = ((view_sign.eq(mean_sign) & valid).sum(dim=(1, 2)).float()
                      / valid.sum(dim=(1, 2)).clamp_min(1).float())
    normalized = views / views.norm(dim=2, keepdim=True).clamp_min(1e-12)
    gram = torch.bmm(normalized, normalized.transpose(1, 2))
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum(dim=1, keepdim=True).clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    )
    return view_cosine, sign_agreement, effective_rank


def clean_gradient(model, pixels: torch.Tensor, labels: torch.Tensor):
    probe = pixels.detach().requires_grad_(True)
    logits = model(normalize(model, probe))
    losses = F.cross_entropy(logits, labels, reduction="none")
    gradient = torch.autograd.grad(losses.sum(), probe)[0]
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    return gradient.detach(), true_logits.detach(), losses.detach(), logits.argmax(dim=1).detach()


def condition_settings(condition: str, frozen_polarity: str):
    if condition == "selected":
        return "patch_score", frozen_polarity
    if condition == "opposite":
        return "patch_score", "low" if frozen_polarity == "high" else "high"
    if condition in {"deviation", "random", "no_drop"}:
        return condition, frozen_polarity
    raise ValueError(f"unknown condition: {condition}")


def make_attacker(model, layer: str, condition: str, polarity: str, args):
    selector, score_mode = condition_settings(condition, polarity)
    return PatchScoreAttacker(
        model,
        epsilon=args.epsilon,
        steps=args.steps,
        attack_method="original_score_postdrop_phase_pair",
        input_diversity_groups=10,
        input_diversity_views_per_group=2,
        input_diversity_phase_shift_set=((4, 4), (8, 8), (12, 12)),
        patch_dropout_ratio=0.3,
        patch_dropout_score_mode=score_mode,
        patch_dropout_sampling_mode="random",
        patch_selector=selector,
        patch_score_layer=layer,
        post_dropout_feature_noise_strength=0.2,
        post_dropout_feature_noise_type="opponent_projected",
        gaussian_sigma=4.0,
        gaussian_alpha=0.75,
        device=model.device,
    )


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
    if args.samples <= 0 or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("samples, batch-size, and steps must be positive.")
    config = FrozenRoutingConfig.load(args.routing_config)
    if args.layers == "frozen":
        layers = [config.layer_for(args.source)]
    elif args.layers == "all":
        layers = list(PATCH_SCORE_LAYER_CANDIDATES[args.source])
    else:
        layers = [item.strip() for item in args.layers.split(",") if item.strip()]
        invalid = set(layers) - set(PATCH_SCORE_LAYER_CANDIDATES[args.source])
        if invalid:
            raise ValueError(f"invalid layers for {args.source}: {sorted(invalid)}")
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    invalid_conditions = set(conditions) - set(CONDITIONS)
    if invalid_conditions:
        raise ValueError(f"invalid conditions: {sorted(invalid_conditions)}")
    targets = (
        [name for name in WHITEBOX_MODEL_CHOICES if name != args.source]
        if args.targets == "auto"
        else [item.strip() for item in args.targets.split(",") if item.strip()]
    )
    if args.source in targets or set(targets) - set(WHITEBOX_MODEL_CHOICES):
        raise ValueError("targets must be registered models other than the source.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    source_model = build_whitebox_model(1000, args.source, pretrained=True, device=device).eval()

    # Cache the source clean gradient once; it is not a selector criterion.
    source_clean_gradients = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        gradient, _, _, _ = clean_gradient(
            source_model,
            pixels_cpu[start:end].to(device),
            labels_cpu[start:end].to(device),
        )
        source_clean_gradients.append(gradient.cpu())
    source_clean = torch.cat(source_clean_gradients)

    route_rows: list[dict[str, object]] = []
    gradients_by_key: dict[str, dict[str, torch.Tensor]] = {}
    for layer in layers:
        for condition in conditions:
            key = f"{layer}:{condition}"
            attacker = make_attacker(
                source_model, layer, condition, config.global_polarity, args
            )
            raw_values, processed_values = [], []
            for start in range(0, args.samples, args.batch_size):
                end = min(args.samples, start + args.batch_size)
                pixels = pixels_cpu[start:end].to(device)
                labels = labels_cpu[start:end].to(device)
                batch_names = names[start:end]

                # Reconstruct the exact clean-fixed production mask for the
                # route-only representation diagnostic.  The production
                # probe uses the same sample-keyed replay context before it
                # executes all 10 groups and 20 views.
                mask_replay = GradientReplay(args.seed)
                mask_replay.begin_batch(batch_names)
                mask_replay.set_context(step=-1, group=-1, view=-1)
                attacker._gradient_replay = mask_replay
                drop_mask, grid_size = attacker._compute_mainline_drop_mask(pixels, labels)
                attacker._gradient_replay = None
                image_mask = attacker._patch_drop_mask_to_image(
                    drop_mask, grid_size, pixels.size(-2), pixels.size(-1)
                ).to(device=device, dtype=pixels.dtype)
                with torch.no_grad():
                    clean_features = source_model.extract_patch_score_features(
                        normalize(source_model, pixels), score_layer=layer
                    )
                    masked_features = source_model.extract_patch_score_features(
                        normalize(source_model, pixels * (1.0 - image_mask)), score_layer=layer
                    )
                    clean_scores = F.cosine_similarity(
                        clean_features.local_tokens,
                        clean_features.global_token.expand_as(clean_features.local_tokens),
                        dim=2,
                    )
                    masked_scores = F.cosine_similarity(
                        masked_features.local_tokens,
                        masked_features.global_token.expand_as(masked_features.local_tokens),
                        dim=2,
                    )
                    global_cosine = F.cosine_similarity(
                        clean_features.global_token[:, 0], masked_features.global_token[:, 0], dim=1
                    )
                    local_cosines = F.cosine_similarity(
                        clean_features.local_tokens, masked_features.local_tokens, dim=2
                    )
                    kept = ~drop_mask
                    kept_cosine = (local_cosines * kept).sum(dim=1) / kept.sum(dim=1).clamp_min(1)
                    score_spearman = row_spearman(clean_scores, masked_scores)

                probe = attacker.probe_attack_gradients(
                    pixels,
                    labels,
                    replay=GradientReplay(args.seed),
                    sample_ids=batch_names,
                )
                raw = probe["raw_mean"]
                processed = probe["processed"]
                raw_values.append(raw.cpu())
                processed_values.append(processed.cpu())
                view_cosine, view_sign, effective_rank = view_metrics(
                    probe["view_gradients"], raw
                )
                clean_cosine = cosine_rows(raw, source_clean[start:end].to(device))
                for index in range(end - start):
                    route_rows.append(
                        {
                            "source_model": args.source,
                            "sample_index": start + index,
                            "image_name": batch_names[index],
                            "layer": layer,
                            "condition": condition,
                            "global_polarity": config.global_polarity,
                            "global_mode": clean_features.global_mode,
                            "grid_h": grid_size[0],
                            "grid_w": grid_size[1],
                            "drop_count": int(drop_mask[index].sum().cpu()),
                            "drop_ratio": float(drop_mask[index].float().mean().cpu()),
                            "global_cosine_clean_masked": float(global_cosine[index].cpu()),
                            "kept_token_cosine_clean_masked": float(kept_cosine[index].cpu()),
                            "score_map_spearman_clean_masked": float(score_spearman[index].cpu()),
                            "clean_route_gradient_cosine": float(clean_cosine[index].cpu()),
                            "view_cosine_to_raw_mean": float(view_cosine[index].cpu()),
                            "view_sign_agreement": float(view_sign[index].cpu()),
                            "view_effective_rank": float(effective_rank[index].cpu()),
                        }
                    )
                del probe, raw, processed, pixels, labels
            gradients_by_key[key] = {
                "raw": torch.cat(raw_values),
                "processed": torch.cat(processed_values),
            }

    cross_rows: list[dict[str, object]] = []
    step_size = args.epsilon / args.steps
    for target_name in targets:
        target_model = build_whitebox_model(1000, target_name, pretrained=True, device=device).eval()
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            target_gradient, clean_true, clean_loss, clean_pred = clean_gradient(
                target_model, pixels, labels
            )
            for layer in layers:
                for condition in conditions:
                    key = f"{layer}:{condition}"
                    raw = gradients_by_key[key]["raw"][start:end].to(device)
                    processed = gradients_by_key[key]["processed"][start:end].to(device)
                    adv = torch.clamp(pixels + step_size * processed.sign(), 0.0, 1.0)
                    with torch.no_grad():
                        adv_logits = target_model(normalize(target_model, adv))
                        adv_true = adv_logits.gather(1, labels[:, None]).squeeze(1)
                        adv_loss = F.cross_entropy(adv_logits, labels, reduction="none")
                    raw_cosine = cosine_rows(raw, target_gradient)
                    processed_cosine = cosine_rows(processed, target_gradient)
                    raw_sign = sign_agreement_rows(raw, target_gradient)
                    processed_sign = sign_agreement_rows(processed, target_gradient)
                    for index in range(end - start):
                        cross_rows.append(
                            {
                                "source_model": args.source,
                                "target_model": target_name,
                                "sample_index": start + index,
                                "image_name": names[start + index],
                                "layer": layer,
                                "condition": condition,
                                "target_clean_correct": bool(clean_pred[index].eq(labels[index]).cpu()),
                                "raw_target_gradient_cosine": float(raw_cosine[index].cpu()),
                                "processed_target_gradient_cosine": float(processed_cosine[index].cpu()),
                                "raw_target_sign_agreement": float(raw_sign[index].cpu()),
                                "processed_target_sign_agreement": float(processed_sign[index].cpu()),
                                "one_step_target_logit_drop": float((clean_true[index] - adv_true[index]).cpu()),
                                "one_step_target_loss_increase": float((adv_loss[index] - clean_loss[index]).cpu()),
                                "one_step_target_prediction_changed": bool(
                                    adv_logits.argmax(dim=1)[index].ne(clean_pred[index]).cpu()
                                ),
                            }
                        )
        del target_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = args.output_dir / args.source
    write_rows(output_dir / "route_metrics.csv", route_rows)
    write_rows(output_dir / "cross_model_metrics.csv", cross_rows)
    summary = {
        "config": {
            "source": args.source,
            "targets": targets,
            "layers": layers,
            "conditions": conditions,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "seed": args.seed,
            "global_polarity": config.global_polarity,
            "epsilon": args.epsilon,
            "steps": args.steps,
            "views": 20,
            "patch_mask_policy": "clean_fixed_per_attack",
            "patch_mask_reference": "clean_pixels",
            "token_score_cls_noise": True,
            "opponent_strength": 0.2,
            "gaussian_sigma": 4.0,
            "gaussian_alpha": 0.75,
        },
        "route_mean": {},
        "cross_model_mean": {},
    }
    for layer in layers:
        summary["route_mean"][layer] = {}
        summary["cross_model_mean"][layer] = {}
        for condition in conditions:
            route_selected = [
                row for row in route_rows
                if row["layer"] == layer and row["condition"] == condition
            ]
            cross_selected = [
                row for row in cross_rows
                if row["layer"] == layer and row["condition"] == condition
            ]
            summary["route_mean"][layer][condition] = {
                metric: sum(float(row[metric]) for row in route_selected) / len(route_selected)
                for metric in (
                    "global_cosine_clean_masked",
                    "kept_token_cosine_clean_masked",
                    "score_map_spearman_clean_masked",
                    "clean_route_gradient_cosine",
                    "view_effective_rank",
                )
            }
            summary["cross_model_mean"][layer][condition] = {
                metric: sum(float(row[metric]) for row in cross_selected) / len(cross_selected)
                for metric in (
                    "raw_target_gradient_cosine",
                    "processed_target_gradient_cosine",
                    "raw_target_sign_agreement",
                    "processed_target_sign_agreement",
                    "one_step_target_logit_drop",
                    "one_step_target_loss_increase",
                )
            }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
