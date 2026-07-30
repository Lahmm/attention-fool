"""E5: semantic-equivalent route coverage and single-source gradient aggregation.

The experiment deliberately never ensembles white-box gradients.  In
``forward`` mode each model is measured independently.  In ``gradient`` mode
one source model generates all gradients; the other registered models are only
used after generation as held-out targets.
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
from experiments.patch_score_promotion_observation import bootstrap_ci, common_map
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
PHASES = ((4, 4), (8, 8), (12, 12))
FORWARD_CONDITIONS = ("opponent", "rgb_gaussian", "feature_gaussian", "phase_only")
GRADIENT_CONDITIONS = (
    "uniform_opponent",
    "global_preserve_opponent",
    "route_balance_opponent",
    "semantic_equivalent_opponent",
    "shuffled_semantic_opponent",
    "semantic_equivalent_feature_gaussian",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("forward", "gradient"), required=True)
    parser.add_argument("--source", choices=WHITEBOX_MODEL_CHOICES)
    parser.add_argument("--targets", default="auto")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=628)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--noise-strength", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=0.75)
    parser.add_argument("--preserve-global-scale", type=float, default=0.02)
    parser.add_argument("--preserve-js-scale", type=float, default=0.01)
    parser.add_argument("--route-density-temperature", type=float, default=0.25)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/semantic_equivalent_route"),
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    a, b = log_a.exp(), log_b.exp()
    midpoint = 0.5 * (a + b)
    return 0.5 * (
        F.kl_div(log_a, midpoint, reduction="none").sum(dim=1)
        + F.kl_div(log_b, midpoint, reduction="none").sum(dim=1)
    )


def feature_score(local: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(
        local, global_token.expand_as(local), dim=-1
    )


def phase_shift(tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return tensor
    padded = F.pad(
        tensor,
        (max(0, dx), max(0, -dx), max(0, dy), max(0, -dy)),
        mode="reflect",
    )
    start_y, start_x = max(0, -dy), max(0, -dx)
    return padded[..., start_y : start_y + tensor.size(-2), start_x : start_x + tensor.size(-1)]


def align_score_map(
    score: torch.Tensor,
    grid: tuple[int, int],
    image_size: tuple[int, int],
    phase: tuple[int, int],
    common_grid: int = 7,
) -> torch.Tensor:
    maps = score.reshape(score.size(0), 1, *grid)
    pixels = F.interpolate(maps, size=image_size, mode="bilinear", align_corners=False)
    if phase != (0, 0):
        pixels = phase_shift(pixels, -phase[0], -phase[1])
    aligned = F.interpolate(pixels, size=(common_grid, common_grid), mode="area")
    return aligned.flatten(1)


def final_from_local(model_name: str, model, state, local_tokens: torch.Tensor):
    """Run the model tail and return final local/global features plus logits."""
    base = model.model
    if model_name == "vit_base_patch16_224":
        tokens = torch.cat((state.context["prefix_tokens"], local_tokens), dim=1)
        tokens = base.blocks(tokens)
        logits = base.forward_head(base.norm(tokens))
        local, global_token = tokens[:, 1:], tokens[:, :1]
        grid = state.grid_size
    elif model_name == "cait_s24_224":
        local, cls_token = model._run_encoder(local_tokens)
        logits = base.forward_head(base.norm(torch.cat((cls_token, local), dim=1)))
        global_token = cls_token
        grid = state.grid_size
    elif model_name == "pit_b_224":
        spatial, cls_token = model._run_transformers(state, local_tokens)
        logits = base.forward_head(base.norm(cls_token))
        local = spatial.flatten(2).transpose(1, 2)
        global_token = cls_token[:, :1]
        grid = (int(spatial.size(-2)), int(spatial.size(-1)))
    elif model_name == "visformer_small":
        spatial = model._run_stages(state, local_tokens)
        logits = base.forward_head(base.norm(spatial))
        local = spatial.flatten(2).transpose(1, 2)
        global_token = local.mean(dim=1, keepdim=True)
        grid = (int(spatial.size(-2)), int(spatial.size(-1)))
    else:
        raise ValueError(f"unsupported model {model_name}")
    return local, global_token, logits, grid


def make_attacker(model, args: argparse.Namespace) -> PatchScoreAttacker:
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


def projected_rgb_gaussian(attacker, state, event: str) -> torch.Tensor:
    batch, count, dimension = state.local_tokens.shape
    kh, kw = state.projection_kernel
    coefficients = attacker._randn_like(
        torch.empty(
            batch,
            count,
            3,
            kh,
            kw,
            device=state.local_tokens.device,
            dtype=state.local_tokens.dtype,
        ),
        event,
    )
    projection = state.rgb_projection_weight.detach().to(state.local_tokens)
    projection = projection.reshape(dimension, -1)
    raw = coefficients.flatten(2).matmul(projection.t())
    return attacker._match_feature_noise_rms(state.local_tokens, raw, event)


def noise_for_kind(attacker, state, kind: str) -> torch.Tensor:
    if kind == "opponent":
        return attacker._strict_opponent_feature_noise(state)
    if kind == "feature_gaussian":
        raw = attacker._randn_like(state.local_tokens, "e5_feature_gaussian")
        return attacker._match_feature_noise_rms(state.local_tokens, raw, kind)
    if kind == "rgb_gaussian":
        return projected_rgb_gaussian(attacker, state, "e5_rgb_gaussian")
    if kind == "none":
        return torch.zeros_like(state.local_tokens)
    raise ValueError(f"unknown noise kind {kind}")


def base_and_view_metadata(
    model_name: str,
    model,
    attacker: PatchScoreAttacker,
    pixels: torch.Tensor,
    noise_kind: str,
    sample_ids: list[str],
    seed: int,
    groups: int,
):
    """Run no-grad views and return route/global metadata plus replay context."""
    replay = GradientReplay(seed)
    replay.begin_batch(sample_ids)
    attacker._gradient_replay = replay
    with torch.no_grad():
        base_state = model.prepare_attack_feature_state(normalize(model, pixels))
        base_local, base_global, base_logits, base_grid = final_from_local(
            model_name, model, base_state, base_state.local_tokens
        )
        base_score = feature_score(base_local, base_global)
        base_common = common_map(base_score, base_grid, 7)
        view_scores, view_globals, view_logits = [], [], []
        view_phases = []
        for group in range(groups):
            phase = PHASES[group % len(PHASES)]
            for view in range(2):
                replay.set_context(step=0, group=group, view=view)
                shifted_pixels = (
                    phase_shift(pixels, *phase) if view == 1 else pixels
                )
                state = model.prepare_attack_feature_state(normalize(model, shifted_pixels))
                local = state.local_tokens + noise_for_kind(attacker, state, noise_kind)
                local_features, global_token, logits, grid = final_from_local(
                    model_name, model, state, local
                )
                score = feature_score(local_features, global_token)
                common = align_score_map(
                    score,
                    grid,
                    tuple(pixels.shape[-2:]),
                    phase if view == 1 else (0, 0),
                )
                view_scores.append(common)
                view_globals.append(global_token.flatten(1))
                view_logits.append(logits)
                view_phases.append((group, view, phase if view == 1 else (0, 0)))
        view_scores = torch.stack(view_scores)
        view_globals = torch.stack(view_globals)
        view_logits = torch.stack(view_logits)
        base_global_flat = base_global.flatten(1)
        global_cos = F.cosine_similarity(
            view_globals, base_global_flat.unsqueeze(0).expand_as(view_globals), dim=-1
        )
        view_logits_flat = view_logits.reshape(-1, view_logits.size(-1))
        base_logits_expand = base_logits.unsqueeze(0).expand(
            view_logits.size(0), -1, -1
        ).reshape(-1, base_logits.size(-1))
        js = js_divergence(view_logits_flat, base_logits_expand).reshape(
            view_logits.size(0), view_logits.size(1)
        )
        route_spearman = torch.stack(
            [row_spearman(view_scores[v], base_common) for v in range(view_scores.size(0))]
        )
        route_pair = torch.stack(
            [
                row_spearman(view_scores[v], view_scores[u])
                for v in range(view_scores.size(0))
                for u in range(view_scores.size(0))
            ]
        ).reshape(view_scores.size(0), view_scores.size(0), view_scores.size(1))
        route_distance = 1.0 - route_pair
        return {
            "base_global": base_global.detach(),
            "base_logits": base_logits.detach(),
            "base_common": base_common.detach(),
            "view_scores": view_scores.detach(),
            "global_cos": global_cos.detach(),
            "js": js.detach(),
            "route_spearman": route_spearman.detach(),
            "route_distance": route_distance.detach(),
            "phases": view_phases,
        }


def route_weights(metadata, condition: str, args: argparse.Namespace) -> torch.Tensor:
    global_drift = (1.0 - metadata["global_cos"]).clamp_min(0.0)
    preserve = torch.exp(
        -global_drift / args.preserve_global_scale
        -metadata["js"] / args.preserve_js_scale
    )
    density = torch.exp(
        -metadata["route_distance"] / args.route_density_temperature
    ).sum(dim=1)
    balance = density.reciprocal()
    if condition in {"uniform_opponent"}:
        weights = torch.ones_like(preserve)
    elif condition == "global_preserve_opponent":
        weights = preserve
    elif condition == "route_balance_opponent":
        weights = balance
    elif condition == "semantic_equivalent_opponent":
        weights = preserve * balance
    elif condition == "shuffled_semantic_opponent":
        shuffled = []
        generator = torch.Generator().manual_seed(20260730)
        for sample in metadata["view_scores"].transpose(0, 1):
            shuffled.append(
                torch.stack(
                    [sample[v][torch.randperm(sample.size(1), generator=generator)] for v in range(sample.size(0))]
                )
            )
        shuffled_scores = torch.stack(shuffled).transpose(0, 1)
        base = metadata["base_common"]
        shuffled_route = torch.stack(
            [row_spearman(shuffled_scores[v], base) for v in range(shuffled_scores.size(0))]
        )
        shuffled_pair = torch.stack(
            [
                row_spearman(shuffled_scores[v], shuffled_scores[u])
                for v in range(shuffled_scores.size(0))
                for u in range(shuffled_scores.size(0))
            ]
        ).reshape(shuffled_scores.size(0), shuffled_scores.size(0), shuffled_scores.size(1))
        shuffled_density = torch.exp(
            -(1.0 - shuffled_pair) / args.route_density_temperature
        ).sum(dim=1)
        weights = preserve * shuffled_density.reciprocal()
    elif condition == "semantic_equivalent_feature_gaussian":
        weights = preserve * balance
    else:
        raise ValueError(f"unknown gradient condition {condition}")
    return weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)


def weighted_gradient_probe(
    model_name: str,
    model,
    attacker: PatchScoreAttacker,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    metadata,
    condition: str,
    sample_ids: list[str],
    seed: int,
    groups: int,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    noise_kind = (
        "feature_gaussian"
        if condition == "semantic_equivalent_feature_gaussian"
        else "opponent"
    )
    weights = route_weights(metadata, condition, args)
    replay = GradientReplay(seed)
    replay.begin_batch(sample_ids)
    probe = pixels.detach().requires_grad_(True)
    attacker._gradient_replay = replay
    gradients = []
    try:
        for group in range(groups):
            phase = PHASES[group % len(PHASES)]
            for view in range(2):
                index = 2 * group + view
                replay.set_context(step=0, group=group, view=view)
                view_pixels = phase_shift(probe, *phase) if view == 1 else probe
                state = model.prepare_attack_feature_state(normalize(model, view_pixels))
                local = state.local_tokens + noise_for_kind(attacker, state, noise_kind)
                _, _, logits, _ = final_from_local(model_name, model, state, local)
                gradients.append(torch.autograd.grad(
                    F.cross_entropy(logits, labels), probe, retain_graph=False
                )[0])
        view_gradients = torch.stack(gradients)
        raw = (view_gradients * weights.to(view_gradients.device)[:, :, None, None, None]).sum(dim=0)
        processed = attacker._apply_gaussian_residual(raw)
        return {
            "view_gradients": view_gradients.detach(),
            "raw_mean": raw.detach(),
            "processed": processed.detach(),
            "weights": weights.detach(),
        }
    finally:
        attacker._gradient_replay = None


def run_forward(args: argparse.Namespace) -> None:
    if args.source is not None:
        raise ValueError("forward mode does not accept --source")
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for model_name in WHITEBOX_MODEL_CHOICES:
        model = build_whitebox_model(1000, model_name, pretrained=True, device=device).eval()
        attacker = make_attacker(model, args)
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            for condition in FORWARD_CONDITIONS:
                noise_kind = "none" if condition == "phase_only" else condition
                metadata = base_and_view_metadata(
                    model_name,
                    model,
                    attacker,
                    pixels,
                    noise_kind,
                    names[start:end],
                    args.seed,
                    args.groups,
                )
                preserved = (
                    (metadata["global_cos"] >= 0.98)
                    & (metadata["js"] <= 0.02)
                )
                preserve_score = torch.exp(
                    -(1.0 - metadata["global_cos"]).clamp_min(0.0)
                    / args.preserve_global_scale
                    - metadata["js"] / args.preserve_js_scale
                )
                top_count = max(1, preserve_score.size(0) // 4)
                for index in range(end - start):
                    keep = preserved[:, index]
                    top = preserve_score[:, index].topk(top_count).indices
                    route_dist = 1.0 - metadata["route_spearman"][:, index]
                    pair = metadata["route_distance"][:, :, index]
                    keep_pair = torch.zeros_like(keep)
                    keep_pair[top] = True
                    pair_mask = keep_pair[:, None] & keep_pair[None, :]
                    pair_values = pair[pair_mask]
                    rows.append(
                        {
                            "model": model_name,
                            "condition": condition,
                            "sample_index": start + index,
                            "image_name": names[start + index],
                            "preserved_view_fraction": float(keep.float().mean()),
                            "preserved_route_distance": float(route_dist[keep].mean()) if keep.any() else 0.0,
                            "preserved_pair_distance": float(pair_values.mean()) if pair_values.numel() else 0.0,
                            "top_preserved_global_cosine": float(metadata["global_cos"][top, index].mean()),
                            "top_preserved_js": float(metadata["js"][top, index].mean()),
                            "top_route_distance": float(route_dist[top].mean()),
                            "top_pair_distance": float(pair_values.mean()) if pair_values.numel() else 0.0,
                            "all_view_global_cosine": float(metadata["global_cos"][:, index].mean()),
                            "all_view_js": float(metadata["js"][:, index].mean()),
                            "all_view_route_distance": float(route_dist.mean()),
                            "route_view_count": int(keep.sum()),
                        }
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "forward_metrics.csv", rows)
    summary = {
        "protocol": {
            "mode": "forward",
            "models": list(WHITEBOX_MODEL_CHOICES),
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "views": args.groups * 2,
            "common_grid": [7, 7],
            "preserved_rule": "global cosine >= 0.98 and JS <= 0.02",
            "no_whitebox_ensemble": True,
            "noise_strength": args.noise_strength,
        },
        "mean": {},
    }
    for model_name in WHITEBOX_MODEL_CHOICES:
        summary["mean"][model_name] = {}
        for condition in FORWARD_CONDITIONS:
            selected = [r for r in rows if r["model"] == model_name and r["condition"] == condition]
            summary["mean"][model_name][condition] = {
                key: mean(selected, key)
                for key in (
                    "preserved_view_fraction",
                    "preserved_route_distance",
                    "preserved_pair_distance",
                    "top_preserved_global_cosine",
                    "top_preserved_js",
                    "top_route_distance",
                    "top_pair_distance",
                    "all_view_global_cosine",
                    "all_view_js",
                    "all_view_route_distance",
                )
            }
    (args.output_dir / "forward_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_gradient(args: argparse.Namespace) -> None:
    if args.source is None:
        raise ValueError("gradient mode requires --source")
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    targets = (
        [name for name in WHITEBOX_MODEL_CHOICES if name != args.source]
        if args.targets == "auto"
        else [item.strip() for item in args.targets.split(",") if item.strip()]
    )
    if args.source in targets or set(targets) - set(WHITEBOX_MODEL_CHOICES):
        raise ValueError("targets must be registered models other than source")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_model = build_whitebox_model(1000, args.source, pretrained=True, device=device).eval()
    attacker = make_attacker(source_model, args)
    clean_parts = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        clean, _, _, _ = clean_gradient(
            source_model,
            pixels_cpu[start:end].to(device),
            labels_cpu[start:end].to(device),
        )
        clean_parts.append(clean.cpu())
    source_clean = torch.cat(clean_parts)
    route_rows, gradients = [], {}
    for condition in GRADIENT_CONDITIONS:
        raw_parts, processed_parts = [], []
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            metadata_kind = "feature_gaussian" if condition == "semantic_equivalent_feature_gaussian" else "opponent"
            metadata = base_and_view_metadata(
                args.source,
                source_model,
                attacker,
                pixels,
                metadata_kind,
                names[start:end],
                args.seed,
                args.groups,
            )
            result = weighted_gradient_probe(
                args.source,
                source_model,
                attacker,
                pixels,
                labels,
                metadata,
                condition,
                names[start:end],
                args.seed,
                args.groups,
                args,
            )
            raw, processed = result["raw_mean"], result["processed"]
            raw_parts.append(raw.cpu())
            processed_parts.append(processed.cpu())
            view_cosine, view_sign, effective_rank = view_metrics(result["view_gradients"], raw)
            clean_cosine = cosine_rows(raw, source_clean[start:end].to(device))
            for local_index, sample_index in enumerate(range(start, end)):
                route_rows.append(
                    {
                        "source_model": args.source,
                        "condition": condition,
                        "sample_index": sample_index,
                        "image_name": names[sample_index],
                        "preserved_view_fraction": float(((metadata["global_cos"][:, local_index] >= 0.98) & (metadata["js"][:, local_index] <= 0.02)).float().mean()),
                        "mean_route_distance": float((1.0 - metadata["route_spearman"][:, local_index]).mean()),
                        "mean_global_cosine": float(metadata["global_cos"][:, local_index].mean()),
                        "mean_view_js": float(metadata["js"][:, local_index].mean()),
                        "weight_entropy": float((-(result["weights"][:, local_index] * result["weights"][:, local_index].clamp_min(1e-12).log()).sum()).cpu()),
                        "clean_route_gradient_cosine": float(clean_cosine[local_index]),
                        "view_cosine_to_raw_mean": float(view_cosine[local_index]),
                        "view_sign_agreement": float(view_sign[local_index]),
                        "view_effective_rank": float(effective_rank[local_index]),
                    }
                )
        gradients[condition] = {"raw": torch.cat(raw_parts), "processed": torch.cat(processed_parts)}

    cross_rows = []
    step_size = args.epsilon / args.steps
    for target_name in targets:
        target_model = build_whitebox_model(1000, target_name, pretrained=True, device=device).eval()
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            target_grad, clean_true, clean_loss, clean_pred = clean_gradient(target_model, pixels, labels)
            for condition in GRADIENT_CONDITIONS:
                raw = gradients[condition]["raw"][start:end].to(device)
                processed = gradients[condition]["processed"][start:end].to(device)
                adversarial = torch.clamp(pixels + step_size * processed.sign(), 0.0, 1.0)
                with torch.no_grad():
                    adv_logits = target_model(normalize(target_model, adversarial))
                    adv_true = adv_logits.gather(1, labels[:, None]).squeeze(1)
                    adv_loss = F.cross_entropy(adv_logits, labels, reduction="none")
                raw_cosine = cosine_rows(raw, target_grad)
                processed_cosine = cosine_rows(processed, target_grad)
                raw_sign = sign_agreement_rows(raw, target_grad)
                processed_sign = sign_agreement_rows(processed, target_grad)
                for local_index, sample_index in enumerate(range(start, end)):
                    cross_rows.append(
                        {
                            "source_model": args.source,
                            "target_model": target_name,
                            "condition": condition,
                            "sample_index": sample_index,
                            "image_name": names[sample_index],
                            "target_clean_correct": bool(clean_pred[local_index].eq(labels[local_index]).cpu()),
                            "raw_target_gradient_cosine": float(raw_cosine[local_index]),
                            "processed_target_gradient_cosine": float(processed_cosine[local_index]),
                            "raw_target_sign_agreement": float(raw_sign[local_index]),
                            "processed_target_sign_agreement": float(processed_sign[local_index]),
                            "one_step_target_logit_drop": float(clean_true[local_index] - adv_true[local_index]),
                            "one_step_target_loss_increase": float(adv_loss[local_index] - clean_loss[local_index]),
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
    summary = {
        "protocol": {
            "mode": "gradient",
            "source": args.source,
            "targets": targets,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "views": args.groups * 2,
            "mask_policy": "no patch drop; semantic score only reweights source views",
            "whitebox_ensemble": False,
            "noise": "opponent projected through initial RGB projection",
        },
        "route_mean": {},
        "cross_model_mean": {},
        "paired_deltas": {},
    }
    for condition in GRADIENT_CONDITIONS:
        selected_route = [r for r in route_rows if r["condition"] == condition]
        selected_cross = [r for r in cross_rows if r["condition"] == condition]
        summary["route_mean"][condition] = {
            key: mean(selected_route, key)
            for key in (
                "preserved_view_fraction",
                "mean_route_distance",
                "mean_global_cosine",
                "mean_view_js",
                "weight_entropy",
                "clean_route_gradient_cosine",
                "view_cosine_to_raw_mean",
                "view_sign_agreement",
                "view_effective_rank",
            )
        }
        summary["cross_model_mean"][condition] = {
            key: mean(selected_cross, key) for key in metrics
        }
    baselines = {
        "uniform_opponent": "uniform_opponent",
        "global_preserve_opponent": "uniform_opponent",
        "route_balance_opponent": "uniform_opponent",
        "semantic_equivalent_opponent": "uniform_opponent",
        "shuffled_semantic_opponent": "semantic_equivalent_opponent",
        "semantic_equivalent_feature_gaussian": "semantic_equivalent_opponent",
    }
    for condition, baseline in baselines.items():
        if condition == baseline:
            continue
        summary["paired_deltas"][condition] = {}
        for metric_index, metric in enumerate(metrics):
            values, base_values = {}, {}
            for row in cross_rows:
                key = (row["target_model"], int(row["sample_index"]))
                if row["condition"] == condition:
                    values[key] = float(row[metric])
                if row["condition"] == baseline:
                    base_values[key] = float(row[metric])
            delta = torch.tensor([values[k] - base_values[k] for k in values if k in base_values])
            summary["paired_deltas"][condition][metric] = {
                "mean": float(delta.mean()),
                "ci95": bootstrap_ci(delta, seed=args.seed + 7000 + metric_index),
                "positive_fraction": float(delta.gt(0).float().mean()),
                "comparisons": int(delta.numel()),
            }
    output_dir = args.output_dir / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "gradient_route_metrics.csv", route_rows)
    write_csv(output_dir / "cross_model_metrics.csv", cross_rows)
    (output_dir / "gradient_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.samples <= 1 or args.batch_size <= 0 or args.groups * 2 != 20:
        raise ValueError("samples > 1, positive batch size, and groups=10 are required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode == "forward":
        run_forward(args)
    else:
        run_gradient(args)


if __name__ == "__main__":
    main()
