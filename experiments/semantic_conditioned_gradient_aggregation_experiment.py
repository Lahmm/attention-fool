"""E8: use a noise-free semantic functional to rank opponent-view gradients.

This experiment does not add the semantic gradient to the attack gradient and
does not use it to gate opponent noise.  A single source model independently
computes (1) noise-free global-discrepancy gradients and (2) the usual twenty
opponent-noise CE gradients.  The former only assigns continuous weights to
the latter.  Target models are loaded afterwards and are used exclusively for
held-out diagnostics or adversarial-example evaluation.

Modes:
  probe: test whether semantic scores predict held-out target-gradient quality;
  attack: run matched iterative attacks with uniform and semantic aggregation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack import PatchScoreAttacker
from experiments.cross_arch_route_vulnerability_experiment import (
    extract_features,
    freeze_model,
    route_layers,
)
from experiments.patch_score_promotion_observation import bootstrap_ci
from experiments.patch_score_routing_gradient_experiment import (
    clean_gradient,
    cosine_rows,
    load_samples,
    normalize,
    row_spearman,
    sign_agreement_rows,
)
from experiments.semantic_equivalent_route_experiment import PHASES, phase_shift
from gradient_replay import GradientReplay
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
DEFAULT_OUTPUT_DIR = Path("outputs/research/semantic_conditioned_gradient_e8")
SEMANTIC_KINDS = ("phase", "flip", "consensus")
AGGREGATION_CONDITIONS = (
    "uniform",
    "phase_weighted",
    "flip_weighted",
    "consensus_weighted",
    "shuffled_semantic_weighted",
    "permuted_weight",
    "consensus_topk",
)
ATTACK_CONDITIONS = ("uniform", "consensus_weighted", "permuted_weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "attack"), required=True)
    parser.add_argument("--source", choices=WHITEBOX_MODEL_CHOICES, required=True)
    parser.add_argument("--targets", default="auto")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=756)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--noise-strength", type=float, default=0.2)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=0.75)
    parser.add_argument("--semantic-phase", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--weight-mixture", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--attack-conditions",
        default=",".join(ATTACK_CONDITIONS),
        help="comma-separated attack conditions",
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


def mean_rows(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


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


def _semantic_gradient(
    model,
    pixels: torch.Tensor,
    transform: str,
    semantic_phase: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a noise-free final-global discrepancy gradient and cosine."""
    probe = pixels.detach().requires_grad_(True)
    if transform == "phase":
        transformed = phase_shift(probe, *semantic_phase)
    elif transform == "flip":
        transformed = torch.flip(probe, dims=(-1,))
    else:
        raise ValueError(f"unknown transform: {transform}")
    layer = route_layers(model.model_name)[1]
    first = extract_features(model, probe, layer).global_token.flatten(1)
    second = extract_features(model, transformed, layer).global_token.flatten(1)
    cosine = F.cosine_similarity(first.detach(), second.detach(), dim=1)
    # Symmetric stop-gradient: both views affect the image Jacobian but neither
    # is allowed to move the other's semantic reference during differentiation.
    loss = 0.5 * (1.0 - F.cosine_similarity(first, second.detach(), dim=1))
    loss = loss + 0.5 * (1.0 - F.cosine_similarity(second, first.detach(), dim=1))
    gradient = torch.autograd.grad(loss.sum(), probe, retain_graph=False)[0]
    return gradient.detach(), cosine


def _semantic_gradients(model, pixels: torch.Tensor, args: argparse.Namespace):
    phase, phase_cosine = _semantic_gradient(
        model, pixels, "phase", tuple(args.semantic_phase)
    )
    flip, flip_cosine = _semantic_gradient(
        model, pixels, "flip", tuple(args.semantic_phase)
    )
    phase_unit = phase / phase.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    flip_unit = flip / flip.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
    consensus = phase_unit + flip_unit
    return {
        "phase": phase,
        "flip": flip,
        "consensus": consensus,
    }, {
        "phase": phase_cosine.detach(),
        "flip": flip_cosine.detach(),
        "phase_flip_gradient_cosine": cosine_rows(phase, flip).detach(),
    }


def _opponent_view_gradients(
    model,
    attacker: PatchScoreAttacker,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    args: argparse.Namespace,
    step_index: int = 0,
) -> torch.Tensor:
    """Compute the matched twenty source CE gradients without semantic gating."""
    replay = GradientReplay(args.seed)
    replay.begin_batch(sample_ids)
    attacker._gradient_replay = replay
    probe = pixels.detach().requires_grad_(True)
    gradients = []
    try:
        for group in range(args.groups):
            phase = PHASES[group % len(PHASES)]
            for view in range(2):
                replay.set_context(step=step_index, group=group, view=view)
                view_pixels = phase_shift(probe, *phase) if view else probe
                state = model.prepare_attack_feature_state(normalize(model, view_pixels))
                local = state.local_tokens + attacker._strict_opponent_feature_noise(state)
                logits = model.forward_from_attack_feature_state(state, local)
                loss = F.cross_entropy(logits, labels)
                gradients.append(torch.autograd.grad(loss, probe, retain_graph=False)[0])
        return torch.stack(gradients).detach()
    finally:
        attacker._gradient_replay = None


def _view_scores(view_gradients: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
    # [V,B,C,H,W] and [B,C,H,W] -> [B,V]
    views = view_gradients.flatten(2).transpose(0, 1)
    reference = semantic.flatten(1)[:, None]
    return F.cosine_similarity(views, reference, dim=2)


def _soft_weights(scores: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    centered = scores - scores.mean(dim=1, keepdim=True)
    standardized = centered / scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    semantic = torch.softmax(standardized / args.temperature, dim=1)
    uniform = torch.full_like(semantic, 1.0 / semantic.size(1))
    return (1.0 - args.weight_mixture) * uniform + args.weight_mixture * semantic


def _permutation_shift(sample_id: str, seed: int, count: int, step: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{step}:weight_control".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "little") % (count - 1)


def _permute_weights(
    weights: torch.Tensor,
    sample_ids: list[str],
    seed: int,
    step: int = 0,
) -> torch.Tensor:
    result = torch.empty_like(weights)
    for index, sample_id in enumerate(sample_ids):
        result[index] = weights[index].roll(
            _permutation_shift(sample_id, seed, weights.size(1), step)
        )
    return result


def _weighted_gradient(view_gradients: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (
        view_gradients
        * weights.transpose(0, 1)[:, :, None, None, None].to(view_gradients)
    ).sum(dim=0)


def _condition_payloads(
    view_gradients: torch.Tensor,
    semantic: dict[str, torch.Tensor],
    shuffled_consensus: torch.Tensor,
    sample_ids: list[str],
    attacker: PatchScoreAttacker,
    args: argparse.Namespace,
    step_index: int = 0,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    scores = {kind: _view_scores(view_gradients, semantic[kind]) for kind in SEMANTIC_KINDS}
    scores["shuffled_semantic"] = _view_scores(view_gradients, shuffled_consensus)
    weights = {kind: _soft_weights(value, args) for kind, value in scores.items()}
    weights["permuted"] = _permute_weights(
        weights["consensus"], sample_ids, args.seed, step_index
    )
    uniform = torch.full_like(weights["consensus"], 1.0 / view_gradients.size(0))
    top_indices = scores["consensus"].topk(args.topk, dim=1).indices
    top_weights = torch.zeros_like(uniform).scatter(1, top_indices, 1.0 / args.topk)
    condition_weights = {
        "uniform": uniform,
        "phase_weighted": weights["phase"],
        "flip_weighted": weights["flip"],
        "consensus_weighted": weights["consensus"],
        "shuffled_semantic_weighted": weights["shuffled_semantic"],
        "permuted_weight": weights["permuted"],
        "consensus_topk": top_weights,
    }
    payloads = {}
    for condition, current_weights in condition_weights.items():
        raw = _weighted_gradient(view_gradients, current_weights)
        payloads[condition] = {
            "raw": raw.detach(),
            "processed": attacker._apply_gaussian_residual(raw).detach(),
            "weights": current_weights.detach(),
        }
    return payloads, scores


def _targets(args: argparse.Namespace) -> list[str]:
    targets = (
        [name for name in WHITEBOX_MODEL_CHOICES if name != args.source]
        if args.targets == "auto"
        else [item.strip() for item in args.targets.split(",") if item.strip()]
    )
    if args.source in targets or set(targets) - set(WHITEBOX_MODEL_CHOICES):
        raise ValueError("targets must be registered models other than source")
    return targets


def _validate_args(args: argparse.Namespace) -> None:
    if args.samples <= 1 or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("samples > 1, batch-size > 0, and steps > 0 are required")
    if args.groups * 2 != 20:
        raise ValueError("E8 requires exactly twenty opponent views")
    if not 0.0 <= args.weight_mixture <= 1.0:
        raise ValueError("weight-mixture must be in [0,1]")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < args.topk <= args.groups * 2:
        raise ValueError("topk must be between 1 and the view count")


def _source_probe_payload(args: argparse.Namespace, names, pixels_cpu, labels_cpu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_whitebox_model(1000, args.source, pretrained=True, device=device).eval()
    freeze_model(model)
    attacker = make_attacker(model, args)
    views_parts = []
    semantic_parts = {kind: [] for kind in SEMANTIC_KINDS}
    semantic_rows = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        pixels = pixels_cpu[start:end].to(device)
        labels = labels_cpu[start:end].to(device)
        current_names = names[start:end]
        semantic, diagnostics = _semantic_gradients(model, pixels, args)
        views = _opponent_view_gradients(
            model, attacker, pixels, labels, current_names, args
        )
        views_parts.append(views.to(dtype=torch.float16).cpu())
        for kind in SEMANTIC_KINDS:
            semantic_parts[kind].append(semantic[kind].to(dtype=torch.float16).cpu())
        for local, sample_index in enumerate(range(start, end)):
            semantic_rows.append({
                "source_model": args.source,
                "sample_index": sample_index,
                "image_name": names[sample_index],
                "phase_global_cosine": float(diagnostics["phase"][local]),
                "flip_global_cosine": float(diagnostics["flip"][local]),
                "phase_flip_gradient_cosine": float(
                    diagnostics["phase_flip_gradient_cosine"][local]
                ),
            })
        if start == 0 or end == args.samples or (start // args.batch_size) % 8 == 0:
            print(f"[{args.source}] source probe {end}/{args.samples}", flush=True)

    all_views = torch.cat(views_parts, dim=1)
    all_semantic = {kind: torch.cat(parts) for kind, parts in semantic_parts.items()}
    aggregate_parts = {condition: {"raw": [], "processed": [], "weights": []}
                       for condition in AGGREGATION_CONDITIONS}
    score_parts = {kind: [] for kind in (*SEMANTIC_KINDS, "shuffled_semantic")}
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        views = all_views[:, start:end].to(device=device, dtype=torch.float32)
        semantic = {
            kind: value[start:end].to(device=device, dtype=torch.float32)
            for kind, value in all_semantic.items()
        }
        shuffled_indices = torch.tensor(
            [(index + 1) % args.samples for index in range(start, end)], dtype=torch.long
        )
        shuffled = all_semantic["consensus"][shuffled_indices].to(
            device=device, dtype=torch.float32
        )
        payloads, scores = _condition_payloads(
            views, semantic, shuffled, names[start:end], attacker, args
        )
        for condition, payload in payloads.items():
            for key, value in payload.items():
                aggregate_parts[condition][key].append(value.cpu())
        for kind, value in scores.items():
            score_parts[kind].append(value.cpu())
    aggregates = {
        condition: {key: torch.cat(parts) for key, parts in payload.items()}
        for condition, payload in aggregate_parts.items()
    }
    scores = {kind: torch.cat(parts) for kind, parts in score_parts.items()}
    del attacker, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return all_views, scores, aggregates, semantic_rows


def run_probe(args: argparse.Namespace) -> None:
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    targets = _targets(args)
    views, semantic_scores, aggregates, semantic_rows = _source_probe_payload(
        args, names, pixels_cpu, labels_cpu
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correlation_rows = []
    cross_rows = []
    step_size = args.epsilon / args.steps
    for target_name in targets:
        print(f"[{args.source}] target diagnostic {target_name}", flush=True)
        target = build_whitebox_model(1000, target_name, pretrained=True, device=device).eval()
        freeze_model(target)
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            target_gradient, clean_true, clean_loss, clean_pred = clean_gradient(
                target, pixels, labels
            )
            current_views = views[:, start:end].to(device=device, dtype=torch.float32)
            view_target_cosine = F.cosine_similarity(
                current_views.flatten(2).transpose(0, 1),
                target_gradient.flatten(1)[:, None],
                dim=2,
            )
            for semantic_kind, all_scores in semantic_scores.items():
                current_scores = all_scores[start:end].to(device)
                correlations = row_spearman(current_scores, view_target_cosine)
                top = current_scores.topk(args.topk, dim=1).indices
                top_quality = view_target_cosine.gather(1, top).mean(dim=1)
                uniform_quality = view_target_cosine.mean(dim=1)
                for local, sample_index in enumerate(range(start, end)):
                    correlation_rows.append({
                        "source_model": args.source,
                        "target_model": target_name,
                        "semantic_kind": semantic_kind,
                        "sample_index": sample_index,
                        "image_name": names[sample_index],
                        "view_quality_spearman": float(correlations[local]),
                        "semantic_topk_view_cosine": float(top_quality[local]),
                        "uniform_view_cosine": float(uniform_quality[local]),
                        "topk_minus_uniform_view_cosine": float(
                            top_quality[local] - uniform_quality[local]
                        ),
                    })
            for condition in AGGREGATION_CONDITIONS:
                raw = aggregates[condition]["raw"][start:end].to(device)
                processed = aggregates[condition]["processed"][start:end].to(device)
                adversarial = torch.clamp(pixels + step_size * processed.sign(), 0.0, 1.0)
                with torch.no_grad():
                    adv_logits = target(normalize(target, adversarial))
                    adv_true = adv_logits.gather(1, labels[:, None]).squeeze(1)
                    adv_loss = F.cross_entropy(adv_logits, labels, reduction="none")
                raw_cosine = cosine_rows(raw, target_gradient)
                processed_cosine = cosine_rows(processed, target_gradient)
                raw_sign = sign_agreement_rows(raw, target_gradient)
                processed_sign = sign_agreement_rows(processed, target_gradient)
                entropy = -(
                    aggregates[condition]["weights"][start:end]
                    * aggregates[condition]["weights"][start:end].clamp_min(1e-12).log()
                ).sum(dim=1)
                for local, sample_index in enumerate(range(start, end)):
                    cross_rows.append({
                        "source_model": args.source,
                        "target_model": target_name,
                        "condition": condition,
                        "sample_index": sample_index,
                        "image_name": names[sample_index],
                        "target_clean_correct": bool(
                            clean_pred[local].eq(labels[local]).cpu()
                        ),
                        "weight_entropy": float(entropy[local]),
                        "raw_target_gradient_cosine": float(raw_cosine[local]),
                        "processed_target_gradient_cosine": float(processed_cosine[local]),
                        "raw_target_sign_agreement": float(raw_sign[local]),
                        "processed_target_sign_agreement": float(processed_sign[local]),
                        "one_step_target_logit_drop": float(
                            clean_true[local] - adv_true[local]
                        ),
                        "one_step_target_loss_increase": float(
                            adv_loss[local] - clean_loss[local]
                        ),
                    })
        del target
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
    condition_mean = {}
    for condition in AGGREGATION_CONDITIONS:
        selected = [row for row in cross_rows if row["condition"] == condition]
        condition_mean[condition] = {metric: mean_rows(selected, metric) for metric in metrics}
        condition_mean[condition]["weight_entropy"] = mean_rows(selected, "weight_entropy")
    correlation_mean = {}
    for kind in (*SEMANTIC_KINDS, "shuffled_semantic"):
        selected = [row for row in correlation_rows if row["semantic_kind"] == kind]
        correlation_mean[kind] = {
            key: mean_rows(selected, key)
            for key in (
                "view_quality_spearman",
                "topk_minus_uniform_view_cosine",
            )
        }

    comparisons = {
        "consensus_weighted_vs_uniform": ("consensus_weighted", "uniform"),
        "consensus_weighted_vs_shuffled": (
            "consensus_weighted", "shuffled_semantic_weighted"
        ),
        "consensus_weighted_vs_permuted": (
            "consensus_weighted", "permuted_weight"
        ),
    }
    paired_deltas = {}
    for name, (condition, baseline) in comparisons.items():
        paired_deltas[name] = {}
        for metric_index, metric in enumerate(metrics):
            current = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in cross_rows if row["condition"] == condition
            }
            base = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in cross_rows if row["condition"] == baseline
            }
            delta = torch.tensor([current[key] - base[key] for key in current])
            paired_deltas[name][metric] = {
                "mean": float(delta.mean()),
                "ci95": bootstrap_ci(
                    delta, seed=args.seed + 1000 + metric_index
                ),
                "positive_fraction": float(delta.gt(0).float().mean()),
                "comparisons": int(delta.numel()),
            }

    prediction_deltas = {}
    for control in ("shuffled_semantic",):
        prediction_deltas[f"consensus_vs_{control}"] = {}
        for metric_index, metric in enumerate(
            ("view_quality_spearman", "topk_minus_uniform_view_cosine")
        ):
            current = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in correlation_rows if row["semantic_kind"] == "consensus"
            }
            base = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in correlation_rows if row["semantic_kind"] == control
            }
            delta = torch.tensor([current[key] - base[key] for key in current])
            prediction_deltas[f"consensus_vs_{control}"][metric] = {
                "mean": float(delta.mean()),
                "ci95": bootstrap_ci(
                    delta, seed=args.seed + 3000 + metric_index
                ),
                "positive_fraction": float(delta.gt(0).float().mean()),
            }

    summary = {
        "protocol": {
            "experiment": "E8_semantic_conditioned_opponent_gradient_probe",
            "source": args.source,
            "targets": targets,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "views": args.groups * 2,
            "semantic_transforms": ["phase", "horizontal_flip"],
            "semantic_phase": list(args.semantic_phase),
            "semantic_branch_noise_free": True,
            "semantic_gradient_added_to_attack": False,
            "opponent_noise_spatially_gated": False,
            "whitebox_ensemble": False,
            "target_used_for_generation": False,
            "weight_mixture": args.weight_mixture,
            "temperature": args.temperature,
            "topk": args.topk,
        },
        "semantic_diagnostics_mean": {
            key: mean_rows(semantic_rows, key)
            for key in (
                "phase_global_cosine",
                "flip_global_cosine",
                "phase_flip_gradient_cosine",
            )
        },
        "correlation_mean": correlation_mean,
        "condition_mean": condition_mean,
        "prediction_deltas": prediction_deltas,
        "paired_deltas": paired_deltas,
    }
    output = args.output_dir / "probe" / args.source
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "semantic_diagnostics.csv", semantic_rows)
    write_csv(output / "view_prediction_metrics.csv", correlation_rows)
    write_csv(output / "cross_model_metrics.csv", cross_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _fixed_clean_consensus(model, clean: torch.Tensor, args: argparse.Namespace):
    semantic, _ = _semantic_gradients(model, clean, args)
    return semantic["consensus"].detach()


def _attack_one_condition(
    model,
    attacker: PatchScoreAttacker,
    clean: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    args: argparse.Namespace,
    condition: str,
) -> torch.Tensor:
    adversarial = clean.detach().clone()
    momentum = torch.zeros_like(adversarial)
    fixed_semantic = _fixed_clean_consensus(model, clean, args)
    for step_index in range(args.steps):
        views = _opponent_view_gradients(
            model, attacker, adversarial, labels, sample_ids, args, step_index
        )
        if condition == "uniform":
            weights = torch.full(
                (clean.size(0), views.size(0)),
                1.0 / views.size(0),
                device=views.device,
            )
        else:
            weights = _soft_weights(_view_scores(views, fixed_semantic), args)
            if condition == "permuted_weight":
                weights = _permute_weights(
                    weights, sample_ids, args.seed, step_index
                )
        raw = _weighted_gradient(views, weights)
        processed = attacker._apply_gaussian_residual(raw).detach()
        momentum = momentum + processed
        with torch.no_grad():
            adversarial = adversarial + (args.epsilon / args.steps) * momentum.sign()
            delta = torch.clamp(adversarial - clean, -args.epsilon, args.epsilon)
            adversarial = torch.clamp(clean + delta, 0.0, 1.0).detach()
    return adversarial.cpu()


def run_attack(args: argparse.Namespace) -> None:
    conditions = [x.strip() for x in args.attack_conditions.split(",") if x.strip()]
    if not conditions or set(conditions) - set(ATTACK_CONDITIONS):
        raise ValueError(f"invalid attack conditions: {conditions}")
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    targets = _targets(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = build_whitebox_model(1000, args.source, pretrained=True, device=device).eval()
    freeze_model(source)
    attacker = make_attacker(source, args)
    adversarial_by_condition = {}
    for condition in conditions:
        generated = []
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            generated.append(_attack_one_condition(
                source,
                attacker,
                pixels_cpu[start:end].to(device),
                labels_cpu[start:end].to(device),
                names[start:end],
                args,
                condition,
            ))
            if start == 0 or end == args.samples or (start // args.batch_size) % 16 == 0:
                print(f"[{args.source}] {condition} attack {end}/{args.samples}", flush=True)
        adversarial_by_condition[condition] = torch.cat(generated)
    del attacker, source
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = []
    for target_name in targets:
        target = build_whitebox_model(1000, target_name, pretrained=True, device=device).eval()
        freeze_model(target)
        for start in range(0, args.samples, max(8, args.batch_size)):
            end = min(args.samples, start + max(8, args.batch_size))
            clean = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            with torch.no_grad():
                clean_pred = target(normalize(target, clean)).argmax(dim=1)
                for condition in conditions:
                    adversarial = adversarial_by_condition[condition][start:end].to(device)
                    adv_pred = target(normalize(target, adversarial)).argmax(dim=1)
                    for local, sample_index in enumerate(range(start, end)):
                        correct = bool(clean_pred[local].eq(labels[local]).cpu())
                        success = bool(adv_pred[local].ne(labels[local]).cpu())
                        rows.append({
                            "source_model": args.source,
                            "target_model": target_name,
                            "condition": condition,
                            "sample_index": sample_index,
                            "image_name": names[sample_index],
                            "clean_correct": correct,
                            "all_success": success,
                            "clean_correct_success": correct and success,
                        })
        del target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    target_mean = {}
    for target_name in targets:
        target_mean[target_name] = {}
        for condition in conditions:
            selected = [row for row in rows if row["target_model"] == target_name
                        and row["condition"] == condition]
            target_mean[target_name][condition] = {
                "all_asr": sum(bool(row["all_success"]) for row in selected) / len(selected),
                "clean_accuracy": sum(bool(row["clean_correct"]) for row in selected) / len(selected),
                "clean_correct_conditional_asr": (
                    sum(bool(row["clean_correct_success"]) for row in selected)
                    / max(1, sum(bool(row["clean_correct"]) for row in selected))
                ),
            }
    paired = {}
    for condition in conditions:
        if condition == "uniform":
            continue
        deltas = []
        for target_name in targets:
            current = {int(row["sample_index"]): float(bool(row["all_success"]))
                       for row in rows if row["target_model"] == target_name
                       and row["condition"] == condition}
            baseline = {int(row["sample_index"]): float(bool(row["all_success"]))
                        for row in rows if row["target_model"] == target_name
                        and row["condition"] == "uniform"}
            deltas.extend(current[index] - baseline[index] for index in current)
        delta = torch.tensor(deltas)
        paired[condition] = {
            "pooled_all_asr_delta": float(delta.mean()),
            "ci95": bootstrap_ci(delta, seed=args.seed + 7000),
            "positive_fraction": float(delta.gt(0).float().mean()),
            "comparisons": int(delta.numel()),
        }
    summary = {
        "protocol": {
            "experiment": "E8_semantic_conditioned_opponent_gradient_attack",
            "source": args.source,
            "targets": targets,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "conditions": conditions,
            "semantic_reference": "fixed_clean_noise_free_phase_flip_consensus_gradient",
            "semantic_gradient_added_to_attack": False,
            "whitebox_ensemble": False,
            "steps": args.steps,
            "epsilon": args.epsilon,
            "weight_mixture": args.weight_mixture,
            "temperature": args.temperature,
        },
        "target_mean": target_mean,
        "paired_asr_deltas": paired,
    }
    output = args.output_dir / "attack" / args.source
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "attack_metrics.csv", rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.mode == "probe":
        run_probe(args)
    else:
        run_attack(args)


if __name__ == "__main__":
    main()
