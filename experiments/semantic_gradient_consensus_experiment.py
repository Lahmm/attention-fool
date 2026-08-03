"""E6: decoupled semantic-gradient and opponent-noise validation.

The semantic branch is deliberately noise-free.  It computes a gradient that
maximizes disagreement between the source model's global representations on
an image and a deterministic phase-equivalent view.  The opponent branch is
computed independently from the usual 20 RGB opponent-noise views.  The two
branches meet only after their image-space gradients have been computed.

No target model participates in gradient generation.  Target models are used
only after generation for held-out gradient and one-step response metrics.
No patch mask or patch drop is used in this experiment; this is intentional so
that semantic routing and noise are not coupled through a mask.
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
from experiments.patch_score_promotion_observation import bootstrap_ci
from experiments.patch_score_routing_gradient_experiment import (
    clean_gradient,
    cosine_rows,
    load_samples,
    normalize,
    sign_agreement_rows,
)
from experiments.semantic_equivalent_route_experiment import PHASES, phase_shift
from gradient_replay import GradientReplay
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
SEMANTIC_PHASE = (8, 8)
CONDITIONS = (
    "opponent_baseline",
    "semantic_only",
    "shuffled_semantic",
    "final_semantic_residual",
    "early_semantic_residual",
    "uniform_multilayer_residual",
    "consensus_multilayer_residual",
)
FINAL_CONDITION = "final_semantic_residual"
CONSENSUS_CONDITION = "consensus_multilayer_residual"

DEFAULT_LAYERS = {
    "vit_base_patch16_224": (
        "block3",
        "block6",
        "block9",
        "block12",
    ),
    "cait_s24_224": (
        "block6_gap",
        "block12_gap",
        "block18_gap",
        "block24_class",
    ),
    "pit_b_224": (
        "stage1_block3",
        "stage2_block6",
        "stage3_block2",
        "stage3_block4",
    ),
    "visformer_small": (
        "stage1_block4",
        "stage2_block4",
        "stage3_block2",
        "stage3_block4",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=WHITEBOX_MODEL_CHOICES, required=True)
    parser.add_argument("--targets", default="auto")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=628)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--semantic-phase", type=int, nargs=2, default=SEMANTIC_PHASE)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--noise-strength", type=float, default=0.2)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=0.75)
    parser.add_argument("--semantic-lambda", type=float, default=0.25)
    parser.add_argument("--consensus-temperature", type=float, default=0.25)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/semantic_gradient_consensus_e6"),
    )
    parser.add_argument(
        "--layers",
        default="default",
        help="default or comma-separated source-model semantic checkpoints",
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
    if not rows:
        raise ValueError(f"cannot average empty rows for {key}")
    return sum(float(row[key]) for row in rows) / len(rows)


def make_attacker(model, args: argparse.Namespace) -> PatchScoreAttacker:
    # attack_method=none is intentional: no patch/drop path is entered.
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


def _freeze_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _normalize_l1(gradient: torch.Tensor) -> torch.Tensor:
    scale = gradient.detach().abs().flatten(1).mean(dim=1).clamp_min(1e-8)
    return gradient / scale.view(-1, 1, 1, 1)


def _scale_to_l2(gradient: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    grad_norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-8)
    ref_norm = reference.flatten(1).norm(dim=1).clamp_min(1e-8)
    return gradient * (ref_norm / grad_norm).view(-1, 1, 1, 1)


def _orthogonal_residual(
    semantic_gradient: torch.Tensor,
    opponent_gradient: torch.Tensor,
) -> torch.Tensor:
    semantic = semantic_gradient.flatten(1)
    opponent = opponent_gradient.flatten(1)
    coefficient = (semantic * opponent).sum(dim=1) / opponent.square().sum(dim=1).clamp_min(1e-8)
    residual = semantic - coefficient[:, None] * opponent
    residual = residual.view_as(semantic_gradient)
    return _scale_to_l2(residual, opponent_gradient)


def _spatially_shuffle_gradient(
    gradient: torch.Tensor,
    sample_ids: list[str],
    seed: int,
    grid: int = 7,
) -> torch.Tensor:
    batch, channels, height, width = gradient.shape
    if height % grid or width % grid:
        raise ValueError("image dimensions must be divisible by the shuffle grid")
    block_h, block_w = height // grid, width // grid
    blocks = gradient.reshape(batch, channels, grid, block_h, grid, block_w)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5).reshape(
        batch, channels, grid * grid, block_h, block_w
    )
    shuffled = torch.empty_like(blocks)
    for index, sample_id in enumerate(sample_ids):
        digest = hashlib.sha256(f"{seed}:{sample_id}:semantic_shuffle".encode()).digest()
        sample_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device=gradient.device).manual_seed(sample_seed)
        permutation = torch.randperm(grid * grid, generator=generator, device=gradient.device)
        shuffled[index] = blocks[index, :, permutation]
    shuffled = shuffled.reshape(batch, channels, grid, grid, block_h, block_w)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5)
    return shuffled.reshape_as(gradient)


def _layer_ids(source: str, specification: str) -> list[str]:
    layers = DEFAULT_LAYERS[source] if specification == "default" else tuple(
        item.strip() for item in specification.split(",") if item.strip()
    )
    if len(layers) < 2:
        raise ValueError("at least two semantic layers are required")
    return list(layers)


def _semantic_gradients(
    model,
    pixels: torch.Tensor,
    layer_ids: list[str],
    semantic_phase: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized per-layer gradients and clean cross-view cosines.

    This function contains no opponent noise and no CE/label objective.  The
    symmetric stop-gradient form makes the two deterministic semantic views
    contribute equally while keeping their reference representations fixed.
    """
    probe = pixels.detach().requires_grad_(True)
    gradients: list[torch.Tensor] = []
    global_cosines: list[torch.Tensor] = []
    for layer_id in layer_ids:
        # Each checkpoint owns an independent phase graph.  Reusing one
        # shifted tensor across autograd.grad calls would reuse a graph whose
        # saved tensors were already released by the previous layer.
        shifted = phase_shift(probe, *semantic_phase)
        first = model.extract_patch_score_features(
            normalize(model, probe), score_layer=layer_id
        ).global_token.flatten(1)
        second = model.extract_patch_score_features(
            normalize(model, shifted), score_layer=layer_id
        ).global_token.flatten(1)
        cosine = F.cosine_similarity(first.detach(), second.detach(), dim=1)
        loss = 0.5 * (
            1.0 - F.cosine_similarity(first, second.detach(), dim=1)
        ) + 0.5 * (
            1.0 - F.cosine_similarity(second, first.detach(), dim=1)
        )
        gradient = torch.autograd.grad(loss.sum(), probe, retain_graph=False)[0]
        gradients.append(_normalize_l1(gradient).detach())
        global_cosines.append(cosine.detach())
    return torch.stack(gradients, dim=1), torch.stack(global_cosines, dim=1)


def _opponent_gradient(
    model,
    attacker: PatchScoreAttacker,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    seed: int,
    groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw and Gaussian-processed 20-view opponent gradients.

    Every local token receives opponent noise.  There is deliberately no
    semantic mask or patch drop in this branch.
    """
    replay = GradientReplay(seed)
    replay.begin_batch(sample_ids)
    attacker._gradient_replay = replay
    probe = pixels.detach().requires_grad_(True)
    gradients: list[torch.Tensor] = []
    try:
        for group in range(groups):
            phase = PHASES[group % len(PHASES)]
            for view in range(2):
                replay.set_context(step=0, group=group, view=view)
                view_pixels = phase_shift(probe, *phase) if view else probe
                state = model.prepare_attack_feature_state(normalize(model, view_pixels))
                local = state.local_tokens + attacker._strict_opponent_feature_noise(state)
                logits = model.forward_from_attack_feature_state(state, local)
                loss = F.cross_entropy(logits, labels)
                gradients.append(torch.autograd.grad(loss, probe, retain_graph=False)[0])
        view_gradients = torch.stack(gradients)
        raw = view_gradients.mean(dim=0).detach()
        processed = attacker._apply_gaussian_residual(raw).detach()
        return raw, processed, view_gradients.detach()
    finally:
        attacker._gradient_replay = None


def _consensus_gradient(
    layer_gradients: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return uniform and soft agreement-weighted multi-layer gradients."""
    # [B,L,C,H,W] -> [B,L,P]
    flat = layer_gradients.flatten(2)
    normalized = flat / flat.norm(dim=2, keepdim=True).clamp_min(1e-8)
    gram = torch.bmm(normalized, normalized.transpose(1, 2))
    layer_count = layer_gradients.size(1)
    agreement = (gram.sum(dim=2) - 1.0) / max(1, layer_count - 1)
    weights = torch.softmax(agreement / temperature, dim=1)
    uniform = layer_gradients.mean(dim=1)
    consensus = (layer_gradients * weights[:, :, None, None, None]).sum(dim=1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
    effective_layers = entropy.exp()
    return uniform, consensus, torch.stack((weights.max(dim=1).values, effective_layers), dim=1)


def _condition_gradients(
    raw_opponent: torch.Tensor,
    layer_gradients: torch.Tensor,
    sample_ids: list[str],
    seed: int,
    semantic_lambda: float,
    consensus_temperature: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    final = layer_gradients[:, -1]
    early = layer_gradients[:, 0]
    uniform, consensus, consensus_stats = _consensus_gradient(
        layer_gradients, consensus_temperature
    )
    shuffled = _spatially_shuffle_gradient(final, sample_ids, seed)
    semantic_directions = {
        "semantic_only": final,
        "shuffled_semantic": shuffled,
        "final_semantic_residual": final,
        "early_semantic_residual": early,
        "uniform_multilayer_residual": uniform,
        "consensus_multilayer_residual": consensus,
    }
    raw_conditions = {"opponent_baseline": raw_opponent}
    raw_conditions["semantic_only"] = _scale_to_l2(final, raw_opponent)
    for condition in (
        "shuffled_semantic",
        "final_semantic_residual",
        "early_semantic_residual",
        "uniform_multilayer_residual",
        "consensus_multilayer_residual",
    ):
        residual = _orthogonal_residual(semantic_directions[condition], raw_opponent)
        raw_conditions[condition] = raw_opponent + semantic_lambda * residual
    return raw_conditions, {
        "consensus_weights": consensus_stats[:, 0].detach(),
        "consensus_effective_layers": consensus_stats[:, 1].detach(),
    }


def _condition_metrics(
    condition: str,
    raw: torch.Tensor,
    processed: torch.Tensor,
    opponent_raw: torch.Tensor,
    semantic_layers: torch.Tensor,
    semantic_cosines: torch.Tensor,
    consensus_stats: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    source_alignment = cosine_rows(raw, opponent_raw)
    layer_cos = []
    for layer_index in range(semantic_layers.size(1)):
        layer_cos.append(cosine_rows(raw, semantic_layers[:, layer_index]))
    layer_cosine = torch.stack(layer_cos, dim=1)
    rows = []
    for index in range(raw.size(0)):
        rows.append(
            {
                "condition": condition,
                "sample_index": index,
                "source_alignment_to_opponent": float(source_alignment[index]),
                "semantic_final_global_cosine": float(semantic_cosines[index, -1]),
                "semantic_early_global_cosine": float(semantic_cosines[index, 0]),
                "semantic_final_gradient_alignment": float(layer_cosine[index, -1]),
                "semantic_early_gradient_alignment": float(layer_cosine[index, 0]),
                "semantic_layer_alignment_mean": float(layer_cosine[index].mean()),
                "consensus_max_weight": float(consensus_stats["consensus_weights"][index]),
                "consensus_effective_layers": float(consensus_stats["consensus_effective_layers"][index]),
            }
        )
    return rows


def run_source(args: argparse.Namespace) -> None:
    if args.groups * 2 != 20:
        raise ValueError("E6 requires groups=10 and exactly 20 opponent views")
    if args.semantic_lambda <= 0:
        raise ValueError("semantic-lambda must be positive")
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    layers = _layer_ids(args.source, args.layers)
    targets = (
        [name for name in WHITEBOX_MODEL_CHOICES if name != args.source]
        if args.targets == "auto"
        else [item.strip() for item in args.targets.split(",") if item.strip()]
    )
    if args.source in targets or set(targets) - set(WHITEBOX_MODEL_CHOICES):
        raise ValueError("targets must be registered models other than source")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_model = build_whitebox_model(1000, args.source, pretrained=True, device=device).eval()
    _freeze_model(source_model)
    attacker = make_attacker(source_model, args)

    all_gradients: dict[str, dict[str, list[torch.Tensor]]] = {
        condition: {"raw": [], "processed": []} for condition in CONDITIONS
    }
    route_rows: list[dict[str, object]] = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        pixels = pixels_cpu[start:end].to(device)
        labels = labels_cpu[start:end].to(device)
        batch_names = names[start:end]
        layer_gradients, semantic_cosines = _semantic_gradients(
            source_model,
            pixels,
            layers,
            tuple(args.semantic_phase),
        )
        raw_opponent, _, _ = _opponent_gradient(
            source_model,
            attacker,
            pixels,
            labels,
            batch_names,
            args.seed,
            args.groups,
        )
        raw_conditions, consensus_stats = _condition_gradients(
            raw_opponent,
            layer_gradients,
            batch_names,
            args.seed,
            args.semantic_lambda,
            args.consensus_temperature,
        )
        for condition, raw in raw_conditions.items():
            processed = attacker._apply_gaussian_residual(raw).detach()
            all_gradients[condition]["raw"].append(raw.detach().cpu())
            all_gradients[condition]["processed"].append(processed.cpu())
            rows = _condition_metrics(
                condition,
                raw,
                processed,
                raw_opponent,
                layer_gradients,
                semantic_cosines,
                consensus_stats,
            )
            for local_index, row in enumerate(rows):
                row["sample_index"] = start + local_index
                row["image_name"] = names[start + local_index]
                route_rows.append(row)
        if (start // args.batch_size) % 8 == 0:
            print(
                f"[{args.source}] semantic/opponent batches {end}/{args.samples}",
                flush=True,
            )

    gradients = {
        condition: {
            key: torch.cat(values)
            for key, values in payload.items()
        }
        for condition, payload in all_gradients.items()
    }
    cross_rows: list[dict[str, object]] = []
    step_size = args.epsilon / args.steps
    for target_name in targets:
        print(f"[{args.source}] loading target {target_name}", flush=True)
        target_model = build_whitebox_model(1000, target_name, pretrained=True, device=device).eval()
        _freeze_model(target_model)
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            target_grad, clean_true, clean_loss, clean_pred = clean_gradient(
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
    cross_mean = {}
    for condition in CONDITIONS:
        selected = [row for row in cross_rows if row["condition"] == condition]
        cross_mean[condition] = {metric: mean(selected, metric) for metric in metrics}
    route_mean = {}
    route_metrics = (
        "source_alignment_to_opponent",
        "semantic_final_global_cosine",
        "semantic_early_global_cosine",
        "semantic_final_gradient_alignment",
        "semantic_early_gradient_alignment",
        "semantic_layer_alignment_mean",
        "consensus_max_weight",
        "consensus_effective_layers",
    )
    for condition in CONDITIONS:
        selected = [row for row in route_rows if row["condition"] == condition]
        route_mean[condition] = {metric: mean(selected, metric) for metric in route_metrics}

    baseline_for = {
        "semantic_only": "opponent_baseline",
        "shuffled_semantic": "opponent_baseline",
        "final_semantic_residual": "opponent_baseline",
        "early_semantic_residual": "opponent_baseline",
        "uniform_multilayer_residual": "final_semantic_residual",
        "consensus_multilayer_residual": "final_semantic_residual",
    }
    paired_deltas: dict[str, dict[str, dict[str, object]]] = {}
    for condition, baseline in baseline_for.items():
        paired_deltas[condition] = {}
        for metric_index, metric in enumerate(metrics):
            values = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in cross_rows
                if row["condition"] == condition
            }
            base_values = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in cross_rows
                if row["condition"] == baseline
            }
            delta = torch.tensor(
                [values[key] - base_values[key] for key in values if key in base_values],
                dtype=torch.float32,
            )
            paired_deltas[condition][metric] = {
                "mean": float(delta.mean()),
                "ci95": bootstrap_ci(delta, seed=args.seed + 1000 + metric_index),
                "positive_fraction": float(delta.gt(0).float().mean()),
                "comparisons": int(delta.numel()),
            }

    output_dir = args.output_dir / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "route_metrics.csv", route_rows)
    write_csv(output_dir / "cross_model_metrics.csv", cross_rows)
    summary = {
        "protocol": {
            "experiment": "E6_decoupled_semantic_gradient",
            "source": args.source,
            "targets": targets,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "layers": layers,
            "semantic_phase": list(args.semantic_phase),
            "semantic_branch": "noise_free_global_consistency_gradient",
            "opponent_branch": "20_view_all_token_rgb_opponent_noise_CE_gradient",
            "patch_drop": False,
            "whitebox_ensemble": False,
            "semantic_lambda": args.semantic_lambda,
            "consensus_temperature": args.consensus_temperature,
            "gaussian_sigma": args.gaussian_sigma,
            "gaussian_alpha": args.gaussian_alpha,
            "step_size": step_size,
        },
        "route_mean": route_mean,
        "cross_model_mean": cross_mean,
        "paired_deltas": paired_deltas,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.samples <= 1 or args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("samples > 1, positive batch size, and positive steps are required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_source(args)


if __name__ == "__main__":
    main()
