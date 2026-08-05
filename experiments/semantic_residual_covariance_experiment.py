"""E10: semantic-conditioned covariance of opponent-view gradient residuals.

The uniform twenty-view opponent gradient remains the immutable base.  A
noise-free phase/flip global-discrepancy gradient is projected through the
normalized covariance of centered source CE view gradients.  This tests
whether the intersection of semantic sensitivity and actual CE-view diversity
is more cross-architecture and more transferable than the bare semantic
gradient.  Target gradients are held-out diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.patch_score_promotion_observation import bootstrap_ci
from experiments.patch_score_routing_gradient_experiment import (
    clean_gradient,
    cosine_rows,
    load_samples,
    normalize,
    sign_agreement_rows,
)
from experiments.semantic_conditioned_gradient_aggregation_experiment import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_IMAGE_DIR,
    _opponent_view_gradients,
    _semantic_gradients,
    freeze_model,
    make_attacker,
)
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_OUTPUT_DIR = Path("outputs/research/semantic_residual_covariance_e10")
DIRECTION_KINDS = (
    "semantic_covariance",
    "shuffled_semantic_covariance",
    "permuted_coefficient_covariance",
    "random_covariance",
    "direct_semantic",
)
CONDITIONS = (
    "uniform",
    "cov_plus",
    "cov_minus",
    "shuffled_plus",
    "shuffled_minus",
    "permuted_plus",
    "permuted_minus",
    "random_plus",
    "random_minus",
    "direct_plus",
    "direct_minus",
)
METRICS = (
    "raw_target_gradient_cosine",
    "processed_target_gradient_cosine",
    "raw_target_sign_agreement",
    "processed_target_sign_agreement",
    "one_step_target_logit_drop",
    "one_step_target_loss_increase",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=756)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--noise-strength", type=float, default=0.2)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=0.75)
    parser.add_argument("--semantic-phase", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--correction-lambda", type=float, default=0.125)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_names(args: argparse.Namespace) -> list[str]:
    names = (
        list(WHITEBOX_MODEL_CHOICES)
        if args.models == "all"
        else [item.strip() for item in args.models.split(",") if item.strip()]
    )
    if len(names) < 2 or set(names) - set(WHITEBOX_MODEL_CHOICES):
        raise ValueError("models must contain at least two registered architectures")
    return names


def validate(args: argparse.Namespace) -> None:
    if args.samples <= 1 or args.batch_size <= 0:
        raise ValueError("samples > 1 and batch-size > 0 are required")
    if args.groups * 2 != 20:
        raise ValueError("E10 requires exactly twenty opponent views")
    if args.correction_lambda <= 0:
        raise ValueError("correction-lambda must be positive")


def unit_rows(values: torch.Tensor) -> torch.Tensor:
    return values / values.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)


def scale_to_reference(direction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    direction_norm = direction.flatten(1).norm(dim=1).clamp_min(1e-12)
    reference_norm = reference.flatten(1).norm(dim=1).clamp_min(1e-12)
    return direction * (reference_norm / direction_norm).view(-1, 1, 1, 1)


def deterministic_random_like(reference: torch.Tensor, names: list[str], seed: int):
    output = torch.empty_like(reference)
    for index, name in enumerate(names):
        digest = hashlib.sha256(f"{seed}:{name}:e10_random_q".encode()).digest()
        generator = torch.Generator(device="cpu").manual_seed(
            int.from_bytes(digest[:8], "little") % (2**63 - 1)
        )
        output[index] = torch.randn(
            reference[index].shape,
            generator=generator,
            dtype=reference.dtype,
        ).to(reference.device)
    return output


def coefficient_shift(name: str, seed: int, views: int) -> int:
    digest = hashlib.sha256(f"{seed}:{name}:e10_coefficients".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "little") % (views - 1)


def covariance_direction(
    normalized_residuals: torch.Tensor,
    semantic: torch.Tensor,
    names: list[str],
    seed: int,
    permute_coefficients: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return C_r q without materializing its input-dimensional covariance."""
    # normalized_residuals: [B,V,C,H,W]
    flattened = normalized_residuals.flatten(2)
    semantic_unit = unit_rows(semantic).flatten(1)
    coefficients = torch.bmm(flattened, semantic_unit[:, :, None]).squeeze(2)
    if permute_coefficients:
        coefficients = torch.stack([
            coefficients[index].roll(
                coefficient_shift(name, seed, coefficients.size(1))
            )
            for index, name in enumerate(names)
        ])
    direction = (
        normalized_residuals
        * coefficients[:, :, None, None, None]
    ).mean(dim=1)
    return direction, coefficients


def residual_spectrum(
    normalized_residuals: torch.Tensor,
    semantic_coefficients: torch.Tensor,
) -> dict[str, torch.Tensor]:
    flattened = normalized_residuals.flatten(2)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum(dim=1, keepdim=True).clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    )
    mode_coefficients = torch.bmm(
        eigenvectors.transpose(1, 2), semantic_coefficients[:, :, None]
    ).squeeze(2)
    mode_energy = mode_coefficients.square()
    mode_energy = mode_energy / mode_energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {
        "residual_effective_rank": effective_rank,
        "top_eigenvalue_fraction": probabilities[:, -1],
        "top3_eigenvalue_fraction": probabilities[:, -3:].sum(dim=1),
        "top_mode_semantic_fraction": mode_energy[:, -1],
        "top3_mode_semantic_fraction": mode_energy[:, -3:].sum(dim=1),
    }


def correction_conditions(
    mean_gradient: torch.Tensor,
    directions: dict[str, torch.Tensor],
    attacker,
    correction_lambda: float,
) -> dict[str, dict[str, torch.Tensor]]:
    mapping = {
        "cov": "semantic_covariance",
        "shuffled": "shuffled_semantic_covariance",
        "permuted": "permuted_coefficient_covariance",
        "random": "random_covariance",
        "direct": "direct_semantic",
    }
    raw = {"uniform": mean_gradient}
    for prefix, direction_name in mapping.items():
        correction = scale_to_reference(directions[direction_name], mean_gradient)
        raw[f"{prefix}_plus"] = mean_gradient + correction_lambda * correction
        raw[f"{prefix}_minus"] = mean_gradient - correction_lambda * correction
    return {
        condition: {
            "raw": gradient.detach(),
            "processed": attacker._apply_gaussian_residual(gradient).detach(),
        }
        for condition, gradient in raw.items()
    }


def compute_source_payload(model_name, names, pixels_cpu, labels_cpu, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_whitebox_model(1000, model_name, pretrained=True, device=device).eval()
    freeze_model(model)
    attacker = make_attacker(model, args)
    view_parts = []
    semantic_parts = []
    source_gradient_parts = []
    semantic_rows = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        pixels = pixels_cpu[start:end].to(device)
        labels = labels_cpu[start:end].to(device)
        semantic, diagnostics = _semantic_gradients(model, pixels, args)
        views = _opponent_view_gradients(
            model, attacker, pixels, labels, names[start:end], args
        )
        source_gradient, _, _, _ = clean_gradient(model, pixels, labels)
        view_parts.append(views.to(torch.float16).cpu())
        semantic_parts.append(semantic["consensus"].to(torch.float16).cpu())
        source_gradient_parts.append(source_gradient.to(torch.float16).cpu())
        for local, sample_index in enumerate(range(start, end)):
            semantic_rows.append({
                "model": model_name,
                "sample_index": sample_index,
                "image_name": names[sample_index],
                "phase_global_cosine": float(diagnostics["phase"][local]),
                "flip_global_cosine": float(diagnostics["flip"][local]),
                "phase_flip_gradient_cosine": float(
                    diagnostics["phase_flip_gradient_cosine"][local]
                ),
            })
        if start == 0 or end == args.samples or (start // args.batch_size) % 8 == 0:
            print(f"[{model_name}] source views {end}/{args.samples}", flush=True)

    all_views = torch.cat(view_parts, dim=1)
    all_semantic = torch.cat(semantic_parts)
    all_source_gradient = torch.cat(source_gradient_parts)
    direction_parts = {kind: [] for kind in DIRECTION_KINDS}
    condition_parts = {
        condition: {"raw": [], "processed": []} for condition in CONDITIONS
    }
    spectrum_rows = []
    source_rows = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        batch_names = names[start:end]
        views = all_views[:, start:end].to(device=device, dtype=torch.float32).transpose(0, 1)
        semantic = all_semantic[start:end].to(device=device, dtype=torch.float32)
        shuffled_index = torch.tensor(
            [(index + 1) % args.samples for index in range(start, end)]
        )
        shuffled_semantic = all_semantic[shuffled_index].to(
            device=device, dtype=torch.float32
        )
        random_semantic = deterministic_random_like(semantic, batch_names, args.seed)
        mean_gradient = views.mean(dim=1)
        residuals = views - mean_gradient[:, None]
        residual_norm = residuals.flatten(2).norm(dim=2).clamp_min(1e-12)
        normalized_residuals = residuals / residual_norm[:, :, None, None, None]
        covariance, coefficients = covariance_direction(
            normalized_residuals, semantic, batch_names, args.seed
        )
        shuffled, _ = covariance_direction(
            normalized_residuals, shuffled_semantic, batch_names, args.seed
        )
        permuted, _ = covariance_direction(
            normalized_residuals,
            semantic,
            batch_names,
            args.seed,
            permute_coefficients=True,
        )
        random_direction, _ = covariance_direction(
            normalized_residuals, random_semantic, batch_names, args.seed
        )
        directions = {
            "semantic_covariance": covariance,
            "shuffled_semantic_covariance": shuffled,
            "permuted_coefficient_covariance": permuted,
            "random_covariance": random_direction,
            "direct_semantic": semantic,
        }
        spectrum = residual_spectrum(normalized_residuals, coefficients)
        conditions = correction_conditions(
            mean_gradient, directions, attacker, args.correction_lambda
        )
        source_gradient = all_source_gradient[start:end].to(
            device=device, dtype=torch.float32
        )
        for kind, direction in directions.items():
            direction_parts[kind].append(direction.to(torch.float16).cpu())
        for condition, payload in conditions.items():
            for key, value in payload.items():
                condition_parts[condition][key].append(value.to(torch.float16).cpu())
            raw_source = cosine_rows(payload["raw"], source_gradient)
            processed_source = cosine_rows(payload["processed"], source_gradient)
            for local, sample_index in enumerate(range(start, end)):
                source_rows.append({
                    "source_model": model_name,
                    "condition": condition,
                    "sample_index": sample_index,
                    "image_name": names[sample_index],
                    "raw_source_gradient_cosine": float(raw_source[local]),
                    "processed_source_gradient_cosine": float(processed_source[local]),
                })
        for local, sample_index in enumerate(range(start, end)):
            spectrum_rows.append({
                "model": model_name,
                "sample_index": sample_index,
                "image_name": names[sample_index],
                **{key: float(value[local]) for key, value in spectrum.items()},
                "mean_residual_l2": float(residual_norm[local].mean()),
                "semantic_covariance_l2": float(covariance[local].norm()),
            })
    payload = {
        "directions": {
            kind: torch.cat(parts) for kind, parts in direction_parts.items()
        },
        "conditions": {
            condition: {key: torch.cat(parts) for key, parts in values.items()}
            for condition, values in condition_parts.items()
        },
        "semantic_rows": semantic_rows,
        "spectrum_rows": spectrum_rows,
        "source_rows": source_rows,
    }
    del attacker, model, all_views, all_semantic, all_source_gradient
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def direction_commonality(payloads, names, args):
    rows = []
    for left, right in itertools.permutations(payloads, 2):
        for kind in DIRECTION_KINDS:
            left_values = payloads[left]["directions"][kind].float()
            right_values = payloads[right]["directions"][kind].float()
            mismatched = right_values.roll(1, dims=0)
            same = cosine_rows(left_values, right_values)
            mismatch = cosine_rows(left_values, mismatched)
            same_sign = sign_agreement_rows(left_values, right_values)
            mismatch_sign = sign_agreement_rows(left_values, mismatched)
            for index, image_name in enumerate(names):
                rows.append({
                    "source_model": left,
                    "target_model": right,
                    "direction_kind": kind,
                    "sample_index": index,
                    "image_name": image_name,
                    "same_image_cosine": float(same[index]),
                    "mismatched_image_cosine": float(mismatch[index]),
                    "same_minus_mismatched_cosine": float(same[index] - mismatch[index]),
                    "same_image_sign_agreement": float(same_sign[index]),
                    "mismatched_image_sign_agreement": float(mismatch_sign[index]),
                })
    return rows


def target_diagnostics(payloads, models, names, pixels_cpu, labels_cpu, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    step_size = args.epsilon / args.steps
    for source_name in models:
        for target_name in models:
            if source_name == target_name:
                continue
            print(f"[{source_name}] held-out target {target_name}", flush=True)
            target = build_whitebox_model(
                1000, target_name, pretrained=True, device=device
            ).eval()
            freeze_model(target)
            for start in range(0, args.samples, args.batch_size):
                end = min(args.samples, start + args.batch_size)
                pixels = pixels_cpu[start:end].to(device)
                labels = labels_cpu[start:end].to(device)
                target_gradient, clean_true, clean_loss, clean_pred = clean_gradient(
                    target, pixels, labels
                )
                for condition in CONDITIONS:
                    raw = payloads[source_name]["conditions"][condition]["raw"][start:end].to(device).float()
                    processed = payloads[source_name]["conditions"][condition]["processed"][start:end].to(device).float()
                    adversarial = torch.clamp(
                        pixels + step_size * processed.sign(), 0.0, 1.0
                    )
                    with torch.no_grad():
                        logits = target(normalize(target, adversarial))
                        adv_true = logits.gather(1, labels[:, None]).squeeze(1)
                        adv_loss = F.cross_entropy(logits, labels, reduction="none")
                    raw_cos = cosine_rows(raw, target_gradient)
                    processed_cos = cosine_rows(processed, target_gradient)
                    raw_sign = sign_agreement_rows(raw, target_gradient)
                    processed_sign = sign_agreement_rows(processed, target_gradient)
                    for local, sample_index in enumerate(range(start, end)):
                        rows.append({
                            "source_model": source_name,
                            "target_model": target_name,
                            "condition": condition,
                            "sample_index": sample_index,
                            "image_name": names[sample_index],
                            "target_clean_correct": bool(
                                clean_pred[local].eq(labels[local]).cpu()
                            ),
                            "raw_target_gradient_cosine": float(raw_cos[local]),
                            "processed_target_gradient_cosine": float(processed_cos[local]),
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
    return rows


def paired_delta(rows, condition, baseline, metric, seed):
    current = {
        (row["source_model"], row["target_model"], int(row["sample_index"])): float(row[metric])
        for row in rows if row["condition"] == condition
    }
    base = {
        (row["source_model"], row["target_model"], int(row["sample_index"])): float(row[metric])
        for row in rows if row["condition"] == baseline
    }
    values = torch.tensor([current[key] - base[key] for key in current])
    return {
        "mean": float(values.mean()),
        "ci95": bootstrap_ci(values, seed=seed),
        "positive_fraction": float(values.gt(0).float().mean()),
        "comparisons": int(values.numel()),
    }


def summarize(payloads, models, commonality_rows, target_rows, args):
    condition_mean = {}
    for condition in CONDITIONS:
        selected = [row for row in target_rows if row["condition"] == condition]
        condition_mean[condition] = {
            metric: sum(float(row[metric]) for row in selected) / len(selected)
            for metric in METRICS
        }
    comparisons = {}
    control_for = {
        "cov_plus_vs_uniform": ("cov_plus", "uniform"),
        "cov_minus_vs_uniform": ("cov_minus", "uniform"),
        "cov_plus_vs_shuffled": ("cov_plus", "shuffled_plus"),
        "cov_minus_vs_shuffled": ("cov_minus", "shuffled_minus"),
        "cov_plus_vs_permuted": ("cov_plus", "permuted_plus"),
        "cov_minus_vs_permuted": ("cov_minus", "permuted_minus"),
        "cov_plus_vs_random": ("cov_plus", "random_plus"),
        "cov_minus_vs_random": ("cov_minus", "random_minus"),
        "cov_plus_vs_direct": ("cov_plus", "direct_plus"),
        "cov_minus_vs_direct": ("cov_minus", "direct_minus"),
    }
    for comparison_index, (name, (condition, baseline)) in enumerate(control_for.items()):
        comparisons[name] = {
            metric: paired_delta(
                target_rows,
                condition,
                baseline,
                metric,
                args.seed + 1000 * (comparison_index + 1) + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }

    commonality_summary = {}
    for kind in DIRECTION_KINDS:
        selected = [
            row for row in commonality_rows if row["direction_kind"] == kind
        ]
        gaps = torch.tensor([
            float(row["same_minus_mismatched_cosine"]) for row in selected
        ])
        commonality_summary[kind] = {
            "same_image_cosine": sum(float(row["same_image_cosine"]) for row in selected) / len(selected),
            "mismatched_image_cosine": sum(float(row["mismatched_image_cosine"]) for row in selected) / len(selected),
            "same_minus_mismatched": {
                "mean": float(gaps.mean()),
                "ci95": bootstrap_ci(gaps, seed=args.seed + 20000 + list(DIRECTION_KINDS).index(kind)),
            },
        }
    covariance_same = {
        (row["source_model"], row["target_model"], int(row["sample_index"])): float(row["same_image_cosine"])
        for row in commonality_rows if row["direction_kind"] == "semantic_covariance"
    }
    direct_same = {
        (row["source_model"], row["target_model"], int(row["sample_index"])): float(row["same_image_cosine"])
        for row in commonality_rows if row["direction_kind"] == "direct_semantic"
    }
    covariance_vs_direct = torch.tensor([
        covariance_same[key] - direct_same[key] for key in covariance_same
    ])
    commonality_summary["semantic_covariance_vs_direct_semantic"] = {
        "mean_same_image_cosine_delta": float(covariance_vs_direct.mean()),
        "ci95": bootstrap_ci(covariance_vs_direct, seed=args.seed + 21000),
    }

    source_positive = {}
    for condition in ("cov_plus", "cov_minus"):
        source_positive[condition] = {}
        for source in models:
            selected = [
                row for row in target_rows
                if row["source_model"] == source and row["condition"] == condition
            ]
            base = {
                (row["target_model"], int(row["sample_index"])): row
                for row in target_rows
                if row["source_model"] == source and row["condition"] == "uniform"
            }
            source_positive[condition][source] = {
                metric: sum(
                    float(row[metric]) - float(base[(row["target_model"], int(row["sample_index"]))][metric])
                    for row in selected
                ) / len(selected)
                for metric in (
                    "processed_target_gradient_cosine",
                    "processed_target_sign_agreement",
                    "one_step_target_loss_increase",
                )
            }

    gate = {"passed": False, "selected_condition": None, "checks": {}}
    covariance_common = commonality_summary["semantic_covariance"]["same_minus_mismatched"]
    gate["checks"]["covariance_same_beats_mismatched"] = covariance_common["ci95"][0] > 0
    gate["checks"]["covariance_beats_direct_commonality"] = commonality_summary[
        "semantic_covariance_vs_direct_semantic"
    ]["ci95"][0] > 0
    for condition, sign_name in (("cov_plus", "plus"), ("cov_minus", "minus")):
        prefix = f"cov_{sign_name}"
        vs_uniform = comparisons[f"{prefix}_vs_uniform"]
        direction_count = sum(
            values["processed_target_gradient_cosine"] > 0
            for values in source_positive[condition].values()
        )
        checks = {
            "target_cosine_ci_positive": vs_uniform["processed_target_gradient_cosine"]["ci95"][0] > 0,
            "target_sign_or_loss_ci_positive": (
                vs_uniform["processed_target_sign_agreement"]["ci95"][0] > 0
                or vs_uniform["one_step_target_loss_increase"]["ci95"][0] > 0
            ),
            "at_least_three_sources_positive": direction_count >= 3,
            "beats_shuffled_target_cosine": comparisons[f"{prefix}_vs_shuffled"]["processed_target_gradient_cosine"]["ci95"][0] > 0,
            "beats_permuted_target_cosine": comparisons[f"{prefix}_vs_permuted"]["processed_target_gradient_cosine"]["ci95"][0] > 0,
            "beats_random_target_cosine": comparisons[f"{prefix}_vs_random"]["processed_target_gradient_cosine"]["ci95"][0] > 0,
            "beats_direct_target_cosine": comparisons[f"{prefix}_vs_direct"]["processed_target_gradient_cosine"]["ci95"][0] > 0,
        }
        gate["checks"][condition] = checks
        if (
            gate["checks"]["covariance_same_beats_mismatched"]
            and gate["checks"]["covariance_beats_direct_commonality"]
            and all(checks.values())
        ):
            gate["passed"] = True
            gate["selected_condition"] = condition

    spectrum_rows = [row for model in models for row in payloads[model]["spectrum_rows"]]
    spectrum_mean = {
        key: sum(float(row[key]) for row in spectrum_rows) / len(spectrum_rows)
        for key in (
            "residual_effective_rank",
            "top_eigenvalue_fraction",
            "top3_eigenvalue_fraction",
            "top_mode_semantic_fraction",
            "top3_mode_semantic_fraction",
        )
    }
    return {
        "protocol": {
            "experiment": "E10_semantic_residual_covariance_gate",
            "models": models,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "views": args.groups * 2,
            "semantic_branch": "noise_free_phase_flip_global_discrepancy",
            "residual_definition": "view_gradient_minus_uniform_mean",
            "covariance_definition": "mean(unit_residual * dot(unit_residual,unit_semantic))",
            "correction_lambda": args.correction_lambda,
            "gaussian_processing": "applied_once_after_raw_correction",
            "target_used_for_generation": False,
            "whitebox_ensemble": False,
        },
        "residual_spectrum_mean": spectrum_mean,
        "direction_commonality": commonality_summary,
        "condition_mean": condition_mean,
        "paired_deltas": comparisons,
        "per_source_deltas": source_positive,
        "gate": gate,
    }


def main() -> None:
    args = parse_args()
    validate(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    models = model_names(args)
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    payloads = {
        model: compute_source_payload(model, names, pixels_cpu, labels_cpu, args)
        for model in models
    }
    commonality_rows = direction_commonality(payloads, names, args)
    target_rows = target_diagnostics(
        payloads, models, names, pixels_cpu, labels_cpu, args
    )
    summary = summarize(payloads, models, commonality_rows, target_rows, args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "direction_commonality.csv", commonality_rows)
    write_csv(output / "target_metrics.csv", target_rows)
    write_csv(
        output / "residual_spectrum.csv",
        [row for model in models for row in payloads[model]["spectrum_rows"]],
    )
    write_csv(
        output / "source_metrics.csv",
        [row for model in models for row in payloads[model]["source_rows"]],
    )
    write_csv(
        output / "semantic_diagnostics.csv",
        [row for model in models for row in payloads[model]["semantic_rows"]],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
