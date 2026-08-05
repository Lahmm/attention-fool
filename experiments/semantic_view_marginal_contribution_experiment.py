"""E9: test semantic scores against a view's aggregation marginal contribution.

E8 showed that a noise-free semantic gradient weakly ranks the held-out target
alignment of individual opponent-noise CE gradients, while semantic weighting
made their aggregate worse.  E9 tests the missing set-level quantity: for each
view, how target alignment changes when that view is removed from the uniform
mean.  Target models remain held-out diagnostics and never affect source-view
generation or semantic scores.

The reported ``removal_delta`` follows the preregistered sign convention:

  cos(mean_without_view, target) - cos(full_mean, target)

Positive means removing the view helps (the view is harmful/redundant);
negative means removing it hurts (the view has positive marginal value).
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

from experiments.patch_score_promotion_observation import bootstrap_ci
from experiments.patch_score_routing_gradient_experiment import (
    clean_gradient,
    cosine_rows,
    load_samples,
    row_spearman,
)
from experiments.semantic_conditioned_gradient_aggregation_experiment import (
    DEFAULT_ANNOTATIONS,
    DEFAULT_IMAGE_DIR,
    _opponent_view_gradients,
    _semantic_gradients,
    _view_scores,
    freeze_model,
    make_attacker,
)
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_OUTPUT_DIR = Path("outputs/research/semantic_view_marginal_e9")
SCORE_KINDS = ("phase", "flip", "consensus", "shuffled_semantic", "permuted_score")
MARGINAL_METRICS = (
    "raw_target_removal_delta",
    "processed_target_removal_delta",
    "raw_source_removal_delta",
    "processed_source_removal_delta",
    "raw_transfer_selective_benefit",
    "processed_transfer_selective_benefit",
    "view_to_uniform_cosine",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--topk", type=int, default=5)
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


def _targets(args: argparse.Namespace) -> list[str]:
    targets = (
        [name for name in WHITEBOX_MODEL_CHOICES if name != args.source]
        if args.targets == "auto"
        else [item.strip() for item in args.targets.split(",") if item.strip()]
    )
    if args.source in targets or set(targets) - set(WHITEBOX_MODEL_CHOICES):
        raise ValueError("targets must be registered models other than source")
    return targets


def _validate(args: argparse.Namespace) -> None:
    if args.samples <= 1 or args.batch_size <= 0:
        raise ValueError("samples > 1 and batch-size > 0 are required")
    if args.groups * 2 != 20:
        raise ValueError("E9 requires exactly twenty opponent views")
    if not 0 < args.topk < args.groups * 2:
        raise ValueError("topk must be between 1 and view_count - 1")


def _permutation_shift(sample_id: str, seed: int, count: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:e9_score_control".encode()).digest()
    return 1 + int.from_bytes(digest[:8], "little") % (count - 1)


def _permute_scores(scores: torch.Tensor, names: list[str], seed: int) -> torch.Tensor:
    result = torch.empty_like(scores)
    for index, name in enumerate(names):
        result[index] = scores[index].roll(
            _permutation_shift(name, seed, scores.size(1))
        )
    return result


def _process_leave_one_out(attacker, gradients: torch.Tensor):
    """Return raw/processed full means and all leave-one-out means."""
    view_count = gradients.size(0)
    total = gradients.sum(dim=0)
    full_raw = total / view_count
    # [V,B,C,H,W] -> [B,V,C,H,W]
    loo_raw = ((total.unsqueeze(0) - gradients) / (view_count - 1)).transpose(0, 1)
    full_processed = attacker._apply_gaussian_residual(full_raw).detach()
    batch, views, channels, height, width = loo_raw.shape
    loo_processed = attacker._apply_gaussian_residual(
        loo_raw.reshape(batch * views, channels, height, width)
    ).reshape_as(loo_raw).detach()
    return full_raw.detach(), loo_raw.detach(), full_processed, loo_processed


def _cosine_to_reference(candidates: torch.Tensor, reference: torch.Tensor):
    """Cosine for [B,V,C,H,W] candidates and [B,C,H,W] references."""
    return F.cosine_similarity(
        candidates.flatten(2), reference.flatten(1)[:, None], dim=2
    )


def _source_payload(args, names, pixels_cpu, labels_cpu):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_whitebox_model(1000, args.source, pretrained=True, device=device).eval()
    freeze_model(model)
    attacker = make_attacker(model, args)
    views_parts = []
    semantic_parts = {kind: [] for kind in ("phase", "flip", "consensus")}
    source_grad_parts = []
    diagnostic_rows = []
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        pixels = pixels_cpu[start:end].to(device)
        labels = labels_cpu[start:end].to(device)
        semantic, diagnostics = _semantic_gradients(model, pixels, args)
        views = _opponent_view_gradients(
            model, attacker, pixels, labels, names[start:end], args
        )
        source_gradient, _, _, _ = clean_gradient(model, pixels, labels)
        views_parts.append(views.to(torch.float16).cpu())
        source_grad_parts.append(source_gradient.to(torch.float16).cpu())
        for kind in semantic_parts:
            semantic_parts[kind].append(semantic[kind].to(torch.float16).cpu())
        for local, sample_index in enumerate(range(start, end)):
            diagnostic_rows.append({
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
            print(f"[{args.source}] source gradients {end}/{args.samples}", flush=True)

    all_views = torch.cat(views_parts, dim=1)
    all_semantic = {kind: torch.cat(parts) for kind, parts in semantic_parts.items()}
    all_source_gradient = torch.cat(source_grad_parts)
    score_parts = {kind: [] for kind in SCORE_KINDS}
    aggregate_parts = {key: [] for key in (
        "full_raw", "loo_raw", "full_processed", "loo_processed",
        "raw_source_removal_delta", "processed_source_removal_delta",
        "view_to_uniform_cosine",
    )}
    for start in range(0, args.samples, args.batch_size):
        end = min(args.samples, start + args.batch_size)
        views = all_views[:, start:end].to(device=device, dtype=torch.float32)
        semantic = {
            kind: value[start:end].to(device=device, dtype=torch.float32)
            for kind, value in all_semantic.items()
        }
        shuffled_index = torch.tensor(
            [(index + 1) % args.samples for index in range(start, end)]
        )
        semantic["shuffled_semantic"] = all_semantic["consensus"][
            shuffled_index
        ].to(device=device, dtype=torch.float32)
        for kind in ("phase", "flip", "consensus", "shuffled_semantic"):
            score_parts[kind].append(_view_scores(views, semantic[kind]).cpu())
        consensus_scores = score_parts["consensus"][-1]
        score_parts["permuted_score"].append(
            _permute_scores(consensus_scores, names[start:end], args.seed)
        )
        full_raw, loo_raw, full_processed, loo_processed = _process_leave_one_out(
            attacker, views
        )
        source_gradient = all_source_gradient[start:end].to(
            device=device, dtype=torch.float32
        )
        full_source_raw = cosine_rows(full_raw, source_gradient)
        loo_source_raw = _cosine_to_reference(loo_raw, source_gradient)
        full_source_processed = cosine_rows(full_processed, source_gradient)
        loo_source_processed = _cosine_to_reference(loo_processed, source_gradient)
        view_to_uniform = F.cosine_similarity(
            views.flatten(2).transpose(0, 1), full_raw.flatten(1)[:, None], dim=2
        )
        values = {
            "full_raw": full_raw,
            "loo_raw": loo_raw,
            "full_processed": full_processed,
            "loo_processed": loo_processed,
            "raw_source_removal_delta": loo_source_raw - full_source_raw[:, None],
            "processed_source_removal_delta": (
                loo_source_processed - full_source_processed[:, None]
            ),
            "view_to_uniform_cosine": view_to_uniform,
        }
        for key, value in values.items():
            aggregate_parts[key].append(value.to(torch.float16).cpu())
    scores = {kind: torch.cat(parts) for kind, parts in score_parts.items()}
    aggregates = {key: torch.cat(parts) for key, parts in aggregate_parts.items()}
    del attacker, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, aggregates, diagnostic_rows


def _metric_correlations(scores: torch.Tensor, metrics: dict[str, torch.Tensor]):
    return {key: row_spearman(scores, value) for key, value in metrics.items()}


def run(args: argparse.Namespace) -> None:
    names, pixels_cpu, labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    targets = _targets(args)
    scores, aggregates, diagnostic_rows = _source_payload(
        args, names, pixels_cpu, labels_cpu
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    marginal_rows = []
    correlation_rows = []
    for target_name in targets:
        print(f"[{args.source}] marginal target {target_name}", flush=True)
        target = build_whitebox_model(1000, target_name, pretrained=True, device=device).eval()
        freeze_model(target)
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            labels = labels_cpu[start:end].to(device)
            target_gradient, _, _, _ = clean_gradient(target, pixels, labels)
            full_raw = aggregates["full_raw"][start:end].to(device).float()
            loo_raw = aggregates["loo_raw"][start:end].to(device).float()
            full_processed = aggregates["full_processed"][start:end].to(device).float()
            loo_processed = aggregates["loo_processed"][start:end].to(device).float()
            full_target_raw = cosine_rows(full_raw, target_gradient)
            loo_target_raw = _cosine_to_reference(loo_raw, target_gradient)
            full_target_processed = cosine_rows(full_processed, target_gradient)
            loo_target_processed = _cosine_to_reference(loo_processed, target_gradient)
            raw_target_delta = loo_target_raw - full_target_raw[:, None]
            processed_target_delta = (
                loo_target_processed - full_target_processed[:, None]
            )
            raw_source_delta = aggregates["raw_source_removal_delta"][start:end].to(device).float()
            processed_source_delta = aggregates[
                "processed_source_removal_delta"
            ][start:end].to(device).float()
            metrics = {
                "raw_target_removal_delta": raw_target_delta,
                "processed_target_removal_delta": processed_target_delta,
                "raw_source_removal_delta": raw_source_delta,
                "processed_source_removal_delta": processed_source_delta,
                # Positive means the view helps target alignment more than it
                # helps source alignment. Benefit is negative removal delta.
                "raw_transfer_selective_benefit": raw_source_delta - raw_target_delta,
                "processed_transfer_selective_benefit": (
                    processed_source_delta - processed_target_delta
                ),
                "view_to_uniform_cosine": aggregates[
                    "view_to_uniform_cosine"
                ][start:end].to(device).float(),
            }
            for kind in SCORE_KINDS:
                current_scores = scores[kind][start:end].to(device).float()
                correlations = _metric_correlations(current_scores, metrics)
                top_indices = current_scores.topk(args.topk, dim=1).indices
                low_indices = current_scores.topk(args.topk, dim=1, largest=False).indices
                for local, sample_index in enumerate(range(start, end)):
                    correlation_rows.append({
                        "source_model": args.source,
                        "target_model": target_name,
                        "score_kind": kind,
                        "sample_index": sample_index,
                        "image_name": names[sample_index],
                        **{
                            f"score_spearman_{key}": float(value[local])
                            for key, value in correlations.items()
                        },
                        "topk_processed_target_removal_delta": float(
                            processed_target_delta[local].gather(
                                0, top_indices[local]
                            ).mean()
                        ),
                        "lowk_processed_target_removal_delta": float(
                            processed_target_delta[local].gather(
                                0, low_indices[local]
                            ).mean()
                        ),
                    })
                if kind == "consensus":
                    for local, sample_index in enumerate(range(start, end)):
                        for view_index in range(current_scores.size(1)):
                            marginal_rows.append({
                                "source_model": args.source,
                                "target_model": target_name,
                                "sample_index": sample_index,
                                "image_name": names[sample_index],
                                "view_index": view_index,
                                "semantic_score": float(current_scores[local, view_index]),
                                **{
                                    key: float(value[local, view_index])
                                    for key, value in metrics.items()
                                },
                            })
        del target
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    correlation_metrics = tuple(
        f"score_spearman_{key}" for key in MARGINAL_METRICS
    ) + (
        "topk_processed_target_removal_delta",
        "lowk_processed_target_removal_delta",
    )
    condition_mean = {}
    for kind in SCORE_KINDS:
        selected = [row for row in correlation_rows if row["score_kind"] == kind]
        condition_mean[kind] = {
            metric: sum(float(row[metric]) for row in selected) / len(selected)
            for metric in correlation_metrics
        }
    paired = {}
    for control in ("shuffled_semantic", "permuted_score"):
        paired[f"consensus_vs_{control}"] = {}
        for metric_index, metric in enumerate(correlation_metrics):
            current = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in correlation_rows if row["score_kind"] == "consensus"
            }
            baseline = {
                (row["target_model"], int(row["sample_index"])): float(row[metric])
                for row in correlation_rows if row["score_kind"] == control
            }
            delta = torch.tensor([current[key] - baseline[key] for key in current])
            paired[f"consensus_vs_{control}"][metric] = {
                "mean": float(delta.mean()),
                "ci95": bootstrap_ci(
                    delta, seed=args.seed + 1000 * (1 + list(("shuffled_semantic", "permuted_score")).index(control)) + metric_index
                ),
                "positive_fraction": float(delta.gt(0).float().mean()),
                "comparisons": int(delta.numel()),
            }

    consensus_rows = [row for row in marginal_rows]
    marginal_mean = {
        metric: sum(float(row[metric]) for row in consensus_rows) / len(consensus_rows)
        for metric in MARGINAL_METRICS
    }
    summary = {
        "protocol": {
            "experiment": "E9_semantic_view_marginal_contribution",
            "source": args.source,
            "targets": targets,
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "views": args.groups * 2,
            "semantic_branch": "noise_free_phase_flip_global_discrepancy",
            "opponent_branch": "matched_twenty_view_opponent_noise_CE_gradients",
            "target_used_for_generation": False,
            "whitebox_ensemble": False,
            "removal_delta_definition": "cos(mean_without_v,target)-cos(full_mean,target)",
            "positive_removal_delta_meaning": "view_is_harmful_to_aggregate_target_alignment",
            "production_processing": "gaussian_residual_applied_after_each_full_or_leave_one_out_mean",
            "topk": args.topk,
        },
        "semantic_diagnostics_mean": {
            key: sum(float(row[key]) for row in diagnostic_rows) / len(diagnostic_rows)
            for key in (
                "phase_global_cosine",
                "flip_global_cosine",
                "phase_flip_gradient_cosine",
            )
        },
        "marginal_mean": marginal_mean,
        "condition_mean": condition_mean,
        "paired_deltas": paired,
    }
    output = args.output_dir / args.source
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "semantic_diagnostics.csv", diagnostic_rows)
    write_csv(output / "view_marginal_metrics.csv", marginal_rows)
    write_csv(output / "correlation_metrics.csv", correlation_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    _validate(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run(args)


if __name__ == "__main__":
    main()
