"""E5-forward: semantic-equivalent route coverage across architectures.

Each model is measured independently.  This retained experiment describes
cross-model forward semantics and does not optimize or aggregate gradients.
"""

from __future__ import annotations

import argparse
import gc
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
from experiments.semantic_forward_utils import (
    common_map,
    load_samples,
    normalize,
    row_spearman,
    write_csv,
    write_json,
)
from gradient_replay import GradientReplay
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "clean_resized_images"
DEFAULT_ANNOTATIONS = REPO_ROOT / "data" / "image_name_to_class_id_and_name.json"
PHASES = ((4, 4), (8, 8), (12, 12))
FORWARD_CONDITIONS = ("opponent", "rgb_gaussian", "feature_gaussian", "phase_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--sample-offset", type=int, default=628)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--noise-strength", type=float, default=0.2)
    parser.add_argument("--preserve-global-scale", type=float, default=0.02)
    parser.add_argument("--preserve-js-scale", type=float, default=0.01)
    parser.add_argument("--common-grid", type=int, default=7)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/semantic_equivalent_route_e5_forward"),
    )
    return parser.parse_args()


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    midpoint = 0.5 * (log_a.exp() + log_b.exp())
    return 0.5 * (
        F.kl_div(log_a, midpoint, reduction="none").sum(dim=1)
        + F.kl_div(log_b, midpoint, reduction="none").sum(dim=1)
    )


def feature_score(local: torch.Tensor, global_token: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(local, global_token.expand_as(local), dim=-1)


def phase_shift(tensor: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return tensor
    padded = F.pad(
        tensor,
        (max(0, dx), max(0, -dx), max(0, dy), max(0, -dy)),
        mode="reflect",
    )
    start_y, start_x = max(0, -dy), max(0, -dx)
    return padded[
        ..., start_y : start_y + tensor.size(-2), start_x : start_x + tensor.size(-1)
    ]


def align_score_map(
    score: torch.Tensor,
    grid: tuple[int, int],
    image_size: tuple[int, int],
    phase: tuple[int, int],
    common_grid_size: int,
) -> torch.Tensor:
    maps = score.reshape(score.size(0), 1, *grid)
    pixels = F.interpolate(maps, size=image_size, mode="bilinear", align_corners=False)
    if phase != (0, 0):
        pixels = phase_shift(pixels, -phase[0], -phase[1])
    return F.interpolate(
        pixels, size=(common_grid_size, common_grid_size), mode="area"
    ).flatten(1)


def final_from_local(
    model_name: str,
    model,
    state,
    local_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Run the model tail from its initial RGB-projected feature state."""
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


def make_noise_adapter(model, args: argparse.Namespace) -> PatchScoreAttacker:
    """Reuse the production projection/noise implementation without attacking."""
    return PatchScoreAttacker(
        model,
        attack_method="none",
        input_diversity_groups=args.groups,
        input_diversity_views_per_group=2,
        post_dropout_feature_noise_strength=args.noise_strength,
        post_dropout_feature_noise_type="opponent_projected",
        device=model.device,
    )


def projected_rgb_gaussian(
    adapter: PatchScoreAttacker,
    state,
    event: str,
) -> torch.Tensor:
    batch, count, dimension = state.local_tokens.shape
    kh, kw = state.projection_kernel
    coefficients = adapter._randn_like(
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
    raw = coefficients.flatten(2).matmul(projection.reshape(dimension, -1).t())
    return adapter._match_feature_noise_rms(state.local_tokens, raw, event)


def noise_for_kind(
    adapter: PatchScoreAttacker,
    state,
    kind: str,
) -> torch.Tensor:
    if kind == "opponent":
        return adapter._strict_opponent_feature_noise(state)
    if kind == "feature_gaussian":
        raw = adapter._randn_like(state.local_tokens, "e5_feature_gaussian")
        return adapter._match_feature_noise_rms(state.local_tokens, raw, kind)
    if kind == "rgb_gaussian":
        return projected_rgb_gaussian(adapter, state, "e5_rgb_gaussian")
    if kind == "none":
        return torch.zeros_like(state.local_tokens)
    raise ValueError(f"unknown noise kind {kind}")


def base_and_view_metadata(
    model_name: str,
    model,
    adapter: PatchScoreAttacker,
    pixels: torch.Tensor,
    noise_kind: str,
    sample_ids: list[str],
    seed: int,
    groups: int,
    common_grid_size: int,
) -> dict[str, torch.Tensor]:
    replay = GradientReplay(seed)
    replay.begin_batch(sample_ids)
    adapter._gradient_replay = replay
    try:
        with torch.no_grad():
            base_state = model.prepare_attack_feature_state(normalize(model, pixels))
            base_local, base_global, base_logits, base_grid = final_from_local(
                model_name, model, base_state, base_state.local_tokens
            )
            base_score = feature_score(base_local, base_global)
            base_common = common_map(base_score, base_grid, common_grid_size)
            view_scores, view_globals, view_logits = [], [], []
            for group in range(groups):
                phase = PHASES[group % len(PHASES)]
                for view in range(2):
                    replay.set_context(step=0, group=group, view=view)
                    shifted_pixels = phase_shift(pixels, *phase) if view == 1 else pixels
                    state = model.prepare_attack_feature_state(
                        normalize(model, shifted_pixels)
                    )
                    local = state.local_tokens + noise_for_kind(
                        adapter, state, noise_kind
                    )
                    local_features, global_token, logits, grid = final_from_local(
                        model_name, model, state, local
                    )
                    score = feature_score(local_features, global_token)
                    view_scores.append(
                        align_score_map(
                            score,
                            grid,
                            tuple(pixels.shape[-2:]),
                            phase if view == 1 else (0, 0),
                            common_grid_size,
                        )
                    )
                    view_globals.append(global_token.flatten(1))
                    view_logits.append(logits)
            scores = torch.stack(view_scores)
            globals_ = torch.stack(view_globals)
            logits_ = torch.stack(view_logits)
            base_global_flat = base_global.flatten(1)
            global_cos = F.cosine_similarity(
                globals_, base_global_flat.unsqueeze(0).expand_as(globals_), dim=-1
            )
            flat_logits = logits_.reshape(-1, logits_.size(-1))
            expanded_base_logits = base_logits.unsqueeze(0).expand(
                logits_.size(0), -1, -1
            ).reshape(-1, base_logits.size(-1))
            js = js_divergence(flat_logits, expanded_base_logits).reshape(
                logits_.size(0), logits_.size(1)
            )
            route_spearman = torch.stack(
                [row_spearman(view, base_common) for view in scores]
            )
            view_count = scores.size(0)
            route_pair = torch.stack(
                [
                    row_spearman(scores[left], scores[right])
                    for left in range(view_count)
                    for right in range(view_count)
                ]
            ).reshape(view_count, view_count, scores.size(1))
            return {
                "global_cos": global_cos.detach(),
                "js": js.detach(),
                "route_spearman": route_spearman.detach(),
                "route_distance": (1.0 - route_pair).detach(),
            }
    finally:
        adapter._gradient_replay = None


def run_forward(args: argparse.Namespace) -> None:
    names, pixels_cpu, _labels_cpu = load_samples(
        args.image_dir, args.annotations, args.sample_offset, args.samples
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    for model_name in WHITEBOX_MODEL_CHOICES:
        model = build_whitebox_model(
            1000, model_name, pretrained=True, device=device
        ).eval()
        adapter = make_noise_adapter(model, args)
        for start in range(0, args.samples, args.batch_size):
            end = min(args.samples, start + args.batch_size)
            pixels = pixels_cpu[start:end].to(device)
            for condition in FORWARD_CONDITIONS:
                noise_kind = "none" if condition == "phase_only" else condition
                metadata = base_and_view_metadata(
                    model_name,
                    model,
                    adapter,
                    pixels,
                    noise_kind,
                    names[start:end],
                    args.seed,
                    args.groups,
                    args.common_grid,
                )
                preserved = (metadata["global_cos"] >= 0.98) & (
                    metadata["js"] <= 0.02
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
                    top_pair_mask = torch.zeros_like(keep)
                    top_pair_mask[top] = True
                    top_pair_values = pair[
                        top_pair_mask[:, None] & top_pair_mask[None, :]
                    ]
                    rows.append(
                        {
                            "model": model_name,
                            "condition": condition,
                            "sample_index": start + index,
                            "image_name": names[start + index],
                            "preserved_view_fraction": float(keep.float().mean()),
                            "preserved_route_distance": float(route_dist[keep].mean())
                            if keep.any()
                            else 0.0,
                            # Historical artifact compatibility: the completed
                            # E5-forward run recorded the top-preserved pair
                            # value under both pair-distance column names.
                            "preserved_pair_distance": float(top_pair_values.mean())
                            if top_pair_values.numel()
                            else 0.0,
                            "top_preserved_global_cosine": float(
                                metadata["global_cos"][top, index].mean()
                            ),
                            "top_preserved_js": float(metadata["js"][top, index].mean()),
                            "top_route_distance": float(route_dist[top].mean()),
                            "top_pair_distance": float(top_pair_values.mean())
                            if top_pair_values.numel()
                            else 0.0,
                            "all_view_global_cosine": float(
                                metadata["global_cos"][:, index].mean()
                            ),
                            "all_view_js": float(metadata["js"][:, index].mean()),
                            "all_view_route_distance": float(route_dist.mean()),
                            "route_view_count": int(keep.sum()),
                        }
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    write_csv(args.output_dir / "forward_metrics.csv", rows)
    summary: dict[str, object] = {
        "protocol": {
            "mode": "forward",
            "models": list(WHITEBOX_MODEL_CHOICES),
            "samples": args.samples,
            "sample_offset": args.sample_offset,
            "views": args.groups * 2,
            "common_grid": [args.common_grid, args.common_grid],
            "preserved_rule": "global cosine >= 0.98 and JS <= 0.02",
            "no_whitebox_ensemble": True,
            "noise_strength": args.noise_strength,
        },
        "mean": {},
    }
    metric_names = (
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
    means = summary["mean"]
    assert isinstance(means, dict)
    for model_name in WHITEBOX_MODEL_CHOICES:
        means[model_name] = {}
        for condition in FORWARD_CONDITIONS:
            selected = [
                row
                for row in rows
                if row["model"] == model_name and row["condition"] == condition
            ]
            means[model_name][condition] = {
                key: mean(selected, key) for key in metric_names
            }
    write_json(args.output_dir / "forward_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if (
        args.samples <= 1
        or args.batch_size <= 0
        or args.groups != 10
        or args.common_grid <= 1
    ):
        raise ValueError(
            "samples > 1, positive batch size, groups=10, and common-grid > 1 are required"
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_forward(args)


if __name__ == "__main__":
    main()
