"""Paired causal study of gradient amplitude and frequency components.

This runner keeps the canonical attack and its 20-view data augmentation fixed.
Only the gradient returned by the existing post-aggregation probe hook changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_gradient_study import (
    BLACKBOX_MODELS,
    DEFAULT_CNN_BLACK_BOX_MODELS,
    DEFAULT_VIT_BLACK_BOX_MODELS,
    architecture_delta,
    evaluate_per_sample,
    paired_comparison,
    run_attack,
    save_json,
)


DEFAULT_PROBES = (
    "amplitude_remove_low_q20",
    "amplitude_remove_high_q95",
    "amplitude_clip_high_q99",
    "coordinate_wiener_floor25",
    "coordinate_wiener_floor50",
    "frequency_high_gain50",
    "spectral_wiener_all_floor50",
    "spectral_wiener_high_floor00",
    "spectral_wiener_high_floor50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient component causal study")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--output-dir", default="outputs/attack/gradient_component_study_screen_s30"
    )
    parser.add_argument("--bootstrap-count", type=int, default=5000)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--probes", nargs="+", default=list(DEFAULT_PROBES))
    return parser.parse_args()


def complete_attack_dir(path: Path, num_samples: int) -> bool:
    return (
        (path / "replay_manifest.json").is_file()
        and (path / "attack_params.json").is_file()
        and len(list(path.glob("adv_*.png"))) == num_samples
    )


def load_or_evaluate(
    image_dir: Path,
    outcomes_path: Path,
    metrics_path: Path,
    reuse_existing: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if reuse_existing and outcomes_path.is_file() and metrics_path.is_file():
        return (
            json.loads(outcomes_path.read_text(encoding="utf-8")),
            json.loads(metrics_path.read_text(encoding="utf-8")),
        )
    outcomes, metrics = evaluate_per_sample(image_dir)
    save_json(outcomes_path, outcomes)
    save_json(metrics_path, metrics)
    return outcomes, metrics


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / "study_config.json", vars(args))

    baseline_dir = root / f"baseline_seed{args.seed}"
    if args.reuse_existing and complete_attack_dir(baseline_dir, args.num_samples):
        manifest = json.loads((baseline_dir / "replay_manifest.json").read_text(encoding="utf-8"))
    else:
        _, manifest = run_attack(
            output_dir=baseline_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            steps=args.steps,
            probe_name=None,
            baseline_sign_dir=None,
            expected_manifest=None,
        )
    baseline_outcomes, baseline_metrics = load_or_evaluate(
        baseline_dir,
        root / "baseline_outcomes.json",
        root / "baseline_metrics.json",
        args.reuse_existing,
    )

    results: dict[str, object] = {}
    results_path = root / "component_results.json"
    if args.reuse_existing and results_path.is_file():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    for probe_index, probe_name in enumerate(args.probes):
        probe_dir = root / f"probe_{probe_name}_seed{args.seed}"
        outcomes_path = root / f"outcomes_{probe_name}.json"
        metrics_path = root / f"metrics_{probe_name}.json"
        if not (args.reuse_existing and complete_attack_dir(probe_dir, args.num_samples)):
            run_attack(
                output_dir=probe_dir,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                seed=args.seed,
                steps=args.steps,
                probe_name=probe_name,
                baseline_sign_dir=baseline_dir,
                expected_manifest=manifest,
            )
        outcomes, metrics = load_or_evaluate(
            probe_dir, outcomes_path, metrics_path, args.reuse_existing
        )
        comparison = paired_comparison(
            baseline_outcomes,
            outcomes,
            args.bootstrap_count,
            args.seed + probe_index + 1,
        )
        vit_delta = architecture_delta(
            baseline_outcomes, outcomes, DEFAULT_VIT_BLACK_BOX_MODELS
        )
        cnn_delta = architecture_delta(
            baseline_outcomes, outcomes, DEFAULT_CNN_BLACK_BOX_MODELS[:4]
        )
        results[probe_name] = {
            "aggregate": metrics,
            "comparison_to_baseline": comparison,
            "vit_delta": vit_delta,
            "cnn_delta": cnn_delta,
            "whitebox_delta": metrics["whitebox"] - baseline_metrics["whitebox"],
            "replay_digest": manifest["event_digest"],
        }
        save_json(results_path, results)

    ranked = sorted(
        (
            {
                "probe": name,
                "overall_delta": result["comparison_to_baseline"]["delta_overall"],
                "vit_delta": result["vit_delta"],
                "cnn_delta": result["cnn_delta"],
                "screen_guard": result["cnn_delta"] >= -0.01
                and result["comparison_to_baseline"]["delta_overall"] >= -0.005,
            }
            for name, result in results.items()
        ),
        key=lambda row: (row["screen_guard"], row["vit_delta"], row["overall_delta"]),
        reverse=True,
    )
    save_json(root / "component_ranking.json", ranked)
    print(f"completed {len(results)} probes over {len(BLACKBOX_MODELS)} black-box models")


if __name__ == "__main__":
    main()
