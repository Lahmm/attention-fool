#!/usr/bin/env python3
"""Summarize the paired high-vs-random selector isolation experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SEEDS = (20260921, 20260922, 20260923)
SELECTORS = ("high", "random")
TARGET_MODELS = (
    "vit_base_patch16_224",
    "levit_256",
    "pit_b_224",
    "deit_base_patch16_224",
    "tnt_s_patch16_224",
    "convit_base",
    "visformer_small",
    "cait_s24_224",
    "inception_v3",
    "inception_v4",
    "inception_resnet_v2",
    "resnet101",
    "inception_v3_adv",
    "inception_resnet_v2_adv",
)


@dataclass(frozen=True)
class Source:
    slug: str
    display_name: str
    model: str


SOURCES = (
    Source("vit", "ViT-B/16", "vit_base_patch16_224"),
    Source("cait", "CaiT-S24", "cait_s24_224"),
    Source("pit", "PiT-B", "pit_b_224"),
    Source("visformer", "Visformer-S", "visformer_small"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/selector_isolation_s500_3seeds.md"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    return parser.parse_args()


def run_stem(source: Source, selector: str, seed: int) -> str:
    return f"selectorisolation500_{source.slug}_{selector}_s500_seed{seed}"


def load_success(
    repo_root: Path, source: Source, selector: str, seed: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    attack_dir = repo_root / "outputs" / "attack" / run_stem(source, selector, seed)
    params = json.loads((attack_dir / "attack_params.json").read_text(encoding="utf-8"))
    expected = {
        "whitebox_model": source.model,
        "progressive_patch_selector": selector,
        "score_global_noise_strength": 0.0,
        "opponent_noise_strength": 0.0,
        "gaussian_alpha": 0.0,
        "max_attacked_samples": 500,
        "seed": seed,
    }
    mismatches = {
        key: (params.get(key), value)
        for key, value in expected.items()
        if params.get(key) != value
    }
    if mismatches:
        raise ValueError(f"attack metadata mismatch in {attack_dir}: {mismatches}")

    artifact = json.loads(
        (attack_dir / "transfer_predictions.json").read_text(encoding="utf-8")
    )
    sample_ids = [str(value) for value in artifact["sample_ids"]]
    labels = np.asarray(artifact["labels"], dtype=np.int64)
    if len(sample_ids) != 500 or labels.shape != (500,):
        raise ValueError(f"expected 500 ordered predictions in {attack_dir}")
    predictions = artifact["predictions"]
    if set(predictions) != set(TARGET_MODELS):
        raise ValueError(f"incomplete target predictions in {attack_dir}")
    predicted = np.stack(
        [np.asarray(predictions[model], dtype=np.int64) for model in TARGET_MODELS],
        axis=1,
    )
    if predicted.shape != (500, len(TARGET_MODELS)):
        raise ValueError(f"prediction shape mismatch in {attack_dir}")
    success = predicted != labels[:, None]
    return sample_ids, labels, success


def bootstrap_delta(
    differences: np.ndarray, samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Two-level bootstrap over seeds and paired images."""
    seed_count, image_count, _ = differences.shape
    per_image_difference = differences.mean(axis=2)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_means = []
        for seed_index in selected_seeds:
            selected_images = rng.integers(0, image_count, size=image_count)
            seed_means.append(
                per_image_difference[seed_index, selected_images].mean()
            )
        values[index] = float(np.mean(seed_means))
    return values


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = repo_root / output
    rng = np.random.default_rng(20260921)

    lines = [
        "# High vs random selector isolation: 500 images × 3 seeds",
        "",
        "Opponent noise, global-score noise, and Gaussian residual are disabled. "
        "ASR is `1 - adversarial accuracy` over every evaluated adversarial sample. "
        "The primary metric is the strict black-box mean, excluding the source-matched target.",
        "",
        "| Source | High strict ASR | Random strict ASR | High−random | 95% paired CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    detail: list[str] = []
    source_bootstraps: list[np.ndarray] = []
    source_overall: list[tuple[float, float, float]] = []

    for source in SOURCES:
        by_selector: dict[str, np.ndarray] = {}
        reference_ids: list[str] | None = None
        reference_labels: np.ndarray | None = None
        for selector in SELECTORS:
            seed_success = []
            for seed in SEEDS:
                sample_ids, labels, success = load_success(
                    repo_root, source, selector, seed
                )
                if reference_ids is None:
                    reference_ids = sample_ids
                    reference_labels = labels
                elif sample_ids != reference_ids or not np.array_equal(
                    labels, reference_labels
                ):
                    raise ValueError(
                        f"paired sample order mismatch for {source.display_name}"
                    )
                seed_success.append(success)
            by_selector[selector] = np.stack(seed_success)

        strict_indices = [
            index for index, model in enumerate(TARGET_MODELS) if model != source.model
        ]
        high_strict = by_selector["high"][:, :, strict_indices]
        random_strict = by_selector["random"][:, :, strict_indices]
        differences = high_strict.astype(np.int8) - random_strict.astype(np.int8)
        bootstrap = bootstrap_delta(differences, args.bootstrap_samples, rng)
        source_bootstraps.append(bootstrap)
        high_mean = float(high_strict.mean())
        random_mean = float(random_strict.mean())
        delta = high_mean - random_mean
        lower, upper = np.quantile(bootstrap, (0.025, 0.975))
        lines.append(
            f"| {source.display_name} | {percent(high_mean)} | {percent(random_mean)} | "
            f"{pp(delta)} | [{pp(float(lower))}, {pp(float(upper))}] |"
        )

        high_overall = float(by_selector["high"].mean())
        random_overall = float(by_selector["random"].mean())
        source_overall.append(
            (high_overall, random_overall, high_overall - random_overall)
        )
        detail.extend(
            [
                "",
                f"## {source.display_name}",
                "",
                "### Per-seed strict black-box ASR",
                "",
                "| Seed | High | Random | Delta |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for seed_index, seed in enumerate(SEEDS):
            high_seed = float(high_strict[seed_index].mean())
            random_seed = float(random_strict[seed_index].mean())
            detail.append(
                f"| {seed} | {percent(high_seed)} | {percent(random_seed)} | "
                f"{pp(high_seed - random_seed)} |"
            )

        detail.extend(
            [
                "",
                "### Per-target ASR averaged over seeds",
                "",
                "| Target | High | Random | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for target_index, target in enumerate(TARGET_MODELS):
            high_target = float(by_selector["high"][:, :, target_index].mean())
            random_target = float(by_selector["random"][:, :, target_index].mean())
            detail.append(
                f"| {target} | {percent(high_target)} | {percent(random_target)} | "
                f"{pp(high_target - random_target)} |"
            )

    pooled_bootstrap = np.stack(source_bootstraps).mean(axis=0)
    pooled_lower, pooled_upper = np.quantile(pooled_bootstrap, (0.025, 0.975))
    pooled_high = float(np.mean([item[0] for item in source_overall]))
    pooled_random = float(np.mean([item[1] for item in source_overall]))
    pooled_delta = pooled_high - pooled_random
    lines.extend(
        [
            "",
            "## Cross-source summary",
            "",
            f"Complete 14-target mean: high {percent(pooled_high)}, random "
            f"{percent(pooled_random)}, delta {pp(pooled_delta)}.",
            "",
            "Strict black-box delta 95% paired bootstrap CI across the four fixed "
            f"source architectures: [{pp(float(pooled_lower))}, "
            f"{pp(float(pooled_upper))}].",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines + detail) + "\n", encoding="utf-8")
    print(f"Wrote isolation report to {output}")


if __name__ == "__main__":
    main()
