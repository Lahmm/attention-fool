#!/usr/bin/env python3
"""Summarize the complete selector/global-score-noise transfer matrix."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


SEED = 20260920
SCORE_SELECTORS = ("high", "low", "extreme-high", "extreme-low")
TRANSFORMER_MODELS = (
    "vit_base_patch16_224",
    "levit_256",
    "pit_b_224",
    "deit_base_patch16_224",
    "tnt_s_patch16_224",
    "convit_base",
    "visformer_small",
    "cait_s24_224",
)
CNN_MODELS = (
    "inception_v3",
    "inception_v4",
    "inception_resnet_v2",
    "resnet101",
    "inception_v3_adv",
    "inception_resnet_v2_adv",
)
TARGET_MODELS = TRANSFORMER_MODELS + CNN_MODELS


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
    parser = argparse.ArgumentParser(
        description="Summarize the 1000-image selector/global-noise ASR matrix."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/reports/selector_global_noise_matrix_s1000_seed20260920.md"
        ),
    )
    return parser.parse_args()


def run_stem(source: Source, selector: str, noise_label: str) -> str:
    return (
        f"selectorgnoise1000_{source.slug}_{selector}_gnoise{noise_label}_"
        f"adapter_defaults_s1000_seed{SEED}"
    )


def latest_complete_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"transfer CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row.get("adv_image_count", "").strip() == "1000"
        and all(row.get(model, "").strip() for model in TARGET_MODELS)
    ]
    if not rows:
        raise ValueError(f"no complete 1000-image, 14-target row in {path}")
    return rows[-1]


def load_run(
    repo_root: Path, source: Source, selector: str, noise_label: str
) -> dict[str, float]:
    stem = run_stem(source, selector, noise_label)
    attack_dir = repo_root / "outputs" / "attack" / stem
    params_path = attack_dir / "attack_params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"attack metadata not found: {params_path}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    expected_strength = 0.2 if noise_label == "on" else 0.0
    expected = {
        "whitebox_model": source.model,
        "progressive_patch_selector": selector,
        "score_global_noise_strength": expected_strength,
        "seed": SEED,
        "max_attacked_samples": 1000,
    }
    mismatches = {
        key: (params.get(key), value)
        for key, value in expected.items()
        if params.get(key) != value
    }
    if mismatches:
        raise ValueError(f"attack metadata mismatch in {params_path}: {mismatches}")

    csv_path = repo_root / "outputs" / "csv" / f"outputs_attack_{stem}.csv"
    row = latest_complete_row(csv_path)
    return {model: float(row[model]) for model in TARGET_MODELS}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def summaries(values: dict[str, float], source_model: str) -> dict[str, float]:
    return {
        "Overall": mean([values[model] for model in TARGET_MODELS]),
        "Transformer": mean([values[model] for model in TRANSFORMER_MODELS]),
        "CNN": mean([values[model] for model in CNN_MODELS]),
        "Strict black-box": mean(
            [value for model, value in values.items() if model != source_model]
        ),
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def delta_pp(on_value: float, off_value: float) -> str:
    return f"{100.0 * (on_value - off_value):+.2f}pp"


def build_report(repo_root: Path) -> str:
    lines = [
        "# Selector × global-score-noise: 1000-image transfer ASR",
        "",
        f"Seed: `{SEED}`. ASR is `1 - adversarial accuracy` over all 1000 "
        "adversarial samples and the same 14 transfer targets. Delta is global "
        "noise on minus off; on uses strength 0.2. Random routing is evaluated "
        "once because it does not read the global score token.",
        "",
        "| Source | Selector | Metric | Noise off / standalone | Noise on | Delta |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    detail: list[str] = []
    for source in SOURCES:
        for selector in SCORE_SELECTORS:
            off = load_run(repo_root, source, selector, "off")
            on = load_run(repo_root, source, selector, "on")
            off_summary = summaries(off, source.model)
            on_summary = summaries(on, source.model)
            for metric in ("Overall", "Transformer", "CNN", "Strict black-box"):
                lines.append(
                    f"| {source.display_name} | {selector} | {metric} | "
                    f"{percent(off_summary[metric])} | {percent(on_summary[metric])} | "
                    f"{delta_pp(on_summary[metric], off_summary[metric])} |"
                )

            detail.extend(
                [
                    "",
                    f"## {source.display_name} / {selector}",
                    "",
                    "| Target | Noise off | Noise on | Delta |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for model in TARGET_MODELS:
                detail.append(
                    f"| {model} | {percent(off[model])} | {percent(on[model])} | "
                    f"{delta_pp(on[model], off[model])} |"
                )
        random = load_run(repo_root, source, "random", "ignored")
        random_summary = summaries(random, source.model)
        for metric in ("Overall", "Transformer", "CNN", "Strict black-box"):
            lines.append(
                f"| {source.display_name} | random | {metric} | "
                f"{percent(random_summary[metric])} | — | — |"
            )
        detail.extend(
            [
                "",
                f"## {source.display_name} / random",
                "",
                "| Target | Standalone ASR |",
                "| --- | ---: |",
            ]
        )
        for model in TARGET_MODELS:
            detail.append(f"| {model} | {percent(random[model])} |")
    return "\n".join(lines + detail) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(repo_root), encoding="utf-8")
    print(f"Wrote matrix report to {output}")


if __name__ == "__main__":
    main()
