#!/usr/bin/env python3
"""Compare the 1000-image random-selector control with the high baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


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
class SourceRun:
    display_name: str
    source_target: str
    high_csv: str
    random_csv: str


SOURCE_RUNS = (
    SourceRun(
        "ViT-B/16",
        "vit_base_patch16_224",
        "outputs/csv/outputs_attack_newconfig1000_vit_b3_b10_c10_10_s1000_offset0_seed20260907.csv",
        "outputs/csv/outputs_attack_randomselector1000_vit_adapter_defaults_s1000_seed20260907.csv",
    ),
    SourceRun(
        "CaiT-S24",
        "cait_s24_224",
        "outputs/csv/outputs_attack_newconfig1000_cait_b17_b23_c02_28_projection_s1000_offset0_seed20260907.csv",
        "outputs/csv/outputs_attack_randomselector1000_cait_adapter_defaults_s1000_seed20260907.csv",
    ),
    SourceRun(
        "PiT-B",
        "pit_b_224",
        "outputs/csv/outputs_attack_newconfig1000_pit_s2b1_s3b2_s3b3_c05_02_06_opp04_s1000_offset0_seed20260907.csv",
        "outputs/csv/outputs_attack_randomselector1000_pit_adapter_defaults_s1000_seed20260907.csv",
    ),
    SourceRun(
        "Visformer-S",
        "visformer_small",
        "outputs/csv/outputs_attack_newconfig1000_vis_s2b1_s3b1_c41_10_projection_opp04_s1000_offset0_seed20260907.csv",
        "outputs/csv/outputs_attack_randomselector1000_visformer_adapter_defaults_s1000_seed20260907.csv",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare complete 14-target random and high transfer ASR records."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/random_selector_vs_high_s1000.md"),
    )
    return parser.parse_args()


def latest_complete_row(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"transfer CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    complete = [
        row
        for row in rows
        if all(row.get(model, "").strip() for model in TARGET_MODELS)
        and row.get("adv_image_count", "").strip() == "1000"
    ]
    if not complete:
        raise ValueError(f"no complete 1000-image, 14-target row in {path}")
    return complete[-1]


def values(row: dict[str, str]) -> dict[str, float]:
    return {model: float(row[model]) for model in TARGET_MODELS}


def mean(items: list[float]) -> float:
    return sum(items) / len(items)


def summaries(model_values: dict[str, float], source_target: str) -> dict[str, float]:
    return {
        "Overall": mean([model_values[model] for model in TARGET_MODELS]),
        "Transformer": mean([model_values[model] for model in TRANSFORMER_MODELS]),
        "CNN": mean([model_values[model] for model in CNN_MODELS]),
        "Strict black-box": mean(
            [value for model, value in model_values.items() if model != source_target]
        ),
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def delta_pp(random_value: float, high_value: float) -> str:
    return f"{100.0 * (random_value - high_value):+.2f}pp"


def build_report(repo_root: Path) -> str:
    lines = [
        "# Random selector vs high selector: 1000-image transfer ASR",
        "",
        "ASR is `1 - adversarial accuracy` over all 1000 adversarial samples. "
        "Each summary uses the same 14 transfer targets; delta is random minus high.",
        "",
        "| Source | Metric | High | Random | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    detailed: list[str] = []
    for run in SOURCE_RUNS:
        high = values(latest_complete_row(repo_root / run.high_csv))
        random = values(latest_complete_row(repo_root / run.random_csv))
        high_summary = summaries(high, run.source_target)
        random_summary = summaries(random, run.source_target)
        for metric in ("Overall", "Transformer", "CNN", "Strict black-box"):
            lines.append(
                f"| {run.display_name} | {metric} | {percent(high_summary[metric])} | "
                f"{percent(random_summary[metric])} | "
                f"{delta_pp(random_summary[metric], high_summary[metric])} |"
            )

        detailed.extend(
            [
                "",
                f"## {run.display_name} per-target ASR",
                "",
                "| Target | High | Random | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for model in TARGET_MODELS:
            detailed.append(
                f"| {model} | {percent(high[model])} | {percent(random[model])} | "
                f"{delta_pp(random[model], high[model])} |"
            )
    return "\n".join(lines + detailed) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output = args.output.expanduser()
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(repo_root), encoding="utf-8")
    print(f"Wrote comparison report to {output}")


if __name__ == "__main__":
    main()
