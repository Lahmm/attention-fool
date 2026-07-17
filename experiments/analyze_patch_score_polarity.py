"""Cross-fitted analysis of patch-score routing polarity.

The mechanism experiment records repeated route outcomes for each image.  This
script uses only those records: a calibration half chooses whether the high
or low score tail is preferable (with score-deviation as a pre-specified
control), and the other half evaluates that choice against uniform random
routing.  No model or attack is rerun here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


CANDIDATES = (
    "high_score_extreme",
    "low_score_extreme",
    "score_deviation_extreme",
)
BASELINE = "random_uniform"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/research/patch_score_paired64"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/research/patch_score_polarity_crossfit"))
    parser.add_argument("--models", type=str, default="vit_base_patch16_224,cait_s24_224,pit_b_224,visformer_small")
    parser.add_argument("--splits", type=int, default=32)
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("sample_index", "repeat"):
            row[key] = int(row[key])
        for key in ("route_logit_drop", "route_loss_increase"):
            row[key] = float(row[key])
    return rows


def aggregate_by_image(rows: list[dict[str, object]]) -> dict[int, dict[str, dict[str, float]]]:
    grouped: dict[int, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        index = int(row["sample_index"])
        strategy = str(row["strategy"])
        grouped.setdefault(index, {}).setdefault(
            strategy, {"logit": [], "loss": []}
        )
        grouped[index][strategy]["logit"].append(float(row["route_logit_drop"]))
        grouped[index][strategy]["loss"].append(float(row["route_loss_increase"]))
    return {
        index: {
            strategy: {
                metric: float(np.mean(values))
                for metric, values in metrics.items()
            }
            for strategy, metrics in strategies.items()
        }
        for index, strategies in grouped.items()
    }


def mean_delta(
    data: dict[int, dict[str, dict[str, float]]],
    indices: list[int],
    strategy: str,
    metric: str,
) -> float:
    return float(np.mean([
        data[index][strategy][metric] - data[index][BASELINE][metric]
        for index in indices
    ]))


def ci95(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        return (float("nan"), float("nan"))
    half_width = 1.96 * float(array.std(ddof=1)) / np.sqrt(len(array))
    return (float(array.mean() - half_width), float(array.mean() + half_width))


def analyze_model(
    model: str,
    input_root: Path,
    output_root: Path,
    split_count: int,
    calibration_size: int,
    seed: int,
) -> dict[str, object]:
    rows = read_rows(input_root / model / "route_per_image_raw.csv")
    data = aggregate_by_image(rows)
    indices = sorted(data)
    if not indices:
        raise ValueError(f"no image rows found for {model}")
    if not all(BASELINE in data[index] for index in indices):
        raise ValueError(f"missing {BASELINE} rows for {model}")
    missing = [
        strategy for strategy in CANDIDATES
        if not all(strategy in data[index] for index in indices)
    ]
    if missing:
        raise ValueError(f"missing candidate strategies for {model}: {missing}")
    if not 1 <= calibration_size < len(indices):
        raise ValueError("calibration-size must be between 1 and the number of images minus 1")

    rng = random.Random(seed)
    split_rows: list[dict[str, object]] = []
    for split in range(split_count):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        calibration = shuffled[:calibration_size]
        evaluation = shuffled[calibration_size:]
        calibration_loss = {
            strategy: mean_delta(data, calibration, strategy, "loss")
            for strategy in CANDIDATES
        }
        selected = max(CANDIDATES, key=lambda strategy: calibration_loss[strategy])
        eval_loss = mean_delta(data, evaluation, selected, "loss")
        eval_logit = mean_delta(data, evaluation, selected, "logit")
        split_rows.append({
            "model": model,
            "split": split,
            "calibration_size": len(calibration),
            "evaluation_size": len(evaluation),
            "selected_strategy": selected,
            "calibration_loss_delta": calibration_loss[selected],
            "calibration_loss_delta_high": calibration_loss["high_score_extreme"],
            "calibration_loss_delta_low": calibration_loss["low_score_extreme"],
            "calibration_loss_delta_deviation": calibration_loss["score_deviation_extreme"],
            "evaluation_loss_delta_vs_random": eval_loss,
            "evaluation_logit_delta_vs_random": eval_logit,
        })

    selected_counts = {
        strategy: sum(row["selected_strategy"] == strategy for row in split_rows)
        for strategy in CANDIDATES
    }
    eval_loss_values = [float(row["evaluation_loss_delta_vs_random"]) for row in split_rows]
    eval_logit_values = [float(row["evaluation_logit_delta_vs_random"]) for row in split_rows]
    summary = {
        "model": model,
        "images": len(indices),
        "splits": split_count,
        "calibration_size": calibration_size,
        "candidate_strategies": list(CANDIDATES),
        "baseline_strategy": BASELINE,
        "selected_strategy_counts": selected_counts,
        "selected_strategy_frequency": {
            strategy: count / split_count for strategy, count in selected_counts.items()
        },
        "evaluation_loss_delta_mean": float(np.mean(eval_loss_values)),
        "evaluation_loss_delta_ci95": ci95(eval_loss_values),
        "evaluation_loss_delta_positive_fraction": float(np.mean(np.asarray(eval_loss_values) > 0)),
        "evaluation_logit_delta_mean": float(np.mean(eval_logit_values)),
        "evaluation_logit_delta_ci95": ci95(eval_logit_values),
        "evaluation_logit_delta_positive_fraction": float(np.mean(np.asarray(eval_logit_values) > 0)),
    }
    model_output = output_root / model
    model_output.mkdir(parents=True, exist_ok=True)
    with (model_output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (model_output / "split_raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)
    return summary


def main() -> None:
    args = parse_args()
    if args.splits <= 0:
        raise ValueError("splits must be positive")
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    summaries = [
        analyze_model(
            model=model,
            input_root=args.input_root,
            output_root=args.output_root,
            split_count=args.splits,
            calibration_size=args.calibration_size,
            seed=args.seed + model_index,
        )
        for model_index, model in enumerate(models)
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
