"""Build and summarize the frozen 500-image selector-only attack suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets import WHITEBOX_MODEL_CHOICES
from routing_config import FrozenRoutingConfig, file_sha256
from transfer_eval import DEFAULT_BLACK_BOX_MODELS


CONDITIONS = (
    "selected",
    "opposite",
    "deviation",
    "random",
    "no_drop",
    "final_layer",
    "gradcam_relu",
)
RESULT_COLUMNS = {
    "source_model",
    "target_model",
    "condition",
    "image_name",
    "clean_correct",
    "adv_correct",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-manifest", type=Path)
    group.add_argument("--results", type=Path)
    parser.add_argument("--routing-config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/research/patch_score_selector_suite/summary.json"),
    )
    return parser.parse_args()


def condition_command(
    source: str,
    condition: str,
    config: FrozenRoutingConfig,
    config_path: Path,
    samples: int,
    sample_offset: int,
    seed: int,
):
    layer = config.layer_for(source)
    polarity = config.global_polarity
    opposite = "low" if polarity == "high" else "high"
    output_dir = Path("outputs/attack/selector_suite") / source / condition
    command = [
        "python",
        "main.py",
        "--whitebox-model",
        source,
        "--max-attacked-samples",
        str(samples),
        "--sample-offset",
        str(sample_offset),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    if condition == "opposite":
        command += [
            "--patch-score-layer",
            layer,
            "--patch-dropout-score-mode",
            opposite,
            "--patch-selector",
            "patch_score",
        ]
    elif condition == "final_layer":
        command += [
            "--patch-score-layer",
            "final",
            "--patch-dropout-score-mode",
            polarity,
            "--patch-selector",
            "patch_score",
        ]
    else:
        selector = {
            "selected": "patch_score",
            "deviation": "deviation",
            "random": "random",
            "no_drop": "no_drop",
            "gradcam_relu": "gradcam_relu",
        }[condition]
        command += [
            "--routing-config",
            str(config_path),
            "--patch-selector",
            selector,
        ]
        if condition == "gradcam_relu":
            command += ["--gradcam-target-mode", "true"]
    return {
        "source_model": source,
        "condition": condition,
        "layer": "final" if condition == "final_layer" else layer,
        "polarity": opposite if condition == "opposite" else polarity,
        "selector": command[command.index("--patch-selector") + 1],
        "attack_output_dir": str(output_dir),
        "attack_command": command,
    }


def build_manifest(
    config: FrozenRoutingConfig,
    config_path: Path,
    *,
    samples: int,
    sample_offset: int,
    seed: int,
):
    if samples <= 0 or sample_offset < 0:
        raise ValueError("samples must be positive and sample_offset non-negative.")
    jobs = [
        condition_command(
            source, condition, config, config_path, samples, sample_offset, seed
        )
        for source in WHITEBOX_MODEL_CHOICES
        for condition in CONDITIONS
    ]
    return {
        "schema_version": 1,
        "protocol": {
            "samples": samples,
            "sample_offset": sample_offset,
            "seed": seed,
            "routing_config": str(config_path),
            "routing_config_sha256": file_sha256(config_path),
            "global_polarity": config.global_polarity,
            "target_models": list(DEFAULT_BLACK_BOX_MODELS),
            "epsilon": 16.0 / 255.0,
            "steps": 10,
            "views": 20,
            "phase_pairs": True,
            "opponent_noise": "initial_projection_kept_only_strength_0.2",
            "gaussian_residual": {"sigma": 4.0, "alpha": 0.75},
        },
        "jobs": jobs,
    }


def read_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def read_results(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not RESULT_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"results require columns {sorted(RESULT_COLUMNS)}")
        for row in reader:
            condition = str(row["condition"])
            if condition not in CONDITIONS:
                raise ValueError(f"unknown condition in results: {condition!r}")
            rows.append(
                {
                    "source_model": str(row["source_model"]),
                    "target_model": str(row["target_model"]),
                    "condition": condition,
                    "image_name": str(row["image_name"]),
                    "clean_correct": read_bool(str(row["clean_correct"])),
                    "adv_correct": read_bool(str(row["adv_correct"])),
                }
            )
    if not rows:
        raise ValueError("results CSV is empty.")
    return rows


def bootstrap_difference(
    selected: dict[str, float],
    gradcam: dict[str, float],
    *,
    repeats: int,
    seed: int,
):
    common = sorted(set(selected) & set(gradcam))
    if not common:
        raise ValueError("selected and Grad-CAM have no paired clean-correct images.")
    differences = [selected[key] - gradcam[key] for key in common]
    point = sum(differences) / len(differences)
    generator = random.Random(seed)
    values = []
    for _ in range(repeats):
        values.append(
            sum(differences[generator.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return {"difference": point, "ci95": [lower, upper], "paired_count": len(common)}


def summarize(rows, *, bootstrap_repeats: int, seed: int):
    if bootstrap_repeats <= 0:
        raise ValueError("bootstrap_repeats must be positive.")
    results = {}
    per_image_macro: dict[tuple[str, str], dict[str, list[float]]] = {}
    for source in WHITEBOX_MODEL_CHOICES:
        results[source] = {}
        for target in DEFAULT_BLACK_BOX_MODELS:
            results[source][target] = {}
            for condition in CONDITIONS:
                selected = [
                    row
                    for row in rows
                    if row["source_model"] == source
                    and row["target_model"] == target
                    and row["condition"] == condition
                    and row["clean_correct"]
                ]
                if not selected:
                    continue
                success = [float(not row["adv_correct"]) for row in selected]
                results[source][target][condition] = {
                    "asr": sum(success) / len(success),
                    "clean_correct_count": len(success),
                }
                for row, value in zip(selected, success):
                    per_image_macro.setdefault((source, condition), {}).setdefault(
                        row["image_name"], []
                    ).append(value)
    macro = {}
    noninferiority = {}
    for source in WHITEBOX_MODEL_CHOICES:
        macro[source] = {}
        for condition in CONDITIONS:
            image_values = per_image_macro.get((source, condition), {})
            averaged = {
                image: sum(values) / len(values) for image, values in image_values.items()
            }
            if averaged:
                macro[source][condition] = sum(averaged.values()) / len(averaged)
        selected = {
            image: sum(values) / len(values)
            for image, values in per_image_macro.get((source, "selected"), {}).items()
        }
        gradcam = {
            image: sum(values) / len(values)
            for image, values in per_image_macro.get((source, "gradcam_relu"), {}).items()
        }
        if selected and gradcam:
            comparison = bootstrap_difference(
                selected, gradcam, repeats=bootstrap_repeats, seed=seed
            )
            comparison["noninferior_at_1pp"] = comparison["ci95"][0] > -0.01
            noninferiority[source] = comparison
    return {
        "per_target": results,
        "per_source_macro": macro,
        "patch_score_vs_gradcam": noninferiority,
        "asr_denominator": "target-clean-correct images only",
        "noninferiority_margin": -0.01,
    }


def main() -> None:
    args = parse_args()
    config = FrozenRoutingConfig.load(args.routing_config)
    if args.write_manifest is not None:
        manifest = build_manifest(
            config,
            args.routing_config,
            samples=args.samples,
            sample_offset=args.sample_offset,
            seed=args.seed,
        )
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"wrote {len(manifest['jobs'])} selector attack jobs to {args.write_manifest}")
        return
    rows = read_results(args.results)
    summary = summarize(
        rows, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed
    )
    summary["routing"] = config.to_dict()
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
