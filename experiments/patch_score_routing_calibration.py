"""Freeze one global patch-score polarity and one layer per source model.

The expensive attacks are deliberately decoupled from selection.  Use
``--write-template`` to create the complete pre-registered result table, fill
its ``asr`` column with 128-image calibration results, then pass it back with
``--results``.  Selection uses only off-diagonal transfer among the other
three registered white-box architectures.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nets import PATCH_SCORE_LAYER_CANDIDATES, WHITEBOX_MODEL_CHOICES
from routing_config import FrozenRoutingConfig


POLARITIES = ("high", "low")
REQUIRED_COLUMNS = {"source_model", "target_model", "polarity", "layer", "asr"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write-template", type=Path)
    group.add_argument("--results", type=Path)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("outputs/research/patch_score_routing_calibration/frozen_routing.json"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/research/patch_score_routing_calibration/summary.json"),
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=500,
        help="Default keeps calibration disjoint from the first 500 final-test images.",
    )
    parser.add_argument("--sample-seed", type=int, default=20260717)
    parser.add_argument("--attack-seed", type=int, default=20260716)
    parser.add_argument(
        "--attack-output-root",
        type=Path,
        default=Path("outputs/attack/routing_calibration"),
        help="Root for generated adversarial images; use a new root for a new protocol.",
    )
    parser.add_argument(
        "--image-ids-sha256",
        help="Required with --results; copy sample_ids_sha256 from a calibration replay manifest.",
    )
    return parser.parse_args()


def template_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in WHITEBOX_MODEL_CHOICES:
        for polarity in POLARITIES:
            for layer in PATCH_SCORE_LAYER_CANDIDATES[source]:
                for target in WHITEBOX_MODEL_CHOICES:
                    if target == source:
                        continue
                    rows.append(
                        {
                            "source_model": source,
                            "target_model": target,
                            "polarity": polarity,
                            "layer": layer,
                            "asr": "",
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    *,
    samples: int,
    sample_offset: int,
    attack_seed: int,
    attack_output_root: Path = Path("outputs/attack/routing_calibration"),
) -> dict[str, object]:
    jobs = []
    for source in WHITEBOX_MODEL_CHOICES:
        for polarity in POLARITIES:
            for layer in PATCH_SCORE_LAYER_CANDIDATES[source]:
                output_dir = attack_output_root / source / polarity / layer
                attack_command = [
                    "python",
                    "main.py",
                    "--whitebox-model",
                    source,
                    "--max-attacked-samples",
                    str(samples),
                    "--sample-offset",
                    str(sample_offset),
                    "--seed",
                    str(attack_seed),
                    "--patch-score-layer",
                    layer,
                    "--patch-dropout-score-mode",
                    polarity,
                    "--patch-selector",
                    "patch_score",
                    "--output-dir",
                    str(output_dir),
                ]
                eval_commands = {
                    target: [
                        "python",
                        "transfer_eval.py",
                        "--image-dir",
                        str(output_dir),
                        "--prefix",
                        "adv_",
                        "--model-name",
                        target,
                        "--no-record",
                    ]
                    for target in WHITEBOX_MODEL_CHOICES
                    if target != source
                }
                jobs.append(
                    {
                        "source_model": source,
                        "polarity": polarity,
                        "layer": layer,
                        "attack_output_dir": str(output_dir),
                        "attack_command": attack_command,
                        "eval_commands": eval_commands,
                    }
                )
    return {
        "protocol": {
            "samples": samples,
            "sample_offset": sample_offset,
            "attack_seed": attack_seed,
            "selectors": ["patch_score"],
            "target_models": list(WHITEBOX_MODEL_CHOICES),
            "patch_mask_policy": "clean_fixed_per_attack",
            "patch_mask_reference": "clean_pixels",
            "token_score_cls_noise": True,
        },
        "jobs": jobs,
    }


def read_results(path: Path) -> dict[tuple[str, str, str, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            raise ValueError(f"results CSV requires columns {sorted(REQUIRED_COLUMNS)}.")
        results: dict[tuple[str, str, str, str], float] = {}
        for line_number, row in enumerate(reader, start=2):
            key = (
                str(row["source_model"]),
                str(row["target_model"]),
                str(row["polarity"]),
                str(row["layer"]),
            )
            if key in results:
                raise ValueError(f"duplicate calibration row at line {line_number}: {key}")
            try:
                value = float(row["asr"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid ASR at line {line_number}: {row['asr']!r}") from exc
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"ASR must be in [0,1] at line {line_number}.")
            results[key] = value
    expected = {
        (
            str(row["source_model"]),
            str(row["target_model"]),
            str(row["polarity"]),
            str(row["layer"]),
        )
        for row in template_rows()
    }
    missing = expected - set(results)
    extra = set(results) - expected
    if missing or extra:
        raise ValueError(
            f"calibration matrix mismatch: missing={len(missing)} extra={len(extra)}."
        )
    return results


def select_config(
    results: dict[tuple[str, str, str, str], float],
    *,
    samples: int,
    sample_seed: int,
    attack_seed: int,
    image_ids_sha256: str,
    results_path: Path,
) -> tuple[FrozenRoutingConfig, dict[str, object]]:
    if samples <= 0:
        raise ValueError("samples must be positive.")
    layer_scores: dict[str, dict[str, dict[str, float]]] = {}
    best_layers: dict[str, dict[str, str]] = {polarity: {} for polarity in POLARITIES}
    best_scores: dict[str, dict[str, float]] = {polarity: {} for polarity in POLARITIES}
    for source in WHITEBOX_MODEL_CHOICES:
        layer_scores[source] = {}
        targets = [target for target in WHITEBOX_MODEL_CHOICES if target != source]
        for polarity in POLARITIES:
            scores = {}
            for layer in PATCH_SCORE_LAYER_CANDIDATES[source]:
                values = [results[(source, target, polarity, layer)] for target in targets]
                scores[layer] = sum(values) / len(values)
            layer_scores[source][polarity] = scores
            # Candidate order runs shallow -> final, so max breaks exact ties
            # toward the final/deepest checkpoint as pre-registered.
            best_layer = max(
                PATCH_SCORE_LAYER_CANDIDATES[source],
                key=lambda layer: (scores[layer], PATCH_SCORE_LAYER_CANDIDATES[source].index(layer)),
            )
            best_layers[polarity][source] = best_layer
            best_scores[polarity][source] = scores[best_layer]
    global_scores = {
        polarity: sum(best_scores[polarity].values()) / len(WHITEBOX_MODEL_CHOICES)
        for polarity in POLARITIES
    }
    global_polarity = max(POLARITIES, key=lambda value: (global_scores[value], value == "high"))
    calibration = {
        "samples": samples,
        "sample_seed": sample_seed,
        "attack_seed": attack_seed,
        "image_ids_sha256": image_ids_sha256,
        "selection_metric": "macro_off_diagonal_transfer_asr_over_other_three_sources",
        "results_path": str(results_path),
        "global_scores": global_scores,
        "best_scores_by_polarity": best_scores,
    }
    config = FrozenRoutingConfig(
        global_polarity=global_polarity,
        model_layers=best_layers[global_polarity],
        calibration=calibration,
    )
    summary = {
        "protocol": calibration,
        "layer_scores": layer_scores,
        "best_layers_by_polarity": best_layers,
        "selected": config.to_dict(),
    }
    return config, summary


def main() -> None:
    args = parse_args()
    if args.write_template is not None:
        rows = template_rows()
        write_csv(args.write_template, rows)
        manifest_path = args.write_template.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(
                build_manifest(
                    samples=args.samples,
                    sample_offset=args.sample_offset,
                    attack_seed=args.attack_seed,
                    attack_output_root=args.attack_output_root,
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"wrote {len(rows)} pre-registered calibration rows to {args.write_template}")
        print(f"wrote executable job manifest to {manifest_path}")
        return
    if args.image_ids_sha256 is None:
        raise ValueError(
            "--image-ids-sha256 is required with --results; copy it from a calibration "
            "replay_manifest.json."
        )
    if len(args.image_ids_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in args.image_ids_sha256
    ):
        raise ValueError("--image-ids-sha256 must be a 64-character hexadecimal SHA-256.")
    results = read_results(args.results)
    config, summary = select_config(
        results,
        samples=args.samples,
        sample_seed=args.sample_seed,
        attack_seed=args.attack_seed,
        image_ids_sha256=args.image_ids_sha256,
        results_path=args.results,
    )
    config.save(args.output_config)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
