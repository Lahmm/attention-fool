"""Reproducible 100-sample gradient direction study.

Stages:
  observe  - canonical replay baseline, diagnostics, all-model outcomes, feature gates
  probe    - run targeted causal probes selected by the observation gate
  all      - observe followed by probes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from attack import PatchScoreAttacker
from gradient_observer import GradientObserver
from gradient_replay import GradientReplay
from gradient_study import analyze_features, build_probe, probe_names_for_family
from main import clear_directory_contents, validate_output_dir
from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from transfer_eval import (
    DEFAULT_CNN_BLACK_BOX_MODELS,
    DEFAULT_VIT_BLACK_BOX_MODELS,
    TransferImageDataset,
    build_black_box_model,
    build_transfer_samples,
    collect_images,
    extract_original_name,
    load_annotations,
)
from utils import DEVICE, load_data, save_adversarial_images


BLACKBOX_MODELS = DEFAULT_VIT_BLACK_BOX_MODELS + DEFAULT_CNN_BLACK_BOX_MODELS[:4]
WHITEBOX_EVAL_MODEL = "vit_base_patch16_224"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproducible gradient direction study")
    parser.add_argument("--stage", choices=("observe", "probe", "all"), default="all")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-dir", default="outputs/attack/gradient_direction_study")
    parser.add_argument("--bootstrap-count", type=int, default=5000)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def build_attacker(model, steps: int = 10) -> PatchScoreAttacker:
    return PatchScoreAttacker(
        model=model,
        epsilon=16.0 / 255.0,
        steps=steps,
        attack_method="original_score_postdrop_phase_pair",
        use_momentum=True,
        momentum_decay=1.0,
        nesterov=False,
        ti_sigma=0.0,
        input_diversity=False,
        input_diversity_groups=10,
        input_diversity_views_per_group=2,
        input_diversity_phase_shift_set=((4, 4), (8, 8), (12, 12)),
        guide_aug_strength=0.2,
        patch_dropout_ratio=0.3,
        patch_dropout_score_mode="high",
        patch_dropout_sampling_mode="random",
        patch_dropout_noise_mode="opponent_channel_gaussian",
        token_score_cls_noise=True,
        token_score_cls_mode="learned",
        token_score_patch_noise=False,
        post_dropout_phase_token_noise=True,
        feature_layer=12,
        gradient_postprocess="mean",
        device=DEVICE,
    )


def run_attack(
    *,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    seed: int,
    steps: int = 10,
    probe_name: str | None,
    baseline_sign_dir: Path | None,
    expected_manifest: dict[str, object] | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    clear_directory_contents(output_dir)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataloader, num_classes = load_data(batch_size=batch_size, num_workers=4, prefetch_factor=4)
    model = build_whitebox_model(num_classes=num_classes, model_name=DEFAULT_MODEL_NAME)
    attacker = build_attacker(model, steps=steps)
    replay = GradientReplay(seed)
    probe = build_probe(probe_name) if probe_name else None
    attacked = 0
    saved_count = 0
    all_sample_ids: list[str] = []
    all_records: list[dict[str, object]] = []

    for batch_index, (images, labels, indices) in enumerate(dataloader):
        if attacked >= num_samples:
            break
        remaining = num_samples - attacked
        images = images[:remaining]
        labels = labels[:remaining]
        indices = indices[:remaining]
        sample_ids = [str(dataloader.dataset.samples[index]["image_name"]) for index in indices.tolist()]
        all_sample_ids.extend(sample_ids)

        reference_signs = None
        if baseline_sign_dir is not None:
            sign_path = baseline_sign_dir / f"signs_batch_{batch_index:03d}.pt"
            reference_signs = torch.load(sign_path, map_location="cpu", weights_only=True)
        observer = GradientObserver(
            enabled=True,
            sample_ids=sample_ids,
            reference_signs=reference_signs,
            capture_signs=probe_name is None,
        )
        adversarial = attacker.attack_batch(
            images,
            labels,
            observer=observer,
            replay=replay,
            sample_ids=sample_ids,
            probe=probe,
        )
        save_adversarial_images(
            images=adversarial,
            output_dir=str(output_dir),
            prefix="adv",
            start_index=saved_count,
            filenames=sample_ids,
        )
        saved_count += images.size(0)
        observer.save(output_dir / f"batch_{batch_index:03d}")
        if probe_name is None:
            torch.save(observer.captured_signs, output_dir / f"signs_batch_{batch_index:03d}.pt")
        all_records.extend(observer.per_sample_summary())
        attacked += images.size(0)
        print(f"attack={probe_name or 'baseline'} samples={attacked}/{num_samples}")

    manifest = replay.manifest(all_sample_ids)
    if expected_manifest is not None:
        for key in ("version", "master_seed", "sample_ids", "event_count", "event_digest", "phase_events"):
            if manifest[key] != expected_manifest[key]:
                raise RuntimeError(f"random replay mismatch for {key} in {probe_name}")
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "attack_params.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "steps": steps,
                "epsilon": 16.0 / 255.0,
                "actual_views": 20,
                "gradient_postprocess": "mean",
                "probe": probe_name,
                "replay_digest": manifest["event_digest"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return all_records, manifest


def evaluate_per_sample(image_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    annotations = load_annotations(Path("data/image_name_to_class_id_and_name.json"))
    image_paths = collect_images(image_dir, "adv_")
    outcomes: dict[str, dict[str, float]] = {
        Path(extract_original_name(path.name, "adv_")).stem: {} for path in image_paths
    }
    aggregate: dict[str, float] = {}

    for model_name in BLACKBOX_MODELS + [WHITEBOX_EVAL_MODEL]:
        model, transform = build_black_box_model(model_name)
        samples, skipped = build_transfer_samples(image_paths, annotations, "adv_")
        if skipped or len(samples) != len(image_paths):
            raise RuntimeError(f"annotation mismatch while evaluating {model_name}")
        dataset = TransferImageDataset(samples, transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        success: list[float] = []
        with torch.inference_mode():
            for images, labels in loader:
                predictions = model(images.to(DEVICE)).argmax(dim=1).cpu()
                success.extend(predictions.ne(labels).float().tolist())
        for path, value in zip(image_paths, success):
            sample_id = Path(extract_original_name(path.name, "adv_")).stem
            outcomes[sample_id][model_name] = float(value)
        aggregate[model_name] = sum(success) / len(success)
        print(f"eval model={model_name} ASR={aggregate[model_name]:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate["overall"] = _model_average(aggregate, BLACKBOX_MODELS)
    aggregate["vit"] = _model_average(aggregate, DEFAULT_VIT_BLACK_BOX_MODELS)
    aggregate["cnn"] = _model_average(aggregate, DEFAULT_CNN_BLACK_BOX_MODELS[:4])
    aggregate["whitebox"] = aggregate[WHITEBOX_EVAL_MODEL]
    return outcomes, aggregate


def _model_average(values: dict[str, float], models: list[str]) -> float:
    return sum(values[model] for model in models) / len(models)


def merge_outcomes(
    records: list[dict[str, object]],
    outcomes: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    hashes = sorted(
        (
            hashlib.sha256(str(record["sample_id"]).encode()).hexdigest(),
            str(record["sample_id"]),
        )
        for record in records
    )
    discovery_ids = {sample_id for _, sample_id in hashes[:60]}
    merged = []
    for record in records:
        sample_id = str(record["sample_id"])
        model_outcomes = outcomes[Path(sample_id).stem]
        row = dict(record)
        row["split"] = "discovery" if sample_id in discovery_ids else "validation"
        for model_name, value in model_outcomes.items():
            row[f"transfer_{model_name}"] = value
        row["transfer_overall"] = _model_average(model_outcomes, BLACKBOX_MODELS)
        row["transfer_vit"] = _model_average(model_outcomes, DEFAULT_VIT_BLACK_BOX_MODELS)
        row["transfer_cnn"] = _model_average(model_outcomes, DEFAULT_CNN_BLACK_BOX_MODELS[:4])
        row["transfer_whitebox"] = model_outcomes[WHITEBOX_EVAL_MODEL]
        merged.append(row)
    return merged


def paired_comparison(
    baseline_outcomes: dict[str, dict[str, float]],
    candidate_outcomes: dict[str, dict[str, float]],
    bootstrap_count: int,
    seed: int,
) -> dict[str, object]:
    sample_ids = sorted(baseline_outcomes)
    differences = np.asarray(
        [
            np.mean([candidate_outcomes[sample][model] - baseline_outcomes[sample][model] for model in BLACKBOX_MODELS])
            for sample in sample_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(bootstrap_count)]
    )
    return {
        "delta_overall": float(differences.mean()),
        "bootstrap_ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "probability_positive": float((bootstrap > 0).mean()),
    }


def architecture_delta(
    reference: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    models: list[str],
) -> float:
    sample_ids = sorted(reference)
    return float(
        np.mean(
            [
                np.mean([candidate[sample][model] - reference[sample][model] for model in models])
                for sample in sample_ids
            ]
        )
    )


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_observation(args: argparse.Namespace, root: Path) -> dict[str, object]:
    baseline_dir = root / f"baseline_seed{args.seed}"
    if args.reuse_existing and (baseline_dir / "replay_manifest.json").is_file():
        records = load_existing_records(baseline_dir)
        manifest = json.loads((baseline_dir / "replay_manifest.json").read_text(encoding="utf-8"))
        if len(records) != args.num_samples:
            raise RuntimeError(
                f"existing baseline has {len(records)} records, expected {args.num_samples}"
            )
    else:
        records, manifest = run_attack(
            output_dir=baseline_dir,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            probe_name=None,
            baseline_sign_dir=None,
            expected_manifest=None,
        )
    outcomes, aggregate = evaluate_per_sample(baseline_dir)
    merged = merge_outcomes(records, outcomes)
    save_json(root / "baseline_outcomes.json", outcomes)
    save_json(root / "baseline_metrics.json", aggregate)
    with (root / "per_sample_features.jsonl").open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    analysis = analyze_features(merged, root / "feature_analysis.json")
    save_json(root / "study_config.json", vars(args))
    return {"manifest": manifest, "analysis": analysis, "outcomes": outcomes, "aggregate": aggregate}


def load_existing_records(output_dir: Path) -> list[dict[str, object]]:
    by_sample: dict[str, dict[str, list[float]]] = {}
    for path in sorted(output_dir.glob("batch_*/gradient_per_sample_step.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            features = by_sample.setdefault(sample_id, {})
            for key, value in row.items():
                if isinstance(value, (int, float)) and key != "step":
                    features.setdefault(key, []).append(float(value))
    return [
        {
            "sample_id": sample_id,
            **{key: sum(values) / len(values) for key, values in features.items()},
        }
        for sample_id, features in by_sample.items()
    ]


def run_probes(args: argparse.Namespace, root: Path) -> dict[str, object]:
    baseline_dir = root / f"baseline_seed{args.seed}"
    manifest = json.loads((baseline_dir / "replay_manifest.json").read_text(encoding="utf-8"))
    analysis = json.loads((root / "feature_analysis.json").read_text(encoding="utf-8"))
    baseline_outcomes = json.loads((root / "baseline_outcomes.json").read_text(encoding="utf-8"))
    evidence_by_family = {row["family"]: row for row in analysis.get("selected_evidence", [])}
    results: dict[str, object] = {}
    for family in analysis.get("selected_families", []):
        for probe_name in probe_names_for_family(family, evidence_by_family.get(family)):
            probe_dir = root / f"probe_{probe_name}_seed{args.seed}"
            _, candidate_manifest = run_attack(
                output_dir=probe_dir,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                seed=args.seed,
                probe_name=probe_name,
                baseline_sign_dir=baseline_dir,
                expected_manifest=manifest,
            )
            outcomes, aggregate = evaluate_per_sample(probe_dir)
            comparison = paired_comparison(
                baseline_outcomes, outcomes, args.bootstrap_count, args.seed + len(results)
            )
            results[probe_name] = {
                "family": family,
                "aggregate": aggregate,
                "comparison_to_baseline": comparison,
                "replay_digest": candidate_manifest["event_digest"],
            }
            save_json(root / f"outcomes_{probe_name}.json", outcomes)
            save_json(root / "probe_results.json", results)
    decisions: dict[str, object] = {}
    confirmed_candidates: list[str] = []
    for family in analysis.get("selected_families", []):
        probe_names = probe_names_for_family(family, evidence_by_family.get(family))
        if len(probe_names) != 3 or any(name not in results for name in probe_names):
            continue
        target_name, opposite_name, random_name = probe_names
        target_outcomes = json.loads((root / f"outcomes_{target_name}.json").read_text(encoding="utf-8"))
        opposite_outcomes = json.loads((root / f"outcomes_{opposite_name}.json").read_text(encoding="utf-8"))
        random_outcomes = json.loads((root / f"outcomes_{random_name}.json").read_text(encoding="utf-8"))
        target_vs_random = paired_comparison(
            random_outcomes, target_outcomes, args.bootstrap_count, args.seed + 101 + len(decisions)
        )
        target_vs_opposite = paired_comparison(
            opposite_outcomes, target_outcomes, args.bootstrap_count, args.seed + 201 + len(decisions)
        )
        vit_delta = architecture_delta(random_outcomes, target_outcomes, DEFAULT_VIT_BLACK_BOX_MODELS)
        cnn_delta = architecture_delta(random_outcomes, target_outcomes, DEFAULT_CNN_BLACK_BOX_MODELS[:4])
        passes = bool(
            float(target_vs_random["delta_overall"]) >= 0.01
            and float(target_vs_random["probability_positive"]) >= 0.90
            and float(target_vs_opposite["delta_overall"]) > 0.0
            and vit_delta >= -0.01
            and cnn_delta >= -0.01
        )
        decisions[family] = {
            "target": target_name,
            "opposite": opposite_name,
            "random": random_name,
            "target_vs_random": target_vs_random,
            "target_vs_opposite": target_vs_opposite,
            "vit_delta_vs_random": vit_delta,
            "cnn_delta_vs_random": cnn_delta,
            "passes_direction_gate": passes,
        }
        if passes:
            confirmed_candidates.append(target_name)
    save_json(root / "probe_decisions.json", decisions)

    confirmations = {}
    for probe_name in confirmed_candidates:
        seed_results = []
        for confirmation_seed in (20260711, 20260712):
            confirmation_root = root / f"confirmation_seed{confirmation_seed}"
            baseline_confirmation = confirmation_root / "baseline"
            _, confirmation_manifest = run_attack(
                output_dir=baseline_confirmation,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                seed=confirmation_seed,
                probe_name=None,
                baseline_sign_dir=None,
                expected_manifest=None,
            )
            baseline_confirmation_outcomes, baseline_metrics = evaluate_per_sample(baseline_confirmation)
            candidate_confirmation = confirmation_root / probe_name
            _, _ = run_attack(
                output_dir=candidate_confirmation,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                seed=confirmation_seed,
                probe_name=probe_name,
                baseline_sign_dir=baseline_confirmation,
                expected_manifest=confirmation_manifest,
            )
            candidate_confirmation_outcomes, candidate_metrics = evaluate_per_sample(candidate_confirmation)
            comparison = paired_comparison(
                baseline_confirmation_outcomes,
                candidate_confirmation_outcomes,
                args.bootstrap_count,
                confirmation_seed,
            )
            seed_results.append(
                {
                    "seed": confirmation_seed,
                    "baseline": baseline_metrics,
                    "candidate": candidate_metrics,
                    "comparison": comparison,
                }
            )
        confirmations[probe_name] = {
            "seeds": seed_results,
            "consistent_positive": all(
                float(result["comparison"]["delta_overall"]) > 0 for result in seed_results
            ),
        }
        save_json(root / "confirmation_results.json", confirmations)
    return {"results": results, "decisions": decisions, "confirmations": confirmations}


def main() -> None:
    args = parse_args()
    root = validate_output_dir(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if args.stage in ("observe", "all"):
        run_observation(args, root)
    if args.stage in ("probe", "all"):
        run_probes(args, root)


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main()
