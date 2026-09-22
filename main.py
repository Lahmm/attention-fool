from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

LOCAL_HF_CACHE = Path(__file__).resolve().parent / "data" / "huggingface"
# Keep model provenance repository-local even when the invoking shell exports
# a different global Hugging Face cache.
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"

from tqdm import tqdm

from gradient_replay import GradientReplay
from nets import DEFAULT_MODEL_NAME, WHITEBOX_MODEL_CHOICES, build_whitebox_model
from progressive_attack import ProgressiveRouteDisruptionAttacker
from utils import DEVICE, load_data, save_adversarial_images


IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"
ATTACK_METHODS = ("progressive",)


def parse_phase_shift(value: str) -> tuple[int, int]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 2:
        raise argparse.ArgumentTypeError("phase shift must be two comma-separated integers")
    return values


def parse_phase_shift_set(value: str) -> tuple[tuple[int, int], ...]:
    shifts = tuple(parse_phase_shift(item) for item in value.split(";") if item.strip())
    if not shifts:
        raise argparse.ArgumentTypeError("phase shift set cannot be empty")
    return shifts


def parse_checkpoint_list(value: str) -> tuple[str | int, ...]:
    checkpoints: list[str | int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        checkpoints.append(int(item) if item.isdigit() else item)
    if not checkpoints:
        raise argparse.ArgumentTypeError("checkpoint list cannot be empty")
    return tuple(checkpoints)


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not values:
        raise argparse.ArgumentTypeError("float list cannot be empty")
    return values


def validate_output_dir(output_dir: str) -> Path:
    repo_root = Path(__file__).resolve().parent
    attack_root = (repo_root / "outputs" / "attack").resolve()
    resolved = Path(output_dir).expanduser()
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(attack_root)
    except ValueError as exc:
        raise ValueError(f"output-dir must be under {attack_root}") from exc
    if resolved == attack_root:
        raise ValueError("output-dir must name a subdirectory under outputs/attack")
    return resolved


def clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def attack_all_samples(
    dataloader,
    attacker,
    output_dir: Path,
    max_attacked_samples: int | None,
    replay: GradientReplay | None = None,
) -> list[str]:
    total = len(dataloader.dataset)
    limit = total if max_attacked_samples is None else min(total, max_attacked_samples)
    progress = tqdm(total=limit, desc="Attacking samples")
    attacked = 0
    saved_count = 0
    all_sample_ids: list[str] = []

    for images, labels, indices in dataloader:
        if attacked >= limit:
            break
        remaining = limit - attacked
        images = images[:remaining]
        labels = labels[:remaining]
        indices = indices[:remaining]
        filenames = [
            str(dataloader.dataset.samples[index]["image_name"])
            for index in indices.tolist()
        ]
        all_sample_ids.extend(filenames)
        adversarial = attacker.attack_batch(
            images,
            labels,
            replay=replay,
            sample_ids=filenames if replay is not None else None,
        )
        saved = save_adversarial_images(
            images=adversarial,
            output_dir=str(output_dir),
            prefix="adv",
            start_index=saved_count,
            filenames=filenames,
        )
        attacked += images.size(0)
        saved_count += len(saved)
        progress.update(images.size(0))

    progress.close()
    print(f"Attacked {attacked} samples and saved {saved_count} images to {output_dir}")
    return all_sample_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressive Route Disruption attack")
    parser.add_argument("--attack-method", choices=ATTACK_METHODS, default="progressive")
    parser.add_argument("--whitebox-model", choices=WHITEBOX_MODEL_CHOICES, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-attacked-samples", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mi", dest="mi", action="store_true")
    parser.add_argument("--no-mi", dest="mi", action="store_false")
    parser.set_defaults(mi=True)
    parser.add_argument("--mi-decay", type=float, default=1.0)
    parser.add_argument("--input-diversity-groups", type=int, default=10)
    parser.add_argument("--input-diversity-views-per-group", type=int, default=2)
    parser.add_argument("--input-diversity-phase-shift-set", type=parse_phase_shift_set, default=((4, 4), (8, 8), (12, 12)))
    parser.add_argument(
        "--checkpoints",
        type=parse_checkpoint_list,
        default=None,
        help="Ordered adapter checkpoint IDs; defaults are architecture-specific.",
    )
    parser.add_argument(
        "--drop-ratios",
        type=parse_float_list,
        default=None,
        help="One ratio per checkpoint; omitted values use the source adapter defaults.",
    )
    parser.add_argument(
        "--opponent-noise-strength",
        type=float,
        default=None,
        help="Opponent-channel noise strength; omitted values use the source adapter default.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=4.0,
        help="Sigma of the Gaussian gradient residual; use alpha=0 to disable it.",
    )
    parser.add_argument(
        "--gaussian-alpha",
        type=float,
        default=0.75,
        help="Weight of the Gaussian-smoothed residual added before MI accumulation.",
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/attack/progressive_route_disruption")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    output_dir = validate_output_dir(args.output_dir)
    dataloader, num_classes = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    model = build_whitebox_model(num_classes=num_classes, model_name=args.whitebox_model)
    if args.input_diversity_views_per_group != 2:
        raise ValueError("the progressive attack requires exactly two views per group.")
    attacker = ProgressiveRouteDisruptionAttacker(
        model=model,
        checkpoints=args.checkpoints,
        drop_ratios=args.drop_ratios,
        opponent_noise_strength=args.opponent_noise_strength,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        use_momentum=args.mi,
        momentum_decay=args.mi_decay,
        input_diversity_groups=args.input_diversity_groups,
        input_diversity_views_per_group=2,
        input_diversity_phase_shift_set=args.input_diversity_phase_shift_set,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_alpha=args.gaussian_alpha,
        device=DEVICE,
    )

    clear_directory_contents(output_dir)
    replay = GradientReplay(args.seed) if args.seed is not None else None
    sample_ids = attack_all_samples(
        dataloader,
        attacker,
        output_dir,
        args.max_attacked_samples,
        replay=replay,
    )
    if replay is not None:
        (output_dir / "replay_manifest.json").write_text(
            json.dumps(replay.manifest(sample_ids), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (output_dir / "gradient_diagnostics.json").write_text(
        json.dumps(attacker.gradient_diagnostics_summary(), indent=2),
        encoding="utf-8",
    )

    params = {
        "attack_method": args.attack_method,
        "whitebox_model": args.whitebox_model,
        "max_attacked_samples": args.max_attacked_samples,
        "epsilon": args.epsilon,
        "step_size": args.step_size if args.step_size is not None else args.epsilon / args.steps,
        "steps": args.steps,
        "seed": args.seed,
        "mi": args.mi,
        "mi_decay": args.mi_decay,
        "input_diversity_groups": args.input_diversity_groups,
        "input_diversity_views_per_group": args.input_diversity_views_per_group,
        "input_diversity_total_views": (
            args.input_diversity_groups * args.input_diversity_views_per_group
        ),
        "input_diversity_phase_shift_set": [list(shift) for shift in args.input_diversity_phase_shift_set],
        "checkpoints": list(attacker.progressive_checkpoints),
        "drop_ratios": list(attacker.progressive_drop_ratios),
        "opponent_noise_strength": attacker.opponent_noise_strength,
        "feature_noise_type": "opponent_channel_rgb_projection",
        "feature_noise_position": "initial_rgb_projection",
        "gradient_postprocess": (
            "raw_mean_plus_gaussian_residual"
            if args.gaussian_alpha != 0
            else "raw_mean"
        ),
        "gaussian_sigma": args.gaussian_sigma,
        "gaussian_alpha": args.gaussian_alpha,
    }
    params.update(attacker.mainline_metadata())
    (output_dir / "attack_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main(parse_args())
