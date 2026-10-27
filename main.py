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

from nets import DEFAULT_MODEL_NAME, WHITEBOX_MODEL_CHOICES, build_whitebox_model
from progressive_attack import NOISE_DIRECTION_VARIANCES, ProgressiveRouteDisruptionAttacker
from utils import DEVICE, SEED, load_data, save_adversarial_images


IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"


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


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not values:
        raise argparse.ArgumentTypeError("float list cannot be empty")
    return values


def parse_checkpoint_list(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("checkpoint list cannot be empty")
    indices = []
    for item in items:
        suffix = item.removeprefix("block")
        if not item.startswith("block") or not suffix.isdigit():
            raise argparse.ArgumentTypeError("checkpoints must use the form block<k>")
        indices.append(int(suffix))
    if any(index < 1 for index in indices):
        raise argparse.ArgumentTypeError("checkpoints must use positive block indices")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("checkpoints must not repeat")
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise argparse.ArgumentTypeError("checkpoints must be strictly increasing")
    return tuple(f"block{index}" for index in indices)


def apply_vit_analysis_checkpoints(model, checkpoints, drop_ratios) -> None:
    if checkpoints is None:
        return
    if model.model_name != "vit_base_patch16_224":
        raise ValueError("--vit-analysis-checkpoints is restricted to vit_base_patch16_224.")
    indices = tuple(int(item.removeprefix("block")) for item in checkpoints)
    block_count = len(model.model.blocks)
    if any(index < 1 or index > block_count for index in indices):
        raise ValueError(f"analysis checkpoints must lie in block1..block{block_count}.")
    ratios = model.drop_ratios if drop_ratios is None else drop_ratios
    if len(ratios) != len(checkpoints):
        raise ValueError("analysis checkpoints and drop ratios must have the same length.")
    model.checkpoints = tuple(checkpoints)


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
) -> list[str]:
    total = len(dataloader.dataset)
    limit = total if max_attacked_samples is None else min(total, max_attacked_samples)
    progress = tqdm(total=limit, desc="Attacking samples")
    attacked = 0
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
        adversarial = attacker.attack_batch(images, labels)
        save_adversarial_images(
            images=adversarial,
            output_dir=str(output_dir),
            filenames=filenames,
        )
        attacked += images.size(0)
        progress.update(images.size(0))

    progress.close()
    print(f"Attacked {attacked} samples and saved {attacked} images to {output_dir}")
    return all_sample_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Progressive Route Disruption attack")
    parser.add_argument("--whitebox-model", choices=WHITEBOX_MODEL_CHOICES, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-attacked-samples", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--input-diversity-groups", type=int, default=10)
    parser.add_argument(
        "--input-diversity-views-per-group", type=int, choices=(1, 2), default=2,
        help="1: original view only; 2: original and phase views (default).",
    )
    parser.add_argument("--input-diversity-phase-shift-set", type=parse_phase_shift_set, default=((4, 4), (8, 8), (12, 12)))
    parser.add_argument(
        "--no-progressive-drop",
        action="store_true",
        help="Disable checkpoint token drops for mechanism ablations.",
    )
    parser.add_argument(
        "--drop-ratios",
        type=parse_float_list,
        default=None,
        help="One ratio per checkpoint; omitted values use the source adapter defaults.",
    )
    parser.add_argument(
        "--vit-analysis-checkpoints",
        type=parse_checkpoint_list,
        default=None,
        help="ViT-B analysis-only checkpoint override, e.g. block5,block8.",
    )
    parser.add_argument(
        "--noise-structure",
        choices=tuple(NOISE_DIRECTION_VARIANCES),
        default="opponent",
        help="RGB noise direction variances: opponent (default) or isotropic_rgb.",
    )
    parser.add_argument(
        "--opponent-noise-strength",
        type=float,
        default=None,
        help="Opponent-channel noise strength; omitted values use the source adapter default.",
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/attack/progressive_route_disruption")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    output_dir = validate_output_dir(args.output_dir)
    dataloader = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    model = build_whitebox_model(model_name=args.whitebox_model)
    apply_vit_analysis_checkpoints(
        model, args.vit_analysis_checkpoints, args.drop_ratios
    )
    attacker = ProgressiveRouteDisruptionAttacker(
        model=model,
        progressive_drop=not args.no_progressive_drop,
        drop_ratios=args.drop_ratios,
        opponent_noise_strength=args.opponent_noise_strength,
        noise_structure=args.noise_structure,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        input_diversity_groups=args.input_diversity_groups,
        input_diversity_views_per_group=args.input_diversity_views_per_group,
        input_diversity_phase_shift_set=args.input_diversity_phase_shift_set,
        device=DEVICE,
    )

    clear_directory_contents(output_dir)
    sample_ids = attack_all_samples(
        dataloader,
        attacker,
        output_dir,
        args.max_attacked_samples,
    )
    params = attacker.parameters()
    params["num_images"] = len(sample_ids)
    (output_dir / "attack_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main(parse_args())
