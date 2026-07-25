from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

LOCAL_HF_CACHE = Path(__file__).resolve().parent / "data" / "huggingface"
os.environ.setdefault("HF_HOME", str(LOCAL_HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(LOCAL_HF_CACHE / "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from tqdm import tqdm

from attack import (
    ATTACK_METHODS,
    PATCH_SELECTORS,
    POST_DROPOUT_NOISE_TYPES,
    PatchScoreAttacker,
)
from gradient_replay import GradientReplay
from nets import DEFAULT_MODEL_NAME, WHITEBOX_MODEL_CHOICES, build_whitebox_model
from routing_config import FrozenRoutingConfig, file_sha256
from utils import DEVICE, load_data, save_adversarial_images


IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"


def parse_float_range(value: str) -> tuple[float, float]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(values) != 2 or not 0.0 < values[0] <= values[1] <= 1.0:
        raise argparse.ArgumentTypeError("range must satisfy 0 < low <= high <= 1")
    return values


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
    attacker: PatchScoreAttacker,
    output_dir: Path,
    max_attacked_samples: int | None,
    sample_offset: int = 0,
    replay: GradientReplay | None = None,
) -> list[str]:
    total = len(dataloader.dataset)
    if sample_offset < 0 or sample_offset >= total:
        raise ValueError(f"sample_offset must be in [0, {total}), got {sample_offset}.")
    available = total - sample_offset
    limit = available if max_attacked_samples is None else min(available, max_attacked_samples)
    progress = tqdm(total=limit, desc="Attacking samples")
    attacked = 0
    saved_count = 0
    seen = 0
    all_sample_ids: list[str] = []

    for images, labels, indices in dataloader:
        if attacked >= limit:
            break
        batch_end = seen + images.size(0)
        if batch_end <= sample_offset:
            seen = batch_end
            continue
        if seen < sample_offset:
            start = sample_offset - seen
            images = images[start:]
            labels = labels[start:]
            indices = indices[start:]
        seen = batch_end
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
    parser = argparse.ArgumentParser(description="Patch-score routing attack")
    parser.add_argument("--attack-method", choices=ATTACK_METHODS, default="original_score_postdrop_phase_pair")
    parser.add_argument("--whitebox-model", choices=WHITEBOX_MODEL_CHOICES, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-attacked-samples", type=int, default=1000)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Skip this many sorted annotated samples before attacking.",
    )
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mi", dest="mi", action="store_true")
    parser.add_argument("--no-mi", dest="mi", action="store_false")
    parser.set_defaults(mi=True)
    parser.add_argument("--mi-decay", type=float, default=1.0)
    parser.add_argument("--ni", action="store_true")
    parser.add_argument("--ti-sigma", type=float, default=0.0)
    parser.add_argument("--dim", action="store_true")
    parser.add_argument("--dim-resize-range", type=parse_float_range, default=(0.85, 1.0))
    parser.add_argument("--guide-aug-copies", type=int, default=20)
    parser.add_argument("--input-diversity-groups", type=int, default=10)
    parser.add_argument("--input-diversity-views-per-group", type=int, default=2)
    parser.add_argument("--input-diversity-phase-shift", type=parse_phase_shift, default=(0, 0))
    parser.add_argument("--input-diversity-phase-shift-set", type=parse_phase_shift_set, default=((4, 4), (8, 8), (12, 12)))
    parser.add_argument("--guide-aug-strength", type=float, default=0.2)
    parser.add_argument("--patch-dropout-ratio", type=float, default=0.3)
    parser.add_argument("--patch-dropout-score-mode", choices=("high", "low", "all"), default="high")
    parser.add_argument("--patch-dropout-sampling-mode", choices=("random", "bernoulli", "extreme", "score_weighted"), default="random")
    parser.add_argument("--patch-dropout-score-quantile-jitter", type=float, default=0.0)
    parser.add_argument("--patch-dropout-score-noise", type=float, default=0.0)
    parser.add_argument("--patch-dropout-noise-mode", choices=("gaussian", "opponent_channel_gaussian"), default="opponent_channel_gaussian")
    parser.add_argument("--token-cls-noise", action="store_true")
    parser.add_argument("--token-score-cls-noise", dest="token_score_cls_noise", action="store_true")
    parser.add_argument("--no-token-score-cls-noise", dest="token_score_cls_noise", action="store_false")
    parser.set_defaults(token_score_cls_noise=True)
    parser.add_argument("--token-score-cls-mode", choices=("learned", "gaussian"), default="learned")
    parser.add_argument("--token-score-patch-noise", action="store_true")
    parser.add_argument("--token-cls-noise-mode", choices=("gaussian", "mahalanobis"), default="gaussian")
    parser.add_argument("--token-cls-noise-strength", type=float, default=None)
    parser.add_argument("--post-dropout-phase-token-noise", dest="post_dropout_phase_token_noise", action="store_true")
    parser.add_argument("--no-post-dropout-phase-token-noise", dest="post_dropout_phase_token_noise", action="store_false")
    parser.set_defaults(post_dropout_phase_token_noise=True)
    parser.add_argument(
        "--post-dropout-feature-noise-strength",
        type=float,
        default=None,
        help="Strength of kept-only post-dropout feature noise; omitted preserves guide-aug strength.",
    )
    parser.add_argument(
        "--post-dropout-feature-noise-type",
        choices=POST_DROPOUT_NOISE_TYPES,
        default="opponent_projected",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=12,
        help="Legacy patch/token-dropout layer; the generalized mainline uses each model's final semantic layer.",
    )
    parser.add_argument(
        "--patch-score-layer",
        default="final",
        help="Registered routing checkpoint for the selected white-box model.",
    )
    parser.add_argument("--patch-selector", choices=PATCH_SELECTORS, default="patch_score")
    parser.add_argument(
        "--gradcam-target-mode",
        choices=("true", "predicted"),
        default="true",
        help="Class target used only by the Grad-CAM selector.",
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/attack/patch_score_routing")
    parser.add_argument(
        "--routing-config",
        type=Path,
        default=None,
        help="Frozen global-polarity/model-layer calibration JSON.",
    )
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
    routing_config = None
    routing_config_sha256 = None
    if args.routing_config is not None:
        routing_config = FrozenRoutingConfig.load(args.routing_config)
        routing_config_sha256 = file_sha256(args.routing_config)
        args.patch_score_layer = routing_config.layer_for(args.whitebox_model)
        args.patch_dropout_score_mode = routing_config.global_polarity
    model = build_whitebox_model(num_classes=num_classes, model_name=args.whitebox_model)
    attacker = PatchScoreAttacker(
        model=model,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        attack_method=args.attack_method,
        use_momentum=args.mi,
        momentum_decay=args.mi_decay,
        nesterov=args.ni,
        ti_sigma=args.ti_sigma,
        input_diversity=args.dim,
        dim_resize_range=args.dim_resize_range,
        guide_aug_copies=args.guide_aug_copies,
        input_diversity_groups=args.input_diversity_groups,
        input_diversity_views_per_group=args.input_diversity_views_per_group,
        input_diversity_phase_shift=args.input_diversity_phase_shift,
        input_diversity_phase_shift_set=args.input_diversity_phase_shift_set,
        guide_aug_strength=args.guide_aug_strength,
        patch_dropout_ratio=args.patch_dropout_ratio,
        patch_dropout_score_mode=args.patch_dropout_score_mode,
        patch_dropout_sampling_mode=args.patch_dropout_sampling_mode,
        patch_dropout_score_quantile_jitter=args.patch_dropout_score_quantile_jitter,
        patch_dropout_score_noise=args.patch_dropout_score_noise,
        patch_dropout_noise_mode=args.patch_dropout_noise_mode,
        token_cls_noise=args.token_cls_noise,
        token_score_cls_noise=args.token_score_cls_noise,
        token_score_cls_mode=args.token_score_cls_mode,
        token_score_patch_noise=args.token_score_patch_noise,
        token_cls_noise_mode=args.token_cls_noise_mode,
        token_cls_noise_strength=args.token_cls_noise_strength,
        post_dropout_phase_token_noise=args.post_dropout_phase_token_noise,
        post_dropout_feature_noise_strength=args.post_dropout_feature_noise_strength,
        post_dropout_feature_noise_type=args.post_dropout_feature_noise_type,
        feature_layer=args.feature_layer,
        patch_score_layer=args.patch_score_layer,
        patch_selector=args.patch_selector,
        gradcam_target_mode=args.gradcam_target_mode,
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
        sample_offset=args.sample_offset,
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
        "sample_offset": args.sample_offset,
        "epsilon": args.epsilon,
        "step_size": args.step_size if args.step_size is not None else args.epsilon / args.steps,
        "steps": args.steps,
        "seed": args.seed,
        "mi": args.mi,
        "mi_decay": args.mi_decay,
        "ni": args.ni,
        "ti_sigma": args.ti_sigma,
        "dim": args.dim,
        "dim_resize_range": list(args.dim_resize_range),
        "guide_aug_copies": args.guide_aug_copies,
        "input_diversity_groups": args.input_diversity_groups,
        "input_diversity_views_per_group": args.input_diversity_views_per_group,
        "input_diversity_total_views": (
            args.input_diversity_groups * args.input_diversity_views_per_group
        ),
        "input_diversity_phase_shift": list(args.input_diversity_phase_shift),
        "input_diversity_phase_shift_set": [list(shift) for shift in args.input_diversity_phase_shift_set],
        "guide_aug_strength": args.guide_aug_strength,
        "patch_dropout_ratio": args.patch_dropout_ratio,
        "patch_dropout_score_mode": args.patch_dropout_score_mode,
        "patch_dropout_sampling_mode": args.patch_dropout_sampling_mode,
        "patch_dropout_score_quantile_jitter": args.patch_dropout_score_quantile_jitter,
        "patch_dropout_score_noise": args.patch_dropout_score_noise,
        "patch_dropout_noise_mode": args.patch_dropout_noise_mode,
        "token_cls_noise": args.token_cls_noise,
        "token_score_cls_noise": args.token_score_cls_noise,
        "token_score_cls_mode": args.token_score_cls_mode,
        "token_score_patch_noise": args.token_score_patch_noise,
        "token_cls_noise_mode": args.token_cls_noise_mode,
        "token_cls_noise_strength": (
            args.token_cls_noise_strength
            if args.token_cls_noise_strength is not None
            else args.guide_aug_strength
        ),
        "post_dropout_phase_token_noise": args.post_dropout_phase_token_noise,
        "post_dropout_feature_noise_strength": (
            args.post_dropout_feature_noise_strength
            if args.post_dropout_feature_noise_strength is not None
            else args.guide_aug_strength
        ),
        "post_dropout_feature_noise_type": args.post_dropout_feature_noise_type,
        "post_dropout_feature_noise_position": "initial",
        "feature_layer": args.feature_layer,
        "patch_score_layer": args.patch_score_layer,
        "patch_selector": args.patch_selector,
        "gradcam_target_mode": args.gradcam_target_mode,
        "routing_config": str(args.routing_config) if args.routing_config is not None else None,
        "routing_config_sha256": routing_config_sha256,
        "gradient_postprocess": "mean",
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
