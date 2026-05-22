import argparse
import csv
import shutil
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from attack import LazyAggregationAttacker
from nets import ViTWithHook, build_vit_model
from utils import (
    DEVICE,
    evaluate_clean_dataset,
    load_data,
    save_adversarial_images,
    save_clean_images,
)

IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"
DEFAULT_IMG_SIZE = 224


def create_attacker(
    model: ViTWithHook,
    epsilon: float,
    step_size: float | None,
    steps: int,
    decay: float,
    layers: tuple[int, ...],
    warmup_steps: int = 3,
    ti_sigma: float = 0.0,
    spectral_transition: float = 0.04,
    fg_top_ratio: float = 0.25,
    lambda_anchor: float = 1.0,
    grad_combine: str = "guide_qk_response",
    si_scales: int = 1,
    nesterov: bool = True,
    eot_iter: int = 1,
    dim_resize_range: tuple[float, float] = (0.85, 1.0),
    perturb_smooth_sigma: float = 0.0,
    anchor_schedule: str = "constant",
    anchor_start_step: int | None = None,
    anchor_end_weight: float | None = None,
    lazy_spectral_delta: bool = False,
    lazy_spectral_cutoff: float = 0.25,
    attention_guide_models: tuple[ViTWithHook, ...] = (),
    guide_type: str = "postsoftmax_cls",
    guide_sample_mode: str = "fixed",
    attention_grad_smooth_sigma: float = 0.0,
    patch_grad_smooth_sigma: float = 0.0,
    guide_entropy_temp: float = 1.0,
    guide_dilate_kernel: int = 1,
    guide_smooth_sigma: float = 0.0,
    guide_dynamic: bool = False,
    guide_update_interval: int = 5,
    guide_ema: float = 0.7,
    guide_aug_copies: int = 3,
    guide_aug_mode: tuple[str, ...] = ("bg_blur",),
    guide_aug_strength: float = 0.3,
    bg_foreground_ratio: float = 0.25,
    bg_background_ratio: float = 0.50,
    bg_fg_dilate_kernel: int = 3,
) -> LazyAggregationAttacker:
    return LazyAggregationAttacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        decay=decay,
        layers=layers,
        fg_top_ratio=fg_top_ratio,
        lambda_anchor=lambda_anchor,
        warmup_steps=warmup_steps,
        grad_combine=grad_combine,
        spectral_transition=spectral_transition,
        ti_sigma=ti_sigma if ti_sigma > 0 else 3.0,
        dim_resize_range=dim_resize_range,
        si_scales=si_scales,
        nesterov=nesterov,
        eot_iter=eot_iter,
        perturb_smooth_sigma=perturb_smooth_sigma,
        anchor_schedule=anchor_schedule,
        anchor_start_step=anchor_start_step,
        anchor_end_weight=anchor_end_weight,
        lazy_spectral_delta=lazy_spectral_delta,
        lazy_spectral_cutoff=lazy_spectral_cutoff,
        attention_guide_models=attention_guide_models,
        guide_type=guide_type,
        guide_sample_mode=guide_sample_mode,
        attention_grad_smooth_sigma=attention_grad_smooth_sigma,
        patch_grad_smooth_sigma=patch_grad_smooth_sigma,
        guide_entropy_temp=guide_entropy_temp,
        guide_dilate_kernel=guide_dilate_kernel,
        guide_smooth_sigma=guide_smooth_sigma,
        guide_dynamic=guide_dynamic,
        guide_update_interval=guide_update_interval,
        guide_ema=guide_ema,
        guide_aug_copies=guide_aug_copies,
        guide_aug_mode=guide_aug_mode,
        guide_aug_strength=guide_aug_strength,
        bg_foreground_ratio=bg_foreground_ratio,
        bg_background_ratio=bg_background_ratio,
        bg_fg_dilate_kernel=bg_fg_dilate_kernel,
        device=DEVICE,
    )


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise argparse.ArgumentTypeError("layers must contain at least one comma-separated integer.")
    return layers


def parse_float_range(value: str) -> tuple[float, float]:
    parts = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range must contain exactly two comma-separated floats, e.g. 0.85,1.0.")
    lo, hi = parts
    if not (0.0 < lo <= hi <= 1.0):
        raise argparse.ArgumentTypeError("range values must satisfy 0 < low <= high <= 1.")
    return lo, hi


def parse_model_names(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def expected_attack_output_dir() -> Path:
    return Path("outputs") / "attack" / "lazyagg"


def validate_attack_output_dir(output_dir: str | None) -> Path:
    expected = expected_attack_output_dir()
    if output_dir is None:
        raise ValueError(
            f"Attack mode requires --output-dir. "
            f"Use --output-dir {expected.as_posix()}."
        )
    provided = Path(output_dir).expanduser()
    if provided.resolve() != expected.resolve():
        if str(provided.resolve()).startswith(str(expected.resolve())):
            return provided
        raise ValueError(
            f"Invalid --output-dir: {provided}. "
            f"Expected exactly: {expected.as_posix()}"
        )
    return provided


def clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def attack_correctly_classified_samples(
    dataloader,
    model: ViTWithHook,
    attacker: LazyAggregationAttacker,
    correct_mask: List[bool],
    output_dir: str | None,
    max_attacked_samples: int | None,
) -> None:
    num_candidates = sum(correct_mask)
    if num_candidates == 0:
        print("No correctly classified samples are available for attack.")
        return

    effective_total = num_candidates if max_attacked_samples is None else min(num_candidates, max_attacked_samples)
    progress = tqdm(total=effective_total, desc="Attacking correctly classified samples")
    attacked = 0
    success_count = 0
    saved_images = 0

    for _batch_idx, (images, labels, indices) in enumerate(dataloader):
        if max_attacked_samples is not None and attacked >= max_attacked_samples:
            break

        batch_indices = indices.tolist()
        mask_list = [correct_mask[idx] for idx in batch_indices]
        if not any(mask_list):
            continue

        batch_mask = torch.tensor(mask_list, dtype=torch.bool)

        if max_attacked_samples is not None:
            remaining = max_attacked_samples - attacked
            if remaining <= 0:
                break

            num_correct_in_batch = int(batch_mask.sum().item())
            if num_correct_in_batch > remaining:
                true_indices = batch_mask.nonzero(as_tuple=False).view(-1)
                keep_true_indices = true_indices[:remaining]
                new_mask = torch.zeros_like(batch_mask)
                new_mask[keep_true_indices] = True
                batch_mask = new_mask

        images_to_attack = images[batch_mask].to(DEVICE, non_blocking=True)
        labels_to_attack = labels[batch_mask].to(DEVICE, non_blocking=True)
        selected_dataset_indices = indices[batch_mask].tolist()
        filenames = [
            str(dataloader.dataset.samples[dataset_idx]["image_name"])
            for dataset_idx in selected_dataset_indices
        ]

        if images_to_attack.numel() == 0:
            continue

        x_adv = attacker.attack_batch(images_to_attack, labels_to_attack)

        dynamic_guide_log = getattr(attacker, "_last_dynamic_guide_log", None)
        if dynamic_guide_log and output_dir is not None:
            stats_path = Path(output_dir) / "dynamic_guide_stats.csv"
            write_header = not stats_path.exists()
            with stats_path.open("a", newline="") as f:
                fieldnames = [
                    "attacked_start",
                    "batch_size",
                    "step",
                    "clean_cosine",
                    "adv_cls_cosine",
                    "entropy",
                    "topk_change_rate",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                for row in dynamic_guide_log:
                    writer.writerow({"attacked_start": attacked, "batch_size": labels_to_attack.size(0), **row})

        with torch.inference_mode():
            logits_adv = model(x_adv, return_attn=False)
            preds_adv = logits_adv.argmax(dim=1)

        successes = (preds_adv != labels_to_attack).sum().item()
        attacked_batch = labels_to_attack.size(0)

        attacked += attacked_batch
        success_count += successes

        saved = save_adversarial_images(
            images=x_adv,
            output_dir=output_dir,
            prefix="adv",
            start_index=saved_images,
            filenames=filenames,
        )
        saved_images += len(saved)

        progress.update(attacked_batch)
        success_rate = success_count / attacked if attacked > 0 else 0.0
        progress.set_postfix(success=f"{success_rate:.4f}", attacked=attacked)

    progress.close()

    if attacked == 0:
        print("No attack was run because no selected correctly classified samples were available.")
        return

    success_rate = success_count / attacked
    print(f"Successfully attacked {success_count} / {attacked} correctly classified images.")
    print(f"Attack success rate: {success_rate:.4f}")
    print(f"Saved {saved_images} adversarial samples to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial samples with lazy aggregation attack.")
    parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0, help="L_inf perturbation budget in pixel range [0, 1].")
    parser.add_argument("--step-size", type=float, default=None, help="Step size. Defaults to epsilon / steps.")
    parser.add_argument("--steps", type=int, default=20, help="Number of attack iterations.")
    parser.add_argument("--decay", type=float, default=1.0, help="Momentum decay factor.")
    parser.add_argument("--layers", type=parse_layers, default=(-6, -5, -4, -3, -2, -1), help='Comma-separated token layers, e.g. "-6,-5,-4,-3,-2,-1".')
    parser.add_argument("--warmup-steps", type=int, default=3, help="Pure-CE steps before adding feature losses.")
    parser.add_argument("--ti-sigma", type=float, default=0.0, help="TI-FGSM Gaussian kernel sigma for gradient smoothing. 0=disabled.")
    parser.add_argument("--spectral-transition", type=float, default=0.04, help="Transition width for spectral filter.")
    parser.add_argument("--fg-top-ratio", type=float, default=0.25, help="Top patch ratio for foreground patches.")
    parser.add_argument("--lambda-anchor", type=float, default=1.0, help="Weight for aggregation hijack loss.")
    parser.add_argument("--grad-combine", type=str, default="guide_qk_response", choices=["guide_response", "dynamic_guide_response", "guide_qk_response", "background_aug_ce"], help="Gradient combination strategy.")
    parser.add_argument("--si-scales", type=int, default=1, help="Number of scale-invariant CE gradient copies.")
    parser.add_argument("--no-nesterov", action="store_true", help="Disable NI-FGSM style lookahead gradients.")
    parser.add_argument("--eot-iter", type=int, default=1, help="Number of DIM samples per SI scale.")
    parser.add_argument("--dim-resize-range", type=parse_float_range, default=(0.85, 1.0), help='DIM resize scale range, e.g. "0.85,1.0".')
    parser.add_argument("--perturb-smooth-sigma", type=float, default=0.0, help="Gaussian sigma for perturbation smoothing. 0=disabled.")
    parser.add_argument("--anchor-schedule", type=str, default="constant", choices=["constant", "linear", "cosine"], help="Anchor modulation schedule.")
    parser.add_argument("--anchor-start-step", type=int, default=None, help="First step where anchor modulation applies. Defaults to warmup steps.")
    parser.add_argument("--anchor-end-weight", type=float, default=None, help="Final anchor modulation weight for linear/cosine schedules.")
    parser.add_argument("--lazy-spectral-delta", action="store_true", help="Enable spectral perturbation filtering in the second half of attack.")
    parser.add_argument("--lazy-spectral-cutoff", type=float, default=0.25, help="Low-pass cutoff ratio for spectral perturbation filtering.")
    parser.add_argument("--attention-guide-models", type=parse_model_names, default=(), help="Comma-separated extra models for clean stable-attention guide maps.")
    parser.add_argument("--guide-type", type=str, default="postsoftmax_cls", help="Comma-separated guide types: postsoftmax_cls,qk_cls,qk_all_queries.")
    parser.add_argument("--guide-sample-mode", type=str, default="fixed", choices=["fixed", "random"], help="How to choose a guide type when listing multiple.")
    parser.add_argument("--attention-grad-smooth-sigma", type=float, default=0.0, help="Gaussian smoothing for attention/QK-response gradients. 0=disabled.")
    parser.add_argument("--patch-grad-smooth-sigma", type=float, default=0.0, help="Gaussian smoothing for guide feature gradients. 0=disabled.")
    parser.add_argument("--guide-entropy-temp", type=float, default=1.0, help="Temperature exponent for guide normalization.")
    parser.add_argument("--guide-dilate-kernel", type=int, default=1, help="Odd patch-grid max-pool kernel for expanded stable-attention guides. 1=disabled.")
    parser.add_argument("--guide-smooth-sigma", type=float, default=0.0, help="Gaussian sigma for smoothing expanded stable-attention guides. 0=disabled.")
    parser.add_argument("--guide-dynamic", action="store_true", help="Update stable-attention guide from adversarial image during attack.")
    parser.add_argument("--guide-update-interval", type=int, default=5, help="Dynamic guide update interval in attack steps.")
    parser.add_argument("--guide-ema", type=float, default=0.7, help="EMA weight for previous dynamic guide.")
    parser.add_argument("--guide-aug-copies", type=int, default=3, help="Number of augmented CE copies per SI/EOT sample.")
    parser.add_argument("--guide-aug-mode", type=parse_model_names, default=("bg_blur",), help="Comma-separated background augmentation modes: bg_blur,bg_jitter,bg_freq.")
    parser.add_argument("--guide-aug-strength", type=float, default=0.3, help="Background augmentation strength.")
    parser.add_argument("--bg-foreground-ratio", type=float, default=0.25, help="Top QK patch ratio protected as foreground by background_aug_ce.")
    parser.add_argument("--bg-background-ratio", type=float, default=0.50, help="Bottom QK patch ratio eligible for background augmentation by background_aug_ce.")
    parser.add_argument("--bg-fg-dilate-kernel", type=int, default=3, help="Odd patch-grid max-pool kernel to dilate protected foreground for background_aug_ce.")
    parser.add_argument("--output-dir", default=None, help="Output directory. In attack mode, use --output-dir outputs/attack/lazyagg.")
    parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
    parser.add_argument("--image-dir", default=IMAGE_DIR, help="Directory containing input images.")
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH, help="Path to image label annotations.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size for clean eval and attack batches.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for image decode/transform.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per DataLoader worker.")
    args = parser.parse_args()

    return args


def main(
    max_attacked_samples: int,
    epsilon: float,
    step_size: float | None,
    steps: int,
    decay: float,
    layers: tuple[int, ...],
    output_dir: str | None,
    mode: str,
    warmup_steps: int = 3,
    ti_sigma: float = 0.0,
    spectral_transition: float = 0.04,
    fg_top_ratio: float = 0.25,
    lambda_anchor: float = 1.0,
    grad_combine: str = "guide_qk_response",
    si_scales: int = 1,
    nesterov: bool = True,
    eot_iter: int = 1,
    dim_resize_range: tuple[float, float] = (0.85, 1.0),
    perturb_smooth_sigma: float = 0.0,
    anchor_schedule: str = "constant",
    anchor_start_step: int | None = None,
    anchor_end_weight: float | None = None,
    lazy_spectral_delta: bool = False,
    lazy_spectral_cutoff: float = 0.25,
    attention_guide_models_arg: tuple[str, ...] = (),
    guide_type: str = "postsoftmax_cls",
    guide_sample_mode: str = "fixed",
    attention_grad_smooth_sigma: float = 0.0,
    patch_grad_smooth_sigma: float = 0.0,
    guide_entropy_temp: float = 1.0,
    guide_dilate_kernel: int = 1,
    guide_smooth_sigma: float = 0.0,
    guide_dynamic: bool = False,
    guide_update_interval: int = 5,
    guide_ema: float = 0.7,
    guide_aug_copies: int = 3,
    guide_aug_mode: tuple[str, ...] = ("bg_blur",),
    guide_aug_strength: float = 0.3,
    bg_foreground_ratio: float = 0.25,
    bg_background_ratio: float = 0.50,
    bg_fg_dilate_kernel: int = 3,
    image_dir: str = IMAGE_DIR,
    annotations_path: str = ANNOTATIONS_PATH,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> None:
    if mode == "attack":
        resolved_output_dir = validate_attack_output_dir(output_dir=output_dir)
    else:
        resolved_output_dir = Path(output_dir) if output_dir is not None else Path("outputs") / "clean"

    dataloader, num_classes = load_data(
        image_dir_arg=image_dir,
        annotations_path_arg=annotations_path,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        img_size=img_size,
    )
    model = build_vit_model(num_classes=num_classes)

    attention_guide_models: tuple[ViTWithHook, ...] = ()
    if attention_guide_models_arg:
        attention_guide_models = tuple(
            build_vit_model(num_classes=num_classes, model_name=model_name)
            for model_name in attention_guide_models_arg
        )

    attacker = create_attacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        decay=decay,
        layers=layers,
        warmup_steps=warmup_steps,
        ti_sigma=ti_sigma,
        spectral_transition=spectral_transition,
        fg_top_ratio=fg_top_ratio,
        lambda_anchor=lambda_anchor,
        grad_combine=grad_combine,
        si_scales=si_scales,
        nesterov=nesterov,
        eot_iter=eot_iter,
        dim_resize_range=dim_resize_range,
        perturb_smooth_sigma=perturb_smooth_sigma,
        anchor_schedule=anchor_schedule,
        anchor_start_step=anchor_start_step,
        anchor_end_weight=anchor_end_weight,
        lazy_spectral_delta=lazy_spectral_delta,
        lazy_spectral_cutoff=lazy_spectral_cutoff,
        attention_guide_models=attention_guide_models,
        guide_type=guide_type,
        guide_sample_mode=guide_sample_mode,
        attention_grad_smooth_sigma=attention_grad_smooth_sigma,
        patch_grad_smooth_sigma=patch_grad_smooth_sigma,
        guide_entropy_temp=guide_entropy_temp,
        guide_dilate_kernel=guide_dilate_kernel,
        guide_smooth_sigma=guide_smooth_sigma,
        guide_dynamic=guide_dynamic,
        guide_update_interval=guide_update_interval,
        guide_ema=guide_ema,
        guide_aug_copies=guide_aug_copies,
        guide_aug_mode=guide_aug_mode,
        guide_aug_strength=guide_aug_strength,
        bg_foreground_ratio=bg_foreground_ratio,
        bg_background_ratio=bg_background_ratio,
        bg_fg_dilate_kernel=bg_fg_dilate_kernel,
    )
    _clean_acc, correct_mask = evaluate_clean_dataset(
        dataloader=dataloader,
        model=model,
    )

    if mode == "clean":
        save_clean_images(
            dataloader=dataloader,
            correct_mask=correct_mask,
            output_dir=str(resolved_output_dir),
            max_samples=max_attacked_samples,
        )
        return

    clear_directory_contents(resolved_output_dir)
    print(f"Cleared adversarial output directory: {resolved_output_dir}")

    attack_correctly_classified_samples(
        dataloader=dataloader,
        model=model,
        attacker=attacker,
        correct_mask=correct_mask,
        output_dir=str(resolved_output_dir),
        max_attacked_samples=max_attacked_samples,
    )


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    args = parse_args()
    main(
        max_attacked_samples=args.max_attacked_samples,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        decay=args.decay,
        layers=args.layers,
        warmup_steps=args.warmup_steps,
        ti_sigma=args.ti_sigma,
        spectral_transition=args.spectral_transition,
        fg_top_ratio=args.fg_top_ratio,
        lambda_anchor=args.lambda_anchor,
        grad_combine=args.grad_combine,
        si_scales=args.si_scales,
        nesterov=not args.no_nesterov,
        eot_iter=args.eot_iter,
        dim_resize_range=args.dim_resize_range,
        perturb_smooth_sigma=args.perturb_smooth_sigma,
        anchor_schedule=args.anchor_schedule,
        anchor_start_step=args.anchor_start_step,
        anchor_end_weight=args.anchor_end_weight,
        lazy_spectral_delta=args.lazy_spectral_delta,
        lazy_spectral_cutoff=args.lazy_spectral_cutoff,
        attention_guide_models_arg=args.attention_guide_models,
        guide_type=args.guide_type,
        guide_sample_mode=args.guide_sample_mode,
        attention_grad_smooth_sigma=args.attention_grad_smooth_sigma,
        patch_grad_smooth_sigma=args.patch_grad_smooth_sigma,
        guide_entropy_temp=args.guide_entropy_temp,
        guide_dilate_kernel=args.guide_dilate_kernel,
        guide_smooth_sigma=args.guide_smooth_sigma,
        guide_dynamic=args.guide_dynamic,
        guide_update_interval=args.guide_update_interval,
        guide_ema=args.guide_ema,
        guide_aug_copies=args.guide_aug_copies,
        guide_aug_mode=args.guide_aug_mode,
        guide_aug_strength=args.guide_aug_strength,
        bg_foreground_ratio=args.bg_foreground_ratio,
        bg_background_ratio=args.bg_background_ratio,
        bg_fg_dilate_kernel=args.bg_fg_dilate_kernel,
        output_dir=args.output_dir,
        mode=args.mode,
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
