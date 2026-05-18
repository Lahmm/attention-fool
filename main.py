import argparse
import shutil
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from attack import FFTCCAttacker, FFTCCAttackerV2, FFTCCAttackerV3, FFTCCAttackerImgFFT, FFTCCAttackerPCGrad, FFTForwardPassAttacker, LazyAggregationAttacker, MIFGSMAttacker
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
ATTACK_OUTPUT_DIR_NAMES = {
    "mifgsm": "mifgsm",
    "fft-cc": "fftcc",
    "fft-cc-v2": "fftccv2",
    "fft-cc-v3": "fftccv3",
    "fft-cc-imgfft": "fftccimgfft",
    "fft-cc-pcgrad": "fftccpcgrad",
    "fft-fpass": "fftfpass",
    "lazy-agg": "lazyagg",
}


def create_attacker(
    model: ViTWithHook,
    attack_type: str,
    epsilon: float,
    step_size: float | None,
    steps: int,
    decay: float,
    layers: tuple[int, ...],
    lambda_contrast: float,
    lambda_attn: float,
    lambda_patch_score: float,
    fft_topk: int,
    pcgrad_mode: str = "asymmetric",
    fpass_mode: str = "combined",
    warmup_steps: int = 3,
    grad_l2_norm: float = 1.0,
    ti_sigma: float = 0.0,
    spectral_cutoff: float = 0.15,
    spectral_transition: float = 0.04,
    anchor_top_ratio: float = 0.25,
    fg_top_ratio: float = 0.25,
    lambda_anchor: float = 1.0,
    grad_combine: str = "anchor_modulate",
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
    anchor_mod_alpha: float = 1.0,
    fg_mod_alpha: float = 0.5,
    anchor_mod_power: float = 1.0,
    ensemble_models: tuple[ViTWithHook, ...] = (),
) -> MIFGSMAttacker:
    if attack_type == "lazy-agg":
        return LazyAggregationAttacker(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            anchor_top_ratio=anchor_top_ratio,
            fg_top_ratio=fg_top_ratio,
            lambda_anchor=lambda_anchor,
            warmup_steps=warmup_steps,
            grad_combine=grad_combine,
            fft_topk=fft_topk,
            spectral_cutoff=spectral_cutoff,
            spectral_transition=spectral_transition,
            grad_l2_norm=grad_l2_norm,
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
            anchor_mod_alpha=anchor_mod_alpha,
            fg_mod_alpha=fg_mod_alpha,
            anchor_mod_power=anchor_mod_power,
            ensemble_models=ensemble_models,
            device=DEVICE,
        )
    if attack_type == "fft-cc-pcgrad":
        return FFTCCAttackerPCGrad(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_contrast=lambda_contrast,
            fft_topk=fft_topk,
            pcgrad_mode=pcgrad_mode,
            warmup_steps=warmup_steps,
            grad_l2_norm=grad_l2_norm,
            ti_sigma=ti_sigma,
            device=DEVICE,
        )
    if attack_type == "fft-fpass":
        # pcgrad_mode maps to ForwardPass "mode", but has a different default
        fpass_mode = fpass_mode if fpass_mode in ("hardmask", "softmask", "spectral", "combined") else "combined"
        return FFTForwardPassAttacker(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            fft_topk=fft_topk,
            lambda_mask=lambda_contrast,
            mode=fpass_mode,
            spectral_cutoff=spectral_cutoff,
            spectral_transition=spectral_transition,
            ti_sigma=ti_sigma,
            device=DEVICE,
        )
    if attack_type == "fft-cc":
        return FFTCCAttacker(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_contrast=lambda_contrast,
            fft_topk=fft_topk,
            device=DEVICE,
        )
    if attack_type == "fft-cc-v2":
        return FFTCCAttackerV2(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_contrast=lambda_contrast,
            lambda_attn=lambda_attn,
            lambda_patch_score=lambda_patch_score,
            fft_topk=fft_topk,
            device=DEVICE,
        )
    if attack_type == "fft-cc-v3":
        return FFTCCAttackerV3(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_contrast=lambda_contrast,
            fft_topk=fft_topk,
            device=DEVICE,
        )
    if attack_type == "fft-cc-imgfft":
        return FFTCCAttackerImgFFT(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_contrast=lambda_contrast,
            device=DEVICE,
        )
    if attack_type != "mifgsm":
        raise ValueError(f"Unknown attack_type: {attack_type}")
    return MIFGSMAttacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        decay=decay,
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


def expected_attack_output_dir(attack_type: str) -> Path:
    return Path("outputs") / "attack" / ATTACK_OUTPUT_DIR_NAMES[attack_type]


def validate_attack_output_dir(attack_type: str, output_dir: str | None) -> Path:
    expected = expected_attack_output_dir(attack_type)
    if output_dir is None:
        raise ValueError(
            f"Attack mode requires --output-dir. For attack_type={attack_type}, "
            f"use --output-dir {expected.as_posix()}."
        )

    provided = Path(output_dir).expanduser()
    if provided.resolve() != expected.resolve():
        # For fft-cc-pcgrad variants, allow subdirectories
        if attack_type in ("fft-cc-pcgrad", "fft-fpass", "lazy-agg") and str(provided.resolve()).startswith(str(expected.resolve())):
            return provided
        raise ValueError(
            f"Invalid --output-dir for attack_type={attack_type}: {provided}. "
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
    attacker: MIFGSMAttacker,
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
    parser = argparse.ArgumentParser(description="Generate adversarial samples with MI-FGSM.")
    parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
    parser.add_argument("--attack-type", choices=["mifgsm", "fft-cc", "fft-cc-v2", "fft-cc-v3", "fft-cc-imgfft", "fft-cc-pcgrad", "fft-fpass", "lazy-agg"], default="mifgsm", help="Attack objective to use.")
    parser.add_argument("--epsilon", type=float, default=8.0 / 255.0, help="L_inf perturbation budget in pixel range [0, 1].")
    parser.add_argument("--step-size", type=float, default=None, help="MI-FGSM step size in pixel range [0, 1]. Defaults to epsilon / steps.")
    parser.add_argument("--steps", type=int, default=10, help="Number of MI-FGSM iterations.")
    parser.add_argument("--decay", type=float, default=1.0, help="Momentum decay factor.")
    parser.add_argument("--layers", type=parse_layers, default=(-4, -2, -1), help='Comma-separated token layers for feature losses, e.g. "-4,-2,-1".')
    parser.add_argument("--lambda-contrast", type=float, default=1.0, help="Weight for FFT foreground/background contrast collapse loss.")
    parser.add_argument("--lambda-attn", type=float, default=0.3, help="Weight for attention-aware loss (fft-cc-v2).")
    parser.add_argument("--lambda-patch-score", type=float, default=0.2, help="Weight for patch score inversion loss (fft-cc-v2).")
    parser.add_argument("--fft-topk", type=int, default=1, help="Per-channel Top-K stable patch count used for FFT stability weights.")
    parser.add_argument("--pcgrad-mode", type=str, default="asymmetric", choices=["asymmetric", "symmetric", "none", "orthogonalize", "adaptive", "modulate"], help="PCGrad gradient surgery mode (fft-cc-pcgrad).")
    parser.add_argument("--fpass-mode", type=str, default="combined", choices=["hardmask", "softmask", "spectral", "combined"], help="FT-ForwardPass operation mode (fft-fpass).")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Pure-CE steps before adding feature losses (fft-cc-pcgrad).")
    parser.add_argument("--grad-l2-norm", type=float, default=1.0, help="L2 target norm per gradient before PCGrad surgery (fft-cc-pcgrad).")
    parser.add_argument("--ti-sigma", type=float, default=0.0, help="TI-FGSM Gaussian kernel sigma for gradient smoothing. 0=disabled (fft-cc-pcgrad / fft-fpass).")
    parser.add_argument("--spectral-cutoff", type=float, default=0.15, help="Low-pass cutoff ratio for spectral perturbation filtering (fft-fpass).")
    parser.add_argument("--spectral-transition", type=float, default=0.04, help="Transition width for spectral filter (fft-fpass).")
    parser.add_argument("--anchor-top-ratio", type=float, default=0.25, help="Top patch ratio used for lazy-agg background anchors.")
    parser.add_argument("--fg-top-ratio", type=float, default=0.25, help="Top patch ratio used for lazy-agg foreground patches.")
    parser.add_argument("--lambda-anchor", type=float, default=1.0, help="Weight for lazy-agg aggregation hijack loss.")
    parser.add_argument("--grad-combine", type=str, default="anchor_modulate", choices=["pcgrad_asymmetric", "sum", "ce", "anchor_modulate"], help="Gradient combination strategy for lazy-agg.")
    parser.add_argument("--si-scales", type=int, default=1, help="Number of scale-invariant CE gradients averaged by lazy-agg anchor_modulate.")
    parser.add_argument("--no-nesterov", action="store_true", help="Disable lazy-agg NI-FGSM style lookahead gradients.")
    parser.add_argument("--eot-iter", type=int, default=1, help="Number of DIM samples averaged per SI scale by lazy-agg anchor_modulate.")
    parser.add_argument("--dim-resize-range", type=parse_float_range, default=(0.85, 1.0), help='DIM resize scale range for lazy-agg, e.g. "0.85,1.0".')
    parser.add_argument("--perturb-smooth-sigma", type=float, default=0.0, help="Gaussian sigma for optional lazy-agg perturbation smoothing after each step. 0=disabled.")
    parser.add_argument("--anchor-schedule", type=str, default="constant", choices=["constant", "linear", "cosine"], help="Lazy-agg anchor modulation schedule.")
    parser.add_argument("--anchor-start-step", type=int, default=None, help="First zero-based step where lazy-agg anchor modulation can apply. Defaults to warmup steps.")
    parser.add_argument("--anchor-end-weight", type=float, default=None, help="Final lazy-agg anchor modulation weight for linear/cosine schedules. Defaults to lambda-anchor.")
    parser.add_argument("--lazy-spectral-delta", action="store_true", help="Enable lazy-agg spectral perturbation filtering in the second half of the attack.")
    parser.add_argument("--lazy-spectral-cutoff", type=float, default=0.25, help="Low-pass cutoff ratio for lazy-agg spectral perturbation filtering.")
    parser.add_argument("--anchor-mod-alpha", type=float, default=1.0, help="Anchor multiplier in anchor_modulation map.")
    parser.add_argument("--fg-mod-alpha", type=float, default=0.5, help="Foreground multiplier in anchor_modulation map.")
    parser.add_argument("--anchor-mod-power", type=float, default=1.0, help="Power exponent applied to token_map before normalization.")
    parser.add_argument("--ensemble-source-models", type=parse_model_names, default=(), help="Comma-separated extra lazy-agg source models whose CE gradients are averaged with the primary ViT.")
    parser.add_argument("--output-dir", default=None, help="Output directory. In attack mode this is required and must match outputs/attack/<attack-name>.")
    parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
    parser.add_argument("--image-dir", default=IMAGE_DIR, help="Directory containing input images.")
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH, help="Path to image label annotations.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size for clean eval and attack batches.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for image decode/transform.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per DataLoader worker.")
    args = parser.parse_args()
    if args.attack_type == "lazy-agg":
        if args.epsilon == 8.0 / 255.0:
            args.epsilon = 16.0 / 255.0
        if args.steps == 10:
            args.steps = 20
        if args.layers == (-4, -2, -1):
            args.layers = (-6, -5, -4, -3, -2, -1)
    return args


def main(
    max_attacked_samples: int,
    attack_type: str,
    epsilon: float,
    step_size: float | None,
    steps: int,
    decay: float,
    layers: tuple[int, ...],
    lambda_contrast: float,
    lambda_attn: float,
    lambda_patch_score: float,
    fft_topk: int,
    output_dir: str | None,
    mode: str,
    pcgrad_mode: str = "asymmetric",
    fpass_mode: str = "combined",
    warmup_steps: int = 3,
    grad_l2_norm: float = 1.0,
    ti_sigma: float = 0.0,
    spectral_cutoff: float = 0.15,
    spectral_transition: float = 0.04,
    anchor_top_ratio: float = 0.25,
    fg_top_ratio: float = 0.25,
    lambda_anchor: float = 1.0,
    grad_combine: str = "anchor_modulate",
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
    anchor_mod_alpha: float = 1.0,
    fg_mod_alpha: float = 0.5,
    anchor_mod_power: float = 1.0,
    ensemble_source_models: tuple[str, ...] = (),
    image_dir: str = IMAGE_DIR,
    annotations_path: str = ANNOTATIONS_PATH,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> None:
    if mode == "attack":
        resolved_output_dir = validate_attack_output_dir(
            attack_type=attack_type,
            output_dir=output_dir,
        )
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
    ensemble_models: tuple[ViTWithHook, ...] = ()
    if attack_type == "lazy-agg" and ensemble_source_models:
        ensemble_models = tuple(
            build_vit_model(num_classes=num_classes, model_name=model_name)
            for model_name in ensemble_source_models
        )
    attacker = create_attacker(
        model=model,
        attack_type=attack_type,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        decay=decay,
        layers=layers,
        lambda_contrast=lambda_contrast,
        lambda_attn=lambda_attn,
        lambda_patch_score=lambda_patch_score,
        fft_topk=fft_topk,
        pcgrad_mode=pcgrad_mode,
        fpass_mode=fpass_mode,
        warmup_steps=warmup_steps,
        grad_l2_norm=grad_l2_norm,
        ti_sigma=ti_sigma,
        spectral_cutoff=spectral_cutoff,
        spectral_transition=spectral_transition,
        anchor_top_ratio=anchor_top_ratio,
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
        anchor_mod_alpha=anchor_mod_alpha,
        fg_mod_alpha=fg_mod_alpha,
        anchor_mod_power=anchor_mod_power,
        ensemble_models=ensemble_models,
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
        attack_type=args.attack_type,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        decay=args.decay,
        layers=args.layers,
        lambda_contrast=args.lambda_contrast,
        lambda_attn=args.lambda_attn,
        lambda_patch_score=args.lambda_patch_score,
        fft_topk=args.fft_topk,
        pcgrad_mode=args.pcgrad_mode,
        fpass_mode=args.fpass_mode,
        warmup_steps=args.warmup_steps,
        grad_l2_norm=args.grad_l2_norm,
        ti_sigma=args.ti_sigma,
        spectral_cutoff=args.spectral_cutoff,
        spectral_transition=args.spectral_transition,
        anchor_top_ratio=args.anchor_top_ratio,
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
        anchor_mod_alpha=args.anchor_mod_alpha,
        fg_mod_alpha=args.fg_mod_alpha,
        anchor_mod_power=args.anchor_mod_power,
        ensemble_source_models=args.ensemble_source_models,
        output_dir=args.output_dir,
        mode=args.mode,
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
