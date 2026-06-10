import argparse
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
    layers: tuple[int, ...],
    ti_sigma: float = 3.0,
    dim: bool = False,
    si: bool = False,
    si_scales: int = 1,
    eot: bool = False,
    eot_iter: int = 1,
    mi: bool = False,
    mi_decay: float = 1.0,
    ni: bool = False,
    normalize_grad: bool = False,
    dim_resize_range: tuple[float, float] = (0.85, 1.0),
    dim_mode: str = "full-random",
    attention_guide_models: tuple[ViTWithHook, ...] = (),
    attention_guide_type: str = "postsoftmax_cls",
    attention_guide_build_method: str = "pixel",
    attention_guide_patch_size: int = 16,
    guide_aug: bool = False,
    guide_aug_area: str = "background",
    guide_aug_methods: tuple[str, ...] = ("dropout",),
    guide_aug_copies: int = 3,
    guide_aug_strength: float = 0.2,
    guide_grad_norm_area: str = "none",
    lowmid_grad_tuning: bool = False,
    lowmid_grad_rotation_strength: float = 0.5,
    lowmid_grad_preserve_norm: bool = True,
    lowmid_dss_filter: bool = False,
    lowmid_dss_consistency: str = "sign",
    lowmid_dss_agreement_threshold: float = 0.67,
    temporal_persistence_filter: bool = False,
    temporal_persistence_k: int = 5,
    spectral_momentum: bool = False,
    spectral_momentum_high_decay: float = 0.7,
    spectral_hook_rotation: bool = False,
    spectral_hook_rotation_strength: float = 0.5,
    project_each_step: bool = True,
) -> LazyAggregationAttacker:
    return LazyAggregationAttacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        layers=layers,
        ti_sigma=ti_sigma,
        input_diversity=dim,
        dim_resize_range=dim_resize_range,
        dim_mode=dim_mode,
        use_si=si,
        si_scales=si_scales,
        use_eot=eot,
        eot_iter=eot_iter,
        use_momentum=mi,
        momentum_decay=mi_decay,
        nesterov=ni,
        normalize_grad=normalize_grad,
        attention_guide_models=attention_guide_models,
        attention_guide_type=attention_guide_type,
        attention_guide_build_method=attention_guide_build_method,
        attention_guide_patch_size=attention_guide_patch_size,
        guide_aug=guide_aug,
        guide_aug_area=guide_aug_area,
        guide_aug_methods=guide_aug_methods,
        guide_aug_copies=guide_aug_copies,
        guide_aug_strength=guide_aug_strength,
        guide_grad_norm_area=guide_grad_norm_area,
        lowmid_grad_tuning=lowmid_grad_tuning,
        lowmid_grad_rotation_strength=lowmid_grad_rotation_strength,
        lowmid_grad_preserve_norm=lowmid_grad_preserve_norm,
        lowmid_dss_filter=lowmid_dss_filter,
        lowmid_dss_consistency=lowmid_dss_consistency,
        lowmid_dss_agreement_threshold=lowmid_dss_agreement_threshold,
        temporal_persistence_filter=temporal_persistence_filter,
        temporal_persistence_k=temporal_persistence_k,
        spectral_momentum=spectral_momentum,
        spectral_momentum_high_decay=spectral_momentum_high_decay,
        spectral_hook_rotation=spectral_hook_rotation,
        spectral_hook_rotation_strength=spectral_hook_rotation_strength,
        project_each_step=project_each_step,
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
    parser = argparse.ArgumentParser(description="Generate adversarial samples with configurable lazy aggregation attack.")
    parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0, help="L_inf perturbation budget in pixel range [0, 1].")
    parser.add_argument("--step-size", type=float, default=None, help="Step size. Defaults to epsilon / steps.")
    parser.add_argument("--steps", type=int, default=20, help="Number of attack iterations.")
    parser.add_argument("--layers", type=parse_layers, default=(-6, -5, -4, -3, -2, -1), help='Comma-separated token layers, e.g. "-6,-5,-4,-3,-2,-1".')
    parser.add_argument("--ti-sigma", type=float, default=3.0, help="TI-FGSM Gaussian kernel sigma for gradient smoothing. 0=disabled.")
    parser.add_argument("--mi", action="store_true", help="Enable momentum iterative gradients.")
    parser.add_argument("--mi-decay", type=float, default=1.0, help="Momentum decay factor used when --mi is enabled.")
    parser.add_argument("--ni", action="store_true", help="Enable Nesterov lookahead. Requires --mi.")
    parser.add_argument("--normalize-grad", action="store_true", help="Normalize input gradients before TI/momentum updates.")
    parser.add_argument("--dim", action="store_true", help="Enable input diversity (DIM).")
    parser.add_argument("--dim-resize-range", type=parse_float_range, default=(0.85, 1.0), help='DIM resize scale range, e.g. "0.85,1.0".')
    parser.add_argument("--si", action="store_true", help="Enable scale-invariant forward copies.")
    parser.add_argument("--si-scales", type=int, default=1, help="Number of scale-invariant CE gradient copies when --si is enabled.")
    parser.add_argument("--eot", action="store_true", help="Enable EOT repeated stochastic forward samples.")
    parser.add_argument("--eot-iter", type=int, default=1, help="Number of EOT samples when --eot is enabled.")
    parser.add_argument("--attention-guide-models", type=parse_model_names, default=(), help="Comma-separated extra models for clean stable-attention guide maps.")
    parser.add_argument("--attention-guide-type", type=str, default="postsoftmax_cls", help="Comma-separated guide types: postsoftmax_cls,qk_cls,qk_all_queries. The first entry is used.")
    parser.add_argument("--attention-guide-build-method", choices=["pixel", "patch"], default="pixel", help="Build guide masks as pixel-level bilinear maps or patch-wise nearest maps.")
    parser.add_argument("--attention-guide-patch-size", type=int, default=16, help="Rendered guide patch size for --attention-guide-build-method patch. Must divide --img-size.")
    parser.add_argument("--guide-aug", action="store_true", help="Enable attention-guided forward augmentation.")
    parser.add_argument("--guide-aug-area", choices=["foreground", "background", "all"], default="background", help="Region affected by guide augmentation. all ignores attention guide maps.")
    parser.add_argument("--guide-aug-method", type=parse_model_names, default=("dropout",), help="Comma-separated guide augmentation methods: dropout,jitter,freq,lowpass_gauss,laplacian_low,fft_lowboost,illumination_low,band_noise,band_noise_low,band_noise_mid,band_noise_high,colored_noise,colored_noise_low,colored_noise_mid,colored_noise_high,progressive_spectral_noise,progressive_spectral_noise_low,progressive_spectral_noise_mid,progressive_spectral_noise_high,wavelet_noise,wavelet_noise_low,wavelet_noise_mid,wavelet_noise_high,wavelet_noise_fglow_bghigh.")
    parser.add_argument("--guide-aug-copies", type=int, default=3, help="Random copies per guide augmentation method.")
    parser.add_argument("--guide-aug-strength", type=float, default=0.2, help="Guide augmentation strength.")
    parser.add_argument("--guide-grad-norm-area", choices=["none", "foreground", "background"], default="none", help="Attention-guide region whose input gradients are normalized after backprop. none disables guided gradient normalization.")
    parser.add_argument("--lowmid-grad-tuning", action="store_true", help="Enable low/mid frequency gradient tuning after TI smoothing and before momentum.")
    parser.add_argument("--lowmid-grad-rotation-strength", type=float, default=0.5, help="Givens rotation strength toward low/mid-frequency gradient subspace when --lowmid-grad-tuning is enabled.")
    parser.add_argument("--no-lowmid-grad-preserve-norm", dest="lowmid_grad_preserve_norm", action="store_false", help="Do not preserve per-sample gradient L2 norm after low/mid gradient tuning.")
    parser.add_argument("--lowmid-dss-filter", action="store_true", help="Filter low/mid gradient components by source-side augmentation direction stability before low/mid rotation and momentum.")
    parser.add_argument("--lowmid-dss-consistency", choices=["sign", "cos"], default="sign", help="Consistency rule for --lowmid-dss-filter: per-element sign agreement or per-sample cosine gate.")
    parser.add_argument("--lowmid-dss-agreement-threshold", type=float, default=0.67, help="Minimum per-element augmentation sign agreement for --lowmid-dss-consistency sign.")
    parser.add_argument("--temporal-persistence-filter", action="store_true", help="Gate gradient elements by temporal sign persistence across the last K steps before momentum accumulation.")
    parser.add_argument("--temporal-persistence-k", type=int, default=5, help="Number of past gradients to buffer for --temporal-persistence-filter.")
    parser.add_argument("--spectral-momentum", action="store_true", help="Frequency-dependent momentum decay: low/mid components accumulate fully (decay=1.0), high components decay faster.")
    parser.add_argument("--spectral-momentum-high-decay", type=float, default=0.7, help="Momentum decay factor for high-frequency gradient components when --spectral-momentum is enabled.")
    parser.add_argument("--spectral-hook-rotation", action="store_true", help="GNS-style: register backward hook on block 0 attn.qkv to rotate V-projection gradients toward low/mid frequencies during backprop.")
    parser.add_argument("--spectral-hook-rotation-strength", type=float, default=0.5, help="Rotation strength for --spectral-hook-rotation.")
    parser.add_argument("--no-step-projection", dest="project_each_step", action="store_false", help="Disable per-step L_inf projection; only clamp pixels to [0, 1] after each IFGSM-style update.")
    parser.set_defaults(lowmid_grad_preserve_norm=True, project_each_step=True)
    parser.add_argument("--output-dir", default=None, help="Output directory. In attack mode, use --output-dir outputs/attack/lazyagg.")
    parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
    parser.add_argument("--image-dir", default=IMAGE_DIR, help="Directory containing input images.")
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH, help="Path to image label annotations.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size for clean eval and attack batches.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for image decode/transform.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per DataLoader worker.")

    return parser.parse_args()


def main(
    max_attacked_samples: int,
    epsilon: float,
    step_size: float | None,
    steps: int,
    layers: tuple[int, ...],
    output_dir: str | None,
    mode: str,
    ti_sigma: float = 3.0,
    mi: bool = False,
    mi_decay: float = 1.0,
    ni: bool = False,
    normalize_grad: bool = False,
    dim: bool = False,
    dim_resize_range: tuple[float, float] = (0.85, 1.0),
    si: bool = False,
    si_scales: int = 1,
    eot: bool = False,
    eot_iter: int = 1,
    attention_guide_models_arg: tuple[str, ...] = (),
    attention_guide_type: str = "postsoftmax_cls",
    attention_guide_build_method: str = "pixel",
    attention_guide_patch_size: int = 16,
    guide_aug: bool = False,
    guide_aug_area: str = "background",
    guide_aug_methods: tuple[str, ...] = ("dropout",),
    guide_aug_copies: int = 3,
    guide_aug_strength: float = 0.2,
    guide_grad_norm_area: str = "none",
    lowmid_grad_tuning: bool = False,
    lowmid_grad_rotation_strength: float = 0.5,
    lowmid_grad_preserve_norm: bool = True,
    lowmid_dss_filter: bool = False,
    lowmid_dss_consistency: str = "sign",
    lowmid_dss_agreement_threshold: float = 0.67,
    temporal_persistence_filter: bool = False,
    temporal_persistence_k: int = 5,
    spectral_momentum: bool = False,
    spectral_momentum_high_decay: float = 0.7,
    spectral_hook_rotation: bool = False,
    spectral_hook_rotation_strength: float = 0.5,
    project_each_step: bool = True,
    image_dir: str = IMAGE_DIR,
    annotations_path: str = ANNOTATIONS_PATH,
    img_size: int = DEFAULT_IMG_SIZE,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> None:
    if attention_guide_patch_size <= 0:
        raise ValueError(f"attention_guide_patch_size must be positive, got {attention_guide_patch_size}.")
    if img_size % attention_guide_patch_size != 0:
        raise ValueError(
            f"attention_guide_patch_size must divide img_size, got "
            f"patch_size={attention_guide_patch_size}, img_size={img_size}."
        )

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
    needs_attention_guide = (guide_aug and guide_aug_area != "all") or guide_grad_norm_area != "none"
    if needs_attention_guide and attention_guide_models_arg:
        attention_guide_models = tuple(
            build_vit_model(num_classes=num_classes, model_name=model_name)
            for model_name in attention_guide_models_arg
        )

    attacker = create_attacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        layers=layers,
        ti_sigma=ti_sigma,
        dim=dim,
        si=si,
        si_scales=si_scales,
        eot=eot,
        eot_iter=eot_iter,
        mi=mi,
        mi_decay=mi_decay,
        ni=ni,
        normalize_grad=normalize_grad,
        dim_resize_range=dim_resize_range,
        attention_guide_models=attention_guide_models,
        attention_guide_type=attention_guide_type,
        attention_guide_build_method=attention_guide_build_method,
        attention_guide_patch_size=attention_guide_patch_size,
        guide_aug=guide_aug,
        guide_aug_area=guide_aug_area,
        guide_aug_methods=guide_aug_methods,
        guide_aug_copies=guide_aug_copies,
        guide_aug_strength=guide_aug_strength,
        guide_grad_norm_area=guide_grad_norm_area,
        lowmid_grad_tuning=lowmid_grad_tuning,
        lowmid_grad_rotation_strength=lowmid_grad_rotation_strength,
        lowmid_grad_preserve_norm=lowmid_grad_preserve_norm,
        lowmid_dss_filter=lowmid_dss_filter,
        lowmid_dss_consistency=lowmid_dss_consistency,
        lowmid_dss_agreement_threshold=lowmid_dss_agreement_threshold,
        temporal_persistence_filter=temporal_persistence_filter,
        temporal_persistence_k=temporal_persistence_k,
        spectral_momentum=spectral_momentum,
        spectral_momentum_high_decay=spectral_momentum_high_decay,
        spectral_hook_rotation=spectral_hook_rotation,
        spectral_hook_rotation_strength=spectral_hook_rotation_strength,
        project_each_step=project_each_step,
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
        layers=args.layers,
        ti_sigma=args.ti_sigma,
        mi=args.mi,
        mi_decay=args.mi_decay,
        ni=args.ni,
        normalize_grad=args.normalize_grad,
        dim=args.dim,
        dim_resize_range=args.dim_resize_range,
        si=args.si,
        si_scales=args.si_scales,
        eot=args.eot,
        eot_iter=args.eot_iter,
        attention_guide_models_arg=args.attention_guide_models,
        attention_guide_type=args.attention_guide_type,
        attention_guide_build_method=args.attention_guide_build_method,
        attention_guide_patch_size=args.attention_guide_patch_size,
        guide_aug=args.guide_aug,
        guide_aug_area=args.guide_aug_area,
        guide_aug_methods=args.guide_aug_method,
        guide_aug_copies=args.guide_aug_copies,
        guide_aug_strength=args.guide_aug_strength,
        guide_grad_norm_area=args.guide_grad_norm_area,
        lowmid_grad_tuning=args.lowmid_grad_tuning,
        lowmid_grad_rotation_strength=args.lowmid_grad_rotation_strength,
        lowmid_grad_preserve_norm=args.lowmid_grad_preserve_norm,
        lowmid_dss_filter=args.lowmid_dss_filter,
        lowmid_dss_consistency=args.lowmid_dss_consistency,
        lowmid_dss_agreement_threshold=args.lowmid_dss_agreement_threshold,
        temporal_persistence_filter=args.temporal_persistence_filter,
        temporal_persistence_k=args.temporal_persistence_k,
        spectral_momentum=args.spectral_momentum,
        spectral_momentum_high_decay=args.spectral_momentum_high_decay,
        spectral_hook_rotation=args.spectral_hook_rotation,
        spectral_hook_rotation_strength=args.spectral_hook_rotation_strength,
        project_each_step=args.project_each_step,
        output_dir=args.output_dir,
        mode=args.mode,
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
