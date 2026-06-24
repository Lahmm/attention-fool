import argparse
import shutil
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from attack import LMDSSAttacker
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
    mi: bool = True,
    mi_decay: float = 1.0,
    ni: bool = False,
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
    dim_adjoint_echo: bool = False,
    lowmid_grad_tuning: bool = False,
    lowmid_grad_rotation_strength: float = 0.5,
    lowmid_grad_preserve_norm: bool = True,
    lowmid_dss_filter: bool = False,
    lowmid_dss_consistency: str = "sign",
    lowmid_dss_agreement_threshold: float = 0.67,
    attack_loss: str = "logits",
    feature_layer: int = 10,
) -> LMDSSAttacker:
    return LMDSSAttacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        layers=layers,
        ti_sigma=ti_sigma,
        input_diversity=dim,
        dim_resize_range=dim_resize_range,
        dim_mode=dim_mode,
        use_momentum=mi,
        momentum_decay=mi_decay,
        nesterov=ni,
        attention_guide_models=attention_guide_models,
        attention_guide_type=attention_guide_type,
        attention_guide_build_method=attention_guide_build_method,
        attention_guide_patch_size=attention_guide_patch_size,
        guide_aug=guide_aug,
        guide_aug_area=guide_aug_area,
        guide_aug_methods=guide_aug_methods,
        guide_aug_copies=guide_aug_copies,
        guide_aug_strength=guide_aug_strength,
        dim_adjoint_echo=dim_adjoint_echo,
        lowmid_grad_tuning=lowmid_grad_tuning,
        lowmid_grad_rotation_strength=lowmid_grad_rotation_strength,
        lowmid_grad_preserve_norm=lowmid_grad_preserve_norm,
        lowmid_dss_filter=lowmid_dss_filter,
        lowmid_dss_consistency=lowmid_dss_consistency,
        lowmid_dss_agreement_threshold=lowmid_dss_agreement_threshold,
        attack_loss=attack_loss,
        feature_layer=feature_layer,
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
    return Path("outputs") / "attack" / "lmdss"


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
    attacker: LMDSSAttacker,
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
    parser = argparse.ArgumentParser(description="Generate adversarial samples with the LMDSS attack.")
    parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0, help="L_inf step budget in pixel range [0, 1].")
    parser.add_argument("--step-size", type=float, default=None, help="Step size. Defaults to epsilon / steps.")
    parser.add_argument("--steps", type=int, default=20, help="Number of attack iterations.")
    parser.add_argument("--layers", type=parse_layers, default=(-6, -5, -4, -3, -2, -1), help='Comma-separated token layers, e.g. "-6,-5,-4,-3,-2,-1".')
    parser.add_argument("--ti-sigma", type=float, default=3.0, help="TI Gaussian kernel sigma for gradient smoothing. 0=disabled.")
    parser.add_argument("--mi", dest="mi", action="store_true", help="Enable momentum iterative gradients. Enabled by default.")
    parser.add_argument("--no-mi", dest="mi", action="store_false", help="Disable momentum iterative gradients.")
    parser.add_argument("--mi-decay", type=float, default=1.0, help="Momentum decay factor used when MI is enabled.")
    parser.add_argument("--ni", action="store_true", help="Enable Nesterov lookahead. Requires MI.")
    parser.add_argument("--dim", action="store_true", help="Enable input diversity (DIM).")
    parser.add_argument("--dim-resize-range", type=parse_float_range, default=(0.85, 1.0), help='DIM resize scale range, e.g. "0.85,1.0".')
    parser.add_argument("--attention-guide-models", type=parse_model_names, default=(), help="Comma-separated extra models for clean stable-attention guide maps.")
    parser.add_argument("--attention-guide-type", type=str, default="postsoftmax_cls", help="Comma-separated guide types: postsoftmax_cls,qk_cls,qk_all_queries. The first entry is used.")
    parser.add_argument("--attention-guide-build-method", choices=["pixel", "patch"], default="pixel", help="Build guide masks as pixel-level bilinear maps or patch-wise nearest maps.")
    parser.add_argument("--attention-guide-patch-size", type=int, default=16, help="Rendered guide patch size for --attention-guide-build-method patch. Must divide --img-size.")
    parser.add_argument("--guide-aug", action="store_true", help="Enable attention-guided forward augmentation.")
    parser.add_argument("--guide-aug-area", choices=["foreground", "background", "all"], default="background", help="Region affected by guide augmentation. all ignores attention guide maps.")
    parser.add_argument("--guide-aug-method", type=parse_model_names, default=("dropout",), help="Comma-separated guide augmentation methods: dropout,jitter,freq,dim_resonance,lowmid_shift,white_noise,antithetic_transport,natural_spectrum_transport,antithetic_filter_bank,multiscale_adjoint_ensemble,orthogonal_photometric_ensemble,orthogonal_spherical_smoothing.")
    parser.add_argument("--guide-aug-copies", type=int, default=3, help="Random copies per guide augmentation method.")
    parser.add_argument("--guide-aug-strength", type=float, default=0.2, help="Guide augmentation strength.")
    parser.add_argument("--dim-adjoint-echo", action="store_true", help="Apply DIM-adjoint echo after guide augmentation and before DIM/normalization.")
    parser.add_argument("--lowmid-grad-tuning", action="store_true", help="Enable low/mid frequency gradient tuning after TI smoothing and before momentum.")
    parser.add_argument("--lowmid-grad-rotation-strength", type=float, default=0.5, help="Givens rotation strength toward low/mid-frequency gradient subspace when --lowmid-grad-tuning is enabled.")
    parser.add_argument("--no-lowmid-grad-preserve-norm", dest="lowmid_grad_preserve_norm", action="store_false", help="Do not preserve per-sample gradient L2 norm after low/mid gradient tuning.")
    parser.add_argument("--lowmid-dss-filter", action="store_true", help="Measure low/mid agreement with historical momentum to modulate low/mid rotation.")
    parser.add_argument("--lowmid-dss-consistency", choices=["sign", "cos"], default="sign", help="Consistency rule for --lowmid-dss-filter: per-element sign agreement or per-sample cosine gate.")
    parser.add_argument("--lowmid-dss-agreement-threshold", type=float, default=0.67, help="Reserved agreement threshold for LMDSS compatibility.")
    parser.add_argument("--attack-loss", choices=["logits", "feature"], default="logits", help="Attack final logits with CE or one block's patch-token features with cosine distance.")
    parser.add_argument("--feature-layer", type=int, default=10, help="Transformer block index used by --attack-loss feature. Negative indices count from the end.")
    parser.set_defaults(mi=True, lowmid_grad_preserve_norm=True)
    parser.add_argument("--output-dir", default=None, help="Output directory. In attack mode, use --output-dir outputs/attack/lmdss.")
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
    mi: bool = True,
    mi_decay: float = 1.0,
    ni: bool = False,
    dim: bool = False,
    dim_resize_range: tuple[float, float] = (0.85, 1.0),
    attention_guide_models_arg: tuple[str, ...] = (),
    attention_guide_type: str = "postsoftmax_cls",
    attention_guide_build_method: str = "pixel",
    attention_guide_patch_size: int = 16,
    guide_aug: bool = False,
    guide_aug_area: str = "background",
    guide_aug_methods: tuple[str, ...] = ("dropout",),
    guide_aug_copies: int = 3,
    guide_aug_strength: float = 0.2,
    dim_adjoint_echo: bool = False,
    lowmid_grad_tuning: bool = False,
    lowmid_grad_rotation_strength: float = 0.5,
    lowmid_grad_preserve_norm: bool = True,
    lowmid_dss_filter: bool = False,
    lowmid_dss_consistency: str = "sign",
    lowmid_dss_agreement_threshold: float = 0.67,
    attack_loss: str = "logits",
    feature_layer: int = 10,
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
    needs_attention_guide = guide_aug and guide_aug_area != "all"
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
        mi=mi,
        mi_decay=mi_decay,
        ni=ni,
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
        dim_adjoint_echo=dim_adjoint_echo,
        lowmid_grad_tuning=lowmid_grad_tuning,
        lowmid_grad_rotation_strength=lowmid_grad_rotation_strength,
        lowmid_grad_preserve_norm=lowmid_grad_preserve_norm,
        lowmid_dss_filter=lowmid_dss_filter,
        lowmid_dss_consistency=lowmid_dss_consistency,
        lowmid_dss_agreement_threshold=lowmid_dss_agreement_threshold,
        attack_loss=attack_loss,
        feature_layer=feature_layer,
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
        dim=args.dim,
        dim_resize_range=args.dim_resize_range,
        attention_guide_models_arg=args.attention_guide_models,
        attention_guide_type=args.attention_guide_type,
        attention_guide_build_method=args.attention_guide_build_method,
        attention_guide_patch_size=args.attention_guide_patch_size,
        guide_aug=args.guide_aug,
        guide_aug_area=args.guide_aug_area,
        guide_aug_methods=args.guide_aug_method,
        guide_aug_copies=args.guide_aug_copies,
        guide_aug_strength=args.guide_aug_strength,
        dim_adjoint_echo=args.dim_adjoint_echo,
        lowmid_grad_tuning=args.lowmid_grad_tuning,
        lowmid_grad_rotation_strength=args.lowmid_grad_rotation_strength,
        lowmid_grad_preserve_norm=args.lowmid_grad_preserve_norm,
        lowmid_dss_filter=args.lowmid_dss_filter,
        lowmid_dss_consistency=args.lowmid_dss_consistency,
        lowmid_dss_agreement_threshold=args.lowmid_dss_agreement_threshold,
        attack_loss=args.attack_loss,
        feature_layer=args.feature_layer,
        output_dir=args.output_dir,
        mode=args.mode,
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
