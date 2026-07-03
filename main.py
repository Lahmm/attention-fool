import argparse
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

from attack import LMDSSAttacker
from nets import DEFAULT_MODEL_NAME, WHITEBOX_MODEL_CHOICES, WhiteBoxWithHook, build_whitebox_model
from utils import (
    DEVICE,
    load_data,
    save_adversarial_images,
)

IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"
DEFAULT_IMG_SIZE = 224


def create_attacker(
    model: WhiteBoxWithHook,
    epsilon: float,
    step_size: float | None,
    steps: int,
    ti_sigma: float = 0.0,
    dim: bool = False,
    mi: bool = True,
    mi_decay: float = 1.0,
    ni: bool = False,
    dim_resize_range: tuple[float, float] = (0.85, 1.0),
    dim_mode: str = "full-random",
    guide_aug: bool = False,
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
    spatial_sign_reinforcement: bool = False,
    spatial_sign_reinforcement_sigma: float = 1.0,
    spatial_sign_reinforcement_strength: float = 0.2,
    fft_sign_regularization: bool = False,
    fft_sign_regularization_cutoff: float = 0.25,
    fft_sign_regularization_strength: float = 0.5,
    attack_loss: str = "logits",
    feature_layer: int = -2,
    feature_scope: str = "block",
) -> LMDSSAttacker:
    return LMDSSAttacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        ti_sigma=ti_sigma,
        input_diversity=dim,
        dim_resize_range=dim_resize_range,
        dim_mode=dim_mode,
        use_momentum=mi,
        momentum_decay=mi_decay,
        nesterov=ni,
        guide_aug=guide_aug,
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
        spatial_sign_reinforcement=spatial_sign_reinforcement,
        spatial_sign_reinforcement_sigma=spatial_sign_reinforcement_sigma,
        spatial_sign_reinforcement_strength=spatial_sign_reinforcement_strength,
        fft_sign_regularization=fft_sign_regularization,
        fft_sign_regularization_cutoff=fft_sign_regularization_cutoff,
        fft_sign_regularization_strength=fft_sign_regularization_strength,
        attack_loss=attack_loss,
        feature_layer=feature_layer,
        feature_scope=feature_scope,
        device=DEVICE,
    )

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


def attack_all_samples(
    dataloader,
    attacker: LMDSSAttacker,
    output_dir: str | None,
    max_attacked_samples: int | None,
) -> None:
    total_samples = len(dataloader.dataset)
    effective_total = total_samples if max_attacked_samples is None else min(total_samples, max_attacked_samples)
    progress = tqdm(total=effective_total, desc="Attacking samples")
    attacked = 0
    saved_images = 0

    for _batch_idx, (images, labels, indices) in enumerate(dataloader):
        if max_attacked_samples is not None and attacked >= max_attacked_samples:
            break

        batch_size_actual = images.size(0)
        remaining = None if max_attacked_samples is None else max_attacked_samples - attacked
        if remaining is not None and remaining <= 0:
            break

        if remaining is not None and batch_size_actual > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
            indices = indices[:remaining]
            batch_size_actual = remaining

        images_to_attack = images.to(DEVICE, non_blocking=True)
        labels_to_attack = labels.to(DEVICE, non_blocking=True)
        selected_dataset_indices = indices.tolist()
        filenames = [
            str(dataloader.dataset.samples[dataset_idx]["image_name"])
            for dataset_idx in selected_dataset_indices
        ]

        if images_to_attack.numel() == 0:
            continue

        x_adv = attacker.attack_batch(images_to_attack, labels_to_attack)

        attacked += batch_size_actual

        saved = save_adversarial_images(
            images=x_adv,
            output_dir=output_dir,
            prefix="adv",
            start_index=saved_images,
            filenames=filenames,
        )
        saved_images += len(saved)

        progress.update(batch_size_actual)
        progress.set_postfix(attacked=attacked)

    progress.close()

    if attacked == 0:
        print("No samples were attacked.")
        return

    print(f"Attacked {attacked} samples.")
    print(f"Saved {saved_images} adversarial samples to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial samples with the LMDSS attack.")
    parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0, help="L_inf step budget in pixel range [0, 1].")
    parser.add_argument("--step-size", type=float, default=None, help="Step size. Defaults to epsilon / steps.")
    parser.add_argument("--steps", type=int, default=20, help="Number of attack iterations.")
    parser.add_argument("--whitebox-model", choices=WHITEBOX_MODEL_CHOICES, default=DEFAULT_MODEL_NAME, help="White-box source model used to generate adversarial samples.")
    parser.add_argument("--ti-sigma", type=float, default=0.0, help="TI Gaussian kernel sigma for gradient smoothing. 0=disabled.")
    parser.add_argument("--mi", dest="mi", action="store_true", help="Enable momentum iterative gradients. Enabled by default.")
    parser.add_argument("--no-mi", dest="mi", action="store_false", help="Disable momentum iterative gradients.")
    parser.add_argument("--mi-decay", type=float, default=1.0, help="Momentum decay factor used when MI is enabled.")
    parser.add_argument("--ni", action="store_true", help="Enable Nesterov lookahead. Requires MI.")
    parser.add_argument("--dim", action="store_true", help="Enable input diversity (DIM).")
    parser.add_argument("--dim-resize-range", type=parse_float_range, default=(0.85, 1.0), help='DIM resize scale range, e.g. "0.85,1.0".')
    parser.add_argument("--guide-aug", action="store_true", help="Enable whole-image forward augmentation.")
    parser.add_argument("--guide-aug-method", type=parse_model_names, default=("dropout",), help="Comma-separated guide augmentation methods: dropout,jitter,freq,dim_resonance,dim_stable_edge,dim_stable_edge_mix,dim_consensus_trajectory,dim_consensus_evidence_trajectory,lowmid_shift,white_noise,antithetic_transport,natural_spectrum_transport,antithetic_filter_bank,multiscale_adjoint_ensemble,orthogonal_photometric_ensemble,orthogonal_spherical_smoothing,antithetic_jitter_cubature,feature_trajectory_dropout.")
    parser.add_argument("--guide-aug-copies", type=int, default=3, help="Random copies per guide augmentation method.")
    parser.add_argument("--guide-aug-strength", type=float, default=0.2, help="Guide augmentation strength.")
    parser.add_argument("--dim-adjoint-echo", action="store_true", help="Apply DIM-adjoint echo after guide augmentation and before DIM/normalization.")
    parser.add_argument("--lowmid-grad-tuning", action="store_true", help="Enable low/mid frequency gradient tuning after TI smoothing and before momentum.")
    parser.add_argument("--lowmid-grad-rotation-strength", type=float, default=0.5, help="Givens rotation strength toward low/mid-frequency gradient subspace when --lowmid-grad-tuning is enabled.")
    parser.add_argument("--no-lowmid-grad-preserve-norm", dest="lowmid_grad_preserve_norm", action="store_false", help="Do not preserve per-sample gradient L2 norm after low/mid gradient tuning.")
    parser.add_argument("--lowmid-dss-filter", action="store_true", help="Measure low/mid agreement with historical momentum to modulate low/mid rotation.")
    parser.add_argument("--lowmid-dss-consistency", choices=["sign", "cos"], default="sign", help="Consistency rule for --lowmid-dss-filter: per-element sign agreement or per-sample cosine gate.")
    parser.add_argument("--lowmid-dss-agreement-threshold", type=float, default=0.67, help="Reserved agreement threshold for LMDSS compatibility.")
    parser.add_argument("--spatial-sign-reinforcement", action="store_true", help="Enable pre-sign reinforcement from spatially stable local update signs.")
    parser.add_argument("--spatial-sign-reinforcement-sigma", type=float, default=1.0, help="Gaussian sigma used to estimate local dominant update signs.")
    parser.add_argument("--spatial-sign-reinforcement-strength", type=float, default=0.2, help="Strength added along confident local sign directions before update.sign().")
    parser.add_argument("--fft-sign-regularization", action="store_true", help="Apply FFT low-pass filtering to update before sign() to suppress high-freq sign-field fragmentation.")
    parser.add_argument("--fft-sign-regularization-cutoff", type=float, default=0.25, help="Frequency cutoff radius for --fft-sign-regularization. Preserves frequencies below this radius.")
    parser.add_argument("--fft-sign-regularization-strength", type=float, default=0.5, help="Interpolation strength (0=keep original, 1=fully filtered) for --fft-sign-regularization.")
    parser.add_argument("--attack-loss", choices=["logits", "feature"], default="logits", help="Attack final logits with CE or one feature layer with cosine distance.")
    parser.add_argument("--feature-layer", type=int, default=-2, help="Feature layer index used by --attack-loss feature. Negative indices count from the end.")
    parser.add_argument("--feature-scope", choices=["block", "stage"], default="block", help="Feature output sequence used by --attack-loss feature: block layers or stage outputs.")
    parser.add_argument("--output-dir", default=None, help="Output directory. In attack mode, use --output-dir outputs/attack/lmdss.")
    parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
    parser.add_argument("--image-dir", default=IMAGE_DIR, help="Directory containing input images.")
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH, help="Path to image label annotations.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size for clean eval and attack batches.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes for image decode/transform.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per DataLoader worker.")
    parser.set_defaults(mi=True, lowmid_grad_preserve_norm=True)


    return parser.parse_args()


def main(
    max_attacked_samples: int,
    epsilon: float,
    step_size: float | None,
    steps: int,
    output_dir: str | None,
    mode: str,
    whitebox_model: str = DEFAULT_MODEL_NAME,
    ti_sigma: float = 0.0,
    mi: bool = True,
    mi_decay: float = 1.0,
    ni: bool = False,
    dim: bool = False,
    dim_resize_range: tuple[float, float] = (0.85, 1.0),
    guide_aug: bool = False,
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
    spatial_sign_reinforcement: bool = False,
    spatial_sign_reinforcement_sigma: float = 1.0,
    spatial_sign_reinforcement_strength: float = 0.2,
    fft_sign_regularization: bool = False,
    fft_sign_regularization_cutoff: float = 0.25,
    fft_sign_regularization_strength: float = 0.5,
    attack_loss: str = "logits",
    feature_layer: int = -2,
    feature_scope: str = "block",
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
    model = build_whitebox_model(num_classes=num_classes, model_name=whitebox_model)

    attacker = create_attacker(
        model=model,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        ti_sigma=ti_sigma,
        dim=dim,
        mi=mi,
        mi_decay=mi_decay,
        ni=ni,
        dim_resize_range=dim_resize_range,
        guide_aug=guide_aug,
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
        spatial_sign_reinforcement=spatial_sign_reinforcement,
        spatial_sign_reinforcement_sigma=spatial_sign_reinforcement_sigma,
        spatial_sign_reinforcement_strength=spatial_sign_reinforcement_strength,
        fft_sign_regularization=fft_sign_regularization,
        fft_sign_regularization_cutoff=fft_sign_regularization_cutoff,
        fft_sign_regularization_strength=fft_sign_regularization_strength,
        attack_loss=attack_loss,
        feature_layer=feature_layer,
        feature_scope=feature_scope,
    )
    if mode == "clean":
        raise NotImplementedError("clean mode is not supported in this branch.")

    clear_directory_contents(resolved_output_dir)
    print(f"Cleared adversarial output directory: {resolved_output_dir}")

    attack_all_samples(
        dataloader=dataloader,
        attacker=attacker,
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
        ti_sigma=args.ti_sigma,
        mi=args.mi,
        mi_decay=args.mi_decay,
        ni=args.ni,
        dim=args.dim,
        dim_resize_range=args.dim_resize_range,
        whitebox_model=args.whitebox_model,
        guide_aug=args.guide_aug,
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
        spatial_sign_reinforcement=args.spatial_sign_reinforcement,
        spatial_sign_reinforcement_sigma=args.spatial_sign_reinforcement_sigma,
        spatial_sign_reinforcement_strength=args.spatial_sign_reinforcement_strength,
        fft_sign_regularization=args.fft_sign_regularization,
        fft_sign_regularization_cutoff=args.fft_sign_regularization_cutoff,
        fft_sign_regularization_strength=args.fft_sign_regularization_strength,
        attack_loss=args.attack_loss,
        feature_layer=args.feature_layer,
        feature_scope=args.feature_scope,
        output_dir=args.output_dir,
        mode=args.mode,
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
