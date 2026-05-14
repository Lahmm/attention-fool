import argparse
from typing import List

import torch
from tqdm import tqdm

from attack import FFTResidualPollutionMIFGSMAttacker, MIFGSMAttacker
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
    attack_type: str,
    epsilon: float,
    step_size: float | None,
    steps: int,
    decay: float,
    layers: tuple[int, ...],
    lambda_ce: float,
    lambda_pollution: float,
    lambda_residual: float,
    fft_topk: int,
) -> MIFGSMAttacker:
    if attack_type == "fft-residual-pollution":
        return FFTResidualPollutionMIFGSMAttacker(
            model=model,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            decay=decay,
            layers=layers,
            lambda_ce=lambda_ce,
            lambda_pollution=lambda_pollution,
            lambda_residual=lambda_residual,
            fft_topk=fft_topk,
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


def attack_correctly_classified_samples(
    dataloader,
    model: ViTWithHook,
    attacker: MIFGSMAttacker,
    correct_mask: List[bool],
    output_dir: str,
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

        images_to_attack = images[batch_mask].to(DEVICE)
        labels_to_attack = labels[batch_mask].to(DEVICE)
        selected_dataset_indices = indices[batch_mask].tolist()
        filenames = [
            str(dataloader.dataset.samples[dataset_idx]["image_name"])
            for dataset_idx in selected_dataset_indices
        ]

        if images_to_attack.numel() == 0:
            continue

        x_adv = attacker.attack_batch(images_to_attack, labels_to_attack)

        with torch.no_grad():
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
    parser.add_argument("--max-attacked-samples", type=int, default=20, help="Maximum number of correctly classified samples to attack.")
    parser.add_argument("--attack-type", choices=["mifgsm", "fft-residual-pollution"], default="mifgsm", help="Attack objective to use.")
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0, help="L_inf perturbation budget in pixel range [0, 1].")
    parser.add_argument("--step-size", type=float, default=None, help="MI-FGSM step size in pixel range [0, 1]. Defaults to epsilon / steps.")
    parser.add_argument("--steps", type=int, default=10, help="Number of MI-FGSM iterations.")
    parser.add_argument("--decay", type=float, default=1.0, help="Momentum decay factor.")
    parser.add_argument("--layers", type=parse_layers, default=(-4, -2, -1), help='Comma-separated token layers for feature losses, e.g. "-4,-2,-1".')
    parser.add_argument("--lambda-ce", type=float, default=1.0, help="Weight for cross-entropy classification loss. Use 0 to disable CE attack.")
    parser.add_argument("--lambda-pollution", type=float, default=1.0, help="Weight for FFT-stable patch score pollution loss.")
    parser.add_argument("--lambda-residual", type=float, default=1.0, help="Weight for multi-layer CLS residual drift loss.")
    parser.add_argument("--fft-topk", type=int, default=1, help="Per-channel Top-K stable patch count used for FFT stability weights.")
    parser.add_argument("--output-dir", default="outputs", help="Directory used to store adversarial samples.")
    parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
    parser.add_argument("--image-dir", default=IMAGE_DIR, help="Directory containing input images.")
    parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH, help="Path to image label annotations.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input image size.")
    return parser.parse_args()


def main(
    max_attacked_samples: int,
    attack_type: str,
    epsilon: float,
    step_size: float | None,
    steps: int,
    decay: float,
    layers: tuple[int, ...],
    lambda_ce: float,
    lambda_pollution: float,
    lambda_residual: float,
    fft_topk: int,
    output_dir: str,
    mode: str,
    image_dir: str = IMAGE_DIR,
    annotations_path: str = ANNOTATIONS_PATH,
    img_size: int = DEFAULT_IMG_SIZE,
) -> None:
    dataloader, num_classes = load_data(
        image_dir_arg=image_dir,
        annotations_path_arg=annotations_path,
        img_size=img_size,
    )
    model = build_vit_model(num_classes=num_classes)
    attacker = create_attacker(
        model=model,
        attack_type=attack_type,
        epsilon=epsilon,
        step_size=step_size,
        steps=steps,
        decay=decay,
        layers=layers,
        lambda_ce=lambda_ce,
        lambda_pollution=lambda_pollution,
        lambda_residual=lambda_residual,
        fft_topk=fft_topk,
    )
    _clean_acc, correct_mask = evaluate_clean_dataset(
        dataloader=dataloader,
        model=model,
    )

    if mode == "clean":
        save_clean_images(
            dataloader=dataloader,
            correct_mask=correct_mask,
            output_dir=output_dir,
            max_samples=max_attacked_samples,
        )
        return

    attack_correctly_classified_samples(
        dataloader=dataloader,
        model=model,
        attacker=attacker,
        correct_mask=correct_mask,
        output_dir=output_dir,
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
        lambda_ce=args.lambda_ce,
        lambda_pollution=args.lambda_pollution,
        lambda_residual=args.lambda_residual,
        fft_topk=args.fft_topk,
        output_dir=args.output_dir,
        mode=args.mode,
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        img_size=args.img_size,
    )
