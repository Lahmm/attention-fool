import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from attack import AttentionFoolImageAttacker
from nets import ViTWithAttn, build_vit_model
from utils import DEVICE, evaluate_clean_dataset, load_data, save_adversarial_images, save_clean_images

IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"
DEFAULT_IMG_SIZE = 224


def create_attacker(
    model: ViTWithAttn,
    img_size: int,
    step_size: float,
    epsilon: float,
    region_topk: int,
    num_views: int,
    noise_eps: float,
    tau: float,
    lambda_cls: float,
    lambda_align: float,
    lambda_compact: float,
    lambda_couple: float,
    norm_type: str,
    momentum_mu: float,
    log_every: int,
    steps: int,
) -> AttentionFoolImageAttacker:
    return AttentionFoolImageAttacker(
        model=model,
        img_size=img_size,
        step_size=step_size,
        eps=epsilon,
        region_topk=region_topk,
        num_views=num_views,
        noise_eps=noise_eps,
        tau=tau,
        lambda_cls=lambda_cls,
        lambda_align=lambda_align,
        lambda_compact=lambda_compact,
        lambda_couple=lambda_couple,
        norm_type=norm_type,
        momentum_mu=momentum_mu,
        log_every=log_every,
        steps=steps,
        device=DEVICE,
    )


def attack_correctly_classified_samples(
    dataloader,
    model: ViTWithAttn,
    attacker: AttentionFoolImageAttacker,
    correct_mask: list[bool],
    output_dir: str,
    max_attacked_samples: int | None,
    allowed_indices: set[int] | None = None,
) -> None:
    if allowed_indices is None:
        num_candidates = sum(correct_mask)
    else:
        num_candidates = sum(correct_mask[idx] for idx in allowed_indices)
    if num_candidates == 0:
        print("No correctly classified samples are available for attack.")
        return

    effective_total = num_candidates if max_attacked_samples is None else min(num_candidates, max_attacked_samples)
    progress = tqdm(total=effective_total, desc="Attacking correctly classified samples")
    attacked = 0
    success_count = 0
    saved_images = 0

    for images, labels, indices in dataloader:
        if max_attacked_samples is not None and attacked >= max_attacked_samples:
            break

        batch_indices = indices.tolist()
        if allowed_indices is None:
            mask_list = [correct_mask[idx] for idx in batch_indices]
        else:
            mask_list = [correct_mask[idx] and idx in allowed_indices for idx in batch_indices]
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

        images_to_attack = images[batch_mask]
        labels_to_attack = labels[batch_mask]
        if images_to_attack.numel() == 0:
            continue

        selected_indices = [
            idx for idx, keep in zip(batch_indices, batch_mask.tolist()) if keep
        ]
        dataset = dataloader.dataset
        filenames = [
            f"adv_{dataset.samples[dataset_idx]['image_path'].name}"
            for dataset_idx in selected_indices
        ]

        images_to_attack = images_to_attack.to(DEVICE)
        labels_to_attack = labels_to_attack.to(DEVICE)

        x_adv, _delta, masks, _patch_indices, extras = attacker.attack_batch(
            images_to_attack,
            labels_to_attack,
        )

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

        attacker.save_visualizations(
            clean_images=images_to_attack,
            adv_images=x_adv,
            masks=masks,
            output_dir=output_dir,
            filenames=filenames,
            extras=extras,
        )

        progress.update(attacked_batch)
        success_rate = success_count / attacked if attacked > 0 else 0.0
        progress.set_postfix(success=f"{success_rate:.4f}", attacked=attacked)

    progress.close()

    if attacked == 0:
        print("No samples were attacked because none satisfied the filtering conditions.")
        return

    success_rate = success_count / attacked
    print(f"Successful attacks: {success_count} / {attacked}")
    print(f"Attack success rate: {success_rate:.4f}")
    print(f"Saved {saved_images} adversarial images to: {output_dir}")


parser = argparse.ArgumentParser()
parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
parser.add_argument("--step-size", type=float, default=1.0 / 255.0, help="MI-FGSM step size in normalized pixel range [0, 1].")
parser.add_argument("--epsilon", type=float, default=8.0 / 255.0, help="Perturbation budget in pixel range [0, 1].")
parser.add_argument("--region-topk", type=int, default=8, help="Number of patches used to define the compact shared region.")
parser.add_argument("--num-views", type=int, default=8, help="Number of augmented views used during optimization.")
parser.add_argument("--noise-eps", type=float, default=4.0 / 255.0, help="Noise magnitude for noisy views.")
parser.add_argument("--tau", type=float, default=0.07, help="Softmax temperature for token attribution normalization.")
parser.add_argument("--lambda-cls", type=float, default=1.0, help="Misclassification loss weight.")
parser.add_argument("--lambda-align", type=float, default=1.0, help="View attribution alignment loss weight.")
parser.add_argument("--lambda-compact", type=float, default=1.0, help="Compact attribution loss weight.")
parser.add_argument("--lambda-couple", type=float, default=1.0, help="Wrong-vs-true attribution coupling loss weight.")
parser.add_argument("--norm-type", type=str, default="linf", choices=["linf", "l2"], help="Perturbation norm constraint.")
parser.add_argument("--log-every", type=int, default=50, help="Log interval for losses.")
parser.add_argument("--output-dir", default="outputs", help="Directory used to store adversarial samples.")
parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
parser.add_argument("--steps", type=int, default=50, help="Number of MI-FGSM steps.")
parser.add_argument("--momentum-mu", type=float, default=0.9, help="MI-FGSM momentum decay factor.")
parser.add_argument("--image-path", type=str, default=None, help="Optional image path or filename inside the dataset.")


def main(
    max_attacked_samples: int,
    step_size: float,
    epsilon: float,
    region_topk: int,
    num_views: int,
    noise_eps: float,
    tau: float,
    lambda_cls: float,
    lambda_align: float,
    lambda_compact: float,
    lambda_couple: float,
    norm_type: str,
    log_every: int,
    output_dir: str,
    mode: str,
    steps: int,
    momentum_mu: float,
    image_path: str | None,
    image_dir: str = IMAGE_DIR,
    annotations_path: str = ANNOTATIONS_PATH,
    img_size: int = DEFAULT_IMG_SIZE,
) -> None:
    dataloader, num_classes = load_data(
        image_dir_arg=image_dir,
        annotations_path_arg=annotations_path,
    )

    model = build_vit_model(num_classes=num_classes)
    attacker = create_attacker(
        model=model,
        img_size=img_size,
        step_size=step_size,
        epsilon=epsilon,
        region_topk=region_topk,
        num_views=num_views,
        noise_eps=noise_eps,
        tau=tau,
        lambda_cls=lambda_cls,
        lambda_align=lambda_align,
        lambda_compact=lambda_compact,
        lambda_couple=lambda_couple,
        norm_type=norm_type,
        momentum_mu=momentum_mu,
        log_every=log_every,
        steps=steps,
    )

    _, correct_mask = evaluate_clean_dataset(
        dataloader=dataloader,
        model=model,
    )

    allowed_indices: set[int] | None = None
    if image_path:
        target = Path(image_path)
        allowed_indices = set()
        dataset = dataloader.dataset
        for idx, sample in enumerate(dataset.samples):
            sample_path = sample["image_path"]
            if sample_path == target or sample_path.name == target.name:
                allowed_indices.add(idx)
        if not allowed_indices:
            print(f"Image not found in dataset: {image_path}")
            return
        filtered_mask = [False] * len(correct_mask)
        for idx in allowed_indices:
            filtered_mask[idx] = correct_mask[idx]
        correct_mask = filtered_mask

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
        allowed_indices=allowed_indices,
    )


if __name__ == "__main__":
    print(f"Running on {str(DEVICE)}")
    args = parser.parse_args()
    main(
        max_attacked_samples=args.max_attacked_samples,
        step_size=args.step_size,
        epsilon=args.epsilon,
        region_topk=args.region_topk,
        num_views=args.num_views,
        noise_eps=args.noise_eps,
        tau=args.tau,
        lambda_cls=args.lambda_cls,
        lambda_align=args.lambda_align,
        lambda_compact=args.lambda_compact,
        lambda_couple=args.lambda_couple,
        norm_type=args.norm_type,
        log_every=args.log_every,
        output_dir=args.output_dir,
        mode=args.mode,
        steps=args.steps,
        momentum_mu=args.momentum_mu,
        image_path=args.image_path,
    )
