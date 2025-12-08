# main.py
import argparse
from typing import List, Optional

import torch
from tqdm import tqdm

from attack import AttentionFoolPatchAttacker
from evaluation import evaluate_clean_dataset
from nets import ViTWithAttn
from utils import DEVICE, load_data, load_model_weights, save_adversarial_images

IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"
DEFAULT_DATASET = "custom"
DEFAULT_IMG_SIZE = 224

# 构架模型
def create_model(num_classes: int) -> ViTWithAttn:
    model = ViTWithAttn(
        model_name="vit_base_patch16_224",
        num_classes=num_classes,
        pretrained=True,
        device=DEVICE,
    )
    return model

# 构建攻击器
def create_attacker(model: ViTWithAttn, img_size: int, pgd_step_size: float) -> AttentionFoolPatchAttacker:
    attacker = AttentionFoolPatchAttacker(
        model=model,
        img_size=img_size,
        patch_size=16,
        patch_row=0,
        patch_col=0,
        steps=250,
        step_size=pgd_step_size,
        lambda_attn=1.0,
        loss_type="ce+attn",
        use_momentum=False,
        momentum_mu=0.9,
        device=DEVICE,
    )
    return attacker

# 加载数据集
def create_dataloader(image_dir,annotations_path,batch_size,num_workers,img_size,
    shuffle: bool,
    dataset_name: str,
    split: str,
    data_dir: str,
    val_split: float,
    seed: int,
    download: bool,
): # shuffle 用于打乱训练用的数据集
    return load_data(
        image_dir_arg=image_dir,
        annotations_path_arg=annotations_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        shuffle=shuffle,
        dataset_name=dataset_name,
        split=split,
        data_dir_arg=data_dir,
        val_split=val_split,
        seed=seed,
        download=download,
    )

# 开始攻击
def attack_correctly_classified_samples(dataloader, model: ViTWithAttn, attacker: AttentionFoolPatchAttacker, correct_mask: List[bool],
    output_dir: str,
    max_attacked_samples: int | None,
) -> None:
    # 对正确分类的样本进行攻击
    num_candidates = sum(correct_mask)
    if num_candidates == 0:
        print("没有任何正确分类的样本可供攻击。")
        return

    effective_total = num_candidates if max_attacked_samples is None else min(num_candidates, max_attacked_samples)
    progress = tqdm(total=effective_total, desc="Attacking correctly classified samples")
    attacked = 0
    success_count = 0
    saved_images = 0

    # 遍历整个batch 从dataloader中按batch取出 images, labels, indices
    for _batch_idx, (images, labels, indices) in enumerate(dataloader):
        # 如果已经达到攻击样本上限，则提前结束
        if max_attacked_samples is not None and attacked >= max_attacked_samples:
            break

        batch_indices = indices.tolist()
        mask_list = [correct_mask[idx] for idx in batch_indices]
        if not any(mask_list):
            continue

        # 构造当前 batch 中“正确分类样本”的布尔掩码
        batch_mask = torch.tensor(mask_list, dtype=torch.bool)

        # 如果有攻击样本上限，则可能只攻击这一 batch 中的一部分样本
        if max_attacked_samples is not None:
            remaining = max_attacked_samples - attacked
            if remaining <= 0:
                break

            num_correct_in_batch = int(batch_mask.sum().item())
            if num_correct_in_batch > remaining:
                # 只选择前 remaining 个 True 位置
                true_indices = batch_mask.nonzero(as_tuple=False).view(-1)
                keep_true_indices = true_indices[:remaining]
                new_mask = torch.zeros_like(batch_mask)
                new_mask[keep_true_indices] = True
                batch_mask = new_mask

        # 根据最终的 batch_mask 选择要攻击的样本
        images_to_attack = images[batch_mask]
        labels_to_attack = labels[batch_mask]

        if images_to_attack.numel() == 0:
            continue

        images_to_attack = images_to_attack.to(DEVICE)
        labels_to_attack = labels_to_attack.to(DEVICE)

        x_adv, _ = attacker.attack_batch(images_to_attack, labels_to_attack)

        with torch.no_grad():
            logits_adv = model(x_adv, return_attn=False)
            preds_adv = logits_adv.argmax(dim=1)

        successes = (preds_adv != labels_to_attack).sum().item()
        attacked_batch = labels_to_attack.size(0)

        attacked += attacked_batch
        success_count += successes

        saved = save_adversarial_images(
            x_adv,
            output_dir=output_dir,
            prefix="adv",
            start_index=saved_images,
        )
        saved_images += len(saved)

        progress.update(attacked_batch)
        success_rate = success_count / attacked if attacked > 0 else 0.0
        progress.set_postfix(success=f"{success_rate:.4f}", attacked=attacked)

    progress.close()

    if attacked == 0:
        print("由于样本数量限制或缺少正确分类样本，没有执行任何攻击。")
        return

    success_rate = success_count / attacked
    print(f"Successfully attacked {success_count} / {attacked} correctly classified samples.")
    print(f"Attack success rate: {success_rate:.4f}")
    print(f"Adversarial images saved under: {output_dir}")

def save_clean_samples(
    dataloader,
    correct_mask: List[bool],
    output_dir: str,
    max_samples: int | None,
) -> None:
    total_clean = sum(correct_mask)
    if total_clean == 0:
        print("没有任何正确分类的样本可供保存。")
        return

    limit = total_clean if max_samples is None else min(total_clean, max_samples)
    saved_images = 0
    progress = tqdm(total=limit, desc="Saving clean samples")

    for images, _labels, indices in dataloader:
        if max_samples is not None and saved_images >= max_samples:
            break

        mask_list = [correct_mask[idx] for idx in indices.tolist()]
        if not any(mask_list):
            continue

        batch_mask = torch.tensor(mask_list, dtype=torch.bool)
        remaining = limit - saved_images
        if remaining <= 0:
            break

        if batch_mask.sum().item() > remaining:
            true_idx = batch_mask.nonzero(as_tuple=False).view(-1)
            keep = true_idx[:remaining]
            new_mask = torch.zeros_like(batch_mask)
            new_mask[keep] = True
            batch_mask = new_mask

        clean_images = images[batch_mask]
        if clean_images.numel() == 0:
            continue

        saved = save_adversarial_images(
            clean_images,
            output_dir=output_dir,
            prefix="clean",
            start_index=saved_images,
        )
        saved_count = len(saved)
        saved_images += saved_count
        progress.update(saved_count)

    progress.close()
    print(f"保存了 {saved_images} 张干净样本到 {output_dir}")

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
parser.add_argument("--pgd-step-size", type=float, default=8.0 / 255.0, help="PGD step size in normalized pixel range [0, 1].")
parser.add_argument("--output-dir", default="outputs", help="Directory used to store adversarial samples.")
parser.add_argument("--dataset", choices=["custom", "cifar10"], default=DEFAULT_DATASET, help="Select which dataset loader to use.")
parser.add_argument("--image-dir", default=IMAGE_DIR, help="Image directory for custom dataset.")
parser.add_argument("--annotations-path", default=ANNOTATIONS_PATH, help="Annotation file for custom dataset.")
parser.add_argument("--data-dir", default=None, help="Root directory used by torchvision datasets (e.g., CIFAR-10).")
parser.add_argument("--val-split", type=float, default=0.1, help="Validation ratio/count for CIFAR-10 train/val split.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splits.")
parser.add_argument("--download", action="store_true", help="Download torchvision dataset if needed.")
parser.add_argument("--weights-path", type=str, default=None, help="Path to fine-tuned model weights.")
parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Input resolution for ViT." )
parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")


def main(image_dir: str,
    annotations_path: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    shuffle: bool,
    dataset_name: str,
    max_attacked_samples: int | None,
    pgd_step_size: float,
    output_dir: str,
    data_dir: Optional[str],
    val_split: float,
    seed: int,
    download: bool,
    weights_path: Optional[str],
    mode: str,
) -> None:
    dataset_name = dataset_name.lower()
    split = "test" if dataset_name == "cifar10" else "full"
    dataloader, num_classes = create_dataloader(
        image_dir=image_dir,
        annotations_path=annotations_path,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        shuffle=(shuffle if split != "test" else False),
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
        download=download,
    )
    model = create_model(num_classes=num_classes)
    load_model_weights(model, weights_path=weights_path)

    attacker = create_attacker(
        model=model,
        img_size=img_size,
        pgd_step_size=pgd_step_size,
    )

    _, correct_mask = evaluate_clean_dataset(
        dataloader=dataloader,
        model=model,
    )
    if mode == "clean":
        save_clean_samples(
            dataloader=dataloader,
            correct_mask=correct_mask,
            output_dir=output_dir,
            max_samples=max_attacked_samples,
        )
    else:
        attack_correctly_classified_samples(
            dataloader=dataloader,
            model=model,
            attacker=attacker,
            correct_mask=correct_mask,
            output_dir=output_dir,
            max_attacked_samples=max_attacked_samples,
        )

if __name__ == "__main__":
    print("Running Attention-Fool Patch Attack on :", DEVICE)
    args = parser.parse_args()
    main(
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        shuffle=False,
        dataset_name=args.dataset,
        max_attacked_samples=args.max_attacked_samples,
        pgd_step_size=args.pgd_step_size,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
        download=args.download,
        weights_path=args.weights_path,
        mode=args.mode,

    )
