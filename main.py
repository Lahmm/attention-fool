# main.py
import argparse
from typing import List

import torch
from tqdm import tqdm

from attack import AttentionFoolImageAttacker
from nets import ViTWithAttn, build_vit_model
from utils import DEVICE, load_data, save_adversarial_images, save_clean_images, evaluate_clean_dataset

IMAGE_DIR = "data/clean_resized_images"
ANNOTATIONS_PATH = "data/image_name_to_class_id_and_name.json"
DEFAULT_IMG_SIZE = 224

# 构建攻击器
def create_attacker(
    model: ViTWithAttn,
    img_size: int,
    pgd_step_size: float,
    loss_type: str,
    lambda_attn: float,
    steps: int,
    use_momentum: bool,
    momentum_mu: float,
    attn_layer_set: set[int],
    ) -> AttentionFoolImageAttacker:
    attacker = AttentionFoolImageAttacker(model=model,img_size=img_size,step_size=pgd_step_size,
        loss_type=loss_type,
        lambda_attn=lambda_attn,
        steps=steps,
        use_momentum=use_momentum,
        momentum_mu=momentum_mu,
        device=DEVICE,
        attn_layer_set=attn_layer_set,
    )
    return attacker

# 开始攻击
def attack_correctly_classified_samples(dataloader, model: ViTWithAttn, attacker: AttentionFoolImageAttacker, correct_mask: List[bool],
    output_dir: str,
    max_attacked_samples: int | None,
) -> None:
    # 对正确分类的样本进行攻击
    num_candidates = sum(correct_mask)
    if num_candidates == 0:
        print("没有任何正确分类的样本可供攻击。")
        return

    effective_total = num_candidates if max_attacked_samples is None else min(num_candidates, max_attacked_samples)
    progress = tqdm(total=effective_total, desc="攻击分类正确的样本")
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

        batch_mask_list = batch_mask.tolist()
        selected_indices = [idx for idx, keep in zip(batch_indices, batch_mask_list) if keep]
        dataset = dataloader.dataset
        filenames = [
            f"adv_{dataset.samples[dataset_idx]['image_path'].name}"
            for dataset_idx in selected_indices
        ]

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
        print("由于样本数量限制或缺少正确分类样本，没有执行任何攻击。")
        return

    success_rate = success_count / attacked
    print(f"成功攻击了 {success_count} / {attacked} 张正确分类的图片.")
    print(f"攻击成功率: {success_rate:.4f}")
    print(f"保存了{saved_images}张对抗样本至: {output_dir}")

parser = argparse.ArgumentParser()
parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
parser.add_argument("--pgd-step-size", type=float, default=8.0 / 255.0, help="PGD step size in normalized pixel range [0, 1].")
parser.add_argument("--output-dir", default="outputs", help="Directory used to store adversarial samples.")
parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
parser.add_argument("--loss-type", default="ce+qk_dir", help="Loss type (e.g., ce, ce+qk_dir).")
parser.add_argument("--lambda-attn", type=float, default=1.0, help="Weight for attention-related loss.")
parser.add_argument("--steps", type=int, default=250, help="Number of PGD steps.")
parser.add_argument("--use-momentum", action="store_true", help="Enable momentum in PGD.")
parser.add_argument("--momentum-mu", type=float, default=0.9, help="Momentum coefficient.")
parser.add_argument("--attn-layer-set", help="Comma-separated 1-based layer indices for QK direction aggregation.")

def main(
        max_attacked_samples: int,
        pgd_step_size: float,
        output_dir: str,
        mode: str,
        loss_type: str,
        lambda_attn: float,
        steps: int,
        use_momentum: bool,
        momentum_mu: float,
        attn_layer_set: set[int],
        image_dir: str = IMAGE_DIR,
        annotations_path: str = ANNOTATIONS_PATH,
        img_size: int = DEFAULT_IMG_SIZE,
        ) -> None:
    
    dataloader, num_classes = load_data(
        image_dir_arg=image_dir,
        annotations_path_arg=annotations_path,
    )

    model = build_vit_model(
        num_classes=num_classes,
    )

    attacker = create_attacker(
        model=model,
        img_size=img_size,
        pgd_step_size=pgd_step_size,
        loss_type=loss_type,
        lambda_attn=lambda_attn,
        steps=steps,
        use_momentum=use_momentum,
        momentum_mu=momentum_mu,
        attn_layer_set={1,3,6,12},
    )

    _, correct_mask = evaluate_clean_dataset(
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
    print(f"Running on {str(DEVICE)}")
    args = parser.parse_args()
    attn_layer_set: set[int] = set()
    if args.attn_layer_set:
        parts = [p.strip() for p in args.attn_layer_set.split(",") if p.strip()]
        attn_layer_set = {int(p) for p in parts}
    main(
        max_attacked_samples=args.max_attacked_samples,
        pgd_step_size=args.pgd_step_size,
        output_dir=args.output_dir,
        mode=args.mode,
        loss_type=args.loss_type,
        lambda_attn=args.lambda_attn,
        steps=args.steps,
        use_momentum=args.use_momentum,
        momentum_mu=args.momentum_mu,
        attn_layer_set=attn_layer_set,
    )
