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
    models: List[ViTWithAttn],
    img_size: int,
    pgd_step_size: float,
    epsilon: float,
    steps: int,
    stage1_steps: int,
    k_common: int,
    num_views: int,
    noise_eps: float,
    importance_method: str,
    tau: float,
    lambda_focus: float,
    lambda_stab: float,
    lambda_preserve: float,
    lambda_focus2: float | None,
    lambda_stab2: float | None,
    lambda_ce1: float,
    lambda_var_model: float,
    lambda_var_aug: float,
    norm_type: str,
    use_momentum: bool,
    momentum_mu: float,
    log_every: int,
) -> AttentionFoolImageAttacker:
    attacker = AttentionFoolImageAttacker(
        model=model,
        models=models,
        img_size=img_size,
        step_size=pgd_step_size,
        eps=epsilon,
        steps=steps,
        stage1_steps=stage1_steps,
        k_common=k_common,
        num_views=num_views,
        noise_eps=noise_eps,
        importance_method=importance_method,
        tau=tau,
        lambda_focus=lambda_focus,
        lambda_stab=lambda_stab,
        lambda_preserve=lambda_preserve,
        lambda_focus2=lambda_focus2,
        lambda_stab2=lambda_stab2,
        lambda_ce1=lambda_ce1,
        lambda_var_model=lambda_var_model,
        lambda_var_aug=lambda_var_aug,
        norm_type=norm_type,
        use_momentum=use_momentum,
        momentum_mu=momentum_mu,
        log_every=log_every,
        device=DEVICE,
    )
    return attacker

# 开始攻击
def attack_correctly_classified_samples(
    dataloader,
    model: ViTWithAttn,
    attacker: AttentionFoolImageAttacker,
    correct_mask: List[bool],
    output_dir: str,
    max_attacked_samples: int | None,
    allowed_indices: set[int] | None = None,
) -> None:
    # 对正确分类的样本进行攻击
    if allowed_indices is None:
        num_candidates = sum(correct_mask)
    else:
        num_candidates = sum(correct_mask[idx] for idx in allowed_indices)
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
        if allowed_indices is None:
            mask_list = [correct_mask[idx] for idx in batch_indices]
        else:
            mask_list = [correct_mask[idx] and idx in allowed_indices for idx in batch_indices]
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
        print("由于样本数量限制或缺少正确分类样本，没有执行任何攻击。")
        return

    success_rate = success_count / attacked
    print(f"成功攻击了 {success_count} / {attacked} 张正确分类的图片.")
    print(f"攻击成功率: {success_rate:.4f}")
    print(f"保存了{saved_images}张对抗样本至: {output_dir}")

parser = argparse.ArgumentParser()
parser.add_argument("--max-attacked-samples", type=int, default=5, help="Maximum number of correctly classified samples to attack.")
parser.add_argument("--pgd-step-size", type=float, default=1.0 / 255.0, help="PGD step size in normalized pixel range [0, 1].")
parser.add_argument("--epsilon", type=float, default=8.0 / 255.0, help="L-inf perturbation budget in pixel range [0, 1].")
parser.add_argument("--ensemble-models", type=str, default=None, help="Comma-separated surrogate model names for ensemble.")
parser.add_argument("--k-common", dest="k_common", type=int, default=8, help="Top-k common patches.")
parser.add_argument("--k", dest="k_common", type=int, default=None, help="Alias for --k-common.")
parser.add_argument("--num-views", type=int, default=None, help="Total number of views (augmentations + noise).")
parser.add_argument("--num-aug-views", type=int, default=4, help="Legacy: number of light augmentation views.")
parser.add_argument("--num-noise-views", type=int, default=4, help="Legacy: number of small-noise views.")
parser.add_argument("--noise-eps", type=float, default=4.0 / 255.0, help="Noise magnitude for stability views.")
parser.add_argument("--importance-method", type=str, default="grad_token", choices=["grad_token", "legrad", "attn_rollout"], help="Token importance method.")
parser.add_argument("--tau", type=float, default=0.07, help="Softmax temperature for importance normalization.")
parser.add_argument("--stage1-steps", type=int, default=0, help="Number of Stage1 evidence amplification steps.")
parser.add_argument("--lambda-focus", type=float, default=1.0, help="Stage1 focus loss weight.")
parser.add_argument("--lambda-stab", type=float, default=1.0, help="Stage1 stability loss weight.")
parser.add_argument("--lambda-preserve", type=float, default=1.0, help="Stage2 preserve loss weight.")
parser.add_argument("--lambda-focus2", type=float, default=None, help="Stage2 focus loss weight (defaults to lambda-focus).")
parser.add_argument("--lambda-stab2", type=float, default=None, help="Stage2 stability loss weight (defaults to lambda-stab).")
parser.add_argument("--lambda-ce1", type=float, default=0.0, help="Stage1 CE weight (default 0).")
parser.add_argument("--lambda-var-model", type=float, default=1.0, help="Common-evidence variance penalty across models.")
parser.add_argument("--lambda-var-aug", type=float, default=1.0, help="Common-evidence variance penalty across augmentations.")
parser.add_argument("--norm-type", type=str, default="linf", choices=["linf", "l2"], help="Perturbation norm constraint.")
parser.add_argument("--log-every", type=int, default=10, help="Log interval for losses.")
parser.add_argument("--output-dir", default="outputs", help="Directory used to store adversarial samples.")
parser.add_argument("--mode", choices=["attack", "clean"], default="attack", help="attack: generate adversarial samples; clean: save correctly classified clean samples.")
parser.add_argument("--steps", type=int, default=100, help="Number of PGD steps.")
parser.add_argument("--use-momentum", action="store_true", help="Enable momentum in PGD.")
parser.add_argument("--momentum-mu", type=float, default=0.9, help="Momentum coefficient.")
parser.add_argument("--image-path", type=str, default=None, help="Optional image path or filename inside the dataset.")

def main(
        max_attacked_samples: int,
        pgd_step_size: float,
        epsilon: float,
        k_common: int,
        num_views: int | None,
        num_aug_views: int,
        num_noise_views: int,
        noise_eps: float,
        importance_method: str,
        tau: float,
        stage1_steps: int,
        lambda_focus: float,
        lambda_stab: float,
        lambda_preserve: float,
        lambda_focus2: float | None,
        lambda_stab2: float | None,
        lambda_ce1: float,
        lambda_var_model: float,
        lambda_var_aug: float,
        norm_type: str,
        log_every: int,
        output_dir: str,
        mode: str,
        steps: int,
        use_momentum: bool,
        momentum_mu: float,
        image_path: str | None,
        ensemble_models: str | None,
        image_dir: str = IMAGE_DIR,
        annotations_path: str = ANNOTATIONS_PATH,
        img_size: int = DEFAULT_IMG_SIZE,
        ) -> None:
    
    dataloader, num_classes = load_data(
        image_dir_arg=image_dir,
        annotations_path_arg=annotations_path,
    )

    model_names: List[str | None] = []
    if ensemble_models:
        model_names = [name.strip() for name in ensemble_models.split(",") if name.strip()]
    if not model_names:
        model_names = [None]

    models: List[ViTWithAttn] = []
    for name in model_names:
        if name:
            models.append(build_vit_model(num_classes=num_classes, model_name=name))
        else:
            models.append(build_vit_model(num_classes=num_classes))
    model = models[0]

    if num_views is None:
        num_views = max(1, num_aug_views + num_noise_views)

    attacker = create_attacker(
        model=model,
        models=models,
        img_size=img_size,
        pgd_step_size=pgd_step_size,
        epsilon=epsilon,
        steps=steps,
        stage1_steps=stage1_steps,
        k_common=k_common,
        num_views=num_views,
        noise_eps=noise_eps,
        importance_method=importance_method,
        tau=tau,
        lambda_focus=lambda_focus,
        lambda_stab=lambda_stab,
        lambda_preserve=lambda_preserve,
        lambda_focus2=lambda_focus2,
        lambda_stab2=lambda_stab2,
        lambda_ce1=lambda_ce1,
        lambda_var_model=lambda_var_model,
        lambda_var_aug=lambda_var_aug,
        norm_type=norm_type,
        use_momentum=use_momentum,
        momentum_mu=momentum_mu,
        log_every=log_every,
    )

    _, correct_mask = evaluate_clean_dataset(
        dataloader=dataloader,
        model=model,
    )

    allowed_indices: set[int] | None = None
    if image_path:
        from pathlib import Path
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

    else:
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
        pgd_step_size=args.pgd_step_size,
        epsilon=args.epsilon,
        k_common=args.k_common if args.k_common is not None else 8,
        num_views=args.num_views,
        num_aug_views=args.num_aug_views,
        num_noise_views=args.num_noise_views,
        noise_eps=args.noise_eps,
        importance_method=args.importance_method,
        tau=args.tau,
        stage1_steps=args.stage1_steps,
        lambda_focus=args.lambda_focus,
        lambda_stab=args.lambda_stab,
        lambda_preserve=args.lambda_preserve,
        lambda_focus2=args.lambda_focus2,
        lambda_stab2=args.lambda_stab2,
        lambda_ce1=args.lambda_ce1,
        lambda_var_model=args.lambda_var_model,
        lambda_var_aug=args.lambda_var_aug,
        norm_type=args.norm_type,
        log_every=args.log_every,
        output_dir=args.output_dir,
        mode=args.mode,
        steps=args.steps,
        use_momentum=args.use_momentum,
        momentum_mu=args.momentum_mu,
        image_path=args.image_path,
        ensemble_models=args.ensemble_models,
    )
