# evaluate.py
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets import ViTWithAttn
from utils import DEVICE, load_data, load_model_weights


def evaluate_clean_dataset(
    dataloader: DataLoader,
    model,
    device: torch.device = DEVICE,
) -> Tuple[float, List[bool]]:
    """在攻击或训练前评估一次模型的分类准确率。"""
    model.eval()
    dataset_size = len(dataloader.dataset)
    per_sample_correct: List[bool] = [False] * dataset_size

    clean_correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Evaluating clean accuracy")
    with torch.no_grad():
        for images, labels, indices in progress:
            images = images.to(device)
            labels = labels.to(device)

            logits_clean = model(images, return_attn=False)
            preds_clean = logits_clean.argmax(dim=1)
            matches = (preds_clean == labels)

            clean_correct += matches.sum().item()
            total += labels.size(0)

            batch_indices = indices.tolist()
            for dataset_idx, is_correct in zip(batch_indices, matches.detach().cpu().tolist()):
                per_sample_correct[dataset_idx] = bool(is_correct)

            if total > 0:
                progress.set_postfix(acc=f"{clean_correct / total:.4f}")

    progress.close()

    clean_acc = clean_correct / total if total > 0 else 0.0
    print(f"Clean accuracy on dataset: {clean_acc:.4f}")
    return clean_acc, per_sample_correct

# 创建模型
def create_model(num_classes: int) -> ViTWithAttn:
    return ViTWithAttn(
        model_name="vit_base_patch16_224",
        num_classes=num_classes,
        pretrained=True,
        device=DEVICE,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ViT clean accuracy.")
    parser.add_argument("--dataset", choices=["custom", "cifar10"], default="cifar10")
    parser.add_argument("--image-dir", default="data/clean_resized_images", help="Custom dataset root.")
    parser.add_argument("--annotations-path", default="data/image_name_to_class_id_and_name.json")
    parser.add_argument("--data-dir", default=None, help="Root directory for torchvision datasets.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation ratio/count for CIFAR-10 train set.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Which CIFAR-10 split to evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--download", action="store_true", help="Download dataset if missing.")
    parser.add_argument("--weights-path", type=str, default=None, help="Checkpoint to evaluate.")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_name = args.dataset.lower()
    split = args.split
    if dataset_name == "custom":
        split = "full"

    dataloader, num_classes = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        shuffle=False,
        dataset_name=dataset_name,
        split=split,
        data_dir_arg=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
        download=args.download,
    )

    model = create_model(num_classes=num_classes)
    load_model_weights(model, args.weights_path)
    evaluate_clean_dataset(dataloader, model)


if __name__ == "__main__":
    main()
