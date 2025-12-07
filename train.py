import argparse
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from nets import ViTWithAttn
from utils import DEVICE, load_data


def build_model(num_classes: int, freeze_backbone: bool) -> ViTWithAttn:
    model = ViTWithAttn(
        model_name="vit_base_patch16_224",
        num_classes=num_classes,
        pretrained=True,
        device=DEVICE,
    )
    if freeze_backbone:
        for name, param in model.model.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False
    return model


def train_one_epoch(
    model: ViTWithAttn,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for images, labels, _indices in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, return_attn=False)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}" if total else "nan",
            acc=f"{correct / total:.4f}" if total else "nan",
        )

    progress.close()
    avg_loss = running_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


def evaluate(
    model: ViTWithAttn,
    dataloader,
    criterion: nn.Module,
    split_name: str,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"Eval ({split_name})", leave=False)
    with torch.no_grad():
        for images, labels, _indices in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images, return_attn=False)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress.set_postfix(
                loss=f"{running_loss / total:.4f}" if total else "nan",
                acc=f"{correct / total:.4f}" if total else "nan",
            )

    progress.close()
    avg_loss = running_loss / total if total else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    data_dir: Optional[str],
    val_split: float,
    seed: int,
    download: bool,
):
    if dataset_name != "cifar10":
        raise ValueError("当前训练脚本仅支持 dataset='cifar10'")

    train_loader, num_classes = load_data(
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        shuffle=True,
        dataset_name=dataset_name,
        split="train",
        data_dir_arg=data_dir,
        val_split=val_split,
        seed=seed,
        download=download,
    )
    val_loader, _ = load_data(
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        shuffle=False,
        dataset_name=dataset_name,
        split="val",
        data_dir_arg=data_dir,
        val_split=val_split,
        seed=seed,
        download=download,
    )
    return train_loader, val_loader, num_classes


def save_checkpoint(model: ViTWithAttn, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on CIFAR-10")
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--data-dir", default=None, help="Root directory for CIFAR-10 data.")
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation ratio/count inside CIFAR-10 train set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the classification head.")
    parser.add_argument("--output-path", type=str, default="checkpoints/vit_cifar10.pth")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader, val_loader, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        data_dir=args.data_dir,
        val_split=args.val_split,
        seed=args.seed,
        download=args.download,
    )

    model = build_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    output_path = Path(args.output_path)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion, split_name="val")
        scheduler.step()

        print(
            f"Epoch {epoch} Summary | "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, output_path)

    print(f"Training finished. Best val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
