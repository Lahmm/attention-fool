# evaluation.py
# 用于评估模型的方法，在mian.py 和 evaluate.py 中调用
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import DEVICE

# 在干净数据集上评估模型的分类准确率
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
