# utils.py
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

# 默认数据路径（可以在调用 load_data 时覆盖）
image_dir = "data/clean_resized_images"
annotations_path = "data/image_name_to_class_id_and_name.json"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# 选择device
def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")
DEVICE: torch.device = get_device()


# 数据集类定义和load函数 
class ImageDataset(Dataset):
    def __init__(self, image_dir: str = image_dir, annotations_path: str = annotations_path,
        transform = None,
        target_transform = None,
        ) -> None:
        self.image_dir = Path(image_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform
        self.target_transform = target_transform

        if not self.image_dir.is_dir():
            raise ValueError(f"图片地址 {self.image_dir} 不存在")
        if not self.annotations_path.is_file():
            raise ValueError(f"标注文件 {self.annotations_path} 不存在")

        with self.annotations_path.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)

        if not isinstance(annotations, dict):
            raise ValueError("标注文件是Json格式,内部应该是一个字典")

        
        self.samples: List[Dict[str, Any]] = []
        image_names = list(annotations.keys())
        image_names.sort()

        self.missing_images: List[str] = []

        for image_name in image_names:
            label_info = annotations[image_name]
            try:
                image_path = self._find_image_file(image_name)
            except FileNotFoundError:
                self.missing_images.append(image_name)
                continue

            sample = {
                "image_path": image_path,
                "class_id": label_info.get("class_id"),
                "class_name": label_info.get("class_name"),
            }
            self.samples.append(sample)

        if not self.samples:
            raise RuntimeError(
                "ImageDataset 初始化失败：没有任何图像与标注匹配。请检查数据路径。"
            )

    def _find_image_file(self, original_name: str) -> Path:
        direct_path = self.image_dir / original_name
        if direct_path.is_file():
            return direct_path

        stem = Path(original_name).stem
        fallback_extensions = [".png", ".jpg", ".jpeg"]

        for extension in fallback_extensions:
            candidate = self.image_dir / f"{stem}{extension}"
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Unable to locate an image for {original_name} inside {self.image_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = {
            "class_id": sample["class_id"],
            "class_name": sample["class_name"],
        }

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, index

# 扩展图片并裁减后标准化
def _build_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

# 获取标签
def _label_target_transform(label: Dict[str, Any]) -> int:
    return int(label["class_id"])

# 加载数据集
def load_data(image_dir_arg: str = image_dir, annotations_path_arg: str = annotations_path,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
) -> Tuple[DataLoader, int]:
    """
    构建 DataLoader,并返回 (dataloader, num_classes)
    """
    # 构建ImageNet自定义数据集
    # 1) 图像 transform & label 的 target_transform
    transform = _build_transform(img_size=img_size)
    dataset = ImageDataset(
        image_dir=image_dir_arg,
        annotations_path=annotations_path_arg,
        transform=transform,
        target_transform=_label_target_transform,  # dict -> int
        )
    # 2) 根据 dataset.samples
    class_ids = [int(sample["class_id"]) for sample in dataset.samples]
    num_classes = max(class_ids) + 1 if class_ids else 0

    # 3) 构建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return dataloader, num_classes

# 保存图像内核
def save_images(
    images: torch.Tensor,
    output_dir: str = "outputs",
    prefix: str = "adv",
    denormalize: bool = True,
    start_index: int = 0,
    filenames: List[str] = None,
) -> List[Path]:

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    tensor = images.detach().cpu()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    if denormalize:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        tensor = tensor * std + mean

    tensor = torch.clamp(tensor, 0.0, 1.0)

    saved_paths: List[Path] = []
    if filenames is not None and len(filenames) != tensor.size(0):
        raise ValueError("filenames length must match number of images")

    for idx, img in enumerate(tensor):
        if filenames is not None:
            filename = filenames[idx]
        else:
            filename = f"{prefix}_{start_index + idx:05d}.png"
        path = output_dir_path / filename
        save_image(img, str(path))
        saved_paths.append(path)

    return saved_paths

def save_adversarial_images(
        images: torch.Tensor,
        output_dir: str,
        prefix: str,
        start_index: int,
        filenames: List[str] = None):
    saved_adv = save_images(
        images=images,
        output_dir=output_dir,
        prefix=prefix,
        start_index=start_index,
        filenames=filenames,
    )
    return saved_adv

def save_clean_images(
    dataloader,
    correct_mask: List[bool],
    output_dir: str,
    max_samples: int | None,
) -> None:
    
    total_clean = sum(correct_mask)
    if total_clean == 0:
        print("没有任何正确分类的样本可供保存")
        return

    limit = total_clean if max_samples is None else min(total_clean, max_samples)
    saved_images = 0
    progress = tqdm(total=limit, desc="保存干净样本")

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

        batch_mask_list = batch_mask.tolist()
        selected_indices = [idx for idx, keep in zip(indices.tolist(), batch_mask_list) if keep]
        dataset = dataloader.dataset
        filenames = [
            f"clean_{dataset.samples[dataset_idx]['image_path'].name}"
            for dataset_idx in selected_indices
        ]

        saved = save_images(
            clean_images,
            output_dir=output_dir,
            prefix="clean",
            start_index=saved_images,
            filenames=filenames,
        )
        saved_count = len(saved)
        saved_images += saved_count
        progress.update(saved_count)

    progress.close()
    print(f"保存了 {saved_images} 张干净样本到 {output_dir}")

## 评估正确率
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

    progress = tqdm(dataloader, desc="评估干净样本准确率")
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
    print(f"模型准确率{clean_acc:.4f}")
    return clean_acc, per_sample_correct
