# utils.py
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 默认数据路径（可以在调用 load_data 时覆盖）
image_dir = "data/clean_resized_images"
annotations_path = "data/image_name_to_class_id_and_name.json"
CIFAR10_DIR = "data/cifar10"

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

    def __init__(
        self,
        image_dir: str = image_dir,
        annotations_path: str = annotations_path,
        transform = None,
        target_transform = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform
        self.target_transform = target_transform

        if not self.image_dir.is_dir():
            raise ValueError(f"Image directory {self.image_dir} does not exist.")
        if not self.annotations_path.is_file():
            raise ValueError(f"Annotation file {self.annotations_path} does not exist.")

        with self.annotations_path.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)

        if not isinstance(annotations, dict):
            raise ValueError("Annotation file must contain a dictionary.")

        
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
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

# 获取标签
def _label_target_transform(label: Dict[str, Any]) -> int:
    return int(label["class_id"])

# 将原本只有 imaghe 和 label 的 Dataset 包装成返回 (image, label, index) 的 Dataset
class _IndexedSubset(Dataset):

    def __init__(self,
        base_dataset,
        indices: List[int],
        transform = None,
        target_transform = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        dataset_index = self.indices[index]
        image, label = self.base_dataset[dataset_index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label, dataset_index

# cifra10 分训练集和测试集处理
def _cifar10_transforms(img_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        resize_size = int(img_size * 256 / 224)
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

# 训练集划分train/val子集
@lru_cache(maxsize=None)
def _cifar10_train_val_indices(val_count: int, seed: int) -> Tuple[List[int], List[int]]:
    total = 50000  # official CIFAR-10 train split size
    val_count = max(0, min(val_count, total))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator).tolist()

    val_indices = perm[:val_count]
    train_indices = perm[val_count:]
    return train_indices, val_indices


def _resolve_val_count(val_split: float | int, total: int) -> int:
    if isinstance(val_split, float) and 0.0 < val_split < 1.0:
        return int(total * val_split)
    try:
        count = int(val_split)
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, total))

# 构建cifar10数据集
def _build_cifar10_dataset(
    split: str,
    img_size: int,
    data_dir_arg: Optional[str],
    val_split: float | int,
    seed: int,
    download: bool,
):
    data_root = Path(data_dir_arg or CIFAR10_DIR)
    split = split.lower()

    if split not in {"train", "val", "test"}:
        raise ValueError("CIFAR-10 split 必须是 'train'、'val' 或 'test'")

    if split == "test":
        base_dataset = datasets.CIFAR10(
            root=data_root,
            train=False,
            download=download,
            transform=None,
        )
        indices = list(range(len(base_dataset)))
        dataset = _IndexedSubset(
            base_dataset=base_dataset,
            indices=indices,
            transform=_cifar10_transforms(img_size, train=False),
        )
    else:
        base_dataset = datasets.CIFAR10(
            root=data_root,
            train=True,
            download=download,
            transform=None,
        )
        val_count = _resolve_val_count(val_split, len(base_dataset))
        train_indices, val_indices = _cifar10_train_val_indices(val_count=val_count, seed=seed)
        if split == "train":
            indices = train_indices
            transform = _cifar10_transforms(img_size, train=True)
        else:
            if val_count == 0:
                raise ValueError("val_split 需要大于 0 才能构建验证集。")
            indices = val_indices
            transform = _cifar10_transforms(img_size, train=False)
        dataset = _IndexedSubset(
            base_dataset=base_dataset,
            indices=indices,
            transform=transform,
        )

    num_classes = 10
    return dataset, num_classes

# 加载数据集
def load_data(image_dir_arg: str = image_dir, annotations_path_arg: str = annotations_path,
    batch_size: int = 16,
    num_workers: int = 4,
    img_size: int = 224,
    shuffle: bool = False,
    dataset_name: str = "custom",
    split: str = "full",
    data_dir_arg: Optional[str] = None,
    val_split: float | int = 0.1,
    seed: int = 42,
    download: bool = False,
) -> Tuple[DataLoader, int]:
    """
    构建 DataLoader，并返回 (dataloader, num_classes)

    - dataloader: batch 输出为 (images, labels, indices)
        * images: [B, 3, H, W] 的 float tensor
        * labels: [B] 的 LongTensor（class_id）
        * indices: [B] 的 LongTensor，表示样本在数据集中的索引
    - num_classes: 根据标注文件自动推断出来的类别数（最大 class_id + 1）
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "custom":
        # 1) 图像 transform & label 的 target_transform
        transform = _build_transform(img_size=img_size)
        dataset = ImageDataset(
            image_dir=image_dir_arg,
            annotations_path=annotations_path_arg,
            transform=transform,
            target_transform=_label_target_transform,  # dict -> int
        )

        # 2) 根据 dataset.samples 自动计算类别数（不写死）
        class_ids = [int(sample["class_id"]) for sample in dataset.samples]
        num_classes = max(class_ids) + 1 if class_ids else 0

        if split != "full":
            raise ValueError("当前自定义数据集仅支持 split='full'。")

    elif dataset_name == "cifar10":
        dataset, num_classes = _build_cifar10_dataset(
            split=split,
            img_size=img_size,
            data_dir_arg=data_dir_arg,
            val_split=val_split,
            seed=seed,
            download=download,
        )
    else:
        raise ValueError(f"未知数据集：{dataset_name}")

    # 3) 构建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return dataloader, num_classes

# 保存攻击后的对抗图像
def save_adversarial_images(
    images: torch.Tensor,
    output_dir: str = "outputs",
    prefix: str = "adv",
    denormalize: bool = True,
    start_index: int = 0,
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
    for idx, img in enumerate(tensor):
        filename = f"{prefix}_{start_index + idx:05d}.png"
        path = output_dir_path / filename
        save_image(img, str(path))
        saved_paths.append(path)

    return saved_paths

# 加载模型权重
def load_model_weights(model: torch.nn.Module, weights_path: Optional[str], device: torch.device = DEVICE) -> None:
    """Load model state dict if a valid path is provided."""
    if not weights_path:
        return
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"权重文件 {path} 不存在。")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"Loaded weights from {path}")
