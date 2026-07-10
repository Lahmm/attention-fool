from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend and backend.is_available())


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, annotations_path: str, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform
        if not self.image_dir.is_dir():
            raise ValueError(f"image directory does not exist: {self.image_dir}")
        if not self.annotations_path.is_file():
            raise ValueError(f"annotation file does not exist: {self.annotations_path}")

        annotations = json.loads(self.annotations_path.read_text(encoding="utf-8"))
        if not isinstance(annotations, dict):
            raise ValueError("annotations must be a JSON object")

        self.samples: list[dict[str, Any]] = []
        for image_name in sorted(annotations):
            image_path = self._find_image_file(image_name)
            if image_path is None:
                continue
            label = annotations[image_name]
            self.samples.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "class_id": int(label["class_id"]),
                }
            )
        if not self.samples:
            raise RuntimeError("no annotated images were found")

    def _find_image_file(self, image_name: str) -> Path | None:
        direct = self.image_dir / image_name
        if direct.is_file():
            return direct
        stem = Path(image_name).stem
        for suffix in (".png", ".jpg", ".jpeg"):
            candidate = self.image_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform is not None else image.copy()
        return tensor, sample["class_id"], index


def load_data(
    image_dir_arg: str = "data/clean_resized_images",
    annotations_path_arg: str = "data/image_name_to_class_id_and_name.json",
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> tuple[DataLoader, int]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    dataset = ImageDataset(image_dir_arg, annotations_path_arg, transform=transform)
    num_classes = max(sample["class_id"] for sample in dataset.samples) + 1
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return dataloader, num_classes


def save_adversarial_images(
    images: torch.Tensor,
    output_dir: str,
    prefix: str,
    start_index: int,
    filenames: list[str] | None = None,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tensor = images.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = torch.clamp(tensor * std + mean, 0.0, 1.0)
    if filenames is not None and len(filenames) != tensor.size(0):
        raise ValueError("filenames must match the image batch size")

    saved: list[Path] = []
    for index, image in enumerate(tensor):
        if filenames is None:
            filename = f"{prefix}_{start_index + index:05d}.png"
        else:
            filename = f"{prefix}_{Path(filenames[index]).stem}.png"
        path = output_path / filename
        save_image(image, str(path))
        saved.append(path)
    return saved
