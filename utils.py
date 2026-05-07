# utils.py
from functools import lru_cache
import math
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
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


def gaussian_kernel_1d(
    kernel_size: int,
    sigma: float | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gaussian low-pass mask along the hidden/channel axis."""
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}.")

    if sigma is None:
        sigma = kernel_size ** 0.5
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")

    x = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / torch.max(kernel)


def circular_low_pass_mask_2d(
    height: int,
    width: int,
    cutoff_ratio: float = 0.15,
    transition_ratio: float = 0.04,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a centered circular 2D FFT low-pass mask."""
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}.")
    if not (0.0 < cutoff_ratio <= 0.5):
        raise ValueError(f"cutoff_ratio must be in (0, 0.5], got {cutoff_ratio}.")
    if transition_ratio < 0.0:
        raise ValueError(f"transition_ratio must be non-negative, got {transition_ratio}.")

    yy = torch.arange(
        -height // 2 + 1,
        height // 2 + 1,
        device=device,
        dtype=dtype,
    ).view(height, 1)
    xx = torch.arange(
        -width // 2 + 1,
        width // 2 + 1,
        device=device,
        dtype=dtype,
    ).view(1, width)
    radius = torch.sqrt((yy / max(height, 1)) ** 2 + (xx / max(width, 1)) ** 2)

    if transition_ratio == 0.0:
        return (radius <= cutoff_ratio).to(dtype=dtype)

    transition = max(transition_ratio, torch.finfo(dtype).eps)
    return torch.sigmoid((cutoff_ratio - radius) / transition)


def image_2d_fft_low_high_maps(
    image: torch.Tensor,
    cutoff_ratio: float = 0.15,
    transition_ratio: float = 0.04,
) -> Dict[str, torch.Tensor]:
    """
    Compute model-independent image-space 2D FFT low/high frequency maps.

    image must be RGB or grayscale in [C, H, W] or [B, C, H, W]. The FFT is
    applied on the spatial dimensions of a grayscale image, unlike the ViT token
    stability helpers that operate on hidden channels.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
        squeeze_batch = True
    elif image.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError(f"image must have shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}.")

    if not torch.is_floating_point(image):
        raise TypeError("image must be a floating point tensor.")

    channels = image.size(1)
    if channels == 1:
        gray = image[:, 0]
    elif channels == 3:
        weights = torch.tensor(
            [0.2989, 0.5870, 0.1140],
            device=image.device,
            dtype=image.dtype,
        ).view(1, 3, 1, 1)
        gray = (image * weights).sum(dim=1)
    else:
        gray = image.mean(dim=1)

    height, width = gray.shape[-2:]
    work_dtype = _as_fft_float_dtype(gray.dtype)
    gray_work = gray.to(dtype=work_dtype)
    mask = circular_low_pass_mask_2d(
        height,
        width,
        cutoff_ratio=cutoff_ratio,
        transition_ratio=transition_ratio,
        device=gray_work.device,
        dtype=gray_work.dtype,
    ).view(1, height, width)

    fft = torch.fft.fft2(gray_work, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    low_fft = fft_shifted * mask
    high_fft = fft_shifted * (1.0 - mask)
    low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
    high = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real

    low_energy = low.abs()
    high_energy = high.abs()
    total_energy = low_energy + high_energy + 1e-8
    low_ratio = low_energy / total_energy
    high_ratio = high_energy / total_energy

    result: Dict[str, torch.Tensor] = {
        "low": low.to(dtype=image.dtype),
        "high": high.to(dtype=image.dtype),
        "low_energy": low_energy.to(dtype=image.dtype),
        "high_energy": high_energy.to(dtype=image.dtype),
        "low_ratio": low_ratio.to(dtype=image.dtype),
        "high_ratio": high_ratio.to(dtype=image.dtype),
    }
    if squeeze_batch:
        result = {key: value.squeeze(0) for key, value in result.items()}
    return result


def _as_fft_float_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _split_patch_tokens(
    tokens: torch.Tensor,
    has_cls_token: bool = True,
) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B, N, D], got {tuple(tokens.shape)}.")
    if has_cls_token:
        if tokens.size(1) < 2:
            raise ValueError("tokens must contain CLS plus at least one patch token.")
        return tokens[:, 1:, :]
    return tokens


def last_vit_low_pass_patch_features(
    patch_features: torch.Tensor,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Apply channel-wise 1D FFT Gaussian low-pass filtering.

    patch_features: [B, N_patch, D]. FFT is applied on D, not on the 2D patch grid.
    """
    if patch_features.ndim != 3:
        raise ValueError(
            f"patch_features must have shape [B, N_patch, D], got {tuple(patch_features.shape)}."
        )
    if not torch.is_floating_point(patch_features):
        raise TypeError("patch_features must be a floating point tensor.")

    hidden_dim = patch_features.size(-1)
    work_dtype = _as_fft_float_dtype(patch_features.dtype)
    x = patch_features.to(dtype=work_dtype)
    kernel = gaussian_kernel_1d(
        hidden_dim,
        sigma=sigma,
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, hidden_dim)

    x_fft = torch.fft.fft(x, dim=-1)
    x_fft = torch.fft.fftshift(x_fft, dim=-1)
    x_fft = x_fft * kernel
    x_fft = torch.fft.ifftshift(x_fft, dim=-1)
    x_low = torch.fft.ifft(x_fft, dim=-1).real
    return x_low.to(dtype=patch_features.dtype)


def last_vit_channel_stability_scores(
    patch_features: torch.Tensor,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Compute channel-wise FFT stability scores for each patch/channel.

    Returns scores with shape [B, N_patch, D], following the official form:
    S = x / abs(x_low - x).
    """
    x_low = last_vit_low_pass_patch_features(patch_features, sigma=sigma)
    return patch_features / torch.abs(x_low - patch_features)


def last_vit_channel_topk_indices(
    patch_features: torch.Tensor,
    topk: int = 1,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Select the most stable patch per channel with per-channel Top-K.

    Returns indices with shape [B, topk, D], indexing the patch dimension.
    """
    if patch_features.ndim != 3:
        raise ValueError(
            f"patch_features must have shape [B, N_patch, D], got {tuple(patch_features.shape)}."
        )
    num_patches = patch_features.size(1)
    if topk <= 0 or topk > num_patches:
        raise ValueError(f"topk must be in [1, {num_patches}], got {topk}.")

    scores = last_vit_channel_stability_scores(
        patch_features,
        sigma=sigma,
    )
    _values, indices = torch.topk(scores, k=topk, dim=1, largest=True)
    return indices


def last_vit_stable_patch_frequency(
    tokens: torch.Tensor,
    topk: int = 1,
    has_cls_token: bool = True,
    sigma: float | None = None,
) -> torch.Tensor:
    """
    Convert channel Top-K selections into a patch-level foreground score.

    The score is the fraction of hidden channels that selected each patch. Higher
    values indicate patches that are more stable under channel-wise low-pass
    filtering and are treated as foreground-like by the FFT heuristic.
    """
    patch_features = _split_patch_tokens(tokens, has_cls_token=has_cls_token)
    indices = last_vit_channel_topk_indices(
        patch_features,
        topk=topk,
        sigma=sigma,
    )

    batch_size, num_patches, hidden_dim = patch_features.shape
    counts = torch.zeros(
        batch_size,
        num_patches,
        device=patch_features.device,
        dtype=patch_features.dtype,
    )
    flat_indices = indices.reshape(batch_size, -1)
    increments = torch.ones_like(flat_indices, dtype=counts.dtype)
    counts.scatter_add_(1, flat_indices, increments)
    return counts / float(hidden_dim)


def last_vit_foreground_background_masks(
    foreground_scores: torch.Tensor,
    foreground_ratio: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build binary foreground/background masks from FFT patch scores.

    foreground_scores must be [B, N_patch]. The top foreground_ratio patches are
    foreground; the remaining patches are background.
    """
    if foreground_scores.ndim != 2:
        raise ValueError(
            f"foreground_scores must have shape [B, N_patch], got {tuple(foreground_scores.shape)}."
        )
    if not (0.0 < foreground_ratio <= 1.0):
        raise ValueError(f"foreground_ratio must be in (0, 1], got {foreground_ratio}.")

    batch_size, num_patches = foreground_scores.shape
    keep = max(1, int(math.ceil(num_patches * foreground_ratio)))
    _values, top_indices = torch.topk(foreground_scores, k=keep, dim=1, largest=True)

    foreground_mask = torch.zeros(
        batch_size,
        num_patches,
        device=foreground_scores.device,
        dtype=torch.bool,
    )
    foreground_mask.scatter_(1, top_indices, torch.ones_like(top_indices, dtype=torch.bool))
    background_mask = ~foreground_mask
    return foreground_mask, background_mask


def last_vit_patch_scores_to_image_map(
    patch_scores: torch.Tensor,
    img_size: int | Tuple[int, int] = 224,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Upsample square-grid patch scores [B, N_patch] to image maps [B, H, W]."""
    if patch_scores.ndim != 2:
        raise ValueError(f"patch_scores must have shape [B, N_patch], got {tuple(patch_scores.shape)}.")

    num_patches = patch_scores.size(1)
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Patch token count {num_patches} is not a square number.")

    if isinstance(img_size, int):
        output_size = (img_size, img_size)
    else:
        output_size = img_size

    grid = patch_scores.reshape(patch_scores.size(0), 1, grid_size, grid_size)
    kwargs = {"size": output_size, "mode": mode}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(grid, **kwargs).squeeze(1)


def last_vit_foreground_background_from_tokens(
    tokens: torch.Tensor,
    topk: int = 1,
    has_cls_token: bool = True,
    foreground_ratio: float = 0.3,
    img_size: int | Tuple[int, int] | None = None,
    sigma: float | None = None,
) -> Dict[str, torch.Tensor]:
    """
    End-to-end FFT foreground/background separation from encoder tokens.

    Returns patch-level scores and masks. If img_size is provided, also returns
    image-level score_map, foreground_map, and background_map.
    """
    foreground_scores = last_vit_stable_patch_frequency(
        tokens=tokens,
        topk=topk,
        has_cls_token=has_cls_token,
        sigma=sigma,
    )
    foreground_mask, background_mask = last_vit_foreground_background_masks(
        foreground_scores=foreground_scores,
        foreground_ratio=foreground_ratio,
    )

    result: Dict[str, torch.Tensor] = {
        "foreground_scores": foreground_scores,
        "foreground_mask": foreground_mask,
        "background_mask": background_mask,
    }

    if img_size is not None:
        result["score_map"] = last_vit_patch_scores_to_image_map(
            foreground_scores,
            img_size=img_size,
            mode="bilinear",
        )
        result["foreground_map"] = last_vit_patch_scores_to_image_map(
            foreground_mask.to(dtype=foreground_scores.dtype),
            img_size=img_size,
            mode="nearest",
        ).to(dtype=torch.bool)
        result["background_map"] = last_vit_patch_scores_to_image_map(
            background_mask.to(dtype=foreground_scores.dtype),
            img_size=img_size,
            mode="nearest",
        ).to(dtype=torch.bool)

    return result


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
                "image_name": image_name,
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
    filenames: List[str] | None = None,
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
    if filenames is not None and len(filenames) != tensor.size(0):
        raise ValueError(f"filenames length ({len(filenames)}) must match image batch size ({tensor.size(0)}).")

    saved_paths: List[Path] = []
    for idx, img in enumerate(tensor):
        if filenames is None:
            filename = f"{prefix}_{start_index + idx:05d}.png"
        else:
            original_name = Path(filenames[idx]).name
            separator = "" if prefix.endswith("_") else "_"
            filename = f"{prefix}{separator}{original_name}"
        path = output_dir_path / filename
        save_image(img, str(path))
        saved_paths.append(path)

    return saved_paths

def save_adversarial_images(
        images: torch.Tensor,
        output_dir: str,
        prefix: str,
        start_index: int,
        filenames: List[str] | None = None):
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
        selected_indices = indices[batch_mask].tolist()
        filenames = [
            str(dataloader.dataset.samples[dataset_idx]["image_name"])
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
