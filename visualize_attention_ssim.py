import argparse
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets import build_vit_model, DEFAULT_MODEL_NAME
from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD, load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute SSIM matrix between attention weight maps of different layers."
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to a single image file.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"outputs/attention_ssim_{DEFAULT_MODEL_NAME}",
        help="Where to store visualization figures.",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes. If omitted, infer from annotations.")
    parser.add_argument("--dataset-dir", type=str, default="data/clean_resized_images", help="Dataset directory used to infer class count.")
    parser.add_argument("--annotations-path", type=str, default="data/image_name_to_class_id_and_name.json", help="Annotations file used to infer class count.")
    parser.add_argument("--c1", type=float, default=1e-4, help="SSIM constant C1.")
    parser.add_argument("--c2", type=float, default=9e-4, help="SSIM constant C2.")
    parser.add_argument("--ssim-threshold", type=float, default=0.75, help="SSIM threshold to select layers.")
    return parser.parse_args()


def build_model(num_classes: int):
    model = build_vit_model(num_classes=num_classes)
    model.eval()
    return model


def resolve_num_classes(num_classes: Optional[int], dataset_dir: Path, annotations_path: Path, img_size: int) -> int:
    if num_classes is not None:
        return num_classes
    _, inferred = load_data(
        image_dir_arg=str(dataset_dir),
        annotations_path_arg=str(annotations_path),
        batch_size=1,
        num_workers=0,
        img_size=img_size,
    )
    return inferred


def preprocess_image(path: Path, img_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BICUBIC)
    np_img = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def attention_weights_from_logits(attn_logits: torch.Tensor) -> np.ndarray:
    # attn_logits: [B, H, N, N]
    attn_weights = torch.softmax(attn_logits, dim=-1)
    attn_weights = attn_weights.mean(dim=1)[0]  # [N, N]
    return attn_weights.detach().cpu().numpy()


def minmax_normalize(a: np.ndarray) -> np.ndarray:
    a_min = a.min()
    a_max = a.max()
    return (a - a_min) / (a_max - a_min + 1e-8)


def ssim(a: np.ndarray, b: np.ndarray, c1: float, c2: float) -> float:
    mu_a = a.mean()
    mu_b = b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / (den + 1e-8))


def compute_ssim_matrix(attn_mats: List[np.ndarray], c1: float, c2: float) -> np.ndarray:
    num_layers = len(attn_mats)
    mats = [minmax_normalize(mat) for mat in attn_mats]
    ssim_mat = np.zeros((num_layers, num_layers), dtype=np.float32)
    for i in range(num_layers):
        for j in range(num_layers):
            ssim_mat[i, j] = ssim(mats[i], mats[j], c1=c1, c2=c2)
    np.fill_diagonal(ssim_mat, 1.0)
    return ssim_mat


def save_heatmap(matrix: np.ndarray, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def compute_attention_heatmap(attn_weights: np.ndarray, img_size: int) -> np.ndarray:
    cls_attn = attn_weights[0, 1:]
    num_tokens = cls_attn.size
    grid_size = int(num_tokens**0.5)
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a perfect square.")
    cls_attn = cls_attn.reshape(1, 1, grid_size, grid_size)
    cls_attn = cls_attn / (cls_attn.max() + 1e-8)
    heatmap = F.interpolate(
        torch.from_numpy(cls_attn).float(),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return heatmap.squeeze().numpy()


def save_overlay(original: np.ndarray, heatmap: np.ndarray, output_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(original)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.6)
    axes[1].set_title("Combined Attention Overlay")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def select_layers_by_threshold(ssim_mat: np.ndarray, threshold: float) -> List[int]:
    num_layers = ssim_mat.shape[0]
    selected = []
    for i in range(num_layers):
        row = np.delete(ssim_mat[i], i)
        if np.any(row > threshold):
            selected.append(i)
    if not selected:
        mean_scores = (ssim_mat.sum(axis=1) - 1.0) / max(num_layers - 1, 1)
        selected = [int(np.argmax(mean_scores))]
    return selected


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    annotations_path = Path(args.annotations_path)
    num_classes = resolve_num_classes(args.num_classes, dataset_dir, annotations_path, args.img_size)
    model = build_model(num_classes=num_classes)

    tensor = preprocess_image(image_path, args.img_size).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        _logits, attn_list = model(tensor, return_attn=True)

    if not attn_list:
        raise ValueError("Model did not return any attention tensors.")

    attn_mats = [attention_weights_from_logits(attn_logits) for attn_logits in attn_list]
    ssim_mat = compute_ssim_matrix(attn_mats, c1=args.c1, c2=args.c2)

    output_path = output_dir / f"{image_path.stem}_attention_ssim.png"
    title = f"{image_path.name} | SSIM between layers"
    save_heatmap(ssim_mat, output_path, title)
    print("SSIM matrix:")
    print(ssim_mat)
    print(f"Saved SSIM heatmap to {output_path}")

    selected_layers = select_layers_by_threshold(ssim_mat, args.ssim_threshold)
    weights = np.ones(len(selected_layers), dtype=np.float32) / max(len(selected_layers), 1)
    combined = np.zeros_like(attn_mats[0], dtype=np.float32)
    for idx, w in zip(selected_layers, weights):
        combined += w * attn_mats[idx]

    npy_path = output_dir / f"{image_path.stem}_attention_ssim_selected.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, combined)
    print(f"Selected layers (threshold {args.ssim_threshold}): {selected_layers}")
    print(f"Layer weights: {weights.tolist()}")
    print(f"Saved combined attention weights to {npy_path}")

    original = np.array(Image.open(image_path).convert("RGB"))
    heatmap = compute_attention_heatmap(combined, args.img_size)
    overlay_path = output_dir / f"{image_path.stem}_attention_ssim_selected_overlay.png"
    overlay_title = f"{image_path.name} | SSIM-selected layers {selected_layers}"
    save_overlay(original, heatmap, overlay_path, overlay_title)
    print(f"Saved combined attention overlay to {overlay_path}")


if __name__ == "__main__":
    main()
