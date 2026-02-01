import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets import build_vit_model, DEFAULT_MODEL_NAME
from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD, load_data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attention maps for all heads at a specific transformer layer."
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to a single image file.")
    parser.add_argument("--layer-idx", type=int, required=True, help="0-based transformer layer index.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"outputs/attention_heads_{DEFAULT_MODEL_NAME}",
        help="Where to store visualization figures.",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes. If omitted, infer from annotations.")
    parser.add_argument("--dataset-dir", type=str, default="data/clean_resized_images", help="Dataset directory used to infer class count.")
    parser.add_argument("--annotations-path", type=str, default="data/image_name_to_class_id_and_name.json", help="Annotations file used to infer class count.")
    parser.add_argument("--save-grid", action="store_true", help="Also save a grid of head heatmaps.")
    parser.add_argument("--grid-cols", type=int, default=4, help="Number of columns in the grid view.")
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


def compute_head_heatmaps(attn_logits: torch.Tensor, img_size: int) -> np.ndarray:
    attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, H, N, N]
    attn_weights = attn_weights[0]  # [H, N, N]

    cls_attn = attn_weights[:, 0, 1:]  # [H, N-1]
    num_tokens = cls_attn.shape[-1]
    grid_size = int(num_tokens**0.5)
    cls_attn = cls_attn.reshape(cls_attn.shape[0], grid_size, grid_size)
    cls_attn = cls_attn / (cls_attn.amax(dim=(1, 2), keepdim=True) + 1e-8)

    heatmap = cls_attn.unsqueeze(1)  # [H, 1, g, g]
    heatmap = F.interpolate(
        heatmap, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
    return heatmap.squeeze(1).cpu().numpy()  # [H, img, img]


def save_overlay(original: np.ndarray, heatmap: np.ndarray, output_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(original)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.6)
    axes[1].set_title("Attention Overlay")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_grid(heatmaps: np.ndarray, output_path: Path, cols: int, title: str):
    num_heads = heatmaps.shape[0]
    cols = max(1, cols)
    rows = int(np.ceil(num_heads / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.axis("off")
        if idx >= num_heads:
            continue
        ax.imshow(heatmaps[idx], cmap="jet")
        ax.set_title(f"Head {idx}")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


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
    original = np.array(Image.open(image_path).convert("RGB"))

    with torch.no_grad():
        logits, attn_list = model(tensor, return_attn=True)
        pred = logits.argmax(dim=1).item()

    if not attn_list:
        raise ValueError("Model did not return any attention tensors.")

    if args.layer_idx < 0 or args.layer_idx >= len(attn_list):
        raise ValueError(f"layer-idx must be in [0, {len(attn_list) - 1}] but got {args.layer_idx}")

    attn_logits = attn_list[args.layer_idx]
    heatmaps = compute_head_heatmaps(attn_logits, args.img_size)

    for head_idx, heatmap in enumerate(heatmaps):
        output_path = output_dir / f"{image_path.stem}_layer_{args.layer_idx:02d}_head_{head_idx:02d}.png"
        title = f"{image_path.name} | layer {args.layer_idx} | head {head_idx} | predicted class {pred}"
        save_overlay(original, heatmap, output_path, title)
        print(f"Saved visualization to {output_path}")

    if args.save_grid:
        grid_path = output_dir / f"{image_path.stem}_layer_{args.layer_idx:02d}_heads_grid.png"
        grid_title = f"{image_path.name} | layer {args.layer_idx} | predicted class {pred}"
        save_grid(heatmaps, grid_path, args.grid_cols, grid_title)
        print(f"Saved grid to {grid_path}")


if __name__ == "__main__":
    main()
