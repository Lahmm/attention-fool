import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets import build_vit_model, ViTWithHook, DEFAULT_MODEL_NAME
from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention patch scores and A@V patch scores for images in one directory.")
    parser.add_argument("--image-dir",type=str,required=True,help="Directory containing images to visualize (e.g., data/clean_resized_images or outputs).")
    parser.add_argument("--pattern",type=str,default="*",help='Filename glob in --image-dir (e.g., "adv_*.png" or "clean_*.png").')
    parser.add_argument("--max-images",type=int,default=5,help="Process first N matched images.")
    parser.add_argument("--block-index",type=int,default=-1,help="Transformer block index to visualize. Supports negative index; -1 means last block.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir",type=str,default=f"outputs/attention_patchscores_{DEFAULT_MODEL_NAME}",help="Directory where visualization figures are saved.")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--dataset-dir", type=str, default="data/clean_resized_images")
    parser.add_argument("--annotations-path",type=str,default="data/image_name_to_class_id_and_name.json")
    return parser.parse_args()


def build_model(num_classes: int) -> ViTWithHook:
    model = build_vit_model(num_classes=num_classes)
    model.eval()
    return model


def resolve_num_classes(
    num_classes: Optional[int],
    dataset_dir: Path,
    annotations_path: Path,
    img_size: int,
) -> int:
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


def list_image_paths(image_dir: Path, pattern: str, max_images: int) -> List[Path]:
    candidates = sorted(image_dir.glob(pattern))
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = [p for p in candidates if p.is_file() and p.suffix.lower() in image_exts]
    if not paths:
        raise FileNotFoundError(f"No image found in {image_dir} with pattern {pattern}.")
    return paths[:max_images] if max_images is not None else paths


def preprocess_image(path: Path, img_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BICUBIC)
    np_img = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def resolve_block_index(block_index: int, total_blocks: int) -> int:
    resolved = block_index if block_index >= 0 else total_blocks + block_index
    if resolved < 0 or resolved >= total_blocks:
        raise ValueError(f"Invalid --block-index {block_index}, total blocks={total_blocks}.")
    return resolved


def tokens_to_heatmap(token_scores: torch.Tensor, img_size: int) -> np.ndarray:
    num_tokens = token_scores.numel()
    grid_size = int(math.sqrt(num_tokens))
    if grid_size * grid_size != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a square number.")
    grid = token_scores.reshape(grid_size, grid_size)
    grid = grid / (grid.max() + 1e-8)
    grid = F.interpolate(
        grid.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    return grid.squeeze().cpu().numpy()


def compute_maps(
    attn_logits: torch.Tensor,
    values: torch.Tensor,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    # attn_logits: [B, H, N, N], values: [B, H, N, d]
    attn_weights = torch.softmax(attn_logits, dim=-1)

    # Map 1: CLS -> patch attention weights
    cls_attn_scores = attn_weights[:, :, 0, 1:].mean(dim=1)[0]
    attn_map = tokens_to_heatmap(cls_attn_scores, img_size)

    # Map 2: CLS query contribution map from (A @ V), per patch token
    cls_patch_contrib = attn_weights[:, :, 0, 1:].unsqueeze(-1) * values[:, :, 1:, :]
    cls_av_scores = cls_patch_contrib.norm(dim=-1).mean(dim=1)[0]
    av_map = tokens_to_heatmap(cls_av_scores, img_size)

    return attn_map, av_map


def compute_patch_token_pca_rgb(tokens_last_block: torch.Tensor, img_size: int) -> np.ndarray:
    # tokens_last_block: [B, N, D], use one image and patch tokens only
    patch_tokens = tokens_last_block[0, 1:, :]  # [N_patch, D]
    num_patches = patch_tokens.size(0)
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Patch token count {num_patches} is not a square number.")

    x = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(x, full_matrices=False)
    comps = vh[:3, :].transpose(0, 1)  # [D, 3]
    proj = x @ comps  # [N_patch, 3]

    pmin = proj.min(dim=0, keepdim=True).values
    pmax = proj.max(dim=0, keepdim=True).values
    proj = (proj - pmin) / (pmax - pmin + 1e-8)

    rgb = proj.reshape(grid_size, grid_size, 3).permute(2, 0, 1).unsqueeze(0)  # [1, 3, g, g]
    rgb = F.interpolate(rgb, size=(img_size, img_size), mode="nearest")
    return rgb.squeeze(0).permute(1, 2, 0).cpu().numpy()


def compute_patch_score_maps(tokens_last_block: torch.Tensor, img_size: int) -> tuple[np.ndarray, np.ndarray]:
    # tokens_last_block: [B, N, D], use one image
    cls_token = tokens_last_block[:, 0, :]      # [B, D]
    patch_tokens = tokens_last_block[:, 1:, :]  # [B, N_patch, D]

    num_patches = patch_tokens.size(1)
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Patch token count {num_patches} is not a square number.")

    cls_token_expanded = cls_token.unsqueeze(1).expand(-1, num_patches, -1)  # [B, N_patch, D]
    patch_scores = F.cosine_similarity(patch_tokens, cls_token_expanded, dim=-1)  # [B, N_patch]

    score_map_2d = patch_scores[0].reshape(grid_size, grid_size).detach().cpu().numpy()

    upsampled = F.interpolate(
        patch_scores[0].reshape(1, 1, grid_size, grid_size),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    overlay_map = upsampled.squeeze().detach().cpu().numpy()
    return score_map_2d, overlay_map


def save_triptych(
    original: np.ndarray,
    attn_map: np.ndarray,
    av_map: np.ndarray,
    pca_rgb_map: np.ndarray,
    patch_score_map_2d: np.ndarray,
    patch_score_overlay_map: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    axes[0].imshow(original)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(original)
    hm1 = axes[1].imshow(attn_map, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
    axes[1].set_title("Attention Scores")
    axes[1].axis("off")
    cbar1 = fig.colorbar(hm1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Normalized score")

    axes[2].imshow(original)
    hm2 = axes[2].imshow(av_map, cmap="jet", alpha=0.6, vmin=0.0, vmax=1.0)
    axes[2].set_title("A@V Scores")
    axes[2].axis("off")
    cbar2 = fig.colorbar(hm2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Normalized score")

    axes[3].imshow(pca_rgb_map)
    axes[3].set_title("Patch Token PCA-RGB")
    axes[3].axis("off")

    axes[4].imshow(original, alpha=0.6)
    patch_score_vis = (patch_score_overlay_map - patch_score_overlay_map.min()) / (
        patch_score_overlay_map.max() - patch_score_overlay_map.min() + 1e-8
    )
    hm3 = axes[4].imshow(
        patch_score_vis,
        cmap="jet",
        alpha=0.6,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    axes[4].set_title(
        f"Patch Score Overlay\nMin: {patch_score_map_2d.min():.3f}, Max: {patch_score_map_2d.max():.3f}"
    )
    axes[4].axis("off")
    cbar3 = fig.colorbar(hm3, ax=axes[4], fraction=0.046, pad=0.04)
    cbar3.set_label("Normalized score")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    annotations_path = Path(args.annotations_path)
    num_classes = resolve_num_classes(args.num_classes, dataset_dir, annotations_path, args.img_size)

    model = build_model(num_classes=num_classes)
    image_paths = list_image_paths(image_dir, args.pattern, args.max_images)

    for path in image_paths:
        print(f"Processing {path}")
        tensor = preprocess_image(path, args.img_size).unsqueeze(0).to(DEVICE)
        original = np.array(Image.open(path).convert("RGB"))

        with torch.no_grad():
            logits, attn_list, value_list, token_list = model(
                tensor,
                return_attn=True,
                return_values=True,
                return_tokens=True,
            )
            pred = logits.argmax(dim=1).item()

        if not attn_list:
            raise ValueError("Model did not return attention tensors.")
        if len(value_list) != len(attn_list):
            raise ValueError(
                f"Hook mismatch: got {len(attn_list)} attention tensors, {len(value_list)} value tensors."
            )
        if not token_list:
            raise ValueError("Model did not return block token outputs.")

        block_idx = resolve_block_index(args.block_index, len(attn_list))
        attn_map, av_map = compute_maps(
            attn_logits=attn_list[block_idx],
            values=value_list[block_idx],
            img_size=args.img_size,
        )
        tokens_last_block = token_list[-1]
        pca_rgb_map = compute_patch_token_pca_rgb(tokens_last_block, args.img_size)
        patch_score_map_2d, patch_score_overlay_map = compute_patch_score_maps(
            tokens_last_block,
            args.img_size,
        )

        output_path = output_dir / f"{path.stem}_block_{block_idx:02d}_patchscores.png"
        title = f"{path.name} | block {block_idx} | predicted class {pred}"
        save_triptych(
            original,
            attn_map,
            av_map,
            pca_rgb_map,
            patch_score_map_2d,
            patch_score_overlay_map,
            output_path,
            title,
        )
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
