import argparse
import math
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets import build_vit_model, ViTWithHook, DEFAULT_MODEL_NAME
from utils import (
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    last_vit_foreground_background_from_tokens,
    load_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize attention patch scores and A@V patch scores for images in one directory.")
    parser.add_argument("--image-dir",type=str,required=True,help="Directory containing images to visualize (e.g., data/clean_resized_images or outputs).")
    parser.add_argument("--pattern",type=str,default="*",help='Filename glob in --image-dir (e.g., "adv_*.png" or "clean_*.png").')
    parser.add_argument("--max-images",type=int,default=5,help="Process first N matched images.")
    parser.add_argument("--block-index",type=int,default=-1,help="Transformer block index to visualize. Supports negative index; -1 means last block.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--fft-topk", type=int, default=1, help="Per-channel Top-K stable patch count used by FFT selection.")
    parser.add_argument("--fft-alpha", type=float, default=0.6, help="FFT stability heatmap overlay opacity.")
    parser.add_argument("--overlap-top-ratio", type=float, default=0.2, help="Top fraction used to mark high FFT, attention, and patch-score regions in the overlap panel.")
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


def load_rgb_image(path: Path, img_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BICUBIC)
    return np.array(image).astype(np.float32) / 255.0


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def preprocess_image(path: Path, img_size: int) -> torch.Tensor:
    return preprocess_rgb(load_rgb_image(path, img_size))


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
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
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


def normalize_map(score_map: np.ndarray) -> np.ndarray:
    return (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-8)


def make_fft_stability_overlay(
    image: np.ndarray,
    score_map: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    score = normalize_map(score_map)
    heatmap = plt.get_cmap("viridis")(score)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image + alpha * heatmap
    return np.clip(overlay, 0.0, 1.0), score


def top_ratio_mask(score_map: np.ndarray, top_ratio: float) -> np.ndarray:
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}.")
    threshold = np.quantile(score_map.reshape(-1), 1.0 - top_ratio)
    return score_map >= threshold


def make_mechanism_overlap_overlay(
    image: np.ndarray,
    fft_score_map: np.ndarray,
    attn_map: np.ndarray,
    patch_score_map: np.ndarray,
    top_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    fft_high = top_ratio_mask(normalize_map(fft_score_map), top_ratio)
    attn_high = top_ratio_mask(normalize_map(attn_map), top_ratio)
    patch_high = top_ratio_mask(normalize_map(patch_score_map), top_ratio)
    attn_patch_high = attn_high & patch_high
    all_high = fft_high & attn_high & patch_high

    overlay = image * 0.30
    fft_color = np.array([0.65, 0.0, 1.0], dtype=np.float32)
    attn_patch_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    both_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    overlay[fft_high] = 0.35 * overlay[fft_high] + 0.65 * fft_color
    overlay[attn_patch_high] = 0.32 * overlay[attn_patch_high] + 0.68 * attn_patch_color
    overlay[all_high] = 0.12 * overlay[all_high] + 0.88 * both_color

    attn_patch_count = float(attn_patch_high.sum())
    fft_capture_ratio = float(all_high.sum()) / attn_patch_count if attn_patch_count > 0 else 0.0
    fft_high_count = float(fft_high.sum())
    fft_precision_ratio = float(all_high.sum()) / fft_high_count if fft_high_count > 0 else 0.0
    return (
        np.clip(overlay, 0.0, 1.0),
        fft_high,
        attn_high,
        patch_high,
        fft_capture_ratio,
        fft_precision_ratio,
    )


def draw_binary_contour(
    ax,
    mask: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str = "solid",
) -> None:
    if mask.any() and (~mask).any():
        ax.contour(mask.astype(np.float32), levels=[0.5], colors=color, linewidths=linewidth, linestyles=linestyle)


def save_triptych(
    original: np.ndarray,
    attn_map: np.ndarray,
    av_map: np.ndarray,
    fft_overlay: np.ndarray,
    mechanism_overlap_overlay: np.ndarray,
    fft_high_mask: np.ndarray,
    attn_high_mask: np.ndarray,
    patch_high_mask: np.ndarray,
    fft_capture_ratio: float,
    fft_precision_ratio: float,
    patch_score_map_2d: np.ndarray,
    patch_score_overlay_map: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(36, 6.5))

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

    axes[3].imshow(fft_overlay)
    axes[3].set_title("FFT Stability Selection")
    axes[3].axis("off")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar_fft = fig.colorbar(
        sm,
        ax=axes[3],
        fraction=0.046,
        pad=0.04,
    )
    cbar_fft.set_label("Selection frequency")

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

    axes[5].imshow(mechanism_overlap_overlay)
    draw_binary_contour(axes[5], fft_high_mask, color="magenta", linewidth=2.4)
    draw_binary_contour(axes[5], attn_high_mask, color="cyan", linewidth=2.2)
    draw_binary_contour(axes[5], patch_high_mask, color="gold", linewidth=2.2)
    axes[5].set_title(
        "Mechanism Overlap\n"
        f"High A & Patch in high FFT: {fft_capture_ratio:.1%}"
    )
    axes[5].axis("off")
    axes[5].legend(
        handles=[
            Patch(facecolor="magenta", edgecolor="black", label="High FFT"),
            Patch(facecolor="cyan", edgecolor="black", label="High Attn"),
            Patch(facecolor="gold", edgecolor="black", label="High Patch"),
            Patch(facecolor="white", edgecolor="black", label="All high"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
        fontsize=8,
        framealpha=0.92,
    )

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
        original = load_rgb_image(path, args.img_size)
        tensor = preprocess_rgb(original).unsqueeze(0).to(DEVICE)

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
        tokens_for_block = token_list[block_idx]
        fft_maps = last_vit_foreground_background_from_tokens(
            tokens=tokens_for_block,
            topk=args.fft_topk,
            has_cls_token=True,
            img_size=args.img_size,
        )
        fft_score_map = fft_maps["score_map"][0].detach().cpu().numpy()
        fft_overlay, fft_stability_map = make_fft_stability_overlay(
            image=original,
            score_map=fft_score_map,
            alpha=args.fft_alpha,
        )
        patch_score_map_2d, patch_score_overlay_map = compute_patch_score_maps(
            tokens_for_block,
            args.img_size,
        )
        (
            mechanism_overlap_overlay,
            fft_high_mask,
            attn_high_mask,
            patch_high_mask,
            fft_capture_ratio,
            fft_precision_ratio,
        ) = make_mechanism_overlap_overlay(
            image=original,
            fft_score_map=fft_stability_map,
            attn_map=attn_map,
            patch_score_map=patch_score_overlay_map,
            top_ratio=args.overlap_top_ratio,
        )

        output_path = output_dir / f"{path.stem}_block_{block_idx:02d}_patchscores.png"
        title = f"{path.name} | block {block_idx} | predicted class {pred}"
        save_triptych(
            original,
            attn_map,
            av_map,
            fft_overlay,
            mechanism_overlap_overlay,
            fft_high_mask,
            attn_high_mask,
            patch_high_mask,
            fft_capture_ratio,
            fft_precision_ratio,
            patch_score_map_2d,
            patch_score_overlay_map,
            output_path,
            title,
        )
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
