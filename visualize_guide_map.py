import argparse
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from attack import LazyAggregationAttacker
from nets import build_vit_model
from utils import DEVICE

HEATMAP_CMAP = "turbo"
HEATMAP_ALPHA = 0.6
TITLE_FONT_SIZE = 13
CORRUPTED_RGB = (0.9, 0.15, 0.15)
CORRUPTED_ALPHA = 0.45
OUTPUT_DPI = 180

# Best attack configuration matching run_effectiveness_s40_500.sh
CONFIG = dict(
    white_box_model="vit_base_patch16_224",
    guide_models=("deit_base_patch16_224", "pit_s_224", "cait_s24_224"),
    attention_guide_type="qk_cls",
    attention_guide_build_method="patch",
    attention_guide_patch_size=16,
    guide_aug_area="background",
    layers=(0, 1, 4, 9, 11),
    guide_aug_methods=("dropout", "jitter", "freq"),
    guide_aug_copies=3,
    guide_aug_strength=0.2,
    epsilon=16.0 / 255.0,
    steps=40,
    use_momentum=True,
    momentum_decay=1.0,
    normalize_grad=True,
    input_diversity=True,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attention guide map from LazyAggregationAttacker."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/clean_resized_images",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*",
        help='Filename glob pattern (default: "*").',
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Number of images to process (default: 5).",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/guide_map_vis",
        help="Output directory for visualization PNGs.",
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    return parser.parse_args()


def list_image_paths(image_dir: Path, pattern: str, max_images: int) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.JPEG")
    paths = []
    for ext in exts:
        paths.extend(sorted(image_dir.glob(ext)))
    if pattern and pattern != "*":
        paths = [p for p in paths if p.match(pattern)]
    return paths[:max_images]


def load_rgb_image(path: Path, img_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BICUBIC)
    return np.array(image).astype(np.float32) / 255.0


def preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def build_attacker(num_classes: int) -> LazyAggregationAttacker:
    c = CONFIG
    white_box = build_vit_model(num_classes=num_classes, model_name=c["white_box_model"])
    white_box.eval()
    guide_models = tuple(
        build_vit_model(num_classes=num_classes, model_name=name).eval()
        for name in c["guide_models"]
    )
    attacker = LazyAggregationAttacker(
        model=white_box,
        epsilon=c["epsilon"],
        steps=c["steps"],
        layers=c["layers"],
        attention_guide_models=guide_models,
        attention_guide_type=c["attention_guide_type"],
        attention_guide_build_method=c["attention_guide_build_method"],
        attention_guide_patch_size=c["attention_guide_patch_size"],
        guide_aug=True,
        guide_aug_area=c["guide_aug_area"],
        guide_aug_methods=c["guide_aug_methods"],
        guide_aug_copies=c["guide_aug_copies"],
        guide_aug_strength=c["guide_aug_strength"],
        use_momentum=c["use_momentum"],
        momentum_decay=c["momentum_decay"],
        normalize_grad=c["normalize_grad"],
        input_diversity=c["input_diversity"],
        device=DEVICE,
    )
    return attacker


@torch.no_grad()
def compute_guide_and_blends(
    attacker: LazyAggregationAttacker, image_tensor: torch.Tensor, img_size: int
):
    clean_pixels = attacker._denormalize(image_tensor).detach()
    guide_pixel_map = attacker._build_guide_pixel_map(image_tensor, img_size)
    guide_np = guide_pixel_map.squeeze().cpu().numpy()
    clean_np = clean_pixels.squeeze(0).permute(1, 2, 0).cpu().numpy()
    blends = {}
    for method in CONFIG["guide_aug_methods"]:
        blended = attacker._guide_augmented_pixels(
            clean_pixels.clone(), guide_pixel_map, method
        )
        blends[method] = blended.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return guide_np, clean_np, blends


def save_figure(
    original: np.ndarray,
    guide_map: np.ndarray,
    blends: dict,
    output_path: Path,
    image_name: str,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    original_clipped = np.clip(original, 0, 1)

    # Panel 0: Original
    axes[0].imshow(original_clipped)
    axes[0].set_title("Original Image", fontsize=TITLE_FONT_SIZE)
    axes[0].axis("off")

    # Panel 1: Guide map heatmap overlay
    axes[1].imshow(original_clipped)
    hm = axes[1].imshow(
        guide_map, cmap=HEATMAP_CMAP, alpha=HEATMAP_ALPHA, vmin=0.0, vmax=1.0
    )
    axes[1].set_title(
        "Attention Guide Map\n(red=preserved, blue=corrupted)",
        fontsize=TITLE_FONT_SIZE,
    )
    axes[1].axis("off")
    cbar = fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Guide value", fontsize=11)

    # Panel 2: Corrupted regions (background area: low guide = corrupted)
    corrupted = guide_map < 0.5
    overlay = original_clipped.copy()
    tint = np.array(CORRUPTED_RGB, dtype=np.float32)
    overlay[corrupted] = (1 - CORRUPTED_ALPHA) * overlay[corrupted] + CORRUPTED_ALPHA * tint
    axes[2].imshow(np.clip(overlay, 0, 1))
    n_corrupted = corrupted.sum()
    n_total = corrupted.size
    axes[2].set_title(
        f"Augmented Regions (guide < 0.5)\n"
        f"{n_corrupted}/{n_total} ({100 * n_corrupted / n_total:.1f}%)",
        fontsize=TITLE_FONT_SIZE,
    )
    axes[2].axis("off")

    # Panels 3-5: Guided blends for each augmentation method
    methods = ["dropout", "jitter", "freq"]
    for idx, method in enumerate(methods):
        ax = axes[3 + idx]
        ax.imshow(np.clip(blends[method], 0, 1))
        ax.set_title(
            f"{method.capitalize()} + Guide Blend", fontsize=TITLE_FONT_SIZE
        )
        ax.axis("off")

    stats = (
        f"min={guide_map.min():.3f}  mean={guide_map.mean():.3f}  max={guide_map.max():.3f}"
    )
    fig.suptitle(
        f"{image_name} | build={CONFIG['attention_guide_build_method']} | "
        f"patch={CONFIG['attention_guide_patch_size']} | "
        f"area={CONFIG['guide_aug_area']} | type={CONFIG['attention_guide_type']} | "
        f"layers={CONFIG['layers']} | {stats}",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    paths = list_image_paths(image_dir, args.pattern, args.max_images)
    if not paths:
        print(f"No images found in {image_dir}")
        return
    print(f"Found {len(paths)} image(s)")

    print("Building models and attacker (this may take a moment)...")
    attacker = build_attacker(args.num_classes)
    print(f"Using device: {DEVICE}")

    for img_path in paths:
        print(f"Processing: {img_path.name}")
        rgb = load_rgb_image(img_path, args.img_size)
        tensor = preprocess_rgb(rgb).unsqueeze(0).to(DEVICE)
        guide_map, clean_np, blends = compute_guide_and_blends(
            attacker, tensor, args.img_size
        )
        out_path = output_dir / f"{img_path.stem}_guide_map.png"
        save_figure(clean_np, guide_map, blends, out_path, img_path.name)
        print(f"  Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
