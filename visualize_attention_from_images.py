import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nets import ViTWithAttn
from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD, load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize attention maps for existing adversarial images."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="outputs",
        help="Directory containing saved adversarial images.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="adv_*.png",
        help="Filename glob used to select images (e.g., adv_*.png).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="Number of images to visualize (processes first N matches).",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Checkpoint for ViT classifier.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Classifier head dimension (e.g., 10 for CIFAR-10).",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--cls-layer",
        choices=["first", "last"],
        default="last",
        help="Which transformer layer attention to render.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/attention_from_images",
        help="Where to store visualization figures.",
    )
    return parser.parse_args()


def build_model(num_classes: int, weights_path: Optional[str]) -> ViTWithAttn:
    model = ViTWithAttn(
        model_name="vit_base_patch16_224",
        num_classes=num_classes,
        pretrained=True,
        device=DEVICE,
    )
    load_model_weights(model, weights_path)
    model.eval()
    return model


def list_image_paths(image_dir: Path, pattern: str, max_images: int) -> List[Path]:
    paths = sorted(image_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"在 {image_dir} 中找不到匹配模式 {pattern} 的图像。"
        )
    return paths[:max_images] if max_images is not None else paths


def preprocess_image(path: Path, img_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((img_size, img_size), Image.BICUBIC)
    np_img = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def compute_attention_heatmap(attn_logits_list, layer_choice: str, img_size: int):
    if not attn_logits_list:
        raise ValueError("模型没有输出任何注意力信息。")

    attn_logits = attn_logits_list[0] if layer_choice == "first" else attn_logits_list[-1]
    attn_weights = torch.softmax(attn_logits, dim=-1)  # [B, H, N, N]
    attn_weights = attn_weights.mean(dim=1)[0]  # [N, N]

    cls_attn = attn_weights[0, 1:]
    num_tokens = cls_attn.numel()
    grid_size = int(num_tokens**0.5)
    cls_attn = cls_attn.reshape(grid_size, grid_size)
    cls_attn = cls_attn / (cls_attn.max() + 1e-8)

    heatmap = cls_attn.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
    return heatmap.squeeze().cpu().numpy()


def save_overlay(original: np.ndarray, heatmap: np.ndarray, output_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[0].set_title("Adversarial Image")
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


def main():
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    model = build_model(num_classes=args.num_classes, weights_path=args.weights_path)
    image_paths = list_image_paths(image_dir, args.pattern, args.max_images)

    for path in image_paths:
        print(f"Processing {path}")
        tensor = preprocess_image(path, args.img_size).unsqueeze(0).to(DEVICE)
        original = np.array(Image.open(path).convert("RGB"))

        with torch.no_grad():
            logits, attn_list = model(tensor, return_attn=True)
            pred = logits.argmax(dim=1).item()

        heatmap = compute_attention_heatmap(attn_list, args.cls_layer, args.img_size)
        output_path = output_dir / f"{path.stem}_attention.png"
        title = f"{path.name} | predicted class {pred}"
        save_overlay(original, heatmap, output_path, title)
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
