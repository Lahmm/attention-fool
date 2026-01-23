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
    parser = argparse.ArgumentParser(description="Visualize attention maps for all transformer layers on one image.")
    parser.add_argument("--image-path",type=str,required=True,help="Path to a single image file.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir",type=str,default=f"outputs/attention_all_layers_{DEFAULT_MODEL_NAME}",help="Where to store visualization figures.")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes. If omitted, infer from annotations.")
    parser.add_argument("--dataset-dir", type=str, default="data/clean_resized_images", help="Dataset directory used to infer class count.")
    parser.add_argument("--annotations-path", type=str, default="data/image_name_to_class_id_and_name.json", help="Annotations file used to infer class count.")
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


def compute_attention_heatmap_from_logits(attn_logits: torch.Tensor, img_size: int) -> np.ndarray:
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

    for layer_idx, attn_logits in enumerate(attn_list):
        heatmap = compute_attention_heatmap_from_logits(attn_logits, args.img_size)
        output_path = output_dir / f"{image_path.stem}_layer_{layer_idx:02d}.png"
        title = f"{image_path.name} | layer {layer_idx} | predicted class {pred}"
        save_overlay(original, heatmap, output_path, title)
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
