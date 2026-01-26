# transfer_eval.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from nets import build_vit_model
from utils import DEVICE, IMAGENET_MEAN, IMAGENET_STD

black_model_list = [
    "deit_base_patch16_224",
]

def load_annotations(path: Path) -> Dict[str, Dict[str, int | str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("annotations must be a json dict")
    return data


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def infer_num_classes(annotations: Dict[str, Dict[str, int | str]]) -> int:
    class_ids = [int(info["class_id"]) for info in annotations.values()]
    return max(class_ids) + 1 if class_ids else 0


def collect_images(image_dir: Path, prefix: str) -> List[Path]:
    if not image_dir.is_dir():
        raise ValueError(f"image dir not found: {image_dir}")
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.name.startswith(prefix)]
    paths.sort()
    return paths


def extract_original_name(filename: str, prefix: str) -> str:
    if not filename.startswith(prefix):
        raise ValueError(f"filename does not start with prefix: {filename}")
    return filename[len(prefix):]


def evaluate(
    image_paths: List[Path],
    annotations: Dict[str, Dict[str, int | str]],
    prefix: str,
    model_name: str
) -> Tuple[int, int]:
    transform = build_transform()
    num_classes = infer_num_classes(annotations)
    model = build_vit_model(num_classes=num_classes, model_name=model_name)
    model.eval()

    correct = 0
    total = 0
    skipped = 0

    progress = tqdm(image_paths, desc="迁移攻击测试")
    with torch.no_grad():
        for path in progress:
            original_name = extract_original_name(path.name, prefix)
            label_info = annotations.get(original_name)
            if label_info is None:
                skipped += 1
                continue

            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            logits = model(tensor, return_attn=False)
            pred = logits.argmax(dim=1).item()
            target = int(label_info["class_id"])
            correct += int(pred == target)
            total += 1

            if total > 0:
                progress.set_postfix(acc=f"{correct / total:.4f}", skipped=skipped)

    return correct, total


def main(image_dir: str, annotations_path: str, prefix: str) -> None:
    image_dir_path = Path(image_dir)
    annotations = load_annotations(Path(annotations_path))
    image_paths = collect_images(image_dir_path, prefix)
    if not image_paths:
        print("no matching images found")
        return
    
    for black_model in black_model_list:
        correct, total = evaluate(image_paths, annotations, prefix, model_name=black_model)
        acc = correct / total if total > 0 else 0.0
        asr = 1.0 - acc
        print(f"评估样本量={total} 正确个数={correct} 准确率={acc:.4f} ASR={asr:.4f} (黑盒模型={black_model})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions on adversarial or clean samples.")
    parser.add_argument("--image-dir", type=str, default="outputs/attack", help="Directory with saved images.")
    parser.add_argument("--annotations-path", type=str, default="data/image_name_to_class_id_and_name.json")
    parser.add_argument("--prefix", type=str, default="adv_", help="Filename prefix used to infer original names.")
    args = parser.parse_args()
    main(args.image_dir, args.annotations_path, args.prefix)
