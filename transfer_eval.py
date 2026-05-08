import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

from utils import DEVICE

DEFAULT_BLACK_BOX_MODELS = [
    "deit_base_patch16_224",
    "beit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "pvt_v2_b2",
    "cait_s24_224",
    "levit_256",
    "pit_s_224",
    "crossvit_15_240",
]


def parse_model_names(value: str) -> List[str]:
    model_names = [item.strip() for item in value.split(",") if item.strip()]
    if not model_names:
        raise argparse.ArgumentTypeError("model-name must contain at least one model.")
    return model_names


def load_annotations(path: Path) -> Dict[str, Dict[str, int | str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("annotations must be a json dict")
    return data


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


def build_black_box_model(model_name: str):
    model = timm.create_model(model_name, pretrained=True)
    model.to(DEVICE)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def evaluate(
    image_paths: List[Path],
    annotations: Dict[str, Dict[str, int | str]],
    prefix: str,
    model_name: str,
) -> Tuple[int, int, int]:
    model, transform = build_black_box_model(model_name)

    correct = 0
    total = 0
    skipped = 0

    progress = tqdm(image_paths, desc=f"transfer eval {model_name}")
    with torch.no_grad():
        for path in progress:
            original_name = extract_original_name(path.name, prefix)
            label_info = annotations.get(original_name)
            if label_info is None:
                skipped += 1
                continue

            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            pred = logits.argmax(dim=1).item()
            target = int(label_info["class_id"])
            correct += int(pred == target)
            total += 1

            if total > 0:
                progress.set_postfix(acc=f"{correct / total:.4f}", skipped=skipped)

    return correct, total, skipped


def main(
    image_dir: str,
    annotations_path: str,
    prefix: str,
    model_names: List[str],
) -> None:
    image_dir_path = Path(image_dir)
    annotations = load_annotations(Path(annotations_path))
    image_paths = collect_images(image_dir_path, prefix)
    if not image_paths:
        print("no matching images found")
        return

    print(f"Device: {DEVICE}")
    print(f"Images: {len(image_paths)}")
    print(f"Black-box models: {', '.join(model_names)}")

    asr_by_model: Dict[str, float] = {}
    metrics_by_model: Dict[str, Dict[str, float | int]] = {}

    for black_model in model_names:
        correct, total, skipped = evaluate(
            image_paths=image_paths,
            annotations=annotations,
            prefix=prefix,
            model_name=black_model,
        )
        acc = correct / total if total > 0 else 0.0
        asr = 1.0 - acc if total > 0 else 0.0
        asr_by_model[black_model] = asr
        metrics_by_model[black_model] = {
            "asr": asr,
            "acc": acc,
            "correct": correct,
            "total": total,
            "skipped": skipped,
        }
        print(
            f"model={black_model} total={total} skipped={skipped} "
            f"correct={correct} acc={acc:.4f} ASR={asr:.4f}"
        )

    print("ASR by model:")
    print({model_name: round(asr, 6) for model_name, asr in asr_by_model.items()})
    print("Metrics by model:")
    print(
        {
            model_name: {
                "asr": round(float(metrics["asr"]), 6),
                "acc": round(float(metrics["acc"]), 6),
                "correct": int(metrics["correct"]),
                "total": int(metrics["total"]),
                "skipped": int(metrics["skipped"]),
            }
            for model_name, metrics in metrics_by_model.items()
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate transferability on adversarial or clean samples.")
    parser.add_argument("--image-dir", type=str, default="outputs/attack", help="Directory with saved images.")
    parser.add_argument("--annotations-path", type=str, default="data/image_name_to_class_id_and_name.json")
    parser.add_argument("--prefix", type=str, default="adv_", help="Filename prefix used to infer original names.")
    parser.add_argument(
        "--model-name",
        type=parse_model_names,
        default=DEFAULT_BLACK_BOX_MODELS,
        help="Black-box timm model name(s), comma-separated for multiple models.",
    )
    args = parser.parse_args()
    main(
        image_dir=args.image_dir,
        annotations_path=args.annotations_path,
        prefix=args.prefix,
        model_names=args.model_name,
    )
