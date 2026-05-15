import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Dict, List, Tuple

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from record_experiment import record_results
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


class TransferImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, int(target)


def build_transfer_samples(
    image_paths: List[Path],
    annotations: Dict[str, Dict[str, int | str]],
    prefix: str,
) -> Tuple[List[Tuple[Path, int]], int]:
    samples: List[Tuple[Path, int]] = []
    skipped = 0
    for path in image_paths:
        original_name = extract_original_name(path.name, prefix)
        label_info = annotations.get(original_name)
        if label_info is None:
            skipped += 1
            continue
        samples.append((path, int(label_info["class_id"])))
    return samples, skipped


def pre_cache_tensors(
    samples: List[Tuple[Path, int]],
    transform,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-transform all images to tensors in RAM to eliminate per-model disk I/O."""
    images_list = []
    labels_list = []
    for path, label in tqdm(samples, desc="pre-cache", leave=False):
        img = Image.open(path).convert("RGB")
        images_list.append(transform(img))
        labels_list.append(label)
    return torch.stack(images_list), torch.tensor(labels_list)


def build_dataloader(
    samples: List[Tuple[Path, int]],
    transform,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    """On-the-fly image loading (fallback when pre-caching is disabled)."""
    dataset = TransferImageDataset(samples=samples, transform=transform)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (DEVICE.type == "cuda"),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_cached_dataloader(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> DataLoader:
    """Pre-cached tensor loading (fast path, eliminates per-model disk I/O)."""
    dataset = TensorDataset(images, labels)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": (DEVICE.type == "cuda"),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def configure_eval_runtime(use_tf32: bool) -> None:
    if DEVICE.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if use_tf32:
        torch.set_float32_matmul_precision("high")


def infer_exp_name(image_dir: Path) -> str:
    return image_dir.name or "transfer_eval"


def evaluate(
    image_paths: List[Path],
    annotations: Dict[str, Dict[str, int | str]],
    prefix: str,
    model_name: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    use_amp: bool,
    pre_cache: bool = True,
) -> Tuple[int, int, int]:
    model, transform = build_black_box_model(model_name)
    samples, skipped = build_transfer_samples(
        image_paths=image_paths,
        annotations=annotations,
        prefix=prefix,
    )
    if not samples:
        return 0, 0, skipped

    if pre_cache:
        images_cached, labels_cached = pre_cache_tensors(samples, transform)
        dataloader = build_cached_dataloader(
            images=images_cached,
            labels=labels_cached,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
    else:
        dataloader = build_dataloader(
            samples=samples,
            transform=transform,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    correct = 0
    total = 0

    autocast_context = (
        torch.cuda.amp.autocast if DEVICE.type == "cuda" and use_amp else nullcontext
    )
    progress = tqdm(dataloader, desc=f"transfer eval {model_name}")
    with torch.inference_mode():
        for images, targets in progress:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with autocast_context():
                logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if total > 0:
                progress.set_postfix(acc=f"{correct / total:.4f}")

    return correct, total, skipped


def main(
    image_dir: str,
    annotations_path: str,
    prefix: str,
    model_names: List[str],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    use_amp: bool,
    use_tf32: bool,
    record_excel: bool,
    exp_name: str | None,
    pre_cache: bool = True,
) -> None:
    configure_eval_runtime(use_tf32=use_tf32)
    image_dir_path = Path(image_dir)
    annotations = load_annotations(Path(annotations_path))
    image_paths = collect_images(image_dir_path, prefix)
    if not image_paths:
        print("no matching images found")
        return

    print(f"Device: {DEVICE}")
    print(f"Images: {len(image_paths)}")
    print(f"Black-box models: {', '.join(model_names)}")
    print(
        f"DataLoader: batch_size={batch_size} num_workers={num_workers} "
        f"prefetch_factor={prefetch_factor}"
    )
    if DEVICE.type == "cuda":
        print(f"CUDA eval: amp={use_amp} tf32={use_tf32}")

    asr_by_model: Dict[str, float] = {}
    metrics_by_model: Dict[str, Dict[str, float | int]] = {}

    for black_model in model_names:
        correct, total, skipped = evaluate(
            image_paths=image_paths,
            annotations=annotations,
            prefix=prefix,
            model_name=black_model,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            use_amp=use_amp,
            pre_cache=pre_cache,
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

    if record_excel:
        repo_path = Path(__file__).resolve().parent
        resolved_image_dir = image_dir_path.resolve()
        record_params = {
            "image_dir": resolved_image_dir.relative_to(repo_path).as_posix()
            if resolved_image_dir.is_relative_to(repo_path)
            else str(resolved_image_dir),
            "prefix": prefix,
            "model_name": ",".join(model_names),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "amp": use_amp,
            "tf32": use_tf32,
            "num_images": len(image_paths),
        }
        csv_path = record_results(
            repo_path=repo_path,
            exp_name=exp_name or infer_exp_name(image_dir_path),
            params_dict=record_params,
            adv_dir_arg=image_dir_path,
            asr_by_model=asr_by_model,
        )
        print(f"Recorded transfer eval to CSV: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate transferability on adversarial or clean samples.")
    parser.add_argument("--image-dir", type=str, default="outputs/attack", help="Directory with saved images.")
    parser.add_argument("--annotations-path", type=str, default="data/image_name_to_class_id_and_name.json")
    parser.add_argument("--prefix", type=str, default="adv_", help="Filename prefix used to infer original names.")
    parser.add_argument("--batch-size", type=int, default=256, help="Images per inference batch.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker processes for image decode/transform.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Batches prefetched per DataLoader worker.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast for faster transfer evaluation.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/cudnn on CUDA.")
    parser.add_argument("--exp-name", default=None, help="Experiment name recorded in the Excel output. Defaults to the image-dir name.")
    parser.add_argument("--no-record", action="store_true", help="Disable automatic Excel recording.")
    parser.add_argument("--no-pre-cache", action="store_true", help="Disable pre-caching images in RAM (slower, lower memory).")
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        use_amp=args.amp,
        use_tf32=not args.no_tf32,
        record_excel=not args.no_record,
        exp_name=args.exp_name,
        pre_cache=not args.no_pre_cache,
    )
