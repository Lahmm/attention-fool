"""Paired clean/adversarial evaluation for a selector-suite manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from patch_score_selector_suite import CONDITIONS, summarize
from transfer_eval import DEFAULT_BLACK_BOX_MODELS, build_black_box_model, load_annotations
from utils import DEVICE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clean-image-dir", type=Path, default=REPO_ROOT / "data/clean_resized_images")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT / "data/image_name_to_class_id_and_name.json",
    )
    parser.add_argument("--targets", default=",".join(DEFAULT_BLACK_BOX_MODELS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/research/patch_score_selector_suite"),
    )
    return parser.parse_args()


def find_clean_path(image_dir: Path, image_name: str) -> Path:
    direct = image_dir / image_name
    if direct.is_file():
        return direct
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{Path(image_name).stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise ValueError(f"clean image not found: {image_name}")


class PairedImageDataset(Dataset):
    def __init__(self, job, clean_image_dir: Path, annotations, transform):
        self.job = job
        self.transform = transform
        self.annotations = annotations
        self.clean_image_dir = clean_image_dir
        adv_dir = (REPO_ROOT / job["attack_output_dir"]).resolve()
        if not adv_dir.is_dir():
            raise ValueError(f"attack output directory not found: {adv_dir}")
        by_stem = {Path(name).stem: name for name in annotations}
        self.samples = []
        for adv_path in sorted(adv_dir.glob("adv_*")):
            original_stem = adv_path.stem.removeprefix("adv_")
            annotation_name = by_stem.get(original_stem)
            if annotation_name is None:
                raise ValueError(f"annotation not found for {adv_path.name}")
            self.samples.append(
                (
                    annotation_name,
                    find_clean_path(clean_image_dir, annotation_name),
                    adv_path,
                    int(annotations[annotation_name]["class_id"]),
                )
            )
        expected = int(job.get("samples", 0) or 0)
        if not self.samples:
            raise ValueError(f"no adversarial images found in {adv_dir}")
        if expected and len(self.samples) != expected:
            raise ValueError(
                f"expected {expected} adversarial images in {adv_dir}, found {len(self.samples)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, clean_path, adv_path, label = self.samples[index]
        with Image.open(clean_path) as image:
            clean = self.transform(image.convert("RGB"))
        with Image.open(adv_path) as image:
            adversarial = self.transform(image.convert("RGB"))
        return clean, adversarial, label, name


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("batch-size and bootstrap-repeats must be positive.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = list(manifest.get("jobs", []))
    if not jobs:
        raise ValueError("selector manifest contains no jobs.")
    expected_samples = int(manifest["protocol"]["samples"])
    for job in jobs:
        job["samples"] = expected_samples
        if job["condition"] not in CONDITIONS:
            raise ValueError(f"unknown selector condition: {job['condition']}")
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    annotations = load_annotations(args.annotations)
    rows = []
    for target_name in targets:
        model, transform = build_black_box_model(target_name)
        for job in jobs:
            dataset = PairedImageDataset(job, args.clean_image_dir, annotations, transform)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=DEVICE.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            with torch.inference_mode():
                for clean, adversarial, labels, names in dataloader:
                    clean = clean.to(DEVICE, non_blocking=True)
                    adversarial = adversarial.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    clean_pred = model(clean).argmax(dim=1)
                    adv_pred = model(adversarial).argmax(dim=1)
                    for index, image_name in enumerate(names):
                        rows.append(
                            {
                                "source_model": job["source_model"],
                                "target_model": target_name,
                                "condition": job["condition"],
                                "image_name": image_name,
                                "label": int(labels[index].cpu()),
                                "clean_correct": bool(clean_pred[index].eq(labels[index]).cpu()),
                                "adv_correct": bool(adv_pred[index].eq(labels[index]).cpu()),
                            }
                        )
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    write_rows(args.output_dir / "per_image.csv", rows)
    summary = summarize(rows, bootstrap_repeats=args.bootstrap_repeats, seed=args.seed)
    summary["manifest"] = str(args.manifest)
    summary["targets"] = targets
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
