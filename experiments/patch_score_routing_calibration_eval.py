"""Paired target-clean-correct evaluation for routing calibration attacks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.patch_score_routing_gradient_experiment import normalize
from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument(
        "--clean-image-dir", type=Path, default=REPO_ROOT / "data/clean_resized_images"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT / "data/image_name_to_class_id_and_name.json",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def paired_asr(
    clean_predictions: torch.Tensor,
    adversarial_predictions: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int, int]:
    clean_correct = clean_predictions.eq(labels)
    denominator = int(clean_correct.sum().item())
    successes = int((adversarial_predictions.ne(labels) & clean_correct).sum().item())
    if denominator == 0:
        raise ValueError("target has no clean-correct samples for this calibration job.")
    return successes / denominator, successes, denominator


def find_clean_path(image_dir: Path, image_name: str) -> Path:
    direct = image_dir / image_name
    if direct.is_file():
        return direct
    for suffix in (".png", ".jpg", ".jpeg"):
        candidate = image_dir / f"{Path(image_name).stem}{suffix}"
        if candidate.is_file():
            return candidate
    raise ValueError(f"clean image not found: {image_name}")


class CalibrationPairDataset(Dataset):
    def __init__(self, job, clean_image_dir: Path, annotations, expected_samples: int):
        output_dir = (REPO_ROOT / job["attack_output_dir"]).resolve()
        replay_path = output_dir / "replay_manifest.json"
        if not replay_path.is_file():
            raise ValueError(f"missing replay manifest: {replay_path}")
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        self.sample_ids_sha256 = str(replay.get("sample_ids_sha256", ""))
        if len(self.sample_ids_sha256) != 64:
            raise ValueError(f"invalid sample ID digest in {replay_path}")
        by_stem = {Path(name).stem: name for name in annotations}
        adversarial_by_stem = {
            path.stem.removeprefix("adv_"): path for path in output_dir.glob("adv_*")
        }
        self.samples = []
        for image_name in replay.get("sample_ids", []):
            annotation_name = by_stem.get(Path(image_name).stem)
            adversarial_path = adversarial_by_stem.get(Path(image_name).stem)
            if annotation_name is None or adversarial_path is None:
                raise ValueError(f"incomplete paired sample for {image_name} in {output_dir}")
            self.samples.append(
                (
                    annotation_name,
                    find_clean_path(clean_image_dir, annotation_name),
                    adversarial_path,
                    int(annotations[annotation_name]["class_id"]),
                )
            )
        if len(self.samples) != expected_samples:
            raise ValueError(
                f"expected {expected_samples} paired samples in {output_dir}, "
                f"found {len(self.samples)}."
            )
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, clean_path, adversarial_path, label = self.samples[index]
        with Image.open(clean_path) as image:
            clean = self.to_tensor(image.convert("RGB"))
        with Image.open(adversarial_path) as image:
            adversarial = self.to_tensor(image.convert("RGB"))
        return clean, adversarial, label, name


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = list(manifest.get("jobs", []))
    expected_samples = int(manifest["protocol"]["samples"])
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    observed_digests: set[str] = set()

    for target_name in WHITEBOX_MODEL_CHOICES:
        model = build_whitebox_model(1000, target_name, pretrained=True, device=device).eval()
        for job in jobs:
            if job["source_model"] == target_name:
                continue
            dataset = CalibrationPairDataset(
                job, args.clean_image_dir, annotations, expected_samples
            )
            observed_digests.add(dataset.sample_ids_sha256)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            clean_predictions = []
            adversarial_predictions = []
            labels_all = []
            with torch.inference_mode():
                for clean, adversarial, labels, _ in loader:
                    clean = clean.to(device, non_blocking=True)
                    adversarial = adversarial.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    clean_predictions.append(model(normalize(model, clean)).argmax(dim=1).cpu())
                    adversarial_predictions.append(
                        model(normalize(model, adversarial)).argmax(dim=1).cpu()
                    )
                    labels_all.append(labels.cpu())
            asr, successes, denominator = paired_asr(
                torch.cat(clean_predictions),
                torch.cat(adversarial_predictions),
                torch.cat(labels_all),
            )
            rows.append(
                {
                    "source_model": job["source_model"],
                    "target_model": target_name,
                    "polarity": job["polarity"],
                    "layer": job["layer"],
                    "asr": asr,
                    "successes": successes,
                    "target_clean_correct": denominator,
                }
            )
            print(
                f"source={job['source_model']} target={target_name} "
                f"polarity={job['polarity']} layer={job['layer']} "
                f"ASR={asr:.6f} ({successes}/{denominator})",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(observed_digests) != 1:
        raise ValueError(
            f"calibration jobs do not share one sample ID digest: {sorted(observed_digests)}"
        )
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    with args.output_results.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "asr_denominator": "target-clean-correct images only",
        "samples": expected_samples,
        "sample_ids_sha256": next(iter(observed_digests)),
        "result_rows": len(rows),
    }
    summary_path = args.output_results.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
