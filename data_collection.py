from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from nets.base import contraction_statistics
from nets.vit import ViTAdapter
from utils import DEVICE, SEED, ImageDataset


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "data_collection"
SAMPLE_COUNT = 50
LAYER_COUNT = 12
GRID_SIZE = (14, 14)


@dataclass
class LayerObservations:
    patch_scores: torch.Tensor
    projections: torch.Tensor
    centered_projections: torch.Tensor
    statistics: dict[str, torch.Tensor]
    global_reference: torch.Tensor
    logits: torch.Tensor
    grid_size: tuple[int, int]
    feature_dimension: int


def validate_adapter(adapter):
    if not isinstance(adapter, ViTAdapter) or adapter.model_name != DEFAULT_MODEL_NAME:
        raise ValueError("collection requires the current ViT-B/16 source adapter")
    if adapter.device.type not in ("cpu", "cuda"):
        raise ValueError("native float64 geometry requires CPU or CUDA")
    if len(adapter.model.blocks) != LAYER_COUNT:
        raise ValueError("collection requires all 12 ViT-B/16 blocks")
    if tuple(adapter.model.patch_embed.grid_size) != GRID_SIZE:
        raise ValueError("collection requires the ViT-B/16 14x14 token grid")
    if adapter.model.num_prefix_tokens != 1:
        raise ValueError("collection requires a single CLS prefix")


@torch.no_grad()
def collect_batch(adapter, pixels):
    """Gather all native block boundaries and one final CLS in a single forward."""
    validate_adapter(adapter)
    adapter.eval()
    if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, 224, 224):
        raise ValueError("expected RGB pixels [B,3,224,224]")
    if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
        raise ValueError("input pixels must be finite and in [0,1]")
    pixels = pixels.to(adapter.device)
    mean = pixels.new_tensor(adapter.model_mean).reshape(1, 3, 1, 1)
    std = pixels.new_tensor(adapter.model_std).reshape(1, 3, 1, 1)
    initial = adapter.begin_gdi_forward((pixels - mean) / std)
    initial.validate()
    prefix = initial.context["prefix_tokens"]
    tokens = torch.cat((prefix, initial.local_tokens), dim=1)
    snapshots = []
    for block in adapter.model.blocks:
        before = tokens[:, prefix.size(1):].clone()
        tokens = block(tokens)
        snapshots.append((before, tokens[:, prefix.size(1):].clone()))
    final_tokens = adapter.model.norm(tokens)
    reference = final_tokens[:, 0].float()
    logits = adapter.model.forward_head(final_tokens)
    patch_scores, projections, centered_projections, statistics = [], [], [], []
    for before, after in snapshots:
        stats = contraction_statistics(after, reference, before)
        statistics.append({key: value.detach().cpu() for key, value in stats.items()})
        local64, global64 = after.float().double(), reference.double()
        unit = local64 / local64.norm(dim=-1, keepdim=True)
        direction = global64 / global64.norm(dim=-1, keepdim=True)
        centered = unit - unit.mean(dim=1, keepdim=True)
        identical = (unit == unit[:, :1]).all(dim=2).all(dim=1)
        centered = torch.where(identical[:, None, None], 0.0, centered)
        projection = (unit * direction[:, None]).sum(dim=2)
        centered_projection = (centered * direction[:, None]).sum(dim=2)
        raw = F.cosine_similarity(after.float(), reference[:, None], dim=-1,
                                  eps=1e-8).clamp(-1, 1)
        patch_scores.append(raw.cpu())
        projections.append(projection.cpu())
        centered_projections.append(centered_projection.cpu())
    return LayerObservations(
        torch.stack(patch_scores, dim=1), torch.stack(projections, dim=1),
        torch.stack(centered_projections, dim=1),
        {key: torch.stack([stats[key] for stats in statistics], dim=1)
         for key in statistics[0]},
        reference.cpu(), logits.cpu(), initial.grid_size, initial.local_tokens.size(2),
    )


def layer_rows(observations, sample):
    stats = observations.statistics
    selected = int(stats["contraction_score"][sample].argmax())
    rows = []
    for layer in range(LAYER_COUNT):
        def number(key):
            return float(stats[key][sample, layer])

        rows.append({
            "block": layer + 1, "input_checkpoint": f"block{layer}",
            "token_count": GRID_SIZE[0] * GRID_SIZE[1],
            "feature_dimension": observations.feature_dimension,
            "patch_score_mean": number("mainline_score_mean"),
            "patch_score_population_variance": number("mainline_score_population_variance"),
            "projection_mean_float64": number("score_mean_float64"),
            "V": number("global_direction_population_variance"),
            "D_in": number("input_total_variance"),
            "D_out": number("output_total_variance"),
            "average_direction_variance": number("average_direction_population_variance"),
            "R": number("direction_variance_ratio"),
            "C": number("contraction_fraction"), "S": number("contraction_score"),
            "selected": int(layer == selected),
        })
    return rows


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_sample(directory, source, observations, sample, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(directory / "input.png")
    scores = observations.patch_scores[sample]
    projections = observations.projections[sample]
    centered = observations.centered_projections[sample]
    centered_scores = scores.double() - scores.double().mean(dim=1, keepdim=True)

    def token_rows():
        for layer in range(LAYER_COUNT):
            for index in range(scores.size(1)):
                row, col = divmod(index, observations.grid_size[1])
                yield {
                    "block": layer + 1, "input_checkpoint": f"block{layer}",
                    "token_index": index, "row": row, "col": col,
                    "patch_score": float(scores[layer, index]),
                    "centered_patch_score": float(centered_scores[layer, index]),
                    "projection_float64": float(projections[layer, index]),
                    "centered_projection_float64": float(centered[layer, index]),
                }

    write_csv(directory / "token_scores.csv", (
        "block", "input_checkpoint", "token_index", "row", "col", "patch_score",
        "centered_patch_score", "projection_float64", "centered_projection_float64",
    ), token_rows())
    rows = layer_rows(observations, sample)
    write_csv(directory / "layer_statistics.csv", list(rows[0]), rows)
    torch.save({
        "patch_scores": scores.clone(), "centered_patch_scores": centered_scores.clone(),
        "projections_float64": projections.clone(),
        "centered_projections_float64": centered.clone(),
        "statistics": {key: value[sample].clone()
                       for key, value in observations.statistics.items()},
        "global_reference": observations.global_reference[sample].clone(),
        "logits": observations.logits[sample].clone(),
    }, directory / "observations.pt")
    selected = next(row for row in rows if row["selected"])
    metadata.update({
        "grid_size": list(observations.grid_size),
        "feature_dimension": observations.feature_dimension,
        "selected_block": selected["block"],
        "selected_input_checkpoint": selected["input_checkpoint"],
        "selected_layer_statistics": selected,
    })
    (directory / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return selected


def package_versions():
    versions = {}
    for name in ("torch", "torchvision", "timm", "Pillow"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def collect_dataset(adapter, dataset, output_dir, *, batch_size=4):
    validate_adapter(adapter)
    if batch_size <= 0 or len(dataset) < SAMPLE_COUNT:
        raise ValueError("positive batch size and at least 50 annotated images are required")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"use a new or empty output directory: {output_dir}")
    samples = dataset.samples[:SAMPLE_COUNT]
    names = [Path(sample["image_name"]).stem for sample in samples]
    if len(set(names)) != SAMPLE_COUNT or any(name in ("", ".", "..") for name in names):
        raise ValueError("the first 50 image names must have valid unique stems")
    for sample in samples:
        with Image.open(sample["image_path"]) as image:
            if image.size != (224, 224):
                raise ValueError(f"expected pre-resized 224x224 image: {sample['image_path']}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "purpose": {"figure2": "Section 3.2 layer observations",
                    "figure3": "Section 3.3 direction variance ratio"},
        "sample_count": SAMPLE_COUNT, "block_count": LAYER_COUNT,
        "seed": SEED, "batch_size": batch_size, "device": str(adapter.device),
        "model_name": adapter.model_name,
        "pretrained_cfg": getattr(adapter.model, "pretrained_cfg", {}),
        "model_mean": list(adapter.model_mean), "model_std": list(adapter.model_std),
        "versions": package_versions(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("data_collection.py", "nets/vit.py", "nets/base.py", "utils.py")},
        "sample_order": "first 50 available annotated image names in sorted order; no prediction filtering",
        "forward": "one clean forward per sample; no phase, noise, drop or attack updates",
        "local_boundaries": "native block inputs and complete outputs before final model.norm; exclude CLS",
        "global_reference": "final encoder CLS after model.norm, shared across all blocks of the same image",
        "geometry": "production contraction_statistics; float64 normalized geometry from float32 features",
        "score_precision": {
            "patch_score": "float32 cosine, eps=1e-8, clamp [-1,1]; retained without display normalization",
            "projection_float64": "dot product of float64 L2-normalized local vectors and global reference",
            "centered_projection_float64": "projection of centered float64 unit local vectors; mean square equals V",
        },
        "statistic_symbols": {
            "V": "global-direction population variance",
            "D_in": "input local population dispersion",
            "D_out": "output local population dispersion",
            "average_direction_variance": "D_out / native feature dimension",
            "R": "V / average_direction_variance; zero for zero output dispersion",
            "C": "max(0, 1 - D_out / D_in); zero for zero input dispersion",
            "S": "R * C",
        },
        "indexing": "block is 1-based; input_checkpoint blockK means K blocks completed; token/row/col are 0-based",
        "selection": "maximum S; earliest native block on ties",
        "heatmap_normalization": "not applied; any per-layer min-max belongs only to later display",
    }
    (output_dir / "collection_metadata.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loader = DataLoader(Subset(dataset, range(SAMPLE_COUNT)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    manifest = []
    for pixels, labels, indices in loader:
        observations = collect_batch(adapter, pixels)
        for offset, index in enumerate(indices.tolist()):
            sample = samples[index]
            source = Path(sample["image_path"])
            metadata = {
                "sample_index": index, "image_name": sample["image_name"],
                "class_id": int(labels[offset]), "image_path": str(source.resolve()),
                "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "collection_metadata": "../collection_metadata.json",
            }
            selected = save_sample(output_dir / names[index], source, observations, offset, metadata)
            manifest.append({"sample_index": index, "image_name": sample["image_name"],
                             "directory": names[index], "selected_block": selected["block"],
                             "selected_input_checkpoint": selected["input_checkpoint"]})
        print(f"Collected layer observations: {int(indices[-1]) + 1}/{SAMPLE_COUNT}", flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect all Figure 2/3 layer observations for the first 50 ViT samples")
    parser.add_argument("--image-dir", default=str(ROOT / "data" / "clean_resized_images"))
    parser.add_argument("--annotations-path", default=str(ROOT / "data" / "image_name_to_class_id_and_name.json"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(args):
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch.manual_seed(SEED)
    automatic = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
    device = automatic if args.device == "auto" else torch.device(args.device)
    dataset = ImageDataset(args.image_dir, args.annotations_path, transform=transforms.ToTensor())
    try:
        adapter = build_whitebox_model(model_name=DEFAULT_MODEL_NAME, device=device)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("Install requirements.txt and prepare pretrained ViT-B/16 weights in data/huggingface.") from exc
    collect_dataset(adapter, dataset, args.output_dir, batch_size=args.batch_size)
    print(f"Saved 50 samples, 600 sample/block observations to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
