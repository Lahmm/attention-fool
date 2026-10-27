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
from utils import DEVICE, SEED, ImageDataset


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "figure1"
SAMPLE_COUNT = 50
LAYER_COUNT = 12
GRID_SIZE = (14, 14)
COLOR_LIMITS = (0.0, 1.0)
COLOR_ANCHORS = (
    (45, 95, 210),
    (215, 55, 70),
)


@dataclass
class LayerwiseObservation:
    scores: torch.Tensor
    logits: torch.Tensor
    grid_size: tuple[int, int]


def compute_patch_scores(local: torch.Tensor, global_repr: torch.Tensor) -> torch.Tensor:
    if local.ndim != 3 or global_repr.shape != (local.size(0), local.size(2)):
        raise ValueError("expected local [B,N,D] and global representation [B,D]")
    local = local.float()
    global_repr = global_repr.float()
    if not torch.isfinite(local).all() or not torch.isfinite(global_repr).all():
        raise ValueError("layer representations contain non-finite values")
    if (local.norm(dim=-1) == 0).any() or (global_repr.norm(dim=-1) == 0).any():
        raise ValueError("cosine similarity is undefined for zero-norm representations")
    scores = F.cosine_similarity(local, global_repr.unsqueeze(1), dim=-1, eps=1e-8)
    return scores.clamp(-1.0, 1.0)


@torch.inference_mode()
def extract_layerwise_scores(adapter, normalized_inputs: torch.Tensor) -> LayerwiseObservation:
    adapter.eval()
    initial = adapter.begin_progressive_forward(normalized_inputs.to(adapter.device))
    initial.validate()
    prefix = initial.context["prefix_tokens"]
    if prefix.size(1) != 1:
        raise ValueError("Figure 1 observation requires the single-CLS ViT-B/16 model")
    tokens = torch.cat((prefix, initial.local_tokens), dim=1)
    local_snapshots = []
    for block in adapter.model.blocks:
        tokens = block(tokens)
        local_snapshots.append(tokens[:, 1:].cpu().clone())
    final_tokens = adapter.model.norm(tokens)
    final_cls = final_tokens[:, 0].cpu()
    scores = [compute_patch_scores(local, final_cls) for local in local_snapshots]
    logits = adapter.model.forward_head(final_tokens).cpu()
    return LayerwiseObservation(torch.stack(scores, dim=1), logits, initial.grid_size)


def render_heatmap(
    scores: torch.Tensor,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> Image.Image:
    rows, cols = grid_size
    if rows <= 0 or cols <= 0 or scores.shape != (rows * cols,) or not torch.isfinite(scores).all():
        raise ValueError("one finite score is required for each grid location")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("heatmap dimensions must be positive")

    values = scores.detach().float().cpu()
    minimum, maximum = values.min(), values.max()
    if maximum > minimum:
        values = ((values - minimum) / (maximum - minimum)).clamp(*COLOR_LIMITS)
    else:
        values = torch.full_like(values, 0.5)
    values = F.interpolate(
        values.reshape(1, 1, rows, cols), size=(height, width),
        mode="bilinear", align_corners=False,
    ).reshape(-1).clamp(*COLOR_LIMITS)
    positions = (values - COLOR_LIMITS[0]) / (COLOR_LIMITS[1] - COLOR_LIMITS[0]) * (len(COLOR_ANCHORS) - 1)
    lower = positions.floor().long().clamp(max=len(COLOR_ANCHORS) - 2)
    fraction = (positions - lower).unsqueeze(1)
    palette = torch.tensor(COLOR_ANCHORS, dtype=torch.float32)
    colors = ((1 - fraction) * palette[lower] + fraction * palette[lower + 1]).round().byte()
    return Image.frombytes("RGB", image_size, bytes(colors.reshape(-1).tolist()))


def save_sample_elements(
    directory: Path,
    image: Image.Image,
    scores: torch.Tensor,
    grid_size: tuple[int, int],
    metadata: dict,
    *,
    alpha: float = 0.45,
) -> None:
    if not 0 < alpha <= 1:
        raise ValueError("overlay alpha must satisfy 0 < alpha <= 1")
    if scores.shape != (LAYER_COUNT, grid_size[0] * grid_size[1]):
        raise ValueError("expected all 12 layers and all local patch scores")
    directory.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(directory / "input.png")
    for layer, layer_scores in enumerate(scores, start=1):
        heatmap = render_heatmap(layer_scores, grid_size, image.size)
        overlay = Image.blend(image.convert("RGB"), heatmap, alpha)
        overlay.save(directory / f"heatmap_block{layer:02d}.png")

    with (directory / "patch_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("layer", "patch_index", "row", "col", "score"))
        for layer, layer_scores in enumerate(scores, start=1):
            for patch_index, score in enumerate(layer_scores.tolist()):
                row, col = divmod(patch_index, grid_size[1])
                writer.writerow((layer, patch_index, row, col, format(score, ".9g")))

    metadata = {
        **metadata,
        "layers": list(range(1, LAYER_COUNT + 1)),
        "grid_size": list(grid_size),
        "image_size_wh": list(image.size),
        "patch_size_hw": [image.height // grid_size[0], image.width // grid_size[1]],
        "observation_boundary": "complete block output, before final model.norm",
        "global_representation": "fixed final encoder CLS token after model.norm, shared by all observation layers",
        "score": "cosine similarity of each block's local token and the final encoder CLS",
        "input": "clean image, no attack, token drop, feature noise, or translation",
        "indices": "layer is 1-based; patch_index, row, col are 0-based; patch_index = row * grid_width + col",
        "heatmap": {
            "alpha": alpha,
            "rendering": "RGB image blended with smooth heatmap; no grid or border",
            "normalization": "per-image, per-layer min-max over all local patch scores; CSV retains raw cosine similarities",
            "constant_score_value": 0.5,
            "colorbar_label": "Normalized patch score",
            "vmin": COLOR_LIMITS[0],
            "vmax": COLOR_LIMITS[1],
            "colormap": "continuous blue-to-red RGB gradient",
            "color_anchors_rgb": [list(color) for color in COLOR_ANCHORS],
            "color_anchor_positions": [index / (len(COLOR_ANCHORS) - 1) for index in range(len(COLOR_ANCHORS))],
            "color_interpolation": "linear RGB between adjacent anchors",
            "resampling": "bilinear on normalized scores before color mapping; align_corners=False",
        },
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def export_figure1_elements(
    adapter,
    dataset: ImageDataset,
    output_dir: Path,
    *,
    batch_size: int = 4,
    alpha: float = 0.45,
) -> None:
    if batch_size <= 0 or not 0 < alpha <= 1:
        raise ValueError("batch size must be positive and overlay alpha must be in (0,1]")
    if len(dataset) < SAMPLE_COUNT:
        raise ValueError(f"Figure 1 requires at least {SAMPLE_COUNT} annotated images")
    selected = dataset.samples[:SAMPLE_COUNT]
    folder_names = [Path(sample["image_name"]).stem for sample in selected]
    if len(set(folder_names)) != SAMPLE_COUNT:
        raise ValueError("the first 50 image names must have unique stems for output folders")
    if any(name in ("", ".", "..") for name in folder_names):
        raise ValueError("invalid sample folder name")
    if len(adapter.model.blocks) != LAYER_COUNT:
        raise ValueError("Figure 1 requires all 12 ViT-B/16 blocks")
    if tuple(adapter.model.patch_embed.grid_size) != GRID_SIZE:
        raise ValueError("Figure 1 requires the ViT-B/16 14x14 patch grid")
    output_dir = Path(output_dir)
    if output_dir.exists():
        extra = {path.name for path in output_dir.iterdir() if path.is_dir()} - set(folder_names)
        if extra:
            raise ValueError(f"output contains other sample folders: {sorted(extra)}")
    for sample in selected:
        with Image.open(sample["image_path"]) as image:
            if image.size != (224, 224):
                raise ValueError(f"expected a pre-resized 224x224 image: {sample['image_path']}")

    mean = torch.tensor(adapter.model_mean, device=adapter.device).view(1, 3, 1, 1)
    std = torch.tensor(adapter.model_std, device=adapter.device).view(1, 3, 1, 1)
    loader = DataLoader(
        Subset(dataset, range(SAMPLE_COUNT)), batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=adapter.device.type == "cuda",
    )
    cfg = adapter.model.pretrained_cfg
    model_metadata = {
        "model_name": adapter.model_name,
        "weights": "timm pretrained=True",
        "pretrained_cfg": {key: cfg.get(key) for key in (
            "architecture", "tag", "hf_hub_id", "input_size", "mean", "std",
        )},
        "model_mean": list(adapter.model_mean),
        "model_std": list(adapter.model_std),
        "seed": SEED,
        "device": str(adapter.device),
        "versions": {name: _package_version(name) for name in ("torch", "torchvision", "timm", "Pillow")},
        "sample_order": "first 50 available annotated images sorted by annotation image_name; no shuffling or prediction filtering",
    }
    with torch.inference_mode():
        for batch_index, (pixels, labels, indices) in enumerate(loader):
            normalized = (pixels.to(adapter.device) - mean) / std
            observation = extract_layerwise_scores(adapter, normalized)
            if batch_index == 0:
                native_logits = adapter(normalized).cpu()
                torch.testing.assert_close(observation.logits, native_logits, rtol=1e-5, atol=1e-6)
            if observation.grid_size != GRID_SIZE:
                raise ValueError("observed grid differs from the 14x14 input patch grid")
            for offset, sample_index in enumerate(indices.tolist()):
                sample = dataset.samples[sample_index]
                image_path = Path(sample["image_path"])
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                metadata = {
                    **model_metadata,
                    "sample_index": sample_index,
                    "image_name": sample["image_name"],
                    "image_path": str(image_path.resolve()),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "class_id": int(labels[offset]),
                }
                save_sample_elements(
                    output_dir / folder_names[sample_index], image,
                    observation.scores[offset], observation.grid_size, metadata, alpha=alpha,
                )
            print(f"Figure 1 elements: {int(indices[-1]) + 1}/{SAMPLE_COUNT}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Figure 1 elements for the first 50 clean samples")
    parser.add_argument("--image-dir", default=str(ROOT / "data" / "clean_resized_images"))
    parser.add_argument("--annotations-path", default=str(ROOT / "data" / "image_name_to_class_id_and_name.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or not 0 < args.overlay_alpha <= 1:
        raise ValueError("batch size must be positive and overlay alpha must be in (0,1]")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = DEVICE if args.device == "auto" else torch.device(args.device)
    dataset = ImageDataset(args.image_dir, args.annotations_path, transform=transforms.ToTensor())
    if len(dataset) < SAMPLE_COUNT:
        raise ValueError(f"Figure 1 requires at least {SAMPLE_COUNT} annotated images")
    try:
        adapter = build_whitebox_model(model_name=DEFAULT_MODEL_NAME, device=device)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not load pretrained ViT-B/16. Install requirements.txt and prepare its "
            "pretrained weights in data/huggingface; model loading uses the repository's offline cache."
        ) from exc
    export_figure1_elements(
        adapter, dataset, OUTPUT_DIR, batch_size=args.batch_size, alpha=args.overlay_alpha,
    )
    print(f"Saved 50 sample folders with 12 heatmap overlays and all patch scores each to {OUTPUT_DIR}")


if __name__ == "__main__":
    main(parse_args())
