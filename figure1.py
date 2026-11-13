from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from torch.overrides import TorchFunctionMode
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from attack import GDIAttacker
from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from nets.vit import ViTAdapter
from utils import DEVICE, SEED, ImageDataset


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "figure1" / "attack_elements"
SAMPLE_COUNT = 50


class RGBNoiseCapture(TorchFunctionMode):
    """Observe the actual Gaussian draw without advancing the RNG a second time."""

    def __init__(self):
        super().__init__()
        self.draws = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.randn:
            self.draws.append(result.detach().cpu().clone())
        return result


@dataclass
class CapturedView:
    pixels: torch.Tensor
    masks: dict = field(default_factory=dict)
    local_after_drop: dict = field(default_factory=dict)
    rgb_noise: torch.Tensor | None = None
    feature_noise: torch.Tensor | None = None
    noise_grid: tuple[int, int] | None = None


class FigureAttacker(GDIAttacker):
    """Capture the first paired forward using the production attack implementation."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.views = []
        self.phases = None
        self._view = None

    def _pick_input_diversity_phases(self, batch_size, device):
        self.phases = super()._pick_input_diversity_phases(batch_size, device)
        return self.phases

    def _projected_gaussian_noise(self, state):
        with RGBNoiseCapture() as capture:
            noise = super()._projected_gaussian_noise(state)
        if len(capture.draws) != 1:
            raise RuntimeError("expected one RGB Gaussian draw per view")
        self._view.rgb_noise = capture.draws[0]
        self._view.feature_noise = noise.detach().cpu().clone()
        self._view.noise_grid = state.grid_size
        return noise

    def _forward_with_schedule(self, pixels, labels, schedule):
        view = CapturedView(pixels=pixels.detach().cpu().clone())
        self._view = view
        handles = []
        try:
            for item in schedule:
                view.masks[item.checkpoint] = item.mask.detach().cpu().clone()

                def record_input(module, inputs, checkpoint=item.checkpoint):
                    local = inputs[0][:, self.model.model.num_prefix_tokens:]
                    view.local_after_drop[checkpoint] = local.detach().cpu().clone()

                block = self.model.model.blocks[self.model.checkpoint_order(item.checkpoint)]
                handles.append(block.register_forward_pre_hook(record_input))
            loss = super()._forward_with_schedule(pixels, labels, schedule)
            for checkpoint, mask in view.masks.items():
                local = view.local_after_drop[checkpoint]
                if local.shape[:2] != mask.shape or torch.count_nonzero(local[mask]):
                    raise RuntimeError("captured block input does not match the executed token drop")
            self.views.append(view)
            return loss
        finally:
            for handle in handles:
                handle.remove()
            self._view = None


@torch.no_grad()
def capture_first_pair(adapter, pixels, labels):
    """Stop after t=0, g=0; no optimization step or later attack views are needed."""
    if not isinstance(adapter, ViTAdapter) or adapter.model_name != DEFAULT_MODEL_NAME:
        raise ValueError("figure1 export uses the current ViT-B/16 source")
    attacker = FigureAttacker(adapter, device=adapter.device)
    pixels, labels = pixels.to(adapter.device), labels.to(adapter.device)
    choices = adapter.select_contraction_checkpoints(attacker._normalize(pixels))
    attacker._prepare_dynamic_mask_layout(choices, pixels.size(0))
    losses = attacker._iter_attack_losses(pixels, labels)
    try:
        next(losses)
        next(losses)
    finally:
        losses.close()
        attacker._dynamic_mask_layout = None
    if len(attacker.views) != 2 or attacker.phases is None:
        raise RuntimeError("could not capture the original/phase pair")
    configuration = attacker.parameters()
    for key in ("sample_checkpoints", "checkpoint_histogram"):
        configuration.pop(key, None)
    return choices, attacker.phases, attacker.views, configuration


def rgb_noise_image(noise, grid):
    """Assemble the sampled RGB patches, with a fixed display range of [-3, 3]."""
    rows, cols = grid
    if noise.ndim != 4 or noise.shape[:2] != (rows * cols, 3):
        raise ValueError("expected RGB noise [N,3,patch_height,patch_width]")
    _, _, height, width = noise.shape
    rgb = noise.reshape(rows, cols, 3, height, width)
    rgb = rgb.permute(2, 0, 3, 1, 4).reshape(3, rows * height, cols * width)
    return (0.5 + rgb / 6).clamp(0, 1)


def spatial_mask(mask, grid, image_size):
    if mask.dtype != torch.bool or mask.shape != (grid[0] * grid[1],):
        raise ValueError("expected one boolean mask on the selected native grid")
    if image_size[0] % grid[0] or image_size[1] % grid[1]:
        raise ValueError("image dimensions must be divisible by the token grid")
    return F.interpolate(mask.float().reshape(1, 1, *grid), size=image_size,
                         mode="nearest")[0].bool()


def save_rgb(tensor, path):
    if tensor.ndim != 3 or tensor.size(0) != 3 or not torch.isfinite(tensor).all():
        raise ValueError("expected finite RGB tensor [3,H,W]")
    pixels = tensor.detach().cpu().clamp(0, 1).mul(255).round().byte()
    Image.fromarray(pixels.permute(1, 2, 0).contiguous().numpy()).save(path)


def drop_mask_image(mask, grid, image_size):
    pixel_mask = spatial_mask(mask, grid, image_size)[0].cpu()
    pixels = (~pixel_mask).byte().mul(240)
    image = Image.fromarray(pixels.numpy())
    draw = ImageDraw.Draw(image)
    height, width = image_size
    for x in (*range(0, width, width // grid[1]), width - 1):
        draw.line((x, 0, x, height - 1), fill=160, width=1)
    for y in (*range(0, height, height // grid[0]), height - 1):
        draw.line((0, y, width - 1, y), fill=160, width=1)
    return image


def save_sample_elements(directory, sample_index, choice, phase, views, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, grid = choice.checkpoint, choice.grid_size
    tensors = {}
    for name, view in zip(("original", "phase"), views):
        image = view.pixels[sample_index]
        mask = view.masks[checkpoint][sample_index]
        local = view.local_after_drop[checkpoint][sample_index]
        pixel_mask = spatial_mask(mask, grid, image.shape[-2:])
        is_original = name == "original"
        save_rgb(image, directory / ("input.png" if is_original else "phase.png"))
        save_rgb(rgb_noise_image(view.rgb_noise[sample_index], view.noise_grid),
                 directory / ("noise.png" if is_original else "phase_noise.png"))
        save_rgb(image.masked_fill(pixel_mask, 0),
                 directory / ("drop.png" if is_original else "phase_drop.png"))
        drop_mask_image(mask, grid, image.shape[-2:]).save(
            directory / ("mask.png" if is_original else "phase_mask.png"))
        tensors[name] = {
            "mask": mask.clone(), "local_after_drop": local.clone(),
            "rgb_noise_patches": view.rgb_noise[sample_index].clone(),
            "projected_scaled_feature_noise": view.feature_noise[sample_index].clone(),
        }
    torch.save(tensors, directory / "capture.pt")
    metadata.update({
        "capture": {"step": 0, "group": 0, "views": ["original", "phase"],
                    "original_mask_draws": 1, "executed_view_forwards": 2,
                    "optimization_updates_executed": 0},
        "selection": asdict(choice),
        "drop_count": int(tensors["original"]["mask"].sum()),
        "phase_drop_count": int(tensors["phase"]["mask"].sum()),
        "phase_shift_xy": list(phase),
        "phase_padding": "reflection; production GDIAttacker transform",
        "noise_initial_grid": list(views[0].noise_grid),
        "noise_visualization": {
            "source": "actual independent RGB Gaussian draw used by each view's feature augmentation",
            "mapping": "clamp(0.5 + RGB_noise / 6, 0, 1); identical display scale for all samples/views",
            "injection": "RGB projection followed by per-image feature-RMS scaling; added to initial local features",
        },
        "drop_visualization": {
            "source": "executed mask mapped to RGB patch locations, with dropped areas painted black",
            "interpretation": "spatial illustration of token zeroing, not an inversion of deep features",
            "actual_features": "capture.pt contains the actual selected-block input after hard zeroing",
            "mask_png": "black (0) = dropped; off-white (240) = retained; gray (160) one-pixel token grid and border",
            "prefix_tokens": "preserved by the production attack; excluded from local_after_drop",
        },
    })
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_figure1_elements(adapter, dataset, output_dir, *, batch_size=4):
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(dataset) < SAMPLE_COUNT:
        raise ValueError(f"at least {SAMPLE_COUNT} annotated images are required")
    if adapter.device.type == "mps":
        raise ValueError("native float64 layer selection requires CPU or CUDA")
    adapter.eval()
    selected = dataset.samples[:SAMPLE_COUNT]
    names = [Path(sample["image_name"]).stem for sample in selected]
    if len(set(names)) != SAMPLE_COUNT or any(name in ("", ".", "..") for name in names):
        raise ValueError("the first 50 sample names must have valid unique stems")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"use a new or empty output directory to avoid mixing exports: {output_dir}")
    for sample in selected:
        with Image.open(sample["image_path"]) as image:
            if image.size != (224, 224):
                raise ValueError(f"expected a pre-resized 224x224 image: {sample['image_path']}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(Subset(dataset, range(SAMPLE_COUNT)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    manifest = []
    for pixels, labels, indices in loader:
        choices, phases, views, parameters = capture_first_pair(adapter, pixels, labels)
        for offset, index in enumerate(indices.tolist()):
            sample = selected[index]
            source = Path(sample["image_path"])
            metadata = {
                "sample_index": index, "image_name": sample["image_name"],
                "class_id": int(labels[offset]), "image_path": str(source.resolve()),
                "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "model_name": adapter.model_name, "model_mean": list(adapter.model_mean),
                "model_std": list(adapter.model_std), "device": str(adapter.device),
                "seed": SEED, "batch_size": batch_size,
                "sample_order": "first 50 available annotated names in sorted order; no prediction filtering",
                "attack_configuration": parameters,
            }
            save_sample_elements(output_dir / names[index], offset, choices[offset],
                                 phases[offset], views, metadata)
            manifest.append({"sample_index": index, "image_name": sample["image_name"],
                             "directory": names[index]})
        print(f"Figure 1 attack elements: {int(indices[-1]) + 1}/{SAMPLE_COUNT}", flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export the first attack pair for the first 50 ViT samples")
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
    automatic_device = torch.device("cpu") if DEVICE.type == "mps" else DEVICE
    device = automatic_device if args.device == "auto" else torch.device(args.device)
    dataset = ImageDataset(args.image_dir, args.annotations_path, transform=transforms.ToTensor())
    try:
        adapter = build_whitebox_model(model_name=DEFAULT_MODEL_NAME, device=device)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("Install requirements.txt and prepare pretrained ViT-B/16 weights in data/huggingface.") from exc
    export_figure1_elements(adapter, dataset, Path(args.output_dir), batch_size=args.batch_size)
    print(f"Saved 50 sample folders to {args.output_dir}")


if __name__ == "__main__":
    main(parse_args())
