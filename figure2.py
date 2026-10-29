from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from progressive_attack import ProgressiveRouteDisruptionAttacker
from utils import DEVICE, SEED, ImageDataset


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "figure2"
SAMPLE_COUNT = 50
NOMINAL_STEPS = 10


def snapshot(tensor):
    return tensor.detach().cpu().clone()


def tile_patches(patches, grid):
    batch, count, channels, height, width = patches.shape
    if count != grid[0] * grid[1]:
        raise ValueError("noise patch count differs from the native grid")
    return patches.reshape(batch, *grid, channels, height, width).permute(
        0, 3, 1, 4, 2, 5
    ).reshape(batch, channels, grid[0] * height, grid[1] * width)


class Figure2Recorder:
    def __init__(self):
        self.views = []
        self.phases = None
        self.x_t = None
        self.x_t_plus_1 = None


class Figure2Attacker(ProgressiveRouteDisruptionAttacker):
    def __init__(self, model, *, device=None):
        super().__init__(
            model, steps=1, step_size=(16 / 255) / NOMINAL_STEPS,
            input_diversity_groups=10, input_diversity_views_per_group=2,
            device=device, dynamic_checkpoint=False,
        )
        self.recorder = Figure2Recorder()
        self._active_view = None
        self._active_pixels = None

    def _pick_input_diversity_phases(self, batch_size, device):
        phases = super()._pick_input_diversity_phases(batch_size, device)
        if self._actual_forward_view_count == 0:
            self.recorder.phases = phases
        return phases

    def _strict_opponent_feature_noise(self, state):
        if self._active_view is None:
            return super()._strict_opponent_feature_noise(state)
        draws = []
        original_randn = torch.randn

        def capture_randn(*args, **kwargs):
            result = original_randn(*args, **kwargs)
            draws.append(result.detach().clone())
            return result

        with patch.object(torch, "randn", new=capture_randn):
            noise = super()._strict_opponent_feature_noise(state)
        if len(draws) != 1:
            raise RuntimeError("expected exactly one RGB coefficient draw per differentiable view")
        with torch.no_grad():
            coefficients = draws[0]
            if coefficients.shape[:3] != (*state.local_tokens.shape[:2], 3):
                raise RuntimeError("unexpected opponent-noise coefficient layout")
            luma, rg, yb = (
                variance**0.5 * coefficients[:, :, index:index + 1]
                for index, variance in enumerate(self.noise_direction_variances)
            )
            rgb = torch.cat((
                3**-0.5 * luma + 2**-0.5 * rg + 6**-0.5 * yb,
                3**-0.5 * luma - 2**-0.5 * rg + 6**-0.5 * yb,
                3**-0.5 * luma - 2 * 6**-0.5 * yb,
            ), dim=2)
            weight = state.rgb_projection_weight.detach().to(state.local_tokens)
            projected = rgb.flatten(2).matmul(weight.reshape(weight.size(0), -1).t())
            local_rms = state.local_tokens.detach().square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            noise_rms = projected.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
            ratio = local_rms / noise_rms
            torch.testing.assert_close(noise, self.opponent_noise_strength * projected * ratio, rtol=0, atol=0)
            scaled_rgb = self.opponent_noise_strength * rgb * ratio.view(-1, 1, 1, 1, 1)
            tiled = tile_patches(scaled_rgb, state.grid_size)
            equivalent = self._active_pixels.detach() + tiled * self.model_std
            equivalent_features = self.model.begin_progressive_forward(self._normalize(equivalent))
            tf32_convolution = equivalent.is_cuda and torch.backends.cudnn.allow_tf32
            equivalent_rtol = 5e-3 if tf32_convolution else 1e-4
            equivalent_atol = 1e-3 if tf32_convolution else 1e-5
            torch.testing.assert_close(
                equivalent_features.local_tokens, state.local_tokens + noise,
                rtol=equivalent_rtol, atol=equivalent_atol,
            )
            self._active_view.update({
                "grid_size": state.grid_size,
                "initial_before_noise": snapshot(state.local_tokens),
                "coefficients": snapshot(coefficients),
                "rgb_noise": snapshot(rgb),
                "scaled_rgb_noise": snapshot(scaled_rgb),
                "feature_noise": snapshot(noise),
                "rms_ratio": snapshot(ratio),
                "equivalent_noisy_pixels_unclipped": snapshot(equivalent),
                "equivalent_feature_max_abs_error": snapshot(
                    (equivalent_features.local_tokens - (state.local_tokens + noise)).abs().flatten(1).amax(1)
                ),
                "equivalent_verification_rtol": equivalent_rtol,
                "equivalent_verification_atol": equivalent_atol,
            })
        return noise

    def _forward_with_schedule(self, pixels, labels, schedule):
        if self._actual_forward_view_count > 2:
            return super()._forward_with_schedule(pixels, labels, schedule)
        view = {"pixels": snapshot(pixels), "checkpoints": []}
        self._active_view, self._active_pixels = view, pixels
        advance = self.model.advance_progressive_state
        apply_mask = self.model.apply_progressive_mask

        def capture_advance(state, checkpoint):
            if "initial_after_noise" not in view:
                view["initial_after_noise"] = snapshot(state.local_tokens)
            result = advance(state, checkpoint)
            view["checkpoints"].append({
                "checkpoint": checkpoint, "grid_size": result.grid_size,
                "before": snapshot(result.local_tokens),
                "prefix_before": snapshot(result.context["prefix_tokens"]),
            })
            return result

        def capture_mask(state, mask):
            result = apply_mask(state, mask)
            item = view["checkpoints"][-1]
            item.update({"mask": snapshot(mask), "after": snapshot(result.local_tokens),
                         "prefix_after": snapshot(result.context["prefix_tokens"])})
            if torch.count_nonzero(result.local_tokens[mask]).item() != 0:
                raise RuntimeError("selected local features were not hard-zeroed")
            torch.testing.assert_close(item["prefix_before"], item["prefix_after"], rtol=0, atol=0)
            return result

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.model, "advance_progressive_state", new=capture_advance))
                stack.enter_context(patch.object(self.model, "apply_progressive_mask", new=capture_mask))
                loss = super()._forward_with_schedule(pixels, labels, schedule)
            torch.testing.assert_close(
                view["initial_after_noise"], view["initial_before_noise"] + view["feature_noise"],
                rtol=1e-6, atol=1e-6,
            )
            self.recorder.views.append(view)
            return loss
        finally:
            self._active_view = self._active_pixels = None

    def attack_batch(self, images, labels):
        self.recorder = Figure2Recorder()
        self.recorder.x_t = snapshot(self._denormalize(images.to(self.device)))
        output = super().attack_batch(images, labels)
        self.recorder.x_t_plus_1 = snapshot(self._denormalize(output))
        if len(self.recorder.views) != 2 or self._actual_forward_view_count != 20:
            raise RuntimeError("expected one captured pair and twenty gradient views")
        return output


def pixel_image(pixels):
    pixels = pixels.detach().float().cpu().clamp(0, 1)
    if pixels.ndim != 3 or pixels.size(0) != 3:
        raise ValueError("expected RGB pixels [3,H,W]")
    data = pixels.mul(255).round().byte().permute(1, 2, 0).contiguous()
    return Image.frombytes("RGB", (pixels.size(2), pixels.size(1)), bytes(data.flatten().tolist()))


def grid_image(image, grid):
    rows, cols = grid
    if image.width % cols or image.height % rows:
        raise ValueError("input size is incompatible with the patch grid")
    result = image.copy()
    draw = ImageDraw.Draw(result, "RGBA")
    for col in range(1, cols):
        x = col * image.width // cols
        draw.line((x, 0, x, image.height - 1), fill=(20, 20, 20, 180))
    for row in range(1, rows):
        y = row * image.height // rows
        draw.line((0, y, image.width - 1, y), fill=(20, 20, 20, 180))
    return result


def spatial_image(values, grid, size, *, maximum):
    gray = (values.detach().float().cpu().reshape(-1) / max(maximum, 1e-12)).clamp(0, 1)
    image = Image.frombytes("L", (grid[1], grid[0]), bytes(gray.mul(255).round().byte().tolist()))
    return image.resize(size, resample=getattr(Image, "Resampling", Image).NEAREST)


def sample_trace(recorder, index):
    views = []
    for view in recorder.views:
        selected = {key: (value[index].clone() if isinstance(value, torch.Tensor) else value)
                    for key, value in view.items() if key != "checkpoints"}
        selected["checkpoints"] = [
            {key: (value[index].clone() if isinstance(value, torch.Tensor) else value)
             for key, value in item.items()}
            for item in view["checkpoints"]
        ]
        views.append(selected)
    return {"x_t": recorder.x_t[index].clone(), "x_t_plus_1": recorder.x_t_plus_1[index].clone(),
            "phase_shift_dx_dy": recorder.phases[index], "views": views}


def save_sample_elements(directory, trace, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    grid = trace["views"][0]["grid_size"]
    image = pixel_image(trace["x_t"])
    image.save(directory / "x_t.png")
    grid_image(image, grid).save(directory / "x_t_grid.png")
    pixel_image(trace["x_t_plus_1"]).save(directory / "x_t_plus_1.png")
    torch.save(trace, directory / "trace.pt")
    rows = []
    noise_limit = max(float(view["scaled_rgb_noise"].abs().max()) for view in trace["views"])
    state_values = []
    for view in trace["views"]:
        state_values.extend(view[key] for key in ("initial_before_noise", "initial_after_noise"))
        for item in view["checkpoints"]:
            state_values.extend((item["before"], item["after"]))
    rms_limit = max(float(value.float().square().mean(-1).sqrt().max()) for value in state_values)
    noise_rms_limit = max(float(view["feature_noise"].float().square().mean(-1).sqrt().max()) for view in trace["views"])
    clip_fractions = {}
    equivalent_verification = {}
    for name, view in zip(("original", "translated"), trace["views"]):
        destination = directory / "group00" / name
        destination.mkdir(parents=True, exist_ok=True)
        view_image = pixel_image(view["pixels"])
        view_image.save(destination / "view.png")
        grid_image(view_image, grid).save(destination / "view_grid.png")
        rgb = tile_patches(view["scaled_rgb_noise"].unsqueeze(0), grid)[0]
        pixel_image(0.5 + rgb / (2 * max(noise_limit, 1e-12))).save(destination / "rgb_noise_display.png")
        equivalent = view["equivalent_noisy_pixels_unclipped"]
        pixel_image(equivalent).save(destination / "noisy_equivalent_clipped.png")
        clip_fractions[name] = float(((equivalent < 0) | (equivalent > 1)).float().mean())
        equivalent_verification[name] = {
            "max_abs_error": float(view["equivalent_feature_max_abs_error"]),
            "rtol": view["equivalent_verification_rtol"],
            "atol": view["equivalent_verification_atol"],
        }
        for key in ("initial_before_noise", "initial_after_noise", "feature_noise"):
            rms = view[key].float().square().mean(-1).sqrt()
            limit = noise_rms_limit if key == "feature_noise" else rms_limit
            spatial_image(rms, grid, image.size, maximum=limit).save(destination / f"{key}_rms.png")
        for item in view["checkpoints"]:
            checkpoint, native_grid, mask = item["checkpoint"], item["grid_size"], item["mask"]
            spatial_image(mask.float(), native_grid, image.size, maximum=1).save(destination / f"{checkpoint}_mask.png")
            masked_view = view_image.copy()
            draw = ImageDraw.Draw(masked_view)
            for patch_index in mask.nonzero().flatten().tolist():
                row, col = divmod(patch_index, native_grid[1])
                width, height = image.width // native_grid[1], image.height // native_grid[0]
                draw.rectangle((col * width, row * height, (col + 1) * width - 1, (row + 1) * height - 1), fill=(0, 0, 0))
                rows.append((name, checkpoint, patch_index, row, col))
            grid_image(masked_view, native_grid).save(destination / f"{checkpoint}_mask_on_view.png")
            for boundary in ("before", "after"):
                rms = item[boundary].float().square().mean(-1).sqrt()
                spatial_image(rms, native_grid, image.size, maximum=rms_limit).save(destination / f"{checkpoint}_{boundary}_rms.png")
    with (directory / "drop_positions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("view", "checkpoint", "patch_index", "row", "col"))
        writer.writerows(rows)
    metadata.update({
        "step_index": 0, "nominal_steps": NOMINAL_STEPS, "executed_steps": 1,
        "captured_group": 0, "captured_views": ["original", "translated"],
        "phase_shift_dx_dy": list(trace["phase_shift_dx_dy"]),
        "grid_size": list(grid), "indices": "group, step, patch_index, row and col are zero-based",
        "visualization": {
            "rgb_noise": "actual RMS-scaled RGB noise in model-normalized input units; 0 maps to gray",
            "rgb_noise_symmetric_limit": noise_limit,
            "feature_rms": "per-token sqrt(mean(feature**2)); shared scale across both views and state boundaries",
            "feature_rms_max": rms_limit, "feature_noise_rms_max": noise_rms_limit,
            "noisy_equivalent": "equivalent RGB input verified against actual noisy initial features; PNG clipped to [0,1] for display; trace.pt retains unclipped pixels",
            "noisy_equivalent_clipped_fraction": clip_fractions,
            "noisy_equivalent_verification": equivalent_verification,
            "mask": "white marks selected tokens; each checkpoint has its own actual mask",
            "mask_on_view": "spatial illustration of feature-drop positions, not a pixel-masked model input",
            "x_t_plus_1": "actual projected pixel update using all 20 differentiable views; trace.pt retains float pixels",
        },
    })
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_geometry(adapter):
    projection = adapter.model.patch_embed.proj
    if not isinstance(adapter.model.norm_pre, torch.nn.Identity):
        raise ValueError("equivalent RGB previews require identity norm_pre")
    if (tuple(projection.kernel_size) != (16, 16) or tuple(projection.stride) != (16, 16)
            or tuple(projection.padding) != (0, 0) or tuple(projection.dilation) != (1, 1)
            or projection.groups != 1 or tuple(adapter.model.patch_embed.grid_size) != (14, 14)):
        raise ValueError("Figure 2 requires non-overlapping ViT-B/16 224x224 RGB projection")
    if adapter.model.num_prefix_tokens != 1 or tuple(adapter.checkpoints) != ("block3", "block10"):
        raise ValueError("Figure 2 requires ViT-B/16 with block3/block10 checkpoints")


def export_figure2_elements(adapter, dataset, output_dir, *, batch_size=4):
    validate_geometry(adapter)
    if batch_size <= 0 or len(dataset) < SAMPLE_COUNT:
        raise ValueError("positive batch size and at least 50 annotated images are required")
    selected = dataset.samples[:SAMPLE_COUNT]
    names = [Path(sample["image_name"]).stem for sample in selected]
    if len(set(names)) != SAMPLE_COUNT or any(name in ("", ".", "..") for name in names):
        raise ValueError("sample folder names must be valid and unique")
    output_dir = Path(output_dir)
    if output_dir.exists() and {p.name for p in output_dir.iterdir() if p.is_dir()} - set(names):
        raise ValueError("output contains other sample folders; existing data was preserved")
    for sample in selected:
        with Image.open(sample["image_path"]) as source:
            if source.size != (224, 224):
                raise ValueError(f"expected 224x224 input: {sample['image_path']}")
    loader = DataLoader(Subset(dataset, range(SAMPLE_COUNT)), batch_size=batch_size, shuffle=False, num_workers=0)
    attacker = Figure2Attacker(adapter, device=adapter.device)
    for images, labels, indices in loader:
        attacker.attack_batch(images, labels)
        for offset, index in enumerate(indices.tolist()):
            sample = dataset.samples[index]
            source_path = Path(sample["image_path"])
            metadata = {
                "image_name": sample["image_name"], "sample_index": index,
                "class_id": int(labels[offset]), "image_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "sample_order": "first 50 available annotated image names in sorted order; no prediction filtering",
                "attack": attacker.parameters(), "batch_size": batch_size,
                "torch_version": torch.__version__, "device": str(adapter.device),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "batch_sample_indices": indices.tolist(),
                "pretrained_cfg": {key: adapter.model.pretrained_cfg.get(key) for key in ("architecture", "tag", "hf_hub_id")},
            }
            save_sample_elements(output_dir / names[index], sample_trace(attacker.recorder, offset), metadata)
        print(f"Figure 2 elements: {int(indices[-1]) + 1}/{SAMPLE_COUNT}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export first-iteration PRD assets for the first 50 images")
    parser.add_argument("--image-dir", default=str(ROOT / "data" / "clean_resized_images"))
    parser.add_argument("--annotations-path", default=str(ROOT / "data" / "image_name_to_class_id_and_name.json"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args(argv)


def main(args):
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = DEVICE if args.device == "auto" else torch.device(args.device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    dataset = ImageDataset(args.image_dir, args.annotations_path, transform=transform)
    if len(dataset) < SAMPLE_COUNT:
        raise ValueError("at least 50 annotated images are required")
    try:
        adapter = build_whitebox_model(model_name=DEFAULT_MODEL_NAME, device=device)
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("Install requirements.txt and prepare pretrained ViT-B/16 weights in data/huggingface before exporting Figure 2") from exc
    export_figure2_elements(adapter, dataset, OUTPUT_DIR, batch_size=args.batch_size)
    print(f"Saved Figure 2 assets for 50 samples to {OUTPUT_DIR}")


if __name__ == "__main__":
    main(parse_args())
