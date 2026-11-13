from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from attack import GDIAttacker
from figure1 import capture_first_pair, export_figure1_elements, rgb_noise_image
from nets.vit import DEFAULT_MODEL_NAME, ViTAdapter
from utils import ImageDataset, SEED


class PatchEmbed(nn.Module):
    grid_size = (14, 14)

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 4, kernel_size=16, stride=16)

    def forward(self, pixels):
        return self.proj(pixels).flatten(2).transpose(1, 2)


class MixingBlock(nn.Module):
    def forward(self, tokens):
        return tokens + 0.1 * tokens.mean(dim=1, keepdim=True)


class TinyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens = 1
        self.patch_embed = PatchEmbed()
        self.patch_drop = self.norm_pre = nn.Identity()
        self.norm = nn.LayerNorm(4)
        self.blocks = nn.ModuleList(MixingBlock() for _ in range(3))
        self.head = nn.Linear(4, 3)

    def _pos_embed(self, local):
        prefix = local.new_tensor([1., .5, -.25, .75]).reshape(1, 1, 4)
        return torch.cat((prefix.expand(local.size(0), -1, -1), local), dim=1)

    def forward_head(self, tokens):
        return self.head(tokens[:, 0])


def make_adapter():
    torch.manual_seed(SEED)
    adapter = ViTAdapter.__new__(ViTAdapter)
    nn.Module.__init__(adapter)
    adapter.device = torch.device("cpu")
    adapter.model_name = DEFAULT_MODEL_NAME
    adapter.model_mean = (.1, .2, .3)
    adapter.model_std = (.4, .5, .6)
    adapter.model = TinyVisionModel()
    return adapter.eval()


class Figure1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_capture_preserves_production_random_stream_and_actual_block_inputs(self):
        adapter = make_adapter()
        pixels = torch.linspace(0, 1, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
        labels = torch.tensor([0, 1])
        torch.manual_seed(SEED)
        choices, phases, views, _ = capture_first_pair(adapter, pixels, labels)
        actual_rng = torch.get_rng_state().clone()
        self.assertEqual(len(views), 2)
        self.assertFalse(torch.equal(views[0].rgb_noise, views[1].rgb_noise))
        self.assertTrue(all(not block._forward_pre_hooks for block in adapter.model.blocks))

        attacker = GDIAttacker(adapter, device=adapter.device)
        attacker._prepare_dynamic_mask_layout(choices, 2)
        expected_locals = []
        expected_masks = []
        native_drop = adapter.apply_gdi_mask

        def observe_drop(state, mask):
            result = native_drop(state, mask)
            expected_locals.append(result.local_tokens.clone())
            expected_masks.append(mask.clone())
            return result

        torch.manual_seed(SEED)
        with torch.no_grad(), patch.object(adapter, "apply_gdi_mask", side_effect=observe_drop):
            losses = attacker._iter_attack_losses(pixels, labels)
            next(losses)
            next(losses)
            losses.close()
        torch.testing.assert_close(torch.get_rng_state(), actual_rng, rtol=0, atol=0)
        index = 0
        for view in views:
            for checkpoint in view.masks:
                torch.testing.assert_close(view.masks[checkpoint], expected_masks[index], rtol=0, atol=0)
                torch.testing.assert_close(view.local_after_drop[checkpoint], expected_locals[index], rtol=0, atol=0)
                index += 1
        for i, (dx, dy) in enumerate(phases):
            reference = F.pad(pixels[i:i+1], (dx, 0, dy, 0), mode="reflect")[:, :, :224, :224]
            torch.testing.assert_close(views[1].pixels[i:i+1], reference, rtol=0, atol=0)

    def test_captured_noise_is_the_draw_used_by_feature_augmentation(self):
        adapter = make_adapter()
        pixels = torch.full((1, 3, 224, 224), .6)
        _, _, views, _ = capture_first_pair(adapter, pixels, torch.tensor([1]))
        projection = adapter.model.patch_embed.proj.weight.detach().reshape(4, -1)
        attacker = GDIAttacker(adapter, device=adapter.device)
        with torch.no_grad():
            for view in views:
                raw_features = view.rgb_noise.flatten(2).matmul(projection.t())
                initial = adapter.begin_gdi_forward(attacker._normalize(view.pixels)).local_tokens
                rms = initial.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
                noise_rms = raw_features.square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-6)
                expected = attacker.noise_strength * raw_features * (rms / noise_rms)
                torch.testing.assert_close(view.feature_noise, expected, rtol=0, atol=0)

    def test_each_sample_is_captured_at_its_own_selected_input_boundary(self):
        adapter = make_adapter()
        pixels = torch.full((2, 3, 224, 224), .6)
        labels = torch.tensor([0, 1])
        attacker = GDIAttacker(adapter, device=adapter.device)
        choices = adapter.select_contraction_checkpoints(attacker._normalize(pixels))
        choices = (replace(choices[0], checkpoint="block0", peak_layer="block1"),
                   replace(choices[1], checkpoint="block2", peak_layer="block3"))
        with patch.object(adapter, "select_contraction_checkpoints", return_value=choices):
            _, _, views, _ = capture_first_pair(adapter, pixels, labels)
        for view in views:
            self.assertEqual(set(view.masks), {"block0", "block2"})
            self.assertEqual(view.masks["block0"].sum(1).tolist(), [20, 0])
            self.assertEqual(view.masks["block2"].sum(1).tolist(), [0, 20])
            for sample, choice in enumerate(choices):
                local = view.local_after_drop[choice.checkpoint][sample]
                mask = view.masks[choice.checkpoint][sample]
                self.assertEqual(torch.count_nonzero(local[mask]), 0)

    def test_noise_patch_assembly_preserves_channel_and_spatial_order(self):
        noise = torch.arange(4 * 3 * 2 * 2).reshape(4, 3, 2, 2).float() / 10 - 2
        image = rgb_noise_image(noise, (2, 2))
        for index in range(4):
            row, col = divmod(index, 2)
            torch.testing.assert_close(image[:, row*2:row*2+2, col*2:col*2+2],
                                       (.5 + noise[index] / 6).clamp(0, 1), rtol=0, atol=0)

    def test_export_writes_first_50_real_pixels_masks_features_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            annotations = {}
            for index in reversed(range(52)):
                name = f"image_{index:03d}.png"
                Image.new("RGB", (224, 224), (index + 20, 60, 140)).save(images / name)
                annotations[name] = {"class_id": index % 3}
            labels = root / "labels.json"
            labels.write_text(json.dumps(annotations))
            dataset = ImageDataset(str(images), str(labels), transform=transforms.ToTensor())
            output = root / "export"
            with redirect_stdout(StringIO()):
                export_figure1_elements(make_adapter(), dataset, output, batch_size=8)
            folders = sorted(path.name for path in output.iterdir() if path.is_dir())
            self.assertEqual(folders, [f"image_{i:03d}" for i in range(50)])
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([entry["sample_index"] for entry in manifest], list(range(50)))
            for index, folder in enumerate(folders):
                directory = output / folder
                self.assertEqual({path.name for path in directory.iterdir()}, {
                    "input.png", "phase.png", "noise.png", "phase_noise.png", "drop.png",
                    "phase_drop.png", "mask.png", "phase_mask.png", "capture.pt", "metadata.json",
                })
                metadata = json.loads((directory / "metadata.json").read_text())
                self.assertEqual(metadata["sample_index"], index)
                self.assertEqual(metadata["drop_count"], 20)
                self.assertEqual(metadata["phase_drop_count"], 20)
                self.assertEqual(metadata["capture"]["optimization_updates_executed"], 0)
                self.assertIn("not an inversion", metadata["drop_visualization"]["interpretation"])
            directory = output / folders[0]
            with Image.open(directory / "input.png") as image:
                with Image.open(images / "image_000.png") as source:
                    self.assertEqual(image.tobytes(), source.tobytes())
                original = transforms.PILToTensor()(image)
            saved = torch.load(directory / "capture.pt", weights_only=True)
            with Image.open(directory / "mask.png") as image:
                mask = transforms.PILToTensor()(image).bool()
            with Image.open(directory / "drop.png") as image:
                actual = transforms.PILToTensor()(image)
            torch.testing.assert_close(actual, original.masked_fill(mask, 0), rtol=0, atol=0)
            torch.testing.assert_close(mask[0, ::16, ::16].flatten(), saved["original"]["mask"])
            for view in saved.values():
                self.assertEqual(torch.count_nonzero(view["local_after_drop"][view["mask"]]), 0)
            with self.assertRaisesRegex(ValueError, "new or empty"):
                export_figure1_elements(make_adapter(), dataset, output)


if __name__ == "__main__":
    unittest.main()
