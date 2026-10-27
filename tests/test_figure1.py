import csv
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from figure1 import (
    COLOR_ANCHORS,
    compute_patch_scores,
    export_figure1_elements,
    extract_layerwise_scores,
    render_heatmap_overlay,
)
from nets.base import ProgressiveState
from utils import ImageDataset


class MixingBlock(nn.Module):
    def forward(self, tokens):
        return tokens + 0.1 * tokens.mean(dim=1, keepdim=True)


class ObservationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 4, kernel_size=16, stride=16)
        self.patch_embed.grid_size = (14, 14)
        self.cls_token = nn.Parameter(torch.tensor([[[1.0, 0.5, -0.25, 0.75]]]))
        self.blocks = nn.Sequential(*(MixingBlock() for _ in range(12)))
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 3)
        self.pretrained_cfg = {"architecture": "observation-test-fixture", "input_size": (3, 224, 224)}

    def initial_tokens(self, x):
        local = self.patch_embed(x).flatten(2).transpose(1, 2)
        return torch.cat((self.cls_token.expand(x.size(0), -1, -1), local), dim=1)

    def forward_head(self, tokens):
        return self.head(tokens[:, 0])

    def forward(self, x):
        return self.forward_head(self.norm(self.blocks(self.initial_tokens(x))))


class ObservationAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.model_name = "observation-test-fixture"
        self.model_mean = (0.1, 0.2, 0.3)
        self.model_std = (0.4, 0.5, 0.6)
        self.model = ObservationModel()
        self.seen_inputs = []

    def begin_progressive_forward(self, x):
        self.seen_inputs.append(x.clone())
        tokens = self.model.initial_tokens(x)
        return ProgressiveState(tokens[:, 1:], (14, 14), {"prefix_tokens": tokens[:, :1]})

    def forward(self, x):
        return self.model(x)


class Figure1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_scores_use_supplied_global_direction_and_reject_undefined_cosines(self):
        local = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
        global_repr = torch.tensor([[1.0, 0.0]])
        torch.testing.assert_close(compute_patch_scores(local, global_repr), torch.tensor([[1.0, 0.0, -1.0]]))
        torch.testing.assert_close(compute_patch_scores(local, global_repr.flip(1)), torch.tensor([[0.0, 1.0, 0.0]]))
        with self.assertRaises(ValueError):
            compute_patch_scores(local, torch.zeros_like(global_repr))

    def test_observation_preserves_native_forward_and_uses_final_normalized_cls(self):
        adapter = ObservationAdapter()
        x = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
        observed = extract_layerwise_scores(adapter, x)
        self.assertEqual(observed.scores.shape, (1, 12, 196))
        self.assertEqual(observed.grid_size, (14, 14))
        with torch.no_grad():
            torch.testing.assert_close(observed.logits, adapter(x), rtol=0, atol=0)
            tokens = adapter.model.initial_tokens(x)
            snapshots = []
            for block in adapter.model.blocks:
                tokens = block(tokens)
                snapshots.append(tokens.clone())
            final_cls = adapter.model.norm(tokens)[:, :1]
            for layer, snapshot in enumerate(snapshots):
                expected = torch.nn.functional.cosine_similarity(snapshot[:, 1:], final_cls, dim=-1)
                torch.testing.assert_close(observed.scores[:, layer], expected)
            same_layer = torch.nn.functional.cosine_similarity(snapshots[0][:, 1:], snapshots[0][:, :1], dim=-1)
            raw_final = torch.nn.functional.cosine_similarity(snapshots[0][:, 1:], snapshots[-1][:, :1], dim=-1)
            self.assertFalse(torch.allclose(observed.scores[:, 0], same_layer))
            self.assertFalse(torch.allclose(observed.scores[:, 0], raw_final))
        self.assertFalse(observed.scores.requires_grad)
        self.assertFalse(adapter.training)
        self.assertFalse(torch.equal(observed.scores[:, 0], observed.scores[:, -1]))

    def test_overlay_preserves_discrete_grid_and_uses_layerwise_min_max(self):
        image = Image.new("RGB", (32, 32), (80, 90, 100))
        scores = torch.tensor([-1.0, 0.0, 1.0, 0.5])
        overlay = render_heatmap_overlay(image, scores, (2, 2), alpha=1)
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(overlay.getpixel((8, 8)), COLOR_ANCHORS[0])
        self.assertEqual(overlay.getpixel((24, 8)), COLOR_ANCHORS[3])
        self.assertEqual(overlay.getpixel((8, 24)), COLOR_ANCHORS[-1])
        self.assertEqual(overlay.getpixel((24, 24)), (255, 194, 30))
        self.assertEqual(overlay.getpixel((1, 1)), overlay.getpixel((15, 15)))
        self.assertNotEqual(overlay.getpixel((16, 8)), overlay.getpixel((24, 8)))
        narrow_range = render_heatmap_overlay(image, scores * 0.125 + 0.375, (2, 2), alpha=1)
        self.assertEqual(narrow_range.tobytes(), overlay.tobytes())
        constant = render_heatmap_overlay(image, torch.full((4,), 0.3), (2, 2), alpha=1)
        for position in ((8, 8), (24, 8), (8, 24), (24, 24)):
            self.assertEqual(constant.getpixel(position), COLOR_ANCHORS[3])

    def test_export_writes_exact_first_50_sample_folders_and_complete_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            annotations = {}
            for index in reversed(range(52)):
                name = f"image_{index:03d}.png"
                Image.new("RGB", (224, 224), (index, 60, 140)).save(images / name)
                annotations[name] = {"class_id": index % 3}
            annotation_path = root / "labels.json"
            annotation_path.write_text(json.dumps(annotations))
            dataset = ImageDataset(str(images), str(annotation_path), transform=transforms.ToTensor())
            adapter = ObservationAdapter()
            output = root / "figure1"
            with redirect_stdout(StringIO()):
                export_figure1_elements(adapter, dataset, output, batch_size=8)
            self.assertEqual(sorted(path.name for path in output.iterdir()), [f"image_{i:03d}" for i in range(50)])
            for directory in output.iterdir():
                expected = {"input.png", "patch_scores.csv", "metadata.json"}
                expected.update(f"overlay_block{layer:02d}.png" for layer in range(1, 13))
                self.assertEqual({path.name for path in directory.iterdir()}, expected)
            sample = output / "image_000"
            with (sample / "patch_scores.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12 * 196)
            self.assertEqual({int(row["layer"]) for row in rows}, set(range(1, 13)))
            for row in rows:
                self.assertEqual(int(row["patch_index"]), int(row["row"]) * 14 + int(row["col"]))
            normalized = (transforms.ToTensor()(Image.open(images / "image_000.png")).unsqueeze(0)
                          - torch.tensor(adapter.model_mean).view(1, 3, 1, 1)) / torch.tensor(adapter.model_std).view(1, 3, 1, 1)
            torch.testing.assert_close(adapter.seen_inputs[0][:1], normalized)
            expected_scores = extract_layerwise_scores(adapter, normalized).scores[0]
            csv_scores = torch.tensor([float(row["score"]) for row in rows]).reshape(12, 196)
            torch.testing.assert_close(csv_scores, expected_scores, rtol=0, atol=0)
            metadata = json.loads((sample / "metadata.json").read_text())
            self.assertEqual(metadata["sample_index"], 0)
            self.assertEqual(metadata["image_name"], "image_000.png")
            self.assertEqual(metadata["global_representation"],
                             "fixed final encoder CLS token after model.norm, shared by all observation layers")
            self.assertEqual((metadata["overlay"]["vmin"], metadata["overlay"]["vmax"]), (0, 1))
            self.assertEqual(metadata["overlay"]["normalization"],
                             "per-image, per-layer min-max over all local patch scores; CSV retains raw cosine similarities")
            self.assertEqual(metadata["overlay"]["constant_score_value"], 0.5)
            self.assertEqual(metadata["overlay"]["colorbar_label"], "Normalized patch score")
            self.assertEqual(metadata["overlay"]["color_anchors_rgb"], [list(color) for color in COLOR_ANCHORS])
            self.assertEqual(metadata["overlay"]["color_anchor_positions"], [index / 6 for index in range(7)])
            self.assertEqual(metadata["overlay"]["color_interpolation"], "linear RGB between adjacent anchors")
            with Image.open(sample / "input.png") as saved:
                self.assertEqual(saved.getpixel((10, 10)), (0, 60, 140))


if __name__ == "__main__":
    unittest.main()
