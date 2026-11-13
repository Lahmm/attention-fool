from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from data_collection import collect_batch, collect_dataset, layer_rows
from test_figure1 import make_adapter, MixingBlock
from utils import ImageDataset


def observation_adapter():
    adapter = make_adapter()
    adapter.model.blocks = nn.ModuleList(MixingBlock() for _ in range(12))
    return adapter.eval()


class DataCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_one_forward_matches_native_outputs_and_production_selection(self):
        adapter = observation_adapter()
        pixels = torch.linspace(.05, .95, 2 * 3 * 224 * 224).reshape(2, 3, 224, 224)
        counts = [0] * 12
        handles = []
        for index, block in enumerate(adapter.model.blocks):
            def count(module, inputs, output, layer=index):
                counts[layer] += 1
            handles.append(block.register_forward_hook(count))
        try:
            observations = collect_batch(adapter, pixels)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts, [1] * 12)
        self.assertEqual(observations.patch_scores.shape, (2, 12, 196))
        self.assertEqual(observations.projections.dtype, torch.float64)
        self.assertFalse(observations.patch_scores.requires_grad)
        normalized = (pixels - pixels.new_tensor(adapter.model_mean).reshape(1, 3, 1, 1))
        normalized = normalized / pixels.new_tensor(adapter.model_std).reshape(1, 3, 1, 1)
        with torch.no_grad():
            expected_logits = adapter.finish_gdi_forward(adapter.begin_gdi_forward(normalized))
        torch.testing.assert_close(observations.logits, expected_logits, rtol=0, atol=0)
        choices = adapter.select_contraction_checkpoints(normalized)
        for sample, choice in enumerate(choices):
            selected = next(row for row in layer_rows(observations, sample) if row["selected"])
            self.assertEqual(selected["input_checkpoint"], choice.checkpoint)
            self.assertEqual(f"block{selected['block']}", choice.peak_layer)
            self.assertEqual(selected["S"], choice.contraction_score)
            self.assertEqual(selected["R"], choice.direction_variance_ratio)
            self.assertEqual(selected["D_in"], choice.input_total_variance)
            self.assertEqual(selected["D_out"], choice.output_total_variance)

    def test_exported_token_projections_explain_variance_and_block_continuity(self):
        adapter = observation_adapter()
        pixels = torch.linspace(.1, .9, 3 * 224 * 224).reshape(1, 3, 224, 224)
        observed = collect_batch(adapter, pixels)
        statistics = observed.statistics
        torch.testing.assert_close(observed.centered_projections.square().mean(dim=2),
                                   statistics["global_direction_population_variance"], rtol=0, atol=0)
        torch.testing.assert_close(observed.centered_projections.mean(dim=2),
                                   torch.zeros(1, 12, dtype=torch.float64), rtol=0, atol=1e-14)
        torch.testing.assert_close(observed.projections.mean(dim=2),
                                   statistics["score_mean_float64"], rtol=0, atol=0)
        torch.testing.assert_close(statistics["output_total_variance"][:, :-1],
                                   statistics["input_total_variance"][:, 1:], rtol=0, atol=0)
        torch.testing.assert_close(statistics["direction_variance_ratio"]
                                   * statistics["average_direction_population_variance"],
                                   statistics["global_direction_population_variance"])

    def test_first_50_complete_exports_have_lossless_scores_and_matching_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            annotations = {}
            for index in reversed(range(52)):
                name = f"sample_{index:03d}.png"
                Image.new("RGB", (224, 224), (index + 10, 70, 160)).save(images / name)
                annotations[name] = {"class_id": index % 3}
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps(annotations))
            dataset = ImageDataset(str(images), str(annotation_path), transform=transforms.ToTensor())
            output = root / "observations"
            with redirect_stdout(StringIO()):
                collect_dataset(observation_adapter(), dataset, output, batch_size=8)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(len(manifest), 50)
            self.assertEqual([entry["image_name"] for entry in manifest],
                             [f"sample_{i:03d}.png" for i in range(50)])
            self.assertEqual(sum(path.is_dir() for path in output.iterdir()), 50)
            for entry in manifest:
                directory = output / entry["directory"]
                self.assertEqual({path.name for path in directory.iterdir()}, {
                    "input.png", "token_scores.csv", "layer_statistics.csv",
                    "observations.pt", "metadata.json",
                })
                with (directory / "layer_statistics.csv").open(newline="") as handle:
                    layers = list(csv.DictReader(handle))
                self.assertEqual([int(row["block"]) for row in layers], list(range(1, 13)))
                self.assertEqual(sum(int(row["selected"]) for row in layers), 1)
            directory = output / manifest[0]["directory"]
            data = torch.load(directory / "observations.pt", weights_only=True)
            with (directory / "token_scores.csv").open(newline="") as handle:
                tokens = list(csv.DictReader(handle))
            self.assertEqual(len(tokens), 12 * 196)
            for row in tokens:
                layer, token = int(row["block"]) - 1, int(row["token_index"])
                self.assertEqual(token, int(row["row"]) * 14 + int(row["col"]))
                self.assertEqual(row["input_checkpoint"], f"block{layer}")
                self.assertEqual(float(row["patch_score"]), float(data["patch_scores"][layer, token]))
                self.assertEqual(float(row["centered_projection_float64"]),
                                 float(data["centered_projections_float64"][layer, token]))
            with Image.open(directory / "input.png") as image, Image.open(images / "sample_000.png") as original:
                self.assertEqual(image.tobytes(), original.tobytes())
            configuration = json.loads((output / "collection_metadata.json").read_text())
            self.assertEqual(configuration["seed"], 42)
            self.assertEqual(configuration["block_count"], 12)
            self.assertIn("no phase, noise, drop", configuration["forward"])
            with self.assertRaisesRegex(ValueError, "new or empty"):
                collect_dataset(observation_adapter(), dataset, output)


if __name__ == "__main__":
    unittest.main()
