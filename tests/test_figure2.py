from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from figure2 import Figure2Attacker, export_figure2_elements, sample_trace, save_sample_elements
from nets.vit import ViTAdapter
from progressive_attack import ProgressiveMaskSchedule, ProgressiveMaskSelection, ProgressiveRouteDisruptionAttacker
from tests.test_progressive_attack import TinyViT
from utils import ImageDataset


def make_adapter():
    torch.manual_seed(42)
    adapter = ViTAdapter.__new__(ViTAdapter)
    nn.Module.__init__(adapter)
    adapter.device = torch.device("cpu")
    adapter.model_name = "vit_base_patch16_224"
    adapter.model_mean = (0.485, 0.456, 0.406)
    adapter.model_std = (0.229, 0.224, 0.225)
    adapter.model = TinyViT(1)
    adapter.dynamic_checkpoint = False
    adapter.model.patch_embed.proj = nn.Conv2d(3, 4, kernel_size=16, stride=16, bias=False)
    return adapter.eval()


class Figure2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def inputs(self):
        pixels = torch.linspace(0.05, 0.95, 3 * 224 * 224).reshape(1, 3, 224, 224)
        return (pixels - 0.5) / 0.5, torch.tensor([1])

    def capture(self):
        attacker = Figure2Attacker(make_adapter(), device=torch.device("cpu"))
        images, labels = self.inputs()
        torch.manual_seed(42)
        attacker.attack_batch(images, labels)
        return attacker

    def test_capture_preserves_pixels_rng_and_all_twenty_views(self):
        adapter = make_adapter()
        images, labels = self.inputs()
        baseline = ProgressiveRouteDisruptionAttacker(
            adapter, steps=1, step_size=(16 / 255) / 10, device=torch.device("cpu")
        )
        traced = Figure2Attacker(adapter, device=torch.device("cpu"))
        advance, mask, randn = adapter.advance_progressive_state, adapter.apply_progressive_mask, torch.randn
        torch.manual_seed(42)
        expected = baseline.attack_batch(images, labels)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        observed = traced.attack_batch(images, labels)
        torch.testing.assert_close(expected, observed, rtol=0, atol=0)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(traced._actual_forward_view_count, 20)
        self.assertEqual(len(traced.recorder.views), 2)
        self.assertEqual(advance, adapter.advance_progressive_state)
        self.assertEqual(mask, adapter.apply_progressive_mask)
        self.assertIs(randn, torch.randn)
        delta = traced.recorder.x_t_plus_1 - traced.recorder.x_t
        self.assertLessEqual(float(delta.abs().max()), (16 / 255) / 10 + 1e-7)

    def test_recorded_noise_and_masks_match_actual_states_and_spatial_pair(self):
        attacker = self.capture()
        original, translated = attacker.recorder.views
        self.assertFalse(torch.equal(original["coefficients"], translated["coefficients"]))
        schedule = ProgressiveMaskSchedule(tuple(
            ProgressiveMaskSelection(item["checkpoint"], item["mask"], 10, item["grid_size"])
            for item in original["checkpoints"]
        ))
        shifted = attacker._phase_mask_schedule(schedule, attacker.recorder.phases, height=224, width=224)
        torch.testing.assert_close(
            translated["pixels"], attacker._apply_samplewise_phase_shifts(original["pixels"], attacker.recorder.phases),
            rtol=0, atol=0,
        )
        for index, item in enumerate(translated["checkpoints"]):
            self.assertTrue(torch.equal(item["mask"], shifted.masks[index]))
        for view in (original, translated):
            equivalent = attacker.model.begin_progressive_forward(attacker._normalize(view["equivalent_noisy_pixels_unclipped"]))
            torch.testing.assert_close(equivalent.local_tokens, view["initial_after_noise"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(view["initial_after_noise"], view["initial_before_noise"] + view["feature_noise"], rtol=0, atol=0)
            for item in view["checkpoints"]:
                self.assertEqual(int(item["mask"].sum()), 10)
                self.assertEqual(int(item["after"][item["mask"]].count_nonzero()), 0)
                torch.testing.assert_close(item["before"][~item["mask"]], item["after"][~item["mask"]], rtol=0, atol=0)
                torch.testing.assert_close(item["prefix_before"], item["prefix_after"], rtol=0, atol=0)

    def test_hooks_restore_when_forward_fails(self):
        adapter = make_adapter()
        attacker = Figure2Attacker(adapter, device=torch.device("cpu"))
        advance, mask, randn = adapter.advance_progressive_state, adapter.apply_progressive_mask, torch.randn
        with patch.object(adapter, "finish_progressive_forward", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                attacker.attack_batch(*self.inputs())
        self.assertEqual(advance, adapter.advance_progressive_state)
        self.assertEqual(mask, adapter.apply_progressive_mask)
        self.assertIs(randn, torch.randn)
        self.assertIsNone(attacker._active_view)

    def test_saved_assets_and_csv_retain_captured_data(self):
        attacker = self.capture()
        trace = sample_trace(attacker.recorder, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_sample_elements(root, trace, {"image_name": "fixture.png"})
            restored = torch.load(root / "trace.pt", weights_only=True)
            torch.testing.assert_close(restored["x_t_plus_1"], trace["x_t_plus_1"], rtol=0, atol=0)
            with (root / "drop_positions.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 40)
            for row in rows:
                view = trace["views"][0 if row["view"] == "original" else 1]
                checkpoint = next(item for item in view["checkpoints"] if item["checkpoint"] == row["checkpoint"])
                index = int(row["patch_index"])
                self.assertTrue(checkpoint["mask"][index])
                self.assertEqual(index, int(row["row"]) * 14 + int(row["col"]))
            metadata = json.loads((root / "metadata.json").read_text())
            self.assertEqual(metadata["nominal_steps"], 10)
            self.assertEqual(metadata["executed_steps"], 1)
            for name in ("original", "translated"):
                directory = root / "group00" / name
                self.assertEqual(len(list(directory.glob("*.png"))), 15)
                mask = Image.open(directory / "block3_mask.png")
                after = Image.open(directory / "block3_after_rms.png")
                for patch_index in trace["views"][0 if name == "original" else 1]["checkpoints"][0]["mask"].nonzero().flatten().tolist():
                    row, col = divmod(patch_index, 14)
                    self.assertEqual(mask.getpixel((col * 16 + 8, row * 16 + 8)), 255)
                    self.assertEqual(after.getpixel((col * 16 + 8, row * 16 + 8)), 0)

    def test_export_selects_first_fifty_names_without_prediction_filtering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            labels = {}
            for index in reversed(range(52)):
                name = f"sample_{index:03d}.png"
                Image.new("RGB", (224, 224), (index, 100, 150)).save(images / name)
                labels[name] = {"class_id": index % 2}
            annotation = root / "labels.json"
            annotation.write_text(json.dumps(labels))
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
            dataset = ImageDataset(str(images), str(annotation), transform=transform)
            written = []

            def record_save(directory, trace, metadata):
                directory.mkdir(parents=True)
                written.append((directory.name, metadata["sample_index"], trace["x_t"].shape))

            with patch("figure2.save_sample_elements", side_effect=record_save), redirect_stdout(StringIO()):
                export_figure2_elements(make_adapter(), dataset, root / "figure2", batch_size=8)
            self.assertEqual([name for name, _, _ in written], [f"sample_{i:03d}" for i in range(50)])
            self.assertEqual([index for _, index, _ in written], list(range(50)))
            self.assertEqual(len(list((root / "figure2").iterdir())), 50)


if __name__ == "__main__":
    unittest.main()
