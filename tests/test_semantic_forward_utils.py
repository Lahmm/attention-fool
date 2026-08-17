import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from experiments.semantic_forward_utils import (
    capture_patch_score_activation,
    common_map,
    load_samples,
    rank_norm,
    row_spearman,
    top_mask,
)
from nets.base import PatchScoreActivationCapture


class TinyCaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.Conv2d(3, 2, kernel_size=1)
        self.head = torch.nn.Linear(2, 3)

    def patch_score_activation_capture(self, score_layer):
        if score_layer != "final":
            raise ValueError(score_layer)
        return PatchScoreActivationCapture(
            self.activation, "output", "tiny activation output"
        )

    def forward(self, pixels):
        features = self.activation(pixels)
        return self.head(features.mean(dim=(2, 3)))


class SemanticForwardUtilsTests(unittest.TestCase):
    def test_rank_common_grid_and_top_mask(self):
        values = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
        self.assertTrue(
            torch.allclose(
                rank_norm(values), torch.tensor([[1.0, 0.0, 2 / 3, 1 / 3]])
            )
        )
        self.assertAlmostEqual(float(row_spearman(values, values)), 1.0, places=6)
        self.assertEqual(int(top_mask(values, 0.5).sum()), 2)
        self.assertEqual(tuple(common_map(values, (2, 2), 3).shape), (1, 9))

    def test_exact_offset_sample_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            annotations = {}
            for index in range(3):
                name = f"sample_{index}.png"
                Image.new("RGB", (2, 2), color=(index, index, index)).save(
                    image_dir / name
                )
                annotations[name] = {"class_id": index}
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps(annotations), encoding="utf-8")

            names, pixels, labels = load_samples(
                image_dir, annotation_path, offset=1, limit=2
            )

        self.assertEqual(names, ["sample_1.png", "sample_2.png"])
        self.assertEqual(tuple(pixels.shape), (2, 3, 2, 2))
        self.assertTrue(torch.equal(labels, torch.tensor([1, 2])))

    def test_activation_capture_stays_logit_connected(self):
        model = TinyCaptureModel()
        logits, activation, source = capture_patch_score_activation(
            model, torch.randn(2, 3, 4, 4)
        )
        gradient = torch.autograd.grad(logits.sum(), activation)[0]
        self.assertEqual(source, "tiny activation output")
        self.assertEqual(tuple(activation.shape), (2, 2, 4, 4))
        self.assertEqual(gradient.shape, activation.shape)


if __name__ == "__main__":
    unittest.main()
