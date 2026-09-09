import ast
import os
from pathlib import Path
import unittest

import torch

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from nets.cait import CaiTS24WithHook
from nets.pit import PiTB224WithHook
from nets.visformer import VisformerSmallWithHook
from nets.vit import ViTWithHook
from progressive_attack import ProgressivePatchScoreAttacker


class ProgressiveAdapterContractTests(unittest.TestCase):
    EXPECTED_DEFAULTS = {
        ViTWithHook: ("block3", "block7", "block11"),
        CaiTS24WithHook: ("block6_gap", "block17_gap", "block23_gap"),
        PiTB224WithHook: ("stage1_block3", "stage2_block5", "stage3_block3"),
        VisformerSmallWithHook: ("stage1_block4", "stage2_block2", "stage3_block3"),
    }

    def test_each_adapter_registers_three_ordered_defaults(self):
        for adapter, defaults in self.EXPECTED_DEFAULTS.items():
            with self.subTest(adapter=adapter.__name__):
                self.assertEqual(adapter._DEFAULT_PROGRESSIVE_LAYERS, defaults)
                candidates = adapter._PROGRESSIVE_LAYERS
                positions = [candidates.index(item) for item in defaults]
                self.assertEqual(positions, sorted(positions))

    def test_main_has_no_top_level_legacy_attack_import(self):
        source = (Path(__file__).resolve().parents[1] / "main.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        top_level_imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level_imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level_imports.append(node.module)
        self.assertNotIn("attack", top_level_imports)

    @unittest.skipUnless(
        os.environ.get("RUN_REAL_PROGRESSIVE_ADAPTER_SMOKE") == "1",
        "set RUN_REAL_PROGRESSIVE_ADAPTER_SMOKE=1 for four real timm adapters",
    )
    def test_real_models_resume_exactly_and_backpropagate(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for the complete adapter smoke matrix")
        device = torch.device("cuda")
        for model_name in WHITEBOX_MODEL_CHOICES:
            with self.subTest(model=model_name):
                model = build_whitebox_model(
                    num_classes=1000,
                    model_name=model_name,
                    pretrained=False,
                    device=device,
                )
                pixels = torch.rand(1, 3, 224, 224, device=device)
                with torch.no_grad():
                    native = model(pixels)
                    state = model.begin_progressive_forward(pixels)
                    resumed = model.finish_progressive_forward(state)
                self.assertTrue(torch.equal(native, resumed))
                attacker = ProgressivePatchScoreAttacker(
                    model,
                    steps=1,
                    input_diversity_groups=1,
                    input_diversity_phase_shift_set=((0, 0),),
                    use_momentum=False,
                    gaussian_alpha=0.0,
                    device=device,
                )
                probe = attacker.probe_attack_gradients(
                    pixels, torch.tensor([1], device=device)
                )
                self.assertEqual(probe["view_gradients"].size(0), 2)
                self.assertTrue(torch.isfinite(probe["processed"]).all())
                del probe, attacker, model, pixels, native, resumed, state
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
