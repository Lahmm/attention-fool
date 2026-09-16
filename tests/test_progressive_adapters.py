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
        ViTWithHook: ("block3", "block10"),
        CaiTS24WithHook: ("block17_gap", "block23_gap"),
        PiTB224WithHook: ("stage2_block1", "stage3_block2", "stage3_block3"),
        VisformerSmallWithHook: ("stage2_block1", "stage3_block1"),
    }
    EXPECTED_DEFAULT_COUNTS = {
        ViTWithHook: ((196, 196), (10, 10)),
        CaiTS24WithHook: ((196, 196), (2, 28)),
        PiTB224WithHook: ((256, 64, 64), (5, 2, 6)),
        VisformerSmallWithHook: ((196, 49), (41, 10)),
    }

    def test_each_adapter_registers_ordered_defaults(self):
        for adapter, defaults in self.EXPECTED_DEFAULTS.items():
            with self.subTest(adapter=adapter.__name__):
                self.assertEqual(adapter._DEFAULT_PROGRESSIVE_LAYERS, defaults)
                candidates = adapter._PROGRESSIVE_LAYERS
                positions = [candidates.index(item) for item in defaults]
                self.assertEqual(positions, sorted(positions))
                self.assertGreater(len(defaults), 1)

    def test_architecture_specific_default_drop_ratios(self):
        expected_ratios = {
            ViTWithHook: (0.051020408163, 0.051020408163),
            CaiTS24WithHook: (0.010204081633, 0.142857142857),
            PiTB224WithHook: (0.02081165, 0.03125, 0.09375),
            VisformerSmallWithHook: (0.209183673469, 0.204081632653),
        }
        for adapter in self.EXPECTED_DEFAULTS:
            with self.subTest(adapter=adapter.__name__):
                instance = object.__new__(adapter)
                self.assertEqual(
                    instance.default_progressive_drop_ratios(), expected_ratios[adapter]
                )

    def test_architecture_specific_defaults_resolve_expected_drop_counts(self):
        for adapter, (token_counts, expected_counts) in self.EXPECTED_DEFAULT_COUNTS.items():
            with self.subTest(adapter=adapter.__name__):
                instance = object.__new__(adapter)
                ratios = instance.default_progressive_drop_ratios()
                self.assertEqual(len(ratios), len(token_counts))
                self.assertEqual(
                    tuple(round(tokens * ratio) for tokens, ratio in zip(token_counts, ratios)),
                    expected_counts,
                )

    def test_architecture_specific_score_and_opponent_defaults(self):
        expected = {
            ViTWithHook: ("cosine", 0.2),
            CaiTS24WithHook: ("gap_projection", 0.2),
            PiTB224WithHook: ("cosine", 0.4),
            VisformerSmallWithHook: ("gap_projection", 0.4),
        }
        for adapter, defaults in expected.items():
            with self.subTest(adapter=adapter.__name__):
                instance = object.__new__(adapter)
                self.assertEqual(instance.default_progressive_score_mode(), defaults[0])
                self.assertEqual(
                    instance.default_progressive_opponent_noise_strength(), defaults[1]
                )

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
