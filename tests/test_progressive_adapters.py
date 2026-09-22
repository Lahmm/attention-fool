import os
import unittest

import torch

from nets import WHITEBOX_MODEL_CHOICES, build_whitebox_model
from nets.cait import CaiTS24Adapter
from nets.pit import PiTB224Adapter
from nets.visformer import VisformerSmallAdapter
from nets.vit import ViTAdapter
from progressive_attack import ProgressiveRouteDisruptionAttacker


class ProgressiveAdapterContractTests(unittest.TestCase):
    EXPECTED_DEFAULTS = {
        ViTAdapter: ("block3", "block10"),
        CaiTS24Adapter: ("block17", "block23"),
        PiTB224Adapter: ("stage2_block1", "stage3_block2", "stage3_block3"),
        VisformerSmallAdapter: ("stage2_block1", "stage3_block1"),
    }
    EXPECTED_DEFAULT_COUNTS = {
        ViTAdapter: ((196, 196), (10, 10)),
        CaiTS24Adapter: ((196, 196), (2, 28)),
        PiTB224Adapter: ((256, 64, 64), (5, 2, 6)),
        VisformerSmallAdapter: ((196, 49), (41, 10)),
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
            ViTAdapter: (0.051020408163, 0.051020408163),
            CaiTS24Adapter: (0.010204081633, 0.142857142857),
            PiTB224Adapter: (0.02081165, 0.03125, 0.09375),
            VisformerSmallAdapter: (0.209183673469, 0.204081632653),
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

    def test_architecture_specific_opponent_defaults(self):
        expected = {
            ViTAdapter: 0.2,
            CaiTS24Adapter: 0.2,
            PiTB224Adapter: 0.4,
            VisformerSmallAdapter: 0.4,
        }
        for adapter, default in expected.items():
            with self.subTest(adapter=adapter.__name__):
                instance = object.__new__(adapter)
                self.assertEqual(
                    instance.default_progressive_opponent_noise_strength(), default
                )

    @unittest.skipUnless(
        os.environ.get("RUN_REAL_PROGRESSIVE_ADAPTER_SMOKE") == "1",
        "set RUN_REAL_PROGRESSIVE_ADAPTER_SMOKE=1 for four real timm adapters",
    )
    def test_real_models_progressive_forward_and_backpropagate(self):
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
                attacker = ProgressiveRouteDisruptionAttacker(
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
