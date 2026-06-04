import unittest

import numpy as np
import torch

from cross_vit_components import _component_measurements, _effect_tensor, _select_candidates, _summarize_screening
from gradient_analysis import parse_component, screening_component_specs


class ComponentScreeningTests(unittest.TestCase):
    def test_streaming_measurements_match_explicit_projections(self):
        torch.manual_seed(3)
        source = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        target = torch.randn(2, 3, 32, 32, dtype=torch.float64)
        derivative, _normalized, energy = _component_measurements(source, target)
        specs = screening_component_specs()
        for index in (0, 17, 24, 24 + 16 * 11 + 7, len(specs) - 1):
            component = parse_component(specs[index])(source)
            expected = (component * target).flatten(1).sum(1).numpy()
            np.testing.assert_allclose(derivative[:, index], expected, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(energy[:, :24].sum(1), np.ones(2), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(energy[:, 24:].sum(1), np.ones(2), rtol=1e-10, atol=1e-10)

    def test_candidate_selection_prefers_one_fft_and_two_nonoverlapping_local_regions(self):
        def row(component, score):
            return {"component": component, "family": component.split(":")[0], "eligible": True,
                    "mean_normalized_derivative": score}
        rows = [
            row("haar:LLL:0:0", 9), row("haar:LLH:0:0", 8), row("haar:LHL:1:1", 7),
            row("fft:2:horizontal", 6), row("fft:3:vertical", 5), row("haar:HHH:2:2", 4),
        ]
        selected = _select_candidates(rows)
        self.assertEqual([item["component"] for item in selected], [
            "fft:2:horizontal", "haar:LLL:0:0", "haar:LHL:1:1",
        ])

    def test_screening_and_effect_tensor_use_runtime_seed_dimension(self):
        specs = ["fft:0:horizontal"]
        derivative = np.ones((1, 2, 3, 8)); normalized = derivative.copy(); energy = np.ones((1, 2, 3))
        rows = _summarize_screening(specs, derivative, normalized, energy, repeats=20, seed=0)
        self.assertTrue(rows[0]["eligible"]); self.assertEqual(rows[0]["positive_seeds"], 2)
        item = {"clean_correct": torch.tensor([True, True, False]),
                "full": {"4": torch.tensor([True, True, False]), "9": torch.tensor([True, False, False])},
                "candidates": {specs[0]: {"drop": {"4": torch.tensor([False, True, False]), "9": torch.tensor([False, False, False])}}}}
        values = _effect_tensor({"models": {"m": item}}, specs[0], "drop", (4, 9), 3)
        self.assertEqual(values.shape, (2, 3, 1)); self.assertTrue(np.isnan(values[:, 2]).all())


if __name__ == "__main__":
    unittest.main()
