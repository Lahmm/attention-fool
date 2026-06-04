import unittest

import numpy as np
import torch

from cross_vit_components import _component_measurements, _select_candidates
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


if __name__ == "__main__":
    unittest.main()
