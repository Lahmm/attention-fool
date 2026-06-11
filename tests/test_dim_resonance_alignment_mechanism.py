import unittest

from dim_resonance_alignment_mechanism import aggregate_rows, build_conclusion


class DimResonanceAlignmentMechanismTest(unittest.TestCase):
    def test_aggregate_rows_groups_numeric_metrics(self):
        rows = [
            {"scope": "source_dim_alignment", "variant": "dim_resonance_djf", "model": "source", "band_group": "low_mid", "dim_projection_gain": 1.2},
            {"scope": "source_dim_alignment", "variant": "dim_resonance_djf", "model": "source", "band_group": "low_mid", "dim_projection_gain": 1.4},
        ]
        summary = aggregate_rows(rows)
        item = summary["source_dim_alignment/dim_resonance_djf/low_mid"]
        self.assertEqual(item["n"], 2)
        self.assertAlmostEqual(item["dim_projection_gain"], 1.3)

    def test_conclusion_reports_dim_resonance_supported_when_best_on_both_axes(self):
        summary = {}
        for variant in ("reference_djf", "dim_resonance_only", "dim_resonance_djf", "fft_lowboost_djf"):
            for group in ("low_mid", "high"):
                summary[f"source_dim_alignment/{variant}/{group}"] = {
                    "dim_projection_gain": 1.0,
                    "dim_norm_gain": 1.0,
                    "dim_cos": 0.9,
                    "orthogonal_energy_over_dim": 0.1,
                    "dim_sign_agreement": 0.8,
                }
                summary[f"target_alignment/{variant}/{group}"] = {
                    "target_cos_delta_vs_dim": 0.0,
                    "target_dot_delta_vs_dim": 0.0,
                    "target_cos": 0.1,
                    "dim_target_cos": 0.1,
                    "positive_target_dot_delta_fraction": 0.5,
                }
                if variant != "reference_djf":
                    summary[f"increment_vs_reference/{variant}/{group}"] = {
                        "increment_dim_projection": 0.0,
                        "increment_dim_cos": 0.0,
                        "increment_norm_over_dim": 0.0,
                    }
                    summary[f"increment_target_alignment/{variant}/{group}"] = {
                        "increment_target_cos": 0.0,
                        "increment_target_dot": 0.0,
                        "positive_increment_target_dot_fraction": 0.5,
                        "increment_dim_cos": 0.0,
                    }
        summary["source_dim_alignment/dim_resonance_djf/low_mid"]["dim_projection_gain"] = 1.5
        summary["target_alignment/dim_resonance_djf/low_mid"]["target_cos_delta_vs_dim"] = 0.2
        summary["increment_vs_reference/dim_resonance_djf/low_mid"]["increment_dim_projection"] = 0.3
        summary["increment_target_alignment/dim_resonance_djf/low_mid"]["increment_target_dot"] = 0.4
        text = build_conclusion(summary, {"dim_resonance_djf": 0.83})
        self.assertIn("同时最大化低/中频 DIM 同向投影增益", text)
        self.assertIn("avg ASR=0.830000", text)


if __name__ == "__main__":
    unittest.main()
