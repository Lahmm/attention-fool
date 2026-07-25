from pathlib import Path
import tempfile
import unittest

from experiments.patch_score_selector_suite import CONDITIONS, build_manifest, summarize
from nets import PATCH_SCORE_LAYER_CANDIDATES, WHITEBOX_MODEL_CHOICES
from routing_config import FrozenRoutingConfig
from transfer_eval import DEFAULT_BLACK_BOX_MODELS


class SelectorSuiteTests(unittest.TestCase):
    def make_config(self):
        return FrozenRoutingConfig(
            global_polarity="low",
            model_layers={
                model: PATCH_SCORE_LAYER_CANDIDATES[model][-1]
                for model in WHITEBOX_MODEL_CHOICES
            },
            calibration={"samples": 128},
        )

    def test_manifest_has_all_sources_and_conditions(self):
        config = self.make_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            config.save(path)
            manifest = build_manifest(
                config,
                path,
                samples=500,
                sample_offset=0,
                seed=20260716,
            )
        self.assertEqual(len(manifest["jobs"]), 4 * len(CONDITIONS))
        keys = {(job["source_model"], job["condition"]) for job in manifest["jobs"]}
        self.assertEqual(
            keys,
            {(source, condition) for source in WHITEBOX_MODEL_CHOICES for condition in CONDITIONS},
        )
        opposite = next(job for job in manifest["jobs"] if job["condition"] == "opposite")
        self.assertEqual(opposite["polarity"], "high")
        gradcam = next(job for job in manifest["jobs"] if job["condition"] == "gradcam_relu")
        self.assertEqual(gradcam["selector"], "gradcam_relu")
        self.assertIn("--gradcam-target-mode", gradcam["attack_command"])

    def test_summary_uses_only_clean_correct_and_is_paired(self):
        rows = []
        source = WHITEBOX_MODEL_CHOICES[0]
        for target in DEFAULT_BLACK_BOX_MODELS:
            for image_index in range(4):
                for condition in CONDITIONS:
                    rows.append(
                        {
                            "source_model": source,
                            "target_model": target,
                            "condition": condition,
                            "image_name": f"image_{image_index}",
                            "clean_correct": image_index < 3,
                            "adv_correct": condition != "selected",
                        }
                    )
        summary = summarize(rows, bootstrap_repeats=100, seed=7)
        first_target = DEFAULT_BLACK_BOX_MODELS[0]
        self.assertEqual(summary["per_target"][source][first_target]["selected"]["asr"], 1.0)
        self.assertEqual(
            summary["per_target"][source][first_target]["selected"]["clean_correct_count"],
            3,
        )
        comparison = summary["patch_score_vs_gradcam"][source]
        self.assertEqual(comparison["difference"], 1.0)
        self.assertTrue(comparison["noninferior_at_1pp"])


if __name__ == "__main__":
    unittest.main()
