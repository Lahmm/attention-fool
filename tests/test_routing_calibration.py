import json
from pathlib import Path
import tempfile
import unittest

from experiments.patch_score_routing_calibration import (
    POLARITIES,
    build_manifest,
    select_config,
    template_rows,
)
from nets import PATCH_SCORE_LAYER_CANDIDATES, WHITEBOX_MODEL_CHOICES
from routing_config import FrozenRoutingConfig, file_sha256


class RoutingCalibrationTests(unittest.TestCase):
    def synthetic_results(self):
        results = {}
        for row in template_rows():
            source = str(row["source_model"])
            target = str(row["target_model"])
            polarity = str(row["polarity"])
            layer = str(row["layer"])
            layer_index = PATCH_SCORE_LAYER_CANDIDATES[source].index(layer)
            # High wins globally, and every source has a deterministic but
            # architecture-specific best layer.
            preferred = list(WHITEBOX_MODEL_CHOICES).index(source) % len(
                PATCH_SCORE_LAYER_CANDIDATES[source]
            )
            value = 0.4 + (0.1 if polarity == "high" else 0.0)
            value += 0.01 if layer_index == preferred else 0.0
            results[(source, target, polarity, layer)] = value
        return results

    def test_global_polarity_and_per_model_layers_are_frozen(self):
        config, summary = select_config(
            self.synthetic_results(),
            samples=128,
            sample_seed=20260717,
            attack_seed=20260716,
            image_ids_sha256="abc",
            results_path=Path("calibration.csv"),
        )

        self.assertEqual(config.global_polarity, "high")
        for index, source in enumerate(WHITEBOX_MODEL_CHOICES):
            expected = PATCH_SCORE_LAYER_CANDIDATES[source][
                index % len(PATCH_SCORE_LAYER_CANDIDATES[source])
            ]
            self.assertEqual(config.model_layers[source], expected)
        self.assertEqual(summary["selected"], config.to_dict())

    def test_exact_ties_choose_high_and_the_deepest_layer(self):
        results = {
            (
                str(row["source_model"]),
                str(row["target_model"]),
                str(row["polarity"]),
                str(row["layer"]),
            ): 0.5
            for row in template_rows()
        }
        config, _ = select_config(
            results,
            samples=128,
            sample_seed=1,
            attack_seed=2,
            image_ids_sha256="tie",
            results_path=Path("tie.csv"),
        )
        self.assertEqual(config.global_polarity, "high")
        for source in WHITEBOX_MODEL_CHOICES:
            self.assertEqual(config.model_layers[source], PATCH_SCORE_LAYER_CANDIDATES[source][-1])

    def test_config_round_trip_and_digest(self):
        config, _ = select_config(
            self.synthetic_results(),
            samples=128,
            sample_seed=1,
            attack_seed=2,
            image_ids_sha256="roundtrip",
            results_path=Path("results.csv"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            config.save(path)
            self.assertEqual(FrozenRoutingConfig.load(path), config)
            self.assertEqual(len(file_sha256(path)), 64)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["model_layers"]["vit_base_patch16_224"] = "unknown"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid frozen layer"):
                FrozenRoutingConfig.load(path)

    def test_template_is_complete(self):
        expected = sum(
            len(PATCH_SCORE_LAYER_CANDIDATES[source]) * len(POLARITIES) * 3
            for source in WHITEBOX_MODEL_CHOICES
        )
        self.assertEqual(len(template_rows()), expected)
        manifest = build_manifest(samples=128, sample_offset=500, attack_seed=20260716)
        self.assertEqual(len(manifest["jobs"]), expected // 3)
        first_command = manifest["jobs"][0]["attack_command"]
        self.assertIn("--sample-offset", first_command)
        self.assertEqual(
            first_command[first_command.index("--sample-offset") + 1],
            "500",
        )


if __name__ == "__main__":
    unittest.main()
