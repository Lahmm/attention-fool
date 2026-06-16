import unittest

from record_experiment import architecture_avg_asr, format_avg


class RecordExperimentArchitectureAvgTests(unittest.TestCase):
    def test_architecture_avg_asr_splits_default_vit_and_cnn_models(self):
        avgs = architecture_avg_asr(
            {
                "levit_256": 0.8,
                "deit_base_patch16_224": 0.6,
                "inception_v3": 0.4,
                "resnet101": 0.2,
            }
        )

        self.assertAlmostEqual(avgs["avg_vit"], 0.7)
        self.assertAlmostEqual(avgs["avg_cnn"], 0.3)

    def test_architecture_avg_asr_returns_none_for_missing_family(self):
        avgs = architecture_avg_asr({"deit_base_patch16_224": 0.5})

        self.assertAlmostEqual(avgs["avg_vit"], 0.5)
        self.assertIsNone(avgs["avg_cnn"])
        self.assertEqual(format_avg(avgs["avg_cnn"]), "n/a")


if __name__ == "__main__":
    unittest.main()
