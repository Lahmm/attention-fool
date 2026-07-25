import unittest

import torch

from experiments.patch_score_routing_calibration_eval import paired_asr


class RoutingCalibrationEvalTests(unittest.TestCase):
    def test_asr_uses_only_target_clean_correct_samples(self):
        labels = torch.tensor([0, 1, 2, 3])
        clean = torch.tensor([0, 8, 2, 3])
        adversarial = torch.tensor([9, 9, 2, 8])
        asr, successes, denominator = paired_asr(clean, adversarial, labels)
        self.assertEqual(denominator, 3)
        self.assertEqual(successes, 2)
        self.assertAlmostEqual(asr, 2 / 3)

    def test_zero_clean_correct_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no clean-correct"):
            paired_asr(torch.tensor([1]), torch.tensor([1]), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
