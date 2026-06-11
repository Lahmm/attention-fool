import unittest
from pathlib import Path

from experiments.transfer_eval import build_transfer_samples, extract_original_name


class TransferEvalAnnotationTests(unittest.TestCase):
    def test_extract_original_name_strips_prefix_only(self):
        self.assertEqual(
            extract_original_name("adv_ILSVRC2012_val_00000031.png", "adv_"),
            "ILSVRC2012_val_00000031.png",
        )

    def test_build_transfer_samples_matches_annotation_by_stem(self):
        paths = [Path("adv_ILSVRC2012_val_00000031.png")]
        annotations = {"ILSVRC2012_val_00000031.JPEG": {"class_id": 7, "class_name": "x"}}
        samples, skipped = build_transfer_samples(paths, annotations, prefix="adv_")
        self.assertEqual(skipped, 0)
        self.assertEqual(samples, [(paths[0], 7)])

    def test_build_transfer_samples_counts_unmatched_files(self):
        samples, skipped = build_transfer_samples(
            [Path("adv_missing.png")],
            {"ILSVRC2012_val_00000031.JPEG": {"class_id": 7}},
            prefix="adv_",
        )
        self.assertEqual(samples, [])
        self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
