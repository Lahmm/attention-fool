import unittest

from gradient_replay import GradientReplay


class GradientReplayManifestTests(unittest.TestCase):
    def test_sample_id_digest_is_stable_and_order_sensitive(self):
        replay = GradientReplay(7)
        first = replay.manifest(["image_a.png", "image_b.png"])
        repeated = replay.manifest(["image_a.png", "image_b.png"])
        reversed_ids = replay.manifest(["image_b.png", "image_a.png"])
        self.assertEqual(first["sample_ids_sha256"], repeated["sample_ids_sha256"])
        self.assertEqual(len(first["sample_ids_sha256"]), 64)
        self.assertNotEqual(first["sample_ids_sha256"], reversed_ids["sample_ids_sha256"])


if __name__ == "__main__":
    unittest.main()
