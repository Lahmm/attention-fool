import unittest

import torch

from nets.base import PatchScoreFeatures
from nets.cait import CaiTS24WithHook
from nets.pit import PiTB224WithHook
from nets.visformer import VisformerSmallWithHook
from nets.vit import ViTWithHook


class PatchScoreLayerContractTests(unittest.TestCase):
    def test_pre_registered_layer_candidates_are_stable(self):
        expected = {
            ViTWithHook: ("block3", "block6", "block9", "block12"),
            CaiTS24WithHook: (
                "block6_gap",
                "block12_gap",
                "block18_gap",
                "block24_gap",
                "block24_class",
            ),
            PiTB224WithHook: (
                "stage1_block3",
                "stage2_block3",
                "stage2_block6",
                "stage3_block2",
                "stage3_block4",
            ),
            VisformerSmallWithHook: (
                "stage1_block4",
                "stage1_block7",
                "stage2_block4",
                "stage3_block2",
                "stage3_block4",
            ),
        }
        for model_type, candidates in expected.items():
            with self.subTest(model=model_type.__name__):
                instance = object.__new__(model_type)
                self.assertEqual(instance.patch_score_layer_candidates(), candidates)
                self.assertNotIn("final", candidates)

    def test_feature_metadata_is_validated(self):
        features = PatchScoreFeatures(
            local_tokens=torch.zeros(2, 4, 3),
            global_token=torch.zeros(2, 1, 3),
            grid_size=(2, 2),
            source_name="stage1[0]+gap",
            layer_id="stage1_block1",
            global_mode="gap",
        )
        features.validate()

        features.global_mode = "unknown"
        with self.assertRaisesRegex(ValueError, "global_mode"):
            features.validate()

if __name__ == "__main__":
    unittest.main()
