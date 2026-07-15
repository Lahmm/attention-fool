import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import transfer_eval


class HuggingFaceTransferModelTests(unittest.TestCase):
    def test_default_suite_is_thirteen_real_huggingface_models(self):
        self.assertEqual(len(transfer_eval.DEFAULT_BLACK_BOX_MODELS), 13)
        self.assertIn("inception_v3_adv", transfer_eval.DEFAULT_BLACK_BOX_MODELS)
        self.assertIn("inception_resnet_v2_adv", transfer_eval.DEFAULT_BLACK_BOX_MODELS)
        self.assertNotIn("inception_v3_adv_3", transfer_eval.DEFAULT_BLACK_BOX_MODELS)
        self.assertNotIn("inception_v3_adv_4", transfer_eval.DEFAULT_BLACK_BOX_MODELS)

    def test_adversarial_models_use_exact_huggingface_timm_variants(self):
        expected = {
            "inception_v3_adv": "inception_v3.tf_adv_in1k",
            "inception_resnet_v2_adv": "inception_resnet_v2.tf_ens_adv_in1k",
        }
        for requested, timm_name in expected.items():
            model = MagicMock()
            with (
                self.subTest(model=requested),
                patch("transfer_eval.timm.create_model", return_value=model) as create,
                patch("transfer_eval.resolve_data_config", return_value={}),
                patch("transfer_eval.create_transform", return_value=object()),
            ):
                built, _ = transfer_eval.build_black_box_model(requested)
                self.assertIs(built, model)
                create.assert_called_once_with(timm_name, pretrained=True)
                model.to.assert_called_once_with(transfer_eval.DEVICE)
                model.eval.assert_called_once_with()

    def test_recording_reads_attack_params_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack_params.json"
            payload = '{"seed": 20260715}'
            path.write_text(payload, encoding="utf-8")

            self.assertEqual(
                transfer_eval.read_attack_params_for_recording(path), payload
            )
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
