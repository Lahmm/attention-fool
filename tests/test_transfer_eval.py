import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import transfer_eval


class HuggingFaceTransferModelTests(unittest.TestCase):
    def test_fpr_suite_matches_official_table_two(self):
        self.assertEqual(transfer_eval.FPR_MODELS, [
            "vit_base_patch16_224", "cait_s24_224", "pit_b_224", "visformer_small",
            "swin_tiny_patch4_window7_224", "deit_tiny_patch16_224", "coat_tiny",
            "resnet18", "vgg16", "densenet121", "efficientnet_b0",
            "mobilenet_v3_small", "resnext50_32x4d",
        ])

    def test_fpr_cnn_uses_torchvision_default_and_keeps_full_image(self):
        from PIL import Image
        import torch
        model = MagicMock(spec=torch.nn.Module)
        with patch("transfer_eval.torchvision_models.get_model", return_value=model) as create:
            _, transform = transfer_eval.build_black_box_model("resnet18", protocol="fpr")
        create.assert_called_once_with("resnet18", weights="DEFAULT")
        image = Image.new("RGB", (224, 224), (255, 0, 0))
        tensor = transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertAlmostEqual(tensor[0, 0, 0].item(), (1 - .485) / .229, places=5)

    def test_asr_is_one_minus_adversarial_accuracy(self):
        self.assertEqual(transfer_eval.ASR_DEFINITION, "1 - adversarial accuracy")
        self.assertAlmostEqual(
            transfer_eval.adversarial_accuracy_to_asr(correct=23, total=100),
            0.77,
        )
        self.assertEqual(
            transfer_eval.adversarial_accuracy_to_asr(correct=0, total=0),
            0.0,
        )

    def test_asr_rejects_invalid_adversarial_counts(self):
        with self.assertRaises(ValueError):
            transfer_eval.adversarial_accuracy_to_asr(correct=11, total=10)

    def test_default_suite_is_fourteen_real_huggingface_models(self):
        self.assertEqual(len(transfer_eval.DEFAULT_BLACK_BOX_MODELS), 14)
        self.assertIn("vit_base_patch16_224", transfer_eval.DEFAULT_BLACK_BOX_MODELS)
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
