import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
from PIL import Image
from torchvision import transforms

import transfer_eval as evaluation


class TransferTests(unittest.TestCase):
    def test_asr_uses_all_adversarial_samples(self):
        self.assertAlmostEqual(evaluation.adversarial_accuracy_to_asr(23, 100), 0.77)
        self.assertEqual(evaluation.adversarial_accuracy_to_asr(0, 0), 0)
        with self.assertRaises(ValueError):
            evaluation.adversarial_accuracy_to_asr(11, 10)


    def test_fpr_targets_and_transformer_weights(self):
        self.assertEqual(len(evaluation.FPR_MODELS), 13)
        self.assertEqual(len(set(evaluation.FPR_MODELS)), 13)
        model = MagicMock()
        model.to.return_value = model
        model.default_cfg = {'mean': (.5, .5, .5), 'std': (.5, .5, .5)}
        with patch.object(evaluation.timm, 'create_model', return_value=model) as create:
            _, transform = evaluation.build_black_box_model('vit_base_patch16_224')
        create.assert_called_once_with('vit_base_patch16_224', pretrained=True)
        red = transform(Image.new('RGB', (300, 200), (255, 0, 0)))
        self.assertEqual(tuple(red.shape), (3, 224, 224))
        torch.testing.assert_close(red[:, 0, 0], torch.tensor([1., -1., -1.]))
        with self.assertRaises(ValueError):
            evaluation.build_black_box_model('resnet101')

    def test_fpr_cnn_preserves_full_image_and_normalization(self):
        model = MagicMock(spec=torch.nn.Module)
        model.to.return_value = model
        with patch.object(evaluation.torchvision_models, 'get_model', return_value=model) as create:
            _, transform = evaluation.build_black_box_model('resnet18')
        create.assert_called_once_with('resnet18', weights='DEFAULT')
        result = transform(Image.new('RGB', (300, 200), (255, 0, 0)))
        self.assertEqual(tuple(result.shape), (3, 224, 224))
        self.assertAlmostEqual(float(result[0, 0, 0]), (1 - .485) / .229, places=5)

    def test_evaluation_and_csv_use_the_same_ordered_sample_set(self):
        class ConstantModel(torch.nn.Module):
            def forward(self, images):
                return torch.tensor([[1., 0.]]).repeat(images.size(0), 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('a', 'b'):
                Image.new('RGB', (8, 8)).save(root / f'adv_{name}.png')
            annotations = {'a.jpg': {'class_id': 0}, 'b.jpg': {'class_id': 1}}
            path = root / 'labels.json'
            path.write_text(json.dumps(annotations))
            (root / 'attack_params.json').write_text(json.dumps({'whitebox_model': 'vit_base_patch16_224'}))
            samples = evaluation.build_transfer_samples(root, annotations)
            with patch.object(evaluation, 'DEVICE', torch.device('cpu')), \
                 patch.object(evaluation, 'build_black_box_model', return_value=(ConstantModel(), transforms.ToTensor())):
                metrics, predictions = evaluation.evaluate(samples, 'vit_base_patch16_224')
                self.assertEqual(predictions, [0, 0])
                self.assertEqual(metrics['total'], 2)
                self.assertEqual(metrics['asr'], 0.5)
                args = type('Args', (), dict(image_dir=root, annotations_path=path, protocol='fpr',
                    model_name='vit_base_patch16_224,resnet18', batch_size=2, num_workers=0,
                    amp=False, no_tf32=True, output=None, predictions_output=root / 'predictions.json'))()
                evaluation.main(args)
            with (root / 'transfer.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual([int(row['total']) for row in rows], [2, 2])
            summary = json.loads((root / 'transfer.json').read_text())['summary']
            self.assertEqual(summary['strict_black_box_asr'], 0.5)
            output = json.loads((root / 'predictions.json').read_text())
            self.assertEqual(output['labels'], [0, 1])
            self.assertEqual(output['sample_ids'], ['a.png', 'b.png'])
            with self.assertRaises(ValueError):
                evaluation.build_transfer_samples(root, {'a.jpg': annotations['a.jpg']})
