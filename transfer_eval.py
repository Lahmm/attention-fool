import argparse
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / "data/huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / "data/huggingface/hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TORCH_HOME"] = str(ROOT / "data/torch")

import timm
import torch
import torchvision.models as torchvision_models
from torchvision import transforms
from PIL import Image
from timm.data import resolve_data_config, create_transform
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import DEVICE

DEFAULT_VIT_BLACK_BOX_MODELS = [
    "vit_base_patch16_224",
    "levit_256",
    "pit_b_224",
    "deit_base_patch16_224",
    "tnt_s_patch16_224",
    "convit_base",
    "visformer_small",
    "cait_s24_224",
]
DEFAULT_CNN_BLACK_BOX_MODELS = [
    "inception_v3",
    "inception_v4",
    "inception_resnet_v2",
    "resnet101",
    "inception_v3_adv",
    "inception_resnet_v2_adv",
]
DEFAULT_BLACK_BOX_MODELS = DEFAULT_VIT_BLACK_BOX_MODELS + DEFAULT_CNN_BLACK_BOX_MODELS
ASR_DEFINITION = "1 - adversarial accuracy"

# Table 2 and transferattack/utils.py in the official RYC-98/FPR release.
FPR_VIT_MODELS = [
    "vit_base_patch16_224", "cait_s24_224", "pit_b_224", "visformer_small",
    "swin_tiny_patch4_window7_224", "deit_tiny_patch16_224", "coat_tiny",
]
FPR_CNN_MODELS = [
    "resnet18", "vgg16", "densenet121", "efficientnet_b0",
    "mobilenet_v3_small", "resnext50_32x4d",
]
FPR_MODELS = FPR_VIT_MODELS + FPR_CNN_MODELS

HF_ADVERSARIAL_MODELS = {
    "inception_v3_adv": "inception_v3.tf_adv_in1k",
    "inception_resnet_v2_adv": "inception_resnet_v2.tf_ens_adv_in1k",
}

def adversarial_accuracy_to_asr(correct, total):
    if total < 0 or correct < 0 or correct > total:
        raise ValueError("counts must satisfy 0 <= correct <= total.")
    return 1 - correct / total if total else 0.0


def build_black_box_model(model_name, protocol="project"):
    targets = FPR_MODELS if protocol == "fpr" else DEFAULT_BLACK_BOX_MODELS
    if model_name not in targets:
        raise ValueError(f"unsupported target for {protocol}: {model_name}.")
    if protocol == "fpr" and model_name in FPR_CNN_MODELS:
        model = torchvision_models.get_model(model_name, weights="DEFAULT")
    else:
        model = timm.create_model(HF_ADVERSARIAL_MODELS.get(model_name, model_name), pretrained=True)
    model.to(DEVICE).eval()
    if protocol == "fpr":
        config = getattr(model, "default_cfg", {})
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(config.get("mean", (0.485, 0.456, 0.406)),
                                 config.get("std", (0.229, 0.224, 0.225))),
        ])
    else:
        transform = create_transform(**resolve_data_config({}, model=model))
    return model, transform


def build_transfer_samples(image_dir, annotations, prefix="adv_"):
    labels = {Path(name).stem: int(value["class_id"]) for name, value in annotations.items()}
    samples = []
    for path in sorted(Path(image_dir).glob(f"{prefix}*.png")):
        sample_id = path.stem[len(prefix):]
        if sample_id not in labels:
            raise ValueError(f"no label for adversarial image: {path.name}.")
        samples.append((path, labels[sample_id]))
    if not samples:
        raise ValueError("no adversarial PNG images found.")
    return samples


class TransferImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def evaluate(samples, model_name, *, protocol="project", batch_size=32, num_workers=0, use_amp=False):
    model, transform = build_black_box_model(model_name, protocol)
    loader = DataLoader(TransferImageDataset(samples, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=DEVICE.type == "cuda")
    predictions = []
    with torch.inference_mode():
        for images, _ in tqdm(loader, desc=model_name):
            context = torch.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" and use_amp else nullcontext()
            with context:
                logits = model(images.to(DEVICE))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    correct = sum(pred == label for pred, (_, label) in zip(predictions, samples))
    return {
        "target": model_name, "correct": correct, "total": len(samples),
        "acc": correct / len(samples),
        "asr": adversarial_accuracy_to_asr(correct, len(samples)),
    }, predictions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--annotations-path", type=Path, default=ROOT / "data/image_name_to_class_id_and_name.json")
    parser.add_argument("--protocol", choices=("project", "fpr"), default="project")
    parser.add_argument("--model-name", help="Comma-separated subset of the protocol targets.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--output", type=Path, help="CSV path; defaults to image-dir/transfer.csv.")
    parser.add_argument("--predictions-output", type=Path)
    return parser.parse_args()


def main(args):
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    annotations = json.loads(args.annotations_path.read_text())
    samples = build_transfer_samples(args.image_dir, annotations)
    targets = FPR_MODELS if args.protocol == "fpr" else DEFAULT_BLACK_BOX_MODELS
    if args.model_name:
        requested = [item.strip() for item in args.model_name.split(",") if item.strip()]
        if not requested or any(item not in targets for item in requested):
            raise ValueError("requested models must belong to the chosen target protocol.")
        targets = requested
    rows, predictions = [], {}
    for target in targets:
        metrics, prediction = evaluate(samples, target, protocol=args.protocol,
                                      batch_size=args.batch_size, num_workers=args.num_workers, use_amp=args.amp)
        rows.append(metrics)
        predictions[target] = prediction
        print(metrics)
    params_path = args.image_dir / "attack_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {}
    source = params.get("whitebox_model")
    transformer_targets = FPR_VIT_MODELS if args.protocol == "fpr" else DEFAULT_VIT_BLACK_BOX_MODELS
    def average(selected):
        values = [row["asr"] for row in rows if selected(row)]
        return sum(values) / len(values) if values else None
    summary = {
        "overall_asr": average(lambda row: True),
        "strict_black_box_asr": average(lambda row: row["target"] != source) if source else None,
        "transformer_asr": average(lambda row: row["target"] in transformer_targets),
        "cnn_asr": average(lambda row: row["target"] not in transformer_targets),
    }
    output = args.output or args.image_dir / "transfer.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps({
        "protocol": args.protocol, "asr_definition": ASR_DEFINITION,
        "attack_params": params, "summary": summary,
    }, indent=2))
    if args.predictions_output:
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output.write_text(json.dumps({
            "protocol": args.protocol, "asr_definition": ASR_DEFINITION,
            "sample_ids": [path.name[len("adv_"):] for path, _ in samples],
            "labels": [label for _, label in samples], "predictions": predictions,
        }))
    print(summary)


if __name__ == "__main__":
    main(parse_args())
