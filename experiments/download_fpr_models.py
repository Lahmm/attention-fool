"""Cache the seven timm and six torchvision models in FPR Table 2."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "data/huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / "data/huggingface/hub")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TORCH_HOME"] = str(ROOT / "data/torch")

import timm
import torch
import torchvision.models as models
from huggingface_hub import snapshot_download
from timm.models import get_pretrained_cfg

VITS = ["vit_base_patch16_224", "cait_s24_224", "pit_b_224", "visformer_small",
        "swin_tiny_patch4_window7_224", "deit_tiny_patch16_224", "coat_tiny"]
CNNS = ["resnet18", "vgg16", "densenet121", "efficientnet_b0",
        "mobilenet_v3_small", "resnext50_32x4d"]

def download(name):
    if name in VITS:
        repo = get_pretrained_cfg(name).hf_hub_id
        snapshot_download(repo, allow_patterns=["config.json", "pytorch_model.bin"], max_workers=2)
    else:
        weights = models.get_model_weights(name).DEFAULT
        torch.hub.load_state_dict_from_url(weights.url, progress=False, check_hash=True)
    print(f"Cached {name}", flush=True)

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, VITS + CNNS))
