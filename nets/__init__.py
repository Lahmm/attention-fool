from __future__ import annotations

from utils import DEVICE

from .base import DEFAULT_PRETRAINED, ProgressiveAdapter
from .cait import build_cait_s24_model
from .pit import build_pit_b_224_model
from .visformer import build_visformer_small_model
from .vit import DEFAULT_MODEL_NAME, build_vit_model


WHITEBOX_MODEL_CHOICES = (
    "vit_base_patch16_224",
    "cait_s24_224",
    "pit_b_224",
    "visformer_small",
)

def build_whitebox_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=DEVICE,
) -> ProgressiveAdapter:
    if model_name == "vit_base_patch16_224":
        return build_vit_model(num_classes=num_classes, model_name=model_name, pretrained=pretrained, device=device)
    if model_name == "cait_s24_224":
        return build_cait_s24_model(num_classes=num_classes, model_name=model_name, pretrained=pretrained, device=device)
    if model_name == "pit_b_224":
        return build_pit_b_224_model(num_classes=num_classes, model_name=model_name, pretrained=pretrained, device=device)
    if model_name == "visformer_small":
        return build_visformer_small_model(num_classes=num_classes, model_name=model_name, pretrained=pretrained, device=device)
    raise ValueError(f"Unsupported whitebox model: {model_name!r}.")
