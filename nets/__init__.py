from utils import DEVICE

from .cait import CaiTAdapter
from .pit import PiTAdapter
from .vit import DEFAULT_MODEL_NAME, ViTAdapter

SOURCE_ADAPTERS = {
    "vit_base_patch16_224": ViTAdapter,
    "cait_s24_224": CaiTAdapter,
    "pit_ti_224": PiTAdapter,
    "deit_base_distilled_patch16_224": ViTAdapter,
    "pit_b_224": PiTAdapter,
}
WHITEBOX_MODEL_CHOICES = tuple(SOURCE_ADAPTERS)


def build_whitebox_model(model_name=DEFAULT_MODEL_NAME, *, pretrained=True, device=DEVICE):
    if model_name not in SOURCE_ADAPTERS:
        raise ValueError(f"unsupported source model: {model_name!r}.")
    return SOURCE_ADAPTERS[model_name](model_name, pretrained=pretrained, device=device)
