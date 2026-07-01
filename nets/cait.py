from __future__ import annotations

from .base import ClsTokenMixin, DEFAULT_PRETRAINED, WhiteBoxWithHook, sequential_modules


DEFAULT_MODEL_NAME = "cait_s24_224"


class CaiTS24WithHook(ClsTokenMixin, WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    def _feature_modules(self):
        return sequential_modules(getattr(self.model, "blocks", None))


def build_cait_s24_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> CaiTS24WithHook:
    return CaiTS24WithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
