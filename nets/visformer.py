from __future__ import annotations

from .base import DEFAULT_PRETRAINED, WhiteBoxWithHook, sequential_modules


DEFAULT_MODEL_NAME = "visformer_small"


class VisformerSmallWithHook(WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    def _feature_modules(self):
        modules = []
        modules.extend(sequential_modules(getattr(self.model, "stage1", None)))
        modules.extend(sequential_modules(getattr(self.model, "stage2", None)))
        modules.extend(sequential_modules(getattr(self.model, "stage3", None)))
        return modules


def build_visformer_small_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> VisformerSmallWithHook:
    return VisformerSmallWithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
