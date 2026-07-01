from __future__ import annotations

from .base import DEFAULT_PRETRAINED, OptionalClsTokenMixin, WhiteBoxWithHook, nested_stage_blocks


DEFAULT_MODEL_NAME = "pit_b_224"


class PiTB224WithHook(OptionalClsTokenMixin, WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    def _feature_modules(self):
        return nested_stage_blocks(getattr(self.model, "transformers", ()))


def build_pit_b_224_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> PiTB224WithHook:
    return PiTB224WithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
