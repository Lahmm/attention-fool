from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    DEFAULT_PRETRAINED,
    PatchScoreFeatures,
    WhiteBoxWithHook,
    conv2d_attack_metadata,
    sequential_modules,
)


DEFAULT_MODEL_NAME = "vit_base_patch16_224"


class ViTWithHook(WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    def _feature_modules(self):
        return sequential_modules(getattr(self.model, "blocks", None))

    def prepare_attack_feature_state(self, x: torch.Tensor) -> AttackFeatureState:
        base = self.model
        tokens = base.patch_embed(x)
        grid_size = tuple(int(value) for value in base.patch_embed.grid_size)
        tokens = base._pos_embed(tokens)
        tokens = base.patch_drop(tokens)
        tokens = base.norm_pre(tokens)
        prefix_count = int(getattr(base, "num_prefix_tokens", 1))
        state = AttackFeatureState(
            local_tokens=tokens[:, prefix_count:],
            grid_size=grid_size,
            context={"prefix_tokens": tokens[:, :prefix_count]},
            **conv2d_attack_metadata(base.patch_embed.proj),
        )
        state.validate()
        return state

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        if score_layer != "final":
            raise ValueError(f"unsupported ViT patch score layer: {score_layer!r}")
        state = self.prepare_attack_feature_state(x)
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        tokens = self.model.blocks(tokens)
        features = PatchScoreFeatures(
            local_tokens=tokens[:, -state.local_tokens.size(1):],
            global_token=tokens[:, :1],
            grid_size=state.grid_size,
            source_name="blocks[11]",
        )
        features.validate()
        return features

    def forward_from_attack_feature_state(
        self,
        state: AttackFeatureState,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        state.validate()
        if local_tokens.shape != state.local_tokens.shape:
            raise ValueError("replacement ViT local tokens do not match the attack state.")
        tokens = torch.cat((state.context["prefix_tokens"], local_tokens), dim=1)
        tokens = self.model.blocks(tokens)
        return self.model.forward_head(self.model.norm(tokens))


def build_vit_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> ViTWithHook:
    return ViTWithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
