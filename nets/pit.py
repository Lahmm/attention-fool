from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    DEFAULT_PRETRAINED,
    PatchScoreFeatures,
    WhiteBoxWithHook,
    conv2d_attack_metadata,
    nested_stage_blocks,
)


DEFAULT_MODEL_NAME = "pit_b_224"


class PiTB224WithHook(WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    def _feature_modules(self):
        return nested_stage_blocks(getattr(self.model, "transformers", ()))

    def prepare_attack_feature_state(self, x: torch.Tensor) -> AttackFeatureState:
        base = self.model
        spatial = base.patch_embed(x)
        spatial = base.pos_drop(spatial + base.pos_embed)
        grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
        cls_token = base.cls_token.expand(spatial.size(0), -1, -1)
        state = AttackFeatureState(
            local_tokens=spatial.flatten(2).transpose(1, 2),
            grid_size=grid_size,
            context={"cls_token": cls_token},
            **conv2d_attack_metadata(base.patch_embed.conv),
        )
        state.validate()
        return state

    def _run_transformers(
        self,
        state: AttackFeatureState,
        local_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = local_tokens.shape
        height, width = state.grid_size
        spatial = local_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        cls_token = state.context["cls_token"]
        for transformer in self.model.transformers:
            spatial, cls_token = transformer((spatial, cls_token))
        return spatial, cls_token

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        state = self.prepare_attack_feature_state(x)
        if score_layer != "final":
            raise ValueError(f"unsupported PiT patch score layer: {score_layer!r}")
        spatial, cls_token = self._run_transformers(state, state.local_tokens)
        grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
        features = PatchScoreFeatures(
            local_tokens=spatial.flatten(2).transpose(1, 2),
            global_token=cls_token[:, :1],
            grid_size=grid_size,
            source_name="transformers[2].blocks[3]",
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
            raise ValueError("replacement PiT local tokens do not match the attack state.")
        _, cls_token = self._run_transformers(state, local_tokens)
        return self.model.forward_head(self.model.norm(cls_token))

def build_pit_b_224_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> PiTB224WithHook:
    return PiTB224WithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
