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


DEFAULT_MODEL_NAME = "visformer_small"


class VisformerSmallWithHook(WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    def _feature_modules(self):
        modules = []
        modules.extend(sequential_modules(getattr(self.model, "stage1", None)))
        modules.extend(sequential_modules(getattr(self.model, "stage2", None)))
        modules.extend(sequential_modules(getattr(self.model, "stage3", None)))
        return modules

    def prepare_attack_feature_state(self, x: torch.Tensor) -> AttackFeatureState:
        base = self.model
        if base.stem is None or not len(base.stem):
            raise ValueError("Visformer mainline adaptation requires the convolutional RGB stem.")
        spatial = base.stem(x)
        grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
        state = AttackFeatureState(
            local_tokens=spatial.flatten(2).transpose(1, 2),
            grid_size=grid_size,
            context=None,
            **conv2d_attack_metadata(base.stem[0]),
        )
        state.validate()
        return state

    def _run_stages(
        self,
        state: AttackFeatureState,
        local_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, channels = local_tokens.shape
        height, width = state.grid_size
        spatial = local_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        base = self.model

        spatial = base.patch_embed1(spatial)
        if base.pos_embed1 is not None:
            spatial = base.pos_drop(spatial + base.pos_embed1)
        spatial = base.stage1(spatial)

        if base.patch_embed2 is not None:
            spatial = base.patch_embed2(spatial)
            if base.pos_embed2 is not None:
                spatial = base.pos_drop(spatial + base.pos_embed2)
        spatial = base.stage2(spatial)

        if base.patch_embed3 is not None:
            spatial = base.patch_embed3(spatial)
            if base.pos_embed3 is not None:
                spatial = base.pos_drop(spatial + base.pos_embed3)
        spatial = base.stage3(spatial)
        return spatial

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        state = self.prepare_attack_feature_state(x)
        if score_layer != "final":
            raise ValueError(f"unsupported Visformer patch score layer: {score_layer!r}")
        spatial = self._run_stages(state, state.local_tokens)
        local_tokens = spatial.flatten(2).transpose(1, 2)
        features = PatchScoreFeatures(
            local_tokens=local_tokens,
            global_token=local_tokens.mean(dim=1, keepdim=True),
            grid_size=(int(spatial.size(-2)), int(spatial.size(-1))),
            source_name="stage3[3]+gap",
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
            raise ValueError("replacement Visformer local tokens do not match the attack state.")
        spatial = self.model.norm(self._run_stages(state, local_tokens))
        return self.model.forward_head(spatial)

def build_visformer_small_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> VisformerSmallWithHook:
    return VisformerSmallWithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
