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

    def _stage_modules(self):
        return [
            stage
            for stage in (
                getattr(self.model, "stage1", None),
                getattr(self.model, "stage2", None),
                getattr(self.model, "stage3", None),
            )
            if stage is not None
        ]

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
        stop_before_last_downsample: bool = False,
        noise_position: str | None = None,
        noise_builder=None,
    ) -> torch.Tensor:
        batch, _, channels = local_tokens.shape
        height, width = state.grid_size
        spatial = local_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        base = self.model

        if noise_position == "initial" and noise_builder is not None:
            local_tokens = noise_builder(local_tokens, state.grid_size, "initial")
            spatial = local_tokens.transpose(1, 2).reshape(batch, channels, height, width)

        spatial = base.patch_embed1(spatial)
        if base.pos_embed1 is not None:
            spatial = base.pos_drop(spatial + base.pos_embed1)
        spatial = base.stage1(spatial)

        if base.patch_embed2 is not None:
            spatial = base.patch_embed2(spatial)
            if base.pos_embed2 is not None:
                spatial = base.pos_drop(spatial + base.pos_embed2)
        spatial = base.stage2(spatial)

        if noise_position == "pre_last_downsample" and noise_builder is not None:
            tokens = spatial.flatten(2).transpose(1, 2)
            grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
            tokens = tokens + noise_builder(tokens, grid_size, "pre_last_downsample")
            spatial = tokens.transpose(1, 2).reshape_as(spatial)

        if stop_before_last_downsample:
            return spatial

        if base.patch_embed3 is not None:
            spatial = base.patch_embed3(spatial)
            if base.pos_embed3 is not None:
                spatial = base.pos_drop(spatial + base.pos_embed3)
        spatial = base.stage3(spatial)
        if noise_position == "final" and noise_builder is not None:
            tokens = spatial.flatten(2).transpose(1, 2)
            grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
            tokens = tokens + noise_builder(tokens, grid_size, "final")
            spatial = tokens.transpose(1, 2).reshape_as(spatial)
        return spatial

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        state = self.prepare_attack_feature_state(x)
        if score_layer == "final":
            spatial = self._run_stages(state, state.local_tokens)
            source_name = "stage3[3]+gap"
        elif score_layer == "pre_last_downsample":
            spatial = self._run_stages(
                state,
                state.local_tokens,
                stop_before_last_downsample=True,
            )
            source_name = "stage2+pre_patch_embed3+gap"
        else:
            raise ValueError(f"unsupported Visformer patch score layer: {score_layer!r}")
        local_tokens = spatial.flatten(2).transpose(1, 2)
        features = PatchScoreFeatures(
            local_tokens=local_tokens,
            global_token=local_tokens.mean(dim=1, keepdim=True),
            grid_size=(int(spatial.size(-2)), int(spatial.size(-1))),
            source_name=source_name,
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

    def forward_with_attack_noise(
        self,
        state: AttackFeatureState,
        local_tokens: torch.Tensor,
        noise_position: str,
        noise_builder,
    ) -> torch.Tensor:
        state.validate()
        if local_tokens.shape != state.local_tokens.shape:
            raise ValueError("replacement Visformer local tokens do not match the attack state.")
        spatial = self.model.norm(
            self._run_stages(
                state,
                local_tokens,
                noise_position=noise_position,
                noise_builder=noise_builder,
            )
        )
        return self.model.forward_head(spatial)


def build_visformer_small_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> VisformerSmallWithHook:
    return VisformerSmallWithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
