from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    DEFAULT_PRETRAINED,
    OptionalClsTokenMixin,
    PatchScoreFeatures,
    WhiteBoxWithHook,
    conv2d_attack_metadata,
    nested_stage_blocks,
)


DEFAULT_MODEL_NAME = "pit_b_224"


class PiTB224WithHook(OptionalClsTokenMixin, WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    def _feature_modules(self):
        return nested_stage_blocks(getattr(self.model, "transformers", ()))

    def _stage_modules(self):
        return list(getattr(self.model, "transformers", ()))

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
        stage_count: int | None = None,
        noise_position: str | None = None,
        noise_builder=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, channels = local_tokens.shape
        height, width = state.grid_size
        spatial = local_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        cls_token = state.context["cls_token"]
        if noise_position == "initial" and noise_builder is not None:
            local_tokens = noise_builder(local_tokens, state.grid_size, "initial")
            spatial = local_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        total_stages = len(self.model.transformers)
        if stage_count is None:
            stage_count = total_stages
        if not 1 <= stage_count <= total_stages:
            raise ValueError(f"PiT stage_count must be in [1, {total_stages}].")
        for stage_index, transformer in enumerate(self.model.transformers):
            spatial, cls_token = transformer((spatial, cls_token))
            if stage_index + 1 == 2 and noise_position == "pre_last_downsample" and noise_builder is not None:
                grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
                tokens = spatial.flatten(2).transpose(1, 2)
                tokens = tokens + noise_builder(tokens, grid_size, "pre_last_downsample")
                spatial = tokens.transpose(1, 2).reshape_as(spatial)
            if stage_index + 1 == stage_count:
                break
        if noise_position == "final" and noise_builder is not None:
            grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
            tokens = spatial.flatten(2).transpose(1, 2)
            tokens = tokens + noise_builder(tokens, grid_size, "final")
            spatial = tokens.transpose(1, 2).reshape_as(spatial)
        return spatial, cls_token

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        state = self.prepare_attack_feature_state(x)
        if score_layer == "final":
            spatial, cls_token = self._run_transformers(state, state.local_tokens)
            source_name = "transformers[2].blocks[3]"
        elif score_layer == "pre_last_downsample":
            spatial, cls_token = self._run_transformers(state, state.local_tokens, stage_count=2)
            source_name = "transformers[1]+pre_pool"
        else:
            raise ValueError(f"unsupported PiT patch score layer: {score_layer!r}")
        grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
        features = PatchScoreFeatures(
            local_tokens=spatial.flatten(2).transpose(1, 2),
            global_token=cls_token[:, :1],
            grid_size=grid_size,
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
            raise ValueError("replacement PiT local tokens do not match the attack state.")
        _, cls_token = self._run_transformers(state, local_tokens)
        return self.model.forward_head(self.model.norm(cls_token))

    def forward_with_attack_noise(
        self,
        state: AttackFeatureState,
        local_tokens: torch.Tensor,
        noise_position: str,
        noise_builder,
    ) -> torch.Tensor:
        state.validate()
        if local_tokens.shape != state.local_tokens.shape:
            raise ValueError("replacement PiT local tokens do not match the attack state.")
        spatial, cls_token = self._run_transformers(
            state,
            local_tokens,
            noise_position=noise_position,
            noise_builder=noise_builder,
        )
        return self.model.forward_head(self.model.norm(cls_token))


def build_pit_b_224_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> PiTB224WithHook:
    return PiTB224WithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
