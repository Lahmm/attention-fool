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

    _PATCH_SCORE_LAYERS = {
        "stage1_block3": (0, 3),
        "stage2_block3": (1, 3),
        "stage2_block6": (1, 6),
        "stage3_block2": (2, 2),
        "stage3_block4": (2, 4),
    }

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

    def patch_score_layer_candidates(self) -> tuple[str, ...]:
        return tuple(self._PATCH_SCORE_LAYERS)

    @staticmethod
    def _run_stage_to_block(
        stage,
        spatial: torch.Tensor,
        cls_token: torch.Tensor,
        block_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stage.pool is not None:
            spatial, cls_token = stage.pool(spatial, cls_token)
        batch, channels, height, width = spatial.shape
        prefix_count = cls_token.size(1)
        tokens = torch.cat((cls_token, spatial.flatten(2).transpose(1, 2)), dim=1)
        tokens = stage.norm(tokens)
        for block in stage.blocks[:block_count]:
            tokens = block(tokens)
        cls_token = tokens[:, :prefix_count]
        spatial = tokens[:, prefix_count:].transpose(1, 2).reshape(
            batch, channels, height, width
        )
        return spatial, cls_token

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
        canonical = "stage3_block4" if score_layer == "final" else score_layer
        if canonical not in self._PATCH_SCORE_LAYERS:
            raise ValueError(
                f"unsupported PiT patch score layer: {score_layer!r}; "
                f"choose from {self.patch_score_layer_candidates()} or 'final'."
            )
        target_stage, target_block_count = self._PATCH_SCORE_LAYERS[canonical]
        state = self.prepare_attack_feature_state(x)
        batch, _, channels = state.local_tokens.shape
        height, width = state.grid_size
        spatial = state.local_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        cls_token = state.context["cls_token"]
        for stage_index, stage in enumerate(self.model.transformers):
            block_count = target_block_count if stage_index == target_stage else len(stage.blocks)
            spatial, cls_token = self._run_stage_to_block(
                stage, spatial, cls_token, block_count
            )
            if stage_index == target_stage:
                break
        grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
        features = PatchScoreFeatures(
            local_tokens=spatial.flatten(2).transpose(1, 2),
            global_token=cls_token[:, :1],
            grid_size=grid_size,
            source_name=f"transformers[{target_stage}].blocks[{target_block_count - 1}]",
            layer_id=canonical,
            global_mode="cls",
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
