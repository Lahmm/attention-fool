from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    DEFAULT_PRETRAINED,
    PatchScoreActivationCapture,
    PatchScoreFeatures,
    WhiteBoxWithHook,
    conv2d_attack_metadata,
    sequential_modules,
)


DEFAULT_MODEL_NAME = "visformer_small"


class VisformerSmallWithHook(WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    _PATCH_SCORE_LAYERS = {
        "stage1_block4": (1, 4),
        "stage1_block7": (1, 7),
        "stage2_block4": (2, 4),
        "stage3_block2": (3, 2),
        "stage3_block4": (3, 4),
    }

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

    def patch_score_layer_candidates(self) -> tuple[str, ...]:
        return tuple(self._PATCH_SCORE_LAYERS)

    def patch_score_activation_capture(self, score_layer: str):
        canonical = "stage3_block4" if score_layer == "final" else score_layer
        if canonical not in self._PATCH_SCORE_LAYERS:
            raise ValueError(f"unsupported Visformer patch score layer: {score_layer!r}")
        stage_index, block_count = self._PATCH_SCORE_LAYERS[canonical]
        stage = getattr(self.model, f"stage{stage_index}")
        if block_count < len(stage):
            return PatchScoreActivationCapture(
                module=stage[block_count],
                hook_type="input",
                source_name=(
                    f"stage{stage_index}[{block_count}] input "
                    f"(=stage{stage_index}[{block_count - 1}] output)"
                ),
            )
        return PatchScoreActivationCapture(
            module=stage[block_count - 1],
            hook_type="output",
            source_name=f"stage{stage_index}[{block_count - 1}] output",
        )

    def _run_to_checkpoint(
        self,
        state: AttackFeatureState,
        target_stage: int,
        target_block_count: int,
    ) -> torch.Tensor:
        batch, _, channels = state.local_tokens.shape
        height, width = state.grid_size
        spatial = state.local_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        base = self.model
        stages = (
            (base.patch_embed1, base.pos_embed1, base.stage1),
            (base.patch_embed2, base.pos_embed2, base.stage2),
            (base.patch_embed3, base.pos_embed3, base.stage3),
        )
        for stage_index, (patch_embed, pos_embed, blocks) in enumerate(stages, start=1):
            if patch_embed is not None:
                spatial = patch_embed(spatial)
                if pos_embed is not None:
                    spatial = base.pos_drop(spatial + pos_embed)
            block_count = target_block_count if stage_index == target_stage else len(blocks)
            for block in blocks[:block_count]:
                spatial = block(spatial)
            if stage_index == target_stage:
                return spatial
        raise RuntimeError("Visformer routing checkpoint was not reached.")

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
        canonical = "stage3_block4" if score_layer == "final" else score_layer
        if canonical not in self._PATCH_SCORE_LAYERS:
            raise ValueError(
                f"unsupported Visformer patch score layer: {score_layer!r}; "
                f"choose from {self.patch_score_layer_candidates()} or 'final'."
            )
        target_stage, target_block_count = self._PATCH_SCORE_LAYERS[canonical]
        state = self.prepare_attack_feature_state(x)
        spatial = self._run_to_checkpoint(state, target_stage, target_block_count)
        local_tokens = spatial.flatten(2).transpose(1, 2)
        features = PatchScoreFeatures(
            local_tokens=local_tokens,
            global_token=local_tokens.mean(dim=1, keepdim=True),
            grid_size=(int(spatial.size(-2)), int(spatial.size(-1))),
            source_name=f"stage{target_stage}[{target_block_count - 1}]+gap",
            layer_id=canonical,
            global_mode="gap",
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
