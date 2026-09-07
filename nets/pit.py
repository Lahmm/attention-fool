from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    DEFAULT_PRETRAINED,
    PatchScoreFeatures,
    ProgressiveAttackState,
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
    _PROGRESSIVE_LAYERS = (
        "stage1_block3",
        "stage2_block1",
        "stage2_block2",
        "stage2_block3",
        "stage2_block4",
        "stage2_block5",
        "stage2_block6",
        "stage3_block1",
        "stage3_block2",
        "stage3_block3",
    )
    _DEFAULT_PROGRESSIVE_LAYERS = (
        "stage1_block3",
        "stage2_block5",
        "stage3_block3",
    )

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

    def progressive_checkpoint_candidates(self) -> tuple[str, ...]:
        return self._PROGRESSIVE_LAYERS

    def default_progressive_checkpoints(self) -> tuple[str, ...]:
        return self._DEFAULT_PROGRESSIVE_LAYERS

    def begin_progressive_forward(self, x: torch.Tensor) -> ProgressiveAttackState:
        initial = self.prepare_attack_feature_state(x)
        return ProgressiveAttackState(
            initial.local_tokens,
            initial.grid_size,
            {
                "cls_token": initial.context["cls_token"],
                "stage_index": 0,
                "block_index": 0,
                "stage_prepared": False,
            },
        )

    def replace_progressive_local_tokens(
        self, state: ProgressiveAttackState, local_tokens: torch.Tensor
    ) -> ProgressiveAttackState:
        if local_tokens.shape != state.local_tokens.shape:
            raise ValueError("replacement PiT progressive tokens do not match the state.")
        return ProgressiveAttackState(local_tokens, state.grid_size, dict(state.context))

    @staticmethod
    def _parse_progressive_checkpoint(checkpoint_id: str) -> tuple[int, int]:
        stage_text, block_text = checkpoint_id.split("_")
        return int(stage_text.removeprefix("stage")) - 1, int(block_text.removeprefix("block"))

    @staticmethod
    def _pit_spatial(state: ProgressiveAttackState) -> torch.Tensor:
        batch, _, channels = state.local_tokens.shape
        height, width = state.grid_size
        return state.local_tokens.transpose(1, 2).reshape(batch, channels, height, width)

    @staticmethod
    def _split_pit_tokens(
        tokens: torch.Tensor, prefix_count: int, grid_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls = tokens[:, :prefix_count]
        batch, _, channels = tokens.shape
        spatial = tokens[:, prefix_count:].transpose(1, 2).reshape(
            batch, channels, *grid_size
        )
        return spatial, cls

    def _prepare_pit_stage(
        self, stage_index: int, spatial: torch.Tensor, cls: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        stage = self.model.transformers[stage_index]
        if stage.pool is not None:
            spatial, cls = stage.pool(spatial, cls)
        grid = (int(spatial.size(-2)), int(spatial.size(-1)))
        tokens = torch.cat((cls, spatial.flatten(2).transpose(1, 2)), dim=1)
        return stage.norm(tokens), grid

    def advance_progressive_state(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> ProgressiveAttackState:
        if checkpoint_id not in self._PROGRESSIVE_LAYERS:
            raise ValueError(f"unsupported PiT progressive checkpoint: {checkpoint_id!r}.")
        target_stage, target_block = self._parse_progressive_checkpoint(checkpoint_id)
        stage_index = int(state.context["stage_index"])
        block_index = int(state.context["block_index"])
        prepared = bool(state.context["stage_prepared"])
        if (target_stage, target_block) <= (stage_index, block_index):
            raise ValueError("PiT progressive checkpoints must be strictly increasing.")
        spatial = self._pit_spatial(state)
        cls = state.context["cls_token"]
        prefix_count = cls.size(1)
        grid = state.grid_size
        while stage_index <= target_stage:
            stage = self.model.transformers[stage_index]
            if not prepared:
                tokens, grid = self._prepare_pit_stage(stage_index, spatial, cls)
                prepared = True
            else:
                tokens = torch.cat((cls, spatial.flatten(2).transpose(1, 2)), dim=1)
            end = target_block if stage_index == target_stage else len(stage.blocks)
            for block in stage.blocks[block_index:end]:
                tokens = block(tokens)
            spatial, cls = self._split_pit_tokens(tokens, prefix_count, grid)
            block_index = end
            if stage_index == target_stage:
                return ProgressiveAttackState(
                    spatial.flatten(2).transpose(1, 2),
                    grid,
                    {
                        "cls_token": cls,
                        "stage_index": stage_index,
                        "block_index": block_index,
                        "stage_prepared": True,
                    },
                )
            stage_index += 1
            block_index = 0
            prepared = False
        raise RuntimeError("PiT progressive checkpoint was not reached.")

    def progressive_score_features(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> PatchScoreFeatures:
        stage_index = int(state.context["stage_index"])
        block_index = int(state.context["block_index"])
        features = PatchScoreFeatures(
            local_tokens=state.local_tokens,
            global_token=state.context["cls_token"][:, :1],
            grid_size=state.grid_size,
            source_name=f"transformers[{stage_index}].blocks[{block_index - 1}]",
            layer_id=checkpoint_id,
            global_mode="cls",
        )
        features.validate()
        return features

    def apply_progressive_mask(
        self, state: ProgressiveAttackState, mask: torch.Tensor
    ) -> ProgressiveAttackState:
        if mask.shape != state.local_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("PiT progressive mask does not match local tokens.")
        local = torch.where(mask.unsqueeze(-1), torch.zeros_like(state.local_tokens), state.local_tokens)
        return ProgressiveAttackState(local, state.grid_size, dict(state.context))

    def finish_progressive_forward(self, state: ProgressiveAttackState) -> torch.Tensor:
        spatial = self._pit_spatial(state)
        cls = state.context["cls_token"]
        stage_index = int(state.context["stage_index"])
        block_index = int(state.context["block_index"])
        prepared = bool(state.context["stage_prepared"])
        prefix_count = cls.size(1)
        grid = state.grid_size
        while stage_index < len(self.model.transformers):
            stage = self.model.transformers[stage_index]
            if not prepared:
                tokens, grid = self._prepare_pit_stage(stage_index, spatial, cls)
            else:
                tokens = torch.cat((cls, spatial.flatten(2).transpose(1, 2)), dim=1)
            for block in stage.blocks[block_index:]:
                tokens = block(tokens)
            spatial, cls = self._split_pit_tokens(tokens, prefix_count, grid)
            stage_index += 1
            block_index = 0
            prepared = False
        return self.model.forward_head(self.model.norm(cls))

def build_pit_b_224_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> PiTB224WithHook:
    return PiTB224WithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
