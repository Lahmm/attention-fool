from __future__ import annotations

import torch

from .base import (
    DEFAULT_PRETRAINED,
    PatchScoreFeatures,
    ProgressiveAdapter,
    ProgressiveAttackState,
    ProgressiveInputState,
    conv2d_progressive_metadata,
)


DEFAULT_MODEL_NAME = "visformer_small"


class VisformerSmallAdapter(ProgressiveAdapter):
    default_model_name = DEFAULT_MODEL_NAME

    _PROGRESSIVE_LAYERS = (
        "stage1_block1",
        "stage1_block2",
        "stage1_block3",
        "stage1_block4",
        "stage1_block5",
        "stage1_block6",
        "stage1_block7",
        "stage2_block1",
        "stage2_block2",
        "stage2_block3",
        "stage2_block4",
        "stage3_block1",
        "stage3_block2",
        "stage3_block3",
    )
    _DEFAULT_PROGRESSIVE_LAYERS = (
        "stage2_block1",
        "stage3_block1",
    )
    _DEFAULT_PROGRESSIVE_DROP_RATIOS = (0.209183673469, 0.204081632653)

    def prepare_progressive_input(self, x: torch.Tensor) -> ProgressiveInputState:
        base = self.model
        if base.stem is None or not len(base.stem):
            raise ValueError("Visformer mainline adaptation requires the convolutional RGB stem.")
        spatial = base.stem(x)
        grid_size = (int(spatial.size(-2)), int(spatial.size(-1)))
        state = ProgressiveInputState(
            local_tokens=spatial.flatten(2).transpose(1, 2),
            grid_size=grid_size,
            context=None,
            **conv2d_progressive_metadata(base.stem[0]),
        )
        state.validate()
        return state

    def progressive_checkpoint_candidates(self) -> tuple[str, ...]:
        return self._PROGRESSIVE_LAYERS

    def default_progressive_checkpoints(self) -> tuple[str, ...]:
        return self._DEFAULT_PROGRESSIVE_LAYERS

    def default_progressive_drop_ratios(self) -> tuple[float, ...]:
        return self._DEFAULT_PROGRESSIVE_DROP_RATIOS

    def default_progressive_score_mode(self) -> str:
        return "gap_projection"

    def default_progressive_opponent_noise_strength(self) -> float:
        return 0.4

    def begin_progressive_forward(self, x: torch.Tensor) -> ProgressiveAttackState:
        initial = self.prepare_progressive_input(x)
        return ProgressiveAttackState(
            initial.local_tokens,
            initial.grid_size,
            {"stage_index": 1, "block_index": 0, "stage_prepared": False},
        )

    def replace_progressive_local_tokens(
        self, state: ProgressiveAttackState, local_tokens: torch.Tensor
    ) -> ProgressiveAttackState:
        if local_tokens.shape != state.local_tokens.shape:
            raise ValueError("replacement Visformer progressive tokens do not match the state.")
        return ProgressiveAttackState(local_tokens, state.grid_size, dict(state.context))

    @staticmethod
    def _parse_progressive_checkpoint(checkpoint_id: str) -> tuple[int, int]:
        stage_text, block_text = checkpoint_id.split("_")
        return int(stage_text.removeprefix("stage")), int(block_text.removeprefix("block"))

    @staticmethod
    def _visformer_spatial(state: ProgressiveAttackState) -> torch.Tensor:
        batch, _, channels = state.local_tokens.shape
        return state.local_tokens.transpose(1, 2).reshape(batch, channels, *state.grid_size)

    def _visformer_stage_parts(self, stage_index: int):
        base = self.model
        return (
            getattr(base, f"patch_embed{stage_index}"),
            getattr(base, f"pos_embed{stage_index}"),
            getattr(base, f"stage{stage_index}"),
        )

    def _prepare_visformer_stage(
        self, stage_index: int, spatial: torch.Tensor
    ) -> torch.Tensor:
        patch_embed, pos_embed, _ = self._visformer_stage_parts(stage_index)
        if patch_embed is not None:
            spatial = patch_embed(spatial)
            if pos_embed is not None:
                spatial = self.model.pos_drop(spatial + pos_embed)
        return spatial

    def advance_progressive_state(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> ProgressiveAttackState:
        if checkpoint_id not in self._PROGRESSIVE_LAYERS:
            raise ValueError(f"unsupported Visformer progressive checkpoint: {checkpoint_id!r}.")
        target_stage, target_block = self._parse_progressive_checkpoint(checkpoint_id)
        stage_index = int(state.context["stage_index"])
        block_index = int(state.context["block_index"])
        prepared = bool(state.context["stage_prepared"])
        if (target_stage, target_block) <= (stage_index, block_index):
            raise ValueError("Visformer progressive checkpoints must be strictly increasing.")
        spatial = self._visformer_spatial(state)
        while stage_index <= target_stage:
            _, _, blocks = self._visformer_stage_parts(stage_index)
            if not prepared:
                spatial = self._prepare_visformer_stage(stage_index, spatial)
                prepared = True
            end = target_block if stage_index == target_stage else len(blocks)
            for block in blocks[block_index:end]:
                spatial = block(spatial)
            block_index = end
            if stage_index == target_stage:
                grid = (int(spatial.size(-2)), int(spatial.size(-1)))
                return ProgressiveAttackState(
                    spatial.flatten(2).transpose(1, 2),
                    grid,
                    {
                        "stage_index": stage_index,
                        "block_index": block_index,
                        "stage_prepared": True,
                    },
                )
            stage_index += 1
            block_index = 0
            prepared = False
        raise RuntimeError("Visformer progressive checkpoint was not reached.")

    def progressive_score_features(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> PatchScoreFeatures:
        stage_index = int(state.context["stage_index"])
        block_index = int(state.context["block_index"])
        features = PatchScoreFeatures(
            local_tokens=state.local_tokens,
            global_token=state.local_tokens.mean(dim=1, keepdim=True),
            grid_size=state.grid_size,
            source_name=f"stage{stage_index}[{block_index - 1}]+gap",
            layer_id=checkpoint_id,
            global_mode="gap",
        )
        features.validate()
        return features

    def apply_progressive_mask(
        self, state: ProgressiveAttackState, mask: torch.Tensor
    ) -> ProgressiveAttackState:
        if mask.shape != state.local_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Visformer progressive mask does not match local tokens.")
        local = torch.where(mask.unsqueeze(-1), torch.zeros_like(state.local_tokens), state.local_tokens)
        return ProgressiveAttackState(local, state.grid_size, dict(state.context))

    def finish_progressive_forward(self, state: ProgressiveAttackState) -> torch.Tensor:
        spatial = self._visformer_spatial(state)
        stage_index = int(state.context["stage_index"])
        block_index = int(state.context["block_index"])
        prepared = bool(state.context["stage_prepared"])
        while stage_index <= 3:
            _, _, blocks = self._visformer_stage_parts(stage_index)
            if not prepared:
                spatial = self._prepare_visformer_stage(stage_index, spatial)
            for block in blocks[block_index:]:
                spatial = block(spatial)
            stage_index += 1
            block_index = 0
            prepared = False
        return self.model.forward_head(self.model.norm(spatial))


def build_visformer_small_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> VisformerSmallAdapter:
    return VisformerSmallAdapter(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
