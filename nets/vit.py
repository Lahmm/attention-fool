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


DEFAULT_MODEL_NAME = "vit_base_patch16_224"


class ViTAdapter(ProgressiveAdapter):
    default_model_name = DEFAULT_MODEL_NAME

    _PROGRESSIVE_LAYERS = tuple(f"block{index}" for index in range(1, 12))
    _DEFAULT_PROGRESSIVE_LAYERS = ("block3", "block10")
    _DEFAULT_PROGRESSIVE_DROP_RATIOS = (0.051020408163, 0.051020408163)

    def prepare_progressive_input(self, x: torch.Tensor) -> ProgressiveInputState:
        base = self.model
        tokens = base.patch_embed(x)
        grid_size = tuple(int(value) for value in base.patch_embed.grid_size)
        tokens = base._pos_embed(tokens)
        tokens = base.patch_drop(tokens)
        tokens = base.norm_pre(tokens)
        prefix_count = int(getattr(base, "num_prefix_tokens", 1))
        state = ProgressiveInputState(
            local_tokens=tokens[:, prefix_count:],
            grid_size=grid_size,
            context={"prefix_tokens": tokens[:, :prefix_count]},
            **conv2d_progressive_metadata(base.patch_embed.proj),
        )
        state.validate()
        return state

    def progressive_checkpoint_candidates(self) -> tuple[str, ...]:
        return self._PROGRESSIVE_LAYERS

    def default_progressive_checkpoints(self) -> tuple[str, ...]:
        return self._DEFAULT_PROGRESSIVE_LAYERS

    def default_progressive_drop_ratios(self) -> tuple[float, ...]:
        return self._DEFAULT_PROGRESSIVE_DROP_RATIOS

    def begin_progressive_forward(self, x: torch.Tensor) -> ProgressiveAttackState:
        initial = self.prepare_progressive_input(x)
        state = ProgressiveAttackState(
            local_tokens=initial.local_tokens,
            grid_size=initial.grid_size,
            context={"prefix_tokens": initial.context["prefix_tokens"], "block_index": 0},
        )
        state.validate()
        return state

    def replace_progressive_local_tokens(
        self, state: ProgressiveAttackState, local_tokens: torch.Tensor
    ) -> ProgressiveAttackState:
        state.validate()
        if local_tokens.shape != state.local_tokens.shape:
            raise ValueError("replacement ViT progressive tokens do not match the state.")
        return ProgressiveAttackState(local_tokens, state.grid_size, dict(state.context))

    def advance_progressive_state(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> ProgressiveAttackState:
        if checkpoint_id not in self._PROGRESSIVE_LAYERS:
            raise ValueError(f"unsupported ViT progressive checkpoint: {checkpoint_id!r}.")
        target = int(checkpoint_id.removeprefix("block"))
        start = int(state.context["block_index"])
        if target <= start:
            raise ValueError("ViT progressive checkpoints must be strictly increasing.")
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        for block in self.model.blocks[start:target]:
            tokens = block(tokens)
        prefix_count = state.context["prefix_tokens"].size(1)
        return ProgressiveAttackState(
            local_tokens=tokens[:, prefix_count:],
            grid_size=state.grid_size,
            context={"prefix_tokens": tokens[:, :prefix_count], "block_index": target},
        )

    def progressive_score_features(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> PatchScoreFeatures:
        features = PatchScoreFeatures(
            local_tokens=state.local_tokens,
            global_token=state.context["prefix_tokens"][:, :1],
            grid_size=state.grid_size,
            source_name=f"blocks[{state.context['block_index'] - 1}]",
            layer_id=checkpoint_id,
            global_mode="cls",
        )
        features.validate()
        return features

    def apply_progressive_mask(
        self, state: ProgressiveAttackState, mask: torch.Tensor
    ) -> ProgressiveAttackState:
        if mask.shape != state.local_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("ViT progressive mask does not match local tokens.")
        local = torch.where(mask.unsqueeze(-1), torch.zeros_like(state.local_tokens), state.local_tokens)
        return ProgressiveAttackState(local, state.grid_size, dict(state.context))

    def finish_progressive_forward(self, state: ProgressiveAttackState) -> torch.Tensor:
        start = int(state.context["block_index"])
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        for block in self.model.blocks[start:]:
            tokens = block(tokens)
        return self.model.forward_head(self.model.norm(tokens))


def build_vit_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> ViTAdapter:
    return ViTAdapter(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
