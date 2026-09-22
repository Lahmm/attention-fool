from __future__ import annotations

import torch

from .base import (
    DEFAULT_PRETRAINED,
    ProgressiveAdapter,
    ProgressiveAttackState,
    ProgressiveInputState,
    conv2d_progressive_metadata,
)


DEFAULT_MODEL_NAME = "cait_s24_224"


class CaiTS24Adapter(ProgressiveAdapter):
    default_model_name = DEFAULT_MODEL_NAME

    _PROGRESSIVE_LAYERS = tuple(f"block{index}" for index in range(1, 25))
    _DEFAULT_PROGRESSIVE_LAYERS = ("block17", "block23")
    _DEFAULT_PROGRESSIVE_DROP_RATIOS = (0.010204081633, 0.142857142857)

    def prepare_progressive_input(self, x: torch.Tensor) -> ProgressiveInputState:
        base = self.model
        local_tokens = base.patch_embed(x)
        grid_size = tuple(int(value) for value in base.patch_embed.grid_size)
        local_tokens = base.pos_drop(local_tokens + base.pos_embed)
        state = ProgressiveInputState(
            local_tokens=local_tokens,
            grid_size=grid_size,
            context=None,
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
        return ProgressiveAttackState(
            initial.local_tokens, initial.grid_size, {"block_index": 0}
        )

    def replace_progressive_local_tokens(
        self, state: ProgressiveAttackState, local_tokens: torch.Tensor
    ) -> ProgressiveAttackState:
        if local_tokens.shape != state.local_tokens.shape:
            raise ValueError("replacement CaiT progressive tokens do not match the state.")
        return ProgressiveAttackState(local_tokens, state.grid_size, dict(state.context))

    def advance_progressive_state(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> ProgressiveAttackState:
        if checkpoint_id not in self._PROGRESSIVE_LAYERS:
            raise ValueError(f"unsupported CaiT progressive checkpoint: {checkpoint_id!r}.")
        target = int(checkpoint_id.removeprefix("block"))
        start = int(state.context["block_index"])
        if target <= start:
            raise ValueError("CaiT progressive checkpoints must be strictly increasing.")
        local = state.local_tokens
        for block in self.model.blocks[start:target]:
            local = block(local)
        return ProgressiveAttackState(local, state.grid_size, {"block_index": target})

    def apply_progressive_mask(
        self, state: ProgressiveAttackState, mask: torch.Tensor
    ) -> ProgressiveAttackState:
        if mask.shape != state.local_tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("CaiT progressive mask does not match local tokens.")
        local = torch.where(mask.unsqueeze(-1), torch.zeros_like(state.local_tokens), state.local_tokens)
        return ProgressiveAttackState(local, state.grid_size, dict(state.context))

    def finish_progressive_forward(self, state: ProgressiveAttackState) -> torch.Tensor:
        local = state.local_tokens
        start = int(state.context["block_index"])
        for block in self.model.blocks[start:]:
            local = block(local)
        cls = self.model.cls_token.expand(local.size(0), -1, -1)
        for block in self.model.blocks_token_only:
            cls = block(local, cls)
        tokens = self.model.norm(torch.cat((cls, local), dim=1))
        return self.model.forward_head(tokens)


def build_cait_s24_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> CaiTS24Adapter:
    return CaiTS24Adapter(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
