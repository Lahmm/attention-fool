from __future__ import annotations

import torch

from .base import ProgressiveAdapter, ProgressiveState

DEFAULT_MODEL_NAME = "vit_base_patch16_224"


class ViTAdapter(ProgressiveAdapter):
    """PRD traversal for ViT-B/16 and distilled DeiT-B, preserving all prefixes."""

    default_model_name = DEFAULT_MODEL_NAME
    checkpoints = ("block3", "block10")
    drop_ratios = (0.051020408163, 0.051020408163)

    def begin_progressive_forward(self, x):
        base = self.model
        tokens = base.norm_pre(base.patch_drop(base._pos_embed(base.patch_embed(x))))
        prefix_count = base.num_prefix_tokens
        state = ProgressiveState(
            local_tokens=tokens[:, prefix_count:],
            grid_size=tuple(base.patch_embed.grid_size),
            context={"prefix_tokens": tokens[:, :prefix_count], "block_index": 0},
            rgb_projection_weight=base.patch_embed.proj.weight,
        )
        state.validate()
        return state

    def advance_progressive_state(self, state, checkpoint_id):
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"unsupported ViT checkpoint: {checkpoint_id!r}.")
        target = int(checkpoint_id.removeprefix("block"))
        start = state.context["block_index"]
        if target <= start:
            raise ValueError("checkpoints must be strictly increasing.")
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        for block in self.model.blocks[start:target]:
            tokens = block(tokens)
        prefix_count = state.context["prefix_tokens"].size(1)
        return ProgressiveState(
            tokens[:, prefix_count:], state.grid_size,
            {"prefix_tokens": tokens[:, :prefix_count], "block_index": target},
        )

    def finish_progressive_forward(self, state):
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        for block in self.model.blocks[state.context["block_index"]:]:
            tokens = block(tokens)
        return self.model.forward_head(self.model.norm(tokens))
