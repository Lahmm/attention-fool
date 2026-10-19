from __future__ import annotations

import torch

from .base import ProgressiveAdapter, ProgressiveState


class CaiTAdapter(ProgressiveAdapter):
    default_model_name = "cait_s24_224"
    checkpoints = ("block17", "block23")
    drop_ratios = (0.010204081633, 0.142857142857)

    def begin_progressive_forward(self, x):
        base = self.model
        local = base.pos_drop(base.patch_embed(x) + base.pos_embed)
        state = ProgressiveState(
            local, tuple(base.patch_embed.grid_size), {"block_index": 0},
            base.patch_embed.proj.weight,
        )
        state.validate()
        return state

    def advance_progressive_state(self, state, checkpoint_id):
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"unsupported CaiT checkpoint: {checkpoint_id!r}.")
        target = int(checkpoint_id.removeprefix("block"))
        start = state.context["block_index"]
        if target <= start:
            raise ValueError("checkpoints must be strictly increasing.")
        local = state.local_tokens
        for block in self.model.blocks[start:target]:
            local = block(local)
        return ProgressiveState(local, state.grid_size, {"block_index": target})

    def finish_progressive_forward(self, state):
        local = state.local_tokens
        for block in self.model.blocks[state.context["block_index"]:]:
            local = block(local)
        cls = self.model.cls_token.expand(local.size(0), -1, -1)
        for block in self.model.blocks_token_only:
            cls = block(local, cls)
        return self.model.forward_head(self.model.norm(torch.cat((cls, local), dim=1)))
