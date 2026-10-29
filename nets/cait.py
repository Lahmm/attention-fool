from __future__ import annotations

import torch

from .base import ProgressiveAdapter, ProgressiveState, select_variance_peaks


class CaiTAdapter(ProgressiveAdapter):
    default_model_name = "cait_s24_224"
    checkpoints = ("block17", "block23")
    drop_ratios = (0.010204081633, 0.142857142857)
    dynamic_checkpoint = True
    checkpoint_score_reference = "final CLS after class-attention, before final LayerNorm; raw local block outputs"

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
        target = self.checkpoint_order(checkpoint_id)
        start = state.context["block_index"]
        if target < start:
            raise ValueError("CaiT checkpoint is outside the remaining block trajectory.")
        if target == start:
            return state
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

    def checkpoint_order(self, checkpoint_id):
        suffix = checkpoint_id.removeprefix("block")
        if (not checkpoint_id.startswith("block") or not suffix.isdigit()
                or not 0 <= int(suffix) <= len(self.model.blocks)):
            raise ValueError(f"unsupported CaiT checkpoint: {checkpoint_id!r}.")
        return int(suffix)

    def checkpoint_grid_size(self, checkpoint_id):
        self.checkpoint_order(checkpoint_id)
        return tuple(self.model.patch_embed.grid_size)

    @torch.no_grad()
    def select_variance_checkpoints(self, normalized_inputs):
        self.eval()
        state = self.begin_progressive_forward(normalized_inputs.to(self.device))
        local = state.local_tokens
        snapshots = []
        for block in self.model.blocks:
            local = block(local)
            snapshots.append(local.clone())
        cls = self.model.cls_token.expand(local.size(0), -1, -1)
        for block in self.model.blocks_token_only:
            cls = block(local, cls)
        reference = cls[:, 0]
        return select_variance_peaks([
            (f"block{index + 1}", f"block{index}", state.grid_size, values, reference)
            for index, values in enumerate(snapshots)
        ])
