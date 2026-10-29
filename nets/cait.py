from __future__ import annotations

import torch

from .base import ProgressiveAdapter, ProgressiveState, select_variance_peaks


class CaiTAdapter(ProgressiveAdapter):
    default_model_name = "cait_s24_224"
    checkpoints = ("block17", "block23")
    drop_ratios = (0.010204081633, 0.142857142857)
    dynamic_checkpoint = True

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
        suffix = checkpoint_id.removeprefix("block")
        if not checkpoint_id.startswith("block") or not suffix.isdigit():
            raise ValueError(f"unsupported CaiT checkpoint: {checkpoint_id!r}.")
        target = int(suffix)
        start = state.context["block_index"]
        if target < start or target > len(self.model.blocks):
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

    @torch.no_grad()
    def select_variance_checkpoints(self, normalized_inputs):
        self.eval()
        state = self.begin_progressive_forward(normalized_inputs.to(self.device))
        local = state.local_tokens
        snapshots = []
        for block in self.model.blocks:
            local = block(local)
            snapshots.append(local.cpu().clone())
        cls = self.model.cls_token.expand(local.size(0), -1, -1)
        for block in self.model.blocks_token_only:
            cls = block(local, cls)
        reference = self.model.norm(torch.cat((cls, local), dim=1))[:, 0].cpu()
        return select_variance_peaks([
            (f"block{index + 1}", f"block{index}", state.grid_size, values, reference)
            for index, values in enumerate(snapshots)
        ])
