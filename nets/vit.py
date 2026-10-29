from __future__ import annotations

import torch

from .base import ProgressiveAdapter, ProgressiveState, select_variance_peaks

DEFAULT_MODEL_NAME = "vit_base_patch16_224"


class ViTAdapter(ProgressiveAdapter):
    """PRD traversal for ViT-B/16 and distilled DeiT-B, preserving all prefixes."""

    default_model_name = DEFAULT_MODEL_NAME
    checkpoints = ("block3", "block10")
    drop_ratios = (0.051020408163, 0.051020408163)
    dynamic_checkpoint = True

    def __init__(self, model_name=None, *, pretrained=True, device=None):
        super().__init__(model_name, pretrained=pretrained, device=device)
        if self.model_name == "deit_base_distilled_patch16_224":
            self.checkpoints = ("block2", "block9")

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
        suffix = checkpoint_id.removeprefix("block")
        if not checkpoint_id.startswith("block") or not suffix.isdigit():
            raise ValueError(f"unsupported ViT checkpoint: {checkpoint_id!r}.")
        target = int(suffix)
        start = state.context["block_index"]
        if target < start or target > len(self.model.blocks):
            raise ValueError("ViT checkpoint is outside the remaining block trajectory.")
        if target == start:
            return state
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

    @torch.no_grad()
    def select_variance_checkpoints(self, normalized_inputs):
        self.eval()
        state = self.begin_progressive_forward(normalized_inputs.to(self.device))
        prefix = state.context["prefix_tokens"]
        tokens = torch.cat((prefix, state.local_tokens), dim=1)
        snapshots = []
        for block in self.model.blocks:
            tokens = block(tokens)
            snapshots.append(tokens[:, prefix.size(1):].cpu().clone())
        reference = self.model.norm(tokens)[:, 0].cpu()
        return select_variance_peaks([
            (f"block{index + 1}", f"block{index}", state.grid_size, local, reference)
            for index, local in enumerate(snapshots)
        ])
