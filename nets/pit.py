from __future__ import annotations

import torch

from .base import ProgressiveAdapter, ProgressiveState, select_variance_peaks


class PiTAdapter(ProgressiveAdapter):
    """Native hierarchical traversal shared by PiT-T and PiT-B."""

    default_model_name = "pit_ti_224"
    checkpoints = ("stage2_block1", "stage3_block2", "stage3_block3")
    drop_ratios = (0.07142857142857142, 0.04081632653061224, 0.04081632653061224)
    opponent_noise_strength = 0.4
    dynamic_checkpoint = True

    def __init__(self, model_name=None, *, pretrained=True, device=None):
        super().__init__(model_name, pretrained=pretrained, device=device)
        if self.model_name == "pit_b_224":
            self.drop_ratios = (0.02081165, 0.03125, 0.09375)
            self.dynamic_checkpoint = False

    def begin_progressive_forward(self, x):
        base = self.model
        spatial = base.pos_drop(base.patch_embed(x) + base.pos_embed)
        state = ProgressiveState(
            spatial.flatten(2).transpose(1, 2),
            (int(spatial.size(-2)), int(spatial.size(-1))),
            {"cls_token": base.cls_token.expand(spatial.size(0), -1, -1),
             "stage_index": 0, "block_index": 0, "stage_prepared": False},
            base.patch_embed.conv.weight,
        )
        state.validate()
        return state

    @staticmethod
    def _parse_progressive_checkpoint(checkpoint_id: str) -> tuple[int, int]:
        stage_text, block_text = checkpoint_id.split("_")
        return int(stage_text.removeprefix("stage")) - 1, int(block_text.removeprefix("block"))

    def checkpoint_order(self, checkpoint_id):
        try:
            stage, block = self._parse_progressive_checkpoint(checkpoint_id)
        except (ValueError, TypeError):
            raise ValueError(f"unsupported PiT checkpoint: {checkpoint_id!r}.") from None
        if (not 0 <= stage < len(self.model.transformers)
                or not 0 <= block <= len(self.model.transformers[stage].blocks)
                or checkpoint_id != f"stage{stage + 1}_block{block}"):
            raise ValueError(f"unsupported PiT checkpoint: {checkpoint_id!r}.")
        return stage, block

    def checkpoint_grid_size(self, checkpoint_id):
        target_stage, _ = self.checkpoint_order(checkpoint_id)
        grid = tuple(self.model.patch_embed.grid_size)
        for stage in self.model.transformers[:target_stage + 1]:
            if stage.pool is not None:
                conv = stage.pool.conv
                grid = tuple((size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
                             for size, padding, dilation, kernel, stride in zip(
                                 grid, conv.padding, conv.dilation, conv.kernel_size, conv.stride))
        return grid

    @staticmethod
    def _pit_spatial(state: ProgressiveState) -> torch.Tensor:
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
        self, state: ProgressiveState, checkpoint_id: str
    ) -> ProgressiveState:
        target_stage, target_block = self.checkpoint_order(checkpoint_id)
        stage_index = int(state.context["stage_index"])
        block_index = int(state.context["block_index"])
        prepared = bool(state.context["stage_prepared"])
        if (target_stage, target_block) < (stage_index, block_index):
            raise ValueError("PiT checkpoint is before the current state.")
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
                return ProgressiveState(
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

    def finish_progressive_forward(self, state: ProgressiveState) -> torch.Tensor:
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

    @torch.no_grad()
    def select_variance_checkpoints(self, normalized_inputs):
        self.eval()
        state = self.begin_progressive_forward(normalized_inputs.to(self.device))
        spatial = self._pit_spatial(state)
        cls = state.context["cls_token"]
        observations = []
        for stage_index, stage in enumerate(self.model.transformers):
            tokens, grid = self._prepare_pit_stage(stage_index, spatial, cls)
            snapshots = []
            for block in stage.blocks:
                tokens = block(tokens)
                snapshots.append(tokens[:, cls.size(1):].clone())
            spatial, cls = self._split_pit_tokens(tokens, cls.size(1), grid)
            reference = (self.model.norm(cls) if stage_index == len(self.model.transformers) - 1
                         else cls)[:, 0]
            for index, local in enumerate(snapshots):
                observations.append((f"stage{stage_index + 1}_block{index + 1}",
                                     f"stage{stage_index + 1}_block{index}", grid, local, reference))
        return select_variance_peaks(observations)
