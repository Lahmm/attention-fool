from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    DEFAULT_PRETRAINED,
    PatchScoreFeatures,
    ProgressiveAttackState,
    WhiteBoxWithHook,
    conv2d_attack_metadata,
    sequential_modules,
)


DEFAULT_MODEL_NAME = "vit_base_patch16_224"


class ViTWithHook(WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    _PATCH_SCORE_LAYERS = {
        "block3": 3,
        "block6": 6,
        "block9": 9,
        "block12": 12,
    }
    _PROGRESSIVE_LAYERS = tuple(f"block{index}" for index in range(1, 12))
    _DEFAULT_PROGRESSIVE_LAYERS = ("block3", "block7", "block11")

    def _feature_modules(self):
        return sequential_modules(getattr(self.model, "blocks", None))

    def prepare_attack_feature_state(self, x: torch.Tensor) -> AttackFeatureState:
        base = self.model
        tokens = base.patch_embed(x)
        grid_size = tuple(int(value) for value in base.patch_embed.grid_size)
        tokens = base._pos_embed(tokens)
        tokens = base.patch_drop(tokens)
        tokens = base.norm_pre(tokens)
        prefix_count = int(getattr(base, "num_prefix_tokens", 1))
        state = AttackFeatureState(
            local_tokens=tokens[:, prefix_count:],
            grid_size=grid_size,
            context={"prefix_tokens": tokens[:, :prefix_count]},
            **conv2d_attack_metadata(base.patch_embed.proj),
        )
        state.validate()
        return state

    def patch_score_layer_candidates(self) -> tuple[str, ...]:
        return tuple(self._PATCH_SCORE_LAYERS)

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        canonical = "block12" if score_layer == "final" else score_layer
        if canonical not in self._PATCH_SCORE_LAYERS:
            raise ValueError(
                f"unsupported ViT patch score layer: {score_layer!r}; "
                f"choose from {self.patch_score_layer_candidates()} or 'final'."
            )
        state = self.prepare_attack_feature_state(x)
        tokens = torch.cat((state.context["prefix_tokens"], state.local_tokens), dim=1)
        block_count = self._PATCH_SCORE_LAYERS[canonical]
        for block in self.model.blocks[:block_count]:
            tokens = block(tokens)
        features = PatchScoreFeatures(
            local_tokens=tokens[:, -state.local_tokens.size(1):],
            global_token=tokens[:, :1],
            grid_size=state.grid_size,
            source_name=f"blocks[{block_count - 1}]",
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
            raise ValueError("replacement ViT local tokens do not match the attack state.")
        tokens = torch.cat((state.context["prefix_tokens"], local_tokens), dim=1)
        tokens = self.model.blocks(tokens)
        return self.model.forward_head(self.model.norm(tokens))

    def progressive_checkpoint_candidates(self) -> tuple[str, ...]:
        return self._PROGRESSIVE_LAYERS

    def default_progressive_checkpoints(self) -> tuple[str, ...]:
        return self._DEFAULT_PROGRESSIVE_LAYERS

    def begin_progressive_forward(self, x: torch.Tensor) -> ProgressiveAttackState:
        initial = self.prepare_attack_feature_state(x)
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
) -> ViTWithHook:
    return ViTWithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
