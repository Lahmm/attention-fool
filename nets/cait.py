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


DEFAULT_MODEL_NAME = "cait_s24_224"


class CaiTS24WithHook(WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

    _PATCH_SCORE_LAYERS = {
        "block6_gap": 6,
        "block12_gap": 12,
        "block18_gap": 18,
        "block24_gap": 24,
        "block24_class": 24,
    }
    _PROGRESSIVE_LAYERS = tuple(f"block{index}_gap" for index in range(1, 25))
    _DEFAULT_PROGRESSIVE_LAYERS = ("block5_gap", "block17_gap", "block23_gap")

    def _feature_modules(self):
        return sequential_modules(getattr(self.model, "blocks", None))

    def prepare_attack_feature_state(self, x: torch.Tensor) -> AttackFeatureState:
        base = self.model
        local_tokens = base.patch_embed(x)
        grid_size = tuple(int(value) for value in base.patch_embed.grid_size)
        local_tokens = base.pos_drop(local_tokens + base.pos_embed)
        state = AttackFeatureState(
            local_tokens=local_tokens,
            grid_size=grid_size,
            context=None,
            **conv2d_attack_metadata(base.patch_embed.proj),
        )
        state.validate()
        return state

    def patch_score_layer_candidates(self) -> tuple[str, ...]:
        return tuple(self._PATCH_SCORE_LAYERS)

    def _run_encoder(self, local_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = self.model.blocks(local_tokens)
        cls_token = self.model.cls_token.expand(local_tokens.size(0), -1, -1)
        for block in self.model.blocks_token_only:
            cls_token = block(local_tokens, cls_token)
        return local_tokens, cls_token

    def extract_patch_score_features(
        self,
        x: torch.Tensor,
        *,
        score_layer: str = "final",
    ) -> PatchScoreFeatures:
        canonical = "block24_class" if score_layer == "final" else score_layer
        if canonical not in self._PATCH_SCORE_LAYERS:
            raise ValueError(
                f"unsupported CaiT patch score layer: {score_layer!r}; "
                f"choose from {self.patch_score_layer_candidates()} or 'final'."
            )
        state = self.prepare_attack_feature_state(x)
        local_tokens = state.local_tokens
        block_count = self._PATCH_SCORE_LAYERS[canonical]
        for block in self.model.blocks[:block_count]:
            local_tokens = block(local_tokens)
        if canonical == "block24_class":
            cls_token = self.model.cls_token.expand(local_tokens.size(0), -1, -1)
            for block in self.model.blocks_token_only:
                cls_token = block(local_tokens, cls_token)
            global_token = cls_token
            global_mode = "class_attention_cls"
            source_name = "blocks[23]+blocks_token_only[1]"
        else:
            global_token = local_tokens.mean(dim=1, keepdim=True)
            global_mode = "gap"
            source_name = f"blocks[{block_count - 1}]+gap"
        features = PatchScoreFeatures(
            local_tokens=local_tokens,
            global_token=global_token,
            grid_size=state.grid_size,
            source_name=source_name,
            layer_id=canonical,
            global_mode=global_mode,
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
            raise ValueError("replacement CaiT local tokens do not match the attack state.")
        local_tokens, cls_token = self._run_encoder(local_tokens)
        tokens = self.model.norm(torch.cat((cls_token, local_tokens), dim=1))
        return self.model.forward_head(tokens)

    def progressive_checkpoint_candidates(self) -> tuple[str, ...]:
        return self._PROGRESSIVE_LAYERS

    def default_progressive_checkpoints(self) -> tuple[str, ...]:
        return self._DEFAULT_PROGRESSIVE_LAYERS

    def begin_progressive_forward(self, x: torch.Tensor) -> ProgressiveAttackState:
        initial = self.prepare_attack_feature_state(x)
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
        target = int(checkpoint_id.removeprefix("block").removesuffix("_gap"))
        start = int(state.context["block_index"])
        if target <= start:
            raise ValueError("CaiT progressive checkpoints must be strictly increasing.")
        local = state.local_tokens
        for block in self.model.blocks[start:target]:
            local = block(local)
        return ProgressiveAttackState(local, state.grid_size, {"block_index": target})

    def progressive_score_features(
        self, state: ProgressiveAttackState, checkpoint_id: str
    ) -> PatchScoreFeatures:
        features = PatchScoreFeatures(
            local_tokens=state.local_tokens,
            global_token=state.local_tokens.mean(dim=1, keepdim=True),
            grid_size=state.grid_size,
            source_name=f"blocks[{state.context['block_index'] - 1}]+gap",
            layer_id=checkpoint_id,
            global_mode="gap",
        )
        features.validate()
        return features

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
) -> CaiTS24WithHook:
    return CaiTS24WithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
