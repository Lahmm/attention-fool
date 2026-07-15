from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    ClsTokenMixin,
    DEFAULT_PRETRAINED,
    PatchScoreFeatures,
    WhiteBoxWithHook,
    conv2d_attack_metadata,
    sequential_modules,
)


DEFAULT_MODEL_NAME = "cait_s24_224"


class CaiTS24WithHook(ClsTokenMixin, WhiteBoxWithHook):
    default_model_name = DEFAULT_MODEL_NAME

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

    def _run_encoder(self, local_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = self.model.blocks(local_tokens)
        cls_token = self.model.cls_token.expand(local_tokens.size(0), -1, -1)
        for block in self.model.blocks_token_only:
            cls_token = block(local_tokens, cls_token)
        return local_tokens, cls_token

    def extract_patch_score_features(self, x: torch.Tensor) -> PatchScoreFeatures:
        state = self.prepare_attack_feature_state(x)
        local_tokens, cls_token = self._run_encoder(state.local_tokens)
        features = PatchScoreFeatures(
            local_tokens=local_tokens,
            global_token=cls_token,
            grid_size=state.grid_size,
            source_name="blocks[23]+blocks_token_only[1]",
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


def build_cait_s24_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> CaiTS24WithHook:
    return CaiTS24WithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
