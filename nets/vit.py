from __future__ import annotations

import torch

from .base import (
    AttackFeatureState,
    DEFAULT_PRETRAINED,
    PatchScoreActivationCapture,
    PatchScoreFeatures,
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

    def patch_score_activation_capture(self, score_layer: str):
        canonical = "block12" if score_layer == "final" else score_layer
        if canonical not in self._PATCH_SCORE_LAYERS:
            raise ValueError(f"unsupported ViT patch score layer: {score_layer!r}")
        block_count = self._PATCH_SCORE_LAYERS[canonical]
        if block_count < len(self.model.blocks):
            return PatchScoreActivationCapture(
                module=self.model.blocks[block_count],
                hook_type="input",
                source_name=(
                    f"blocks[{block_count}] input (=blocks[{block_count - 1}] output)"
                ),
            )
        return PatchScoreActivationCapture(
            module=self.model.blocks[-1],
            hook_type="input",
            source_name="blocks[11] input (logit-connected fallback)",
        )

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


def build_vit_model(
    num_classes: int,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: bool = DEFAULT_PRETRAINED,
    device=None,
) -> ViTWithHook:
    return ViTWithHook(model_name=model_name, num_classes=num_classes, pretrained=pretrained, device=device)
