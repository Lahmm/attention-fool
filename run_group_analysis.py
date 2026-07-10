"""Analyze within-group vs between-group gradient correlation.

The 20 views are ordered as:
  Views 0,1 = group 0 (same mask), views 2,3 = group 1, ... views 18,19 = group 9.
This script measures whether views sharing a dropout mask produce more similar
gradients than views with different masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attack import PatchScoreAttacker
from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from utils import DEVICE, load_data


class GroupAwareObserver:
    """Captures per-view gradients with group identity tracking."""

    def __init__(self) -> None:
        self.per_step: list[dict] = []

    def record(self, view_grads: torch.Tensor) -> None:
        """view_grads: [20, B, C, H, W]"""
        flat = view_grads.detach().flatten(2)  # [20, B, D]
        flat = flat / flat.norm(p=2, dim=2, keepdim=True).clamp_min(1e-12)
        # Compute pairwise cosine: [B, V, V]
        V, B, D = flat.shape
        pw = torch.bmm(flat.permute(1, 0, 2), flat.permute(1, 2, 0))  # [B, V, V]
        within_group_cos = []
        between_group_cos = []
        for g in range(10):
            w = pw[:, 2 * g, 2 * g + 1]  # cos(view_a, view_b) within group g
            within_group_cos.append(w)
            for h in range(g + 1, 10):
                # Between group g and h: 4 cross-pairs
                for i in range(2):
                    for j in range(2):
                        b = pw[:, 2 * g + i, 2 * h + j]
                        between_group_cos.append(b)

        within = torch.stack(within_group_cos).mean()  # avg over groups and batch
        between = torch.stack(between_group_cos).mean()
        self.per_step.append({
            "within_group_cos": float(within.cpu()),
            "between_group_cos": float(between.cpu()),
            "ratio": float((within / between.clamp_min(1e-12)).cpu()),
        })

    def summary(self) -> dict:
        if not self.per_step:
            return {}
        within = sum(r["within_group_cos"] for r in self.per_step) / len(self.per_step)
        between = sum(r["between_group_cos"] for r in self.per_step) / len(self.per_step)
        return {"within_group_cos": within, "between_group_cos": between,
                "ratio": within / between if between > 0 else 0, "num_steps": len(self.per_step)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataloader, num_classes = load_data(batch_size=4, num_workers=4, prefetch_factor=4)
    model = build_whitebox_model(num_classes=num_classes, model_name=DEFAULT_MODEL_NAME)

    attacker = PatchScoreAttacker(
        model=model, epsilon=16.0 / 255.0, steps=10,
        attack_method="original_score_postdrop_phase_pair",
        use_momentum=True, momentum_decay=1.0, nesterov=False,
        ti_sigma=0.0, input_diversity=False,
        input_diversity_groups=10, input_diversity_views_per_group=2,
        input_diversity_phase_shift_set=((4, 4), (8, 8), (12, 12)),
        guide_aug_strength=0.2, patch_dropout_ratio=0.3,
        patch_dropout_score_mode="high", patch_dropout_sampling_mode="random",
        patch_dropout_noise_mode="opponent_channel_gaussian",
        token_score_cls_noise=True, token_score_cls_mode="learned",
        token_score_patch_noise=False, post_dropout_phase_token_noise=True,
        feature_layer=12, gradient_postprocess="mean", device=DEVICE,
    )

    observer = GroupAwareObserver()

    # Patch _attack_grad to capture view_gradients
    original_attack_grad = attacker._attack_grad

    def hooked_attack_grad(pixels, labels, observer=None):
        gradients = []
        for loss in attacker._iter_attack_losses(pixels, labels):
            gradients.append(torch.autograd.grad(loss, pixels, retain_graph=False)[0])
        view_gradients = torch.stack(gradients, dim=0)
        observer_capture.record(view_gradients)
        # Aggregate manually without re-iterating losses
        aggregated = attacker._aggregate_gradients(
            view_gradients,
            gradient_postprocess=attacker.gradient_postprocess,
            gradient_consensus_lambda=attacker.gradient_consensus_lambda,
        )
        attacker._record_gradient_diagnostics(view_gradients, aggregated)
        return aggregated

    observer_capture = observer
    attacker._attack_grad = hooked_attack_grad

    total = min(args.num_samples, len(dataloader.dataset))
    attacked = 0
    for images, labels, indices in dataloader:
        if attacked >= total:
            break
        remaining = total - attacked
        images = images[:remaining]
        labels = labels[:remaining]
        attacker.attack_batch(images, labels)
        attacked += images.size(0)

    summary = observer.summary()
    print(f"Within-group cos: {summary['within_group_cos']:.4f}")
    print(f"Between-group cos: {summary['between_group_cos']:.4f}")
    print(f"Ratio (within/between): {summary['ratio']:.2f}x")
    print(f"Num steps: {summary['num_steps']}")

    # Per-step breakdown
    for i, rec in enumerate(observer.per_step):
        print(f"  Step {i}: within={rec['within_group_cos']:.4f} between={rec['between_group_cos']:.4f} ratio={rec['ratio']:.2f}x")


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main()
