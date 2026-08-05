"""Isolated original/patch-shuffle two-view transfer attack.

This experiment retains the historical dynamic mainline's 10 x 2 stochastic
gradient estimator, opponent-projected initial-feature noise, Gaussian
residual option, and MI-FGSM update.  By default, every shuffle-view input
gradient is L2-matched to its paired original-view gradient before averaging.
It deliberately removes patch scoring, patch selection, patch dropping,
routing layers, and phase shifts.

For every sample, attack step, and augmentation group, the two views are:

* the current, spatially unchanged adversarial image; and
* the same current image with all patches on a fixed 14 x 14 input grid
  independently permuted for that sample/step/group.

The production attack in ``attack.py`` is not modified by this experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack import POST_DROPOUT_NOISE_TYPES, PatchScoreAttacker
from gradient_replay import GradientReplay
from main import attack_all_samples, validate_output_dir
from nets import DEFAULT_MODEL_NAME, WHITEBOX_MODEL_CHOICES, build_whitebox_model
from utils import DEVICE, load_data


DEFAULT_SEED = 20260805
SHUFFLE_GRID = (14, 14)


class PatchShufflePairAttacker(PatchScoreAttacker):
    """MI-FGSM with an original view and an independently patch-shuffled view."""

    def __init__(
        self,
        *args,
        shuffle_grid: tuple[int, int] = SHUFFLE_GRID,
        shuffle_gradient_norm_match: bool = True,
        **kwargs,
    ) -> None:
        if "attack_method" in kwargs:
            raise ValueError("PatchShufflePairAttacker fixes its isolated attack method internally.")
        if tuple(shuffle_grid) != SHUFFLE_GRID:
            raise ValueError("the isolated attack uses the fixed 14 x 14 input shuffle grid.")

        # ``none`` prevents every production patch-score/fixed-mask path from
        # running.  This subclass owns its view iterator and attack loop below.
        kwargs["attack_method"] = "none"
        super().__init__(*args, **kwargs)
        if self.input_diversity_views_per_group != 2:
            raise ValueError("the patch-shuffle pair requires exactly two views per group.")
        if self.input_diversity:
            raise ValueError("the isolated patch-shuffle pair does not combine with DIM.")
        if not self.post_dropout_phase_token_noise:
            raise ValueError("both patch-shuffle views must receive initial-feature noise.")
        if self.post_dropout_feature_noise_strength <= 0:
            raise ValueError("patch-shuffle feature-noise strength must be positive.")

        self.shuffle_grid = SHUFFLE_GRID
        self.shuffle_gradient_norm_match = bool(shuffle_gradient_norm_match)
        self._shuffle_patch_size: tuple[int, int] | None = None
        self._shuffle_pair_diagnostics = {
            "original_shuffle_gradient_cosine": [],
            "shuffle_to_original_gradient_norm_ratio": [],
            "applied_shuffle_gradient_l2_scale": [],
            "post_scale_shuffle_to_original_gradient_norm_ratio": [],
        }

    def _sample_patch_permutation(
        self,
        count: int,
        sample_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self._gradient_replay is not None:
            return self._gradient_replay.randperm(
                count,
                "patch_shuffle",
                sample_index,
                device=device,
            )
        return torch.randperm(count, device=device)

    def _shuffle_image_patches(self, pixels: torch.Tensor) -> torch.Tensor:
        """Permute intact RGB patches independently for every batch sample."""
        if pixels.ndim != 4:
            raise ValueError("pixels must have shape [B,C,H,W].")
        batch, channels, height, width = pixels.shape
        grid_h, grid_w = self.shuffle_grid
        if height % grid_h or width % grid_w:
            raise ValueError(
                f"image size {(height, width)} must be divisible by the fixed "
                f"shuffle grid {self.shuffle_grid}."
            )
        patch_h, patch_w = height // grid_h, width // grid_w
        self._shuffle_patch_size = (patch_h, patch_w)
        patch_count = grid_h * grid_w

        patches = pixels.reshape(
            batch,
            channels,
            grid_h,
            patch_h,
            grid_w,
            patch_w,
        ).permute(0, 2, 4, 1, 3, 5).reshape(
            batch,
            patch_count,
            channels,
            patch_h,
            patch_w,
        )
        shuffled = torch.stack(
            [
                patches[index, self._sample_patch_permutation(patch_count, index, pixels.device)]
                for index in range(batch)
            ],
            dim=0,
        )
        return shuffled.reshape(
            batch,
            grid_h,
            grid_w,
            channels,
            patch_h,
            patch_w,
        ).permute(0, 3, 1, 4, 2, 5).reshape_as(pixels)

    def _attack_loss_for_patch_shuffle_view(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        prepare = getattr(self.model, "prepare_attack_feature_state", None)
        resume = getattr(self.model, "forward_from_attack_feature_state", None)
        if prepare is None or resume is None:
            raise ValueError("the patch-shuffle attack requires a resumable white-box adapter.")

        state = prepare(self._normalize(pixels))
        state.validate()
        # There is no dropped region.  A zero image mask makes the retained
        # noise builder inject RMS-matched noise into every initial local token.
        no_drop_mask = torch.zeros(
            pixels.size(0),
            1,
            pixels.size(-2),
            pixels.size(-1),
            device=pixels.device,
            dtype=pixels.dtype,
        )
        noise = self._build_post_dropout_feature_noise(
            state.local_tokens,
            state,
            no_drop_mask,
        )
        logits = resume(state, state.local_tokens + noise)
        return F.cross_entropy(logits, labels)

    def _iter_patch_shuffle_pair(
        self,
        pixels: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        """Yield original/shuffled pairs with fresh group-scoped randomness."""
        for group_index in range(self.input_diversity_groups):
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(group=group_index, view=0)
            self._actual_forward_view_count += 1
            yield pixels

            if self._gradient_replay is not None:
                self._gradient_replay.set_context(group=group_index, view=1)
            shuffled = self._shuffle_image_patches(pixels)
            self._actual_forward_view_count += 1
            yield shuffled

    def _iter_attack_losses(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        for view_pixels in self._iter_patch_shuffle_pair(pixels):
            yield self._attack_loss_for_patch_shuffle_view(view_pixels, labels)

    def _norm_match_shuffle_gradients(
        self,
        view_gradients: torch.Tensor,
    ) -> torch.Tensor:
        """Match each shuffle gradient's L2 norm to its paired original gradient."""
        if view_gradients.ndim != 5:
            raise ValueError(
                "view_gradients must have shape [num_views,batch,channels,height,width]."
            )
        if view_gradients.size(0) != self._expected_view_count():
            raise ValueError("patch-shuffle norm matching received an invalid view count.")
        original = view_gradients[0::2]
        shuffled = view_gradients[1::2]
        original_norm = original.flatten(2).norm(dim=-1)
        shuffled_norm = shuffled.flatten(2).norm(dim=-1).clamp_min(1e-12)
        scale = (original_norm / shuffled_norm).detach()
        matched_shuffled = shuffled * scale[..., None, None, None]
        return torch.stack((original, matched_shuffled), dim=1).reshape_as(view_gradients)

    def _aggregate_patch_shuffle_gradients(
        self,
        view_gradients: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the historical raw mean and the production aggregation."""
        raw_mean = self._aggregate_gradients(view_gradients)
        if not self.shuffle_gradient_norm_match:
            return raw_mean, raw_mean
        matched_views = self._norm_match_shuffle_gradients(view_gradients)
        return raw_mean, self._aggregate_gradients(matched_views)

    def _attack_grad(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        gradients = [
            torch.autograd.grad(loss, pixels, retain_graph=False)[0]
            for loss in self._iter_attack_losses(pixels, labels)
        ]
        if len(gradients) != self._expected_view_count():
            raise RuntimeError(
                f"view count mismatch: {len(gradients)} != {self._expected_view_count()}."
            )
        view_gradients = torch.stack(gradients, dim=0)
        _, aggregated = self._aggregate_patch_shuffle_gradients(view_gradients)
        self._record_gradient_diagnostics(view_gradients, aggregated)
        return aggregated

    def _record_gradient_diagnostics(
        self,
        view_gradients: torch.Tensor,
        final_gradient: torch.Tensor,
    ) -> None:
        super()._record_gradient_diagnostics(view_gradients, final_gradient)
        if view_gradients.size(0) != 2 * self.input_diversity_groups:
            raise ValueError("patch-shuffle diagnostics received an invalid view count.")
        with torch.no_grad():
            original = view_gradients[0::2].flatten(2)
            shuffled = view_gradients[1::2].flatten(2)
            pair_cosine = F.cosine_similarity(original, shuffled, dim=-1)
            original_norm = original.norm(dim=-1).clamp_min(1e-12)
            shuffled_norm = shuffled.norm(dim=-1).clamp_min(1e-12)
            norm_ratio = shuffled_norm / original_norm
            norm_match_scale = original_norm / shuffled_norm
            applied_scale = (
                norm_match_scale
                if self.shuffle_gradient_norm_match
                else torch.ones_like(norm_match_scale)
            )
            matched_ratio = shuffled_norm * applied_scale / original_norm
            self._shuffle_pair_diagnostics["original_shuffle_gradient_cosine"].append(
                float(pair_cosine.mean().cpu())
            )
            self._shuffle_pair_diagnostics[
                "shuffle_to_original_gradient_norm_ratio"
            ].append(float(norm_ratio.mean().cpu()))
            self._shuffle_pair_diagnostics["applied_shuffle_gradient_l2_scale"].append(
                float(applied_scale.mean().cpu())
            )
            self._shuffle_pair_diagnostics[
                "post_scale_shuffle_to_original_gradient_norm_ratio"
            ].append(float(matched_ratio.mean().cpu()))

    def gradient_diagnostics_summary(self) -> dict[str, float | int]:
        summary = super().gradient_diagnostics_summary()
        for name, values in self._shuffle_pair_diagnostics.items():
            if values:
                summary[name] = sum(values) / len(values)
        return summary

    def _expected_view_count(self) -> int:
        return 2 * self.input_diversity_groups

    def probe_attack_gradients(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        *,
        replay: GradientReplay | None = None,
        sample_ids: list[str] | None = None,
        step_index: int = 0,
    ) -> dict[str, torch.Tensor]:
        if replay is not None:
            if sample_ids is None or len(sample_ids) != pixels.size(0):
                raise ValueError("sample_ids must match pixels when replay is enabled.")
            replay.begin_batch(sample_ids)
            replay.set_context(step=step_index, group=-1, view=-1)
        self._gradient_replay = replay
        self._actual_forward_view_count = 0
        probe_pixels = pixels.to(self.device).detach().requires_grad_(True)
        labels = labels.to(self.device)
        try:
            gradients = [
                torch.autograd.grad(loss, probe_pixels, retain_graph=False)[0]
                for loss in self._iter_attack_losses(probe_pixels, labels)
            ]
            if len(gradients) != self._expected_view_count():
                raise RuntimeError(
                    f"view count mismatch: {len(gradients)} != {self._expected_view_count()}."
                )
            view_gradients = torch.stack(gradients, dim=0)
            raw_mean, aggregated = self._aggregate_patch_shuffle_gradients(view_gradients)
            self._record_gradient_diagnostics(view_gradients, aggregated)
            processed = self._smooth_grad(self._apply_gaussian_residual(aggregated))
            return {
                "view_gradients": view_gradients.detach(),
                "raw_mean": raw_mean.detach(),
                "aggregated": aggregated.detach(),
                "norm_matched_mean": aggregated.detach(),
                "processed": processed.detach(),
            }
        finally:
            self._gradient_replay = None

    def attack_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        replay: GradientReplay | None = None,
        sample_ids: list[str] | None = None,
    ) -> torch.Tensor:
        if replay is not None:
            if sample_ids is None or len(sample_ids) != images.size(0):
                raise ValueError("sample_ids must match the batch when replay is enabled.")
            replay.begin_batch(sample_ids)
        self._gradient_replay = replay
        images = images.to(self.device)
        labels = labels.to(self.device)
        clean_pixels = self._denormalize(images).detach()
        adv_pixels = clean_pixels.clone()
        momentum = torch.zeros_like(adv_pixels)
        try:
            for step_index in range(self.steps):
                if replay is not None:
                    replay.set_context(step=step_index, group=-1, view=-1)
                self._actual_forward_view_count = 0
                grad_pixels = adv_pixels.detach()
                if self.nesterov and step_index > 0:
                    grad_pixels = grad_pixels + self.decay * self.step_size * momentum.sign()
                    delta = torch.clamp(grad_pixels - clean_pixels, -self.epsilon, self.epsilon)
                    grad_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0)
                grad_pixels = grad_pixels.detach().requires_grad_(True)

                gradient = self._attack_grad(grad_pixels, labels)
                gradient = self._apply_gaussian_residual(gradient)
                gradient = self._smooth_grad(gradient)
                if self._actual_forward_view_count != self._expected_view_count():
                    raise RuntimeError(
                        "view count mismatch: "
                        f"{self._actual_forward_view_count} != {self._expected_view_count()}."
                    )

                if self.use_momentum:
                    momentum = self.decay * momentum + gradient
                    update = momentum
                else:
                    update = gradient
                with torch.no_grad():
                    self._gradient_diagnostics["mi_cumulative_cosine"].append(
                        float(F.cosine_similarity(momentum, gradient, dim=1).mean().cpu())
                        if self.use_momentum
                        else 1.0
                    )

                adv_pixels = adv_pixels + self.step_size * update.sign()
                delta = torch.clamp(adv_pixels - clean_pixels, -self.epsilon, self.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta, 0.0, 1.0).detach()

            return self._normalize_output(adv_pixels)
        finally:
            self._gradient_replay = None

    def mainline_metadata(self) -> dict[str, object]:
        return {
            "experiment": "patch_shuffle_pair_attack",
            "experimental_augmentation": "original_patch_shuffle_pair",
            "mainline_code_modified": False,
            "patch_score": "disabled",
            "patch_drop": "disabled",
            "phase_shift": "disabled",
            "view_0": "current_adversarial_spatially_unchanged",
            "view_1": "current_adversarial_patch_permuted",
            "shuffle_grid": list(self.shuffle_grid),
            "shuffle_patch_size": (
                list(self._shuffle_patch_size) if self._shuffle_patch_size is not None else None
            ),
            "shuffle_scope": "per_sample_per_step_per_group",
            "shuffle_event": "patch_shuffle",
            "input_diversity_groups": self.input_diversity_groups,
            "input_diversity_views_per_group": 2,
            "input_diversity_total_views": self._expected_view_count(),
            "feature_noise_type": self._feature_noise_type,
            "feature_noise_position": "initial_rgb_projection",
            "feature_noise_scope": "all_initial_local_tokens",
            "post_dropout_feature_noise_type": self.post_dropout_feature_noise_type,
            "post_dropout_feature_noise_strength": self.post_dropout_feature_noise_strength,
            "shuffle_gradient_norm_match": self.shuffle_gradient_norm_match,
            "shuffle_gradient_norm": "per_sample_per_group_input_l2",
            "shuffle_gradient_norm_anchor": "paired_original_view",
            "gradient_aggregation": (
                "pairwise_shuffle_l2_match_then_arithmetic_mean"
                if self.shuffle_gradient_norm_match
                else "raw_arithmetic_mean"
            ),
            "gaussian_sigma": self.gaussian_sigma,
            "gaussian_alpha": self.gaussian_alpha,
            "model_mean": self.model_mean.flatten().tolist(),
            "model_std": self.model_std.flatten().tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whitebox-model", choices=WHITEBOX_MODEL_CHOICES, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-attacked-samples", type=int, default=500)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--input-diversity-groups", type=int, default=10)
    parser.add_argument(
        "--post-dropout-feature-noise-type",
        choices=POST_DROPOUT_NOISE_TYPES,
        default="opponent_projected",
    )
    parser.add_argument("--post-dropout-feature-noise-strength", type=float, default=0.2)
    parser.add_argument(
        "--shuffle-gradient-norm-match",
        dest="shuffle_gradient_norm_match",
        action="store_true",
        help="Match every shuffle-view input-gradient L2 norm to its paired original view.",
    )
    parser.add_argument(
        "--no-shuffle-gradient-norm-match",
        dest="shuffle_gradient_norm_match",
        action="store_false",
        help="Reproduce the historical unbalanced raw arithmetic mean.",
    )
    parser.set_defaults(shuffle_gradient_norm_match=True)
    parser.add_argument("--gaussian-sigma", type=float, default=4.0)
    parser.add_argument("--gaussian-alpha", type=float, default=0.75)
    parser.add_argument("--image-dir", default="data/clean_resized_images")
    parser.add_argument("--annotations-path", default="data/image_name_to_class_id_and_name.json")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.max_attacked_samples <= 0:
        raise ValueError("max-attacked-samples must be positive.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = validate_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"patch-shuffle output directory must be empty: {output_dir}")

    dataloader, num_classes = load_data(
        image_dir_arg=args.image_dir,
        annotations_path_arg=args.annotations_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    model = build_whitebox_model(
        num_classes=num_classes,
        model_name=args.whitebox_model,
    )
    attacker = PatchShufflePairAttacker(
        model=model,
        epsilon=args.epsilon,
        step_size=args.step_size,
        steps=args.steps,
        use_momentum=True,
        momentum_decay=1.0,
        nesterov=False,
        ti_sigma=0.0,
        input_diversity=False,
        input_diversity_groups=args.input_diversity_groups,
        input_diversity_views_per_group=2,
        guide_aug_strength=args.post_dropout_feature_noise_strength,
        token_score_cls_noise=False,
        token_score_patch_noise=False,
        post_dropout_phase_token_noise=True,
        post_dropout_feature_noise_strength=args.post_dropout_feature_noise_strength,
        post_dropout_feature_noise_type=args.post_dropout_feature_noise_type,
        shuffle_gradient_norm_match=args.shuffle_gradient_norm_match,
        gaussian_sigma=args.gaussian_sigma,
        gaussian_alpha=args.gaussian_alpha,
        device=DEVICE,
    )
    replay = GradientReplay(args.seed)
    sample_ids = attack_all_samples(
        dataloader,
        attacker,
        output_dir,
        args.max_attacked_samples,
        sample_offset=args.sample_offset,
        replay=replay,
    )
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(replay.manifest(sample_ids), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "gradient_diagnostics.json").write_text(
        json.dumps(attacker.gradient_diagnostics_summary(), indent=2),
        encoding="utf-8",
    )
    params = {
        "whitebox_model": args.whitebox_model,
        "max_attacked_samples": args.max_attacked_samples,
        "sample_offset": args.sample_offset,
        "epsilon": args.epsilon,
        "step_size": args.step_size if args.step_size is not None else args.epsilon / args.steps,
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "mi": True,
        "mi_decay": 1.0,
        "ni": False,
        "dim": False,
        "ti_sigma": 0.0,
    }
    params.update(attacker.mainline_metadata())
    (output_dir / "attack_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(f"Running isolated patch-shuffle pair attack on {DEVICE}")
    main(parse_args())
