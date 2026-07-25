"""Isolated frequency-pair attack experiment.

This module deliberately subclasses the retained mainline attacker instead of
adding another branch to ``attack.py``.  The production default therefore
remains the original/phase-shift pair.  Each augmentation group in this
experiment reuses one patch-score mask for two complementary views:

* a Gaussian low-pass view; and
* a natural-image high-frequency-enhanced view.

Both views retain the mainline's initial-projection, kept-only feature noise.
"""

from __future__ import annotations

import argparse
import json
import math
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


DEFAULT_SEED = 20260720


class FrequencyPairAttacker(PatchScoreAttacker):
    """Patch-score attack with a low/high-frequency view pair.

    ``frequency_low_residual_scale=0`` makes the low view a pure Gaussian
    low-pass image.  ``frequency_high_residual_scale=2`` constructs an
    unsharp, but still natural-image-based, high-frequency view as
    ``low + 2 * high``.  The decomposition itself is complementary before
    clamping: ``pixels == low + high``.
    """

    def __init__(
        self,
        *args,
        frequency_sigma: float = 2.0,
        frequency_low_residual_scale: float = 0.0,
        frequency_high_residual_scale: float = 2.0,
        **kwargs,
    ) -> None:
        if frequency_sigma <= 0:
            raise ValueError("frequency_sigma must be positive.")
        if not 0.0 <= frequency_low_residual_scale <= 1.0:
            raise ValueError("frequency_low_residual_scale must be in [0, 1].")
        if frequency_high_residual_scale < 1.0:
            raise ValueError("frequency_high_residual_scale must be at least 1.")
        if "attack_method" in kwargs:
            raise ValueError("FrequencyPairAttacker fixes its isolated attack method internally.")
        kwargs["attack_method"] = "original_score_postdrop_phase_pair"
        super().__init__(*args, **kwargs)
        self.frequency_sigma = float(frequency_sigma)
        self.frequency_low_residual_scale = float(frequency_low_residual_scale)
        self.frequency_high_residual_scale = float(frequency_high_residual_scale)
        self._frequency_filter_cache: dict[
            tuple[int, int, torch.device, torch.dtype], torch.Tensor
        ] = {}
        self._frequency_diagnostics = {
            "low_high_gradient_cosine": [],
            "high_to_low_gradient_norm_ratio": [],
        }

    def _frequency_lowpass_filter(self, pixels: torch.Tensor) -> torch.Tensor:
        height, width = pixels.shape[-2:]
        key = (height, width, pixels.device, pixels.dtype)
        cached = self._frequency_filter_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(height, device=pixels.device, dtype=pixels.dtype)[:, None]
        fx = torch.fft.rfftfreq(width, device=pixels.device, dtype=pixels.dtype)[None, :]
        radius_squared = fy.square() + fx.square()
        transfer = torch.exp(
            -2.0 * math.pi**2 * self.frequency_sigma**2 * radius_squared
        ).view(1, 1, height, width // 2 + 1)
        self._frequency_filter_cache[key] = transfer
        return transfer

    def _frequency_components(
        self,
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable complementary Gaussian low/high components."""
        if pixels.ndim != 4:
            raise ValueError("pixels must have shape [B,C,H,W].")
        spectrum = torch.fft.rfft2(pixels, norm="ortho")
        low = torch.fft.irfft2(
            spectrum * self._frequency_lowpass_filter(pixels),
            s=pixels.shape[-2:],
            norm="ortho",
        )
        return low, pixels - low

    def _frequency_view(self, pixels: torch.Tensor, residual_scale: float) -> torch.Tensor:
        low, high = self._frequency_components(pixels)
        return torch.clamp(low + residual_scale * high, 0.0, 1.0)

    def _iter_original_score_postdrop_phase_pair(
        self,
        pixels: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield a shared-mask low/high pair without touching mainline code."""
        for group_index in range(self.input_diversity_groups):
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(group=group_index, view=-1)
            drop_mask, grid_size = self._compute_mainline_drop_mask(pixels, labels)
            image_mask = self._patch_drop_mask_to_image(
                drop_mask,
                grid_size,
                pixels.size(-2),
                pixels.size(-1),
            ).to(device=pixels.device, dtype=pixels.dtype)

            low_view = self._frequency_view(
                pixels,
                self.frequency_low_residual_scale,
            )
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=0)
            self._actual_forward_view_count += 1
            yield low_view * (1.0 - image_mask), image_mask

            # View A's backward pass frees its FFT graph.  Rebuild the
            # decomposition so view B owns an independent autograd graph.
            high_view = self._frequency_view(
                pixels,
                self.frequency_high_residual_scale,
            )
            if self._gradient_replay is not None:
                self._gradient_replay.set_context(view=1)
            self._actual_forward_view_count += 1
            yield high_view * (1.0 - image_mask), image_mask

    def _record_gradient_diagnostics(
        self,
        view_gradients: torch.Tensor,
        final_gradient: torch.Tensor,
    ) -> None:
        super()._record_gradient_diagnostics(view_gradients, final_gradient)
        if view_gradients.size(0) % 2:
            raise ValueError("frequency-pair diagnostics require an even view count.")
        with torch.no_grad():
            low = view_gradients[0::2].flatten(2)
            high = view_gradients[1::2].flatten(2)
            pair_cosine = F.cosine_similarity(low, high, dim=-1)
            norm_ratio = high.norm(dim=-1) / low.norm(dim=-1).clamp_min(1e-12)
            self._frequency_diagnostics["low_high_gradient_cosine"].append(
                float(pair_cosine.mean().cpu())
            )
            self._frequency_diagnostics["high_to_low_gradient_norm_ratio"].append(
                float(norm_ratio.mean().cpu())
            )

    def gradient_diagnostics_summary(self) -> dict[str, float | int]:
        summary = super().gradient_diagnostics_summary()
        for name, values in self._frequency_diagnostics.items():
            if values:
                summary[name] = sum(values) / len(values)
        return summary

    def mainline_metadata(self) -> dict[str, object]:
        metadata = super().mainline_metadata()
        metadata.update(
            {
                "experimental_augmentation": "frequency_low_high_pair",
                "frequency_decomposition": "complementary_fourier_gaussian",
                "frequency_sigma": self.frequency_sigma,
                "frequency_low_residual_scale": self.frequency_low_residual_scale,
                "frequency_high_residual_scale": self.frequency_high_residual_scale,
                "frequency_merge": "raw_gradient_mean",
                "mainline_code_modified": False,
            }
        )
        return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--whitebox-model", choices=WHITEBOX_MODEL_CHOICES, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-attacked-samples", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=16.0 / 255.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--input-diversity-groups", type=int, default=10)
    parser.add_argument("--post-dropout-feature-noise-type", choices=POST_DROPOUT_NOISE_TYPES, required=True)
    parser.add_argument("--post-dropout-feature-noise-strength", type=float, default=0.2)
    parser.add_argument("--frequency-sigma", type=float, default=2.0)
    parser.add_argument("--frequency-low-residual-scale", type=float, default=0.0)
    parser.add_argument("--frequency-high-residual-scale", type=float, default=2.0)
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
        raise ValueError(f"frequency experiment output directory must be empty: {output_dir}")

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
    attacker = FrequencyPairAttacker(
        model=model,
        epsilon=args.epsilon,
        steps=args.steps,
        use_momentum=True,
        momentum_decay=1.0,
        input_diversity=False,
        input_diversity_groups=args.input_diversity_groups,
        input_diversity_views_per_group=2,
        guide_aug_strength=args.post_dropout_feature_noise_strength,
        patch_dropout_ratio=0.3,
        patch_dropout_score_mode="high",
        patch_dropout_sampling_mode="random",
        token_score_cls_noise=True,
        post_dropout_phase_token_noise=True,
        post_dropout_feature_noise_strength=args.post_dropout_feature_noise_strength,
        post_dropout_feature_noise_type=args.post_dropout_feature_noise_type,
        patch_score_layer="final",
        gaussian_alpha=0.0,
        frequency_sigma=args.frequency_sigma,
        frequency_low_residual_scale=args.frequency_low_residual_scale,
        frequency_high_residual_scale=args.frequency_high_residual_scale,
        device=DEVICE,
    )
    replay = GradientReplay(args.seed)
    sample_ids = attack_all_samples(
        dataloader,
        attacker,
        output_dir,
        args.max_attacked_samples,
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
        "experiment": "frequency_pair_attack",
        "whitebox_model": args.whitebox_model,
        "max_attacked_samples": args.max_attacked_samples,
        "epsilon": args.epsilon,
        "step_size": args.epsilon / args.steps,
        "steps": args.steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "input_diversity_groups": args.input_diversity_groups,
        "input_diversity_views_per_group": 2,
        "input_diversity_total_views": 2 * args.input_diversity_groups,
        "patch_dropout_ratio": 0.3,
        "patch_dropout_score_mode": "high",
        "patch_dropout_sampling_mode": "random",
        "token_score_cls_noise": True,
        "post_dropout_feature_noise_type": args.post_dropout_feature_noise_type,
        "post_dropout_feature_noise_strength": args.post_dropout_feature_noise_strength,
        "gaussian_alpha": 0.0,
    }
    params.update(attacker.mainline_metadata())
    (output_dir / "attack_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(f"Running isolated frequency-pair experiment on {DEVICE}")
    main(parse_args())
