"""Run the mainline attack with gradient observation on a small sample set.

Captures per-view gradients, aggregated gradients, momentum, and sign updates
at every step, then saves structured diagnostics for analysis.

Usage:
    python run_gradient_diagnostics.py --num-samples 10 --seed 20260710
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from attack import PatchScoreAttacker
from gradient_observer import GradientObserver
from main import clear_directory_contents, validate_output_dir
from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from utils import DEVICE, load_data, save_adversarial_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient diagnostic run")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/attack/gradient_diagnostics")
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    output_dir = validate_output_dir(args.output_dir)
    clear_directory_contents(output_dir)

    dataloader, num_classes = load_data(
        batch_size=args.batch_size,
        num_workers=4,
        prefetch_factor=4,
    )
    model = build_whitebox_model(num_classes=num_classes, model_name=DEFAULT_MODEL_NAME)

    attacker = PatchScoreAttacker(
        model=model,
        epsilon=16.0 / 255.0,
        steps=10,
        attack_method="original_score_postdrop_phase_pair",
        use_momentum=True,
        momentum_decay=1.0,
        nesterov=False,
        ti_sigma=0.0,
        input_diversity=False,
        input_diversity_groups=10,
        input_diversity_views_per_group=2,
        input_diversity_phase_shift_set=((4, 4), (8, 8), (12, 12)),
        guide_aug_strength=0.2,
        patch_dropout_ratio=0.3,
        patch_dropout_score_mode="high",
        patch_dropout_sampling_mode="random",
        patch_dropout_noise_mode="opponent_channel_gaussian",
        token_score_cls_noise=True,
        token_score_cls_mode="learned",
        token_score_patch_noise=False,
        post_dropout_phase_token_noise=True,
        feature_layer=12,
        gradient_postprocess="mean",
        device=DEVICE,
    )

    total = min(args.num_samples, len(dataloader.dataset))
    attacked = 0
    saved_count = 0

    print(f"Running gradient diagnostics on {total} samples with seed={args.seed}")
    print(f"Output dir: {output_dir}")

    for images, labels, indices in dataloader:
        if attacked >= total:
            break
        remaining = total - attacked
        images = images[:remaining]
        labels = labels[:remaining]
        indices = indices[:remaining]
        filenames = [
            str(dataloader.dataset.samples[index]["image_name"])
            for index in indices.tolist()
        ]

        observer = GradientObserver(enabled=True)
        adversarial = attacker.attack_batch(images, labels, observer=observer)

        if args.save_images:
            saved = save_adversarial_images(
                images=adversarial,
                output_dir=str(output_dir),
                prefix="adv",
                start_index=saved_count,
                filenames=filenames,
            )
            saved_count += len(saved)

        # Save per-batch diagnostics
        batch_diag_dir = output_dir / f"batch_{attacked:03d}"
        observer.save(batch_diag_dir)

        summary = observer.summarize()
        print(
            f"Batch {attacked:03d} ({images.size(0)} samples): "
            f"view_to_mean_cos={summary.get('view_to_mean_cosine_mean', 'N/A'):.4f} "
            f"view_sign_agree={summary.get('view_sign_agreement_mean', 'N/A'):.4f} "
            f"view_eff_rank={summary.get('view_effective_rank_mean', 'N/A'):.2f} "
            f"low_freq={summary.get('freq_low_freq_frac', 'N/A'):.3f} "
        )

        attacked += images.size(0)

    # Aggregate across all batches
    all_summaries = []
    for batch_dir in sorted(output_dir.glob("batch_*")):
        summary_file = batch_dir / "gradient_summary.json"
        if summary_file.exists():
            all_summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))

    # Compute global averages
    global_summary: dict = {"num_batches": len(all_summaries)}
    if all_summaries:
        scalar_keys = set()
        for s in all_summaries:
            scalar_keys.update(
                k for k, v in s.items() if isinstance(v, (int, float)) and k != "num_steps"
            )
        for key in sorted(scalar_keys):
            vals = [s[key] for s in all_summaries if key in s]
            if vals:
                global_summary[key] = sum(vals) / len(vals)
                global_summary[f"{key}_std"] = (
                    float(torch.tensor(vals).std().item()) if len(vals) > 1 else 0.0
                )

    (output_dir / "global_summary.json").write_text(
        json.dumps(global_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save attack params
    params = {
        "attack_method": "original_score_postdrop_phase_pair",
        "whitebox_model": DEFAULT_MODEL_NAME,
        "num_samples": total,
        "seed": args.seed,
        "epsilon": 16.0 / 255.0,
        "steps": 10,
        "mi": True,
        "mi_decay": 1.0,
        "input_diversity_groups": 10,
        "input_diversity_views_per_group": 2,
        "gradient_postprocess": "mean",
    }
    params.update(attacker.mainline_metadata())
    (output_dir / "attack_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. Results saved to {output_dir}")
    print(f"Global summary: {json.dumps(global_summary, indent=2)}")


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main(parse_args())
