"""Per-sample gradient-transfer correlation analysis.

Runs the baseline attack on 50 samples, records per-sample gradient
diagnostics, evaluates transfer ASR on quick black-box models, and
correlates gradient features with transfer success to identify which
gradient structures are associated with better transferability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attack import PatchScoreAttacker
from gradient_observer import GradientObserver
from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from utils import DEVICE, load_data, save_adversarial_images


DEFAULT_VIT_EVAL_MODELS = [
    "levit_256",
    "pit_b_224",
    "deit_base_patch16_224",
    "tnt_s_patch16_224",
    "convit_base",
    "visformer_small",
    "cait_s24_224",
]


def build_quick_blackbox_model(model_name: str):
    import timm
    from timm.data import resolve_data_config, create_transform
    model = timm.create_model(model_name, pretrained=True)
    model.to(DEVICE)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def evaluate_per_sample_transfer(
    adv_dir: Path,
    annotations_path: str,
    num_samples: int,
    model_names: list[str],
) -> dict[str, list[float]]:
    """Return per-sample ASR for the requested black-box models."""
    from transfer_eval import (
        collect_images, build_transfer_samples, pre_cache_tensors,
        load_annotations,
    )
    from torch.utils.data import DataLoader, TensorDataset

    annotations = load_annotations(Path(annotations_path))
    image_paths = sorted(collect_images(adv_dir, "adv_"))[:num_samples]
    results: dict[str, list[float]] = {}

    for model_name in model_names:
        try:
            model, transform = build_quick_blackbox_model(model_name)
        except Exception as exc:
            print(f"  skip {model_name}: {exc}")
            results[model_name] = [0.0] * len(image_paths)
            continue

        samples, skipped = build_transfer_samples(image_paths, annotations, "adv_")
        if not samples:
            results[model_name] = [0.0] * len(image_paths)
            continue

        images_cached, labels_cached = pre_cache_tensors(samples, transform, num_workers=4)
        per_sample_asr = []
        with torch.inference_mode():
            for i in range(len(images_cached)):
                img = images_cached[i:i+1].to(DEVICE)
                tgt = labels_cached[i:i+1].to(DEVICE)
                logits = model(img)
                pred = logits.argmax(dim=1).item()
                per_sample_asr.append(1.0 if pred != tgt.item() else 0.0)
        results[model_name] = per_sample_asr

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-dir", default="outputs/attack/per_sample_correlation")
    parser.add_argument(
        "--eval-models",
        default=",".join(DEFAULT_VIT_EVAL_MODELS),
        help="Comma-separated timm models used for per-sample transfer analysis.",
    )
    args = parser.parse_args()
    eval_models = [name.strip() for name in args.eval_models.split(",") if name.strip()]
    if not eval_models:
        parser.error("--eval-models must contain at least one model")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataloader, num_classes = load_data(batch_size=1, num_workers=4, prefetch_factor=4)
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
    per_sample_records = []

    print(f"Running per-sample analysis on {total} samples...")
    for batch_idx, (images, labels, indices) in enumerate(dataloader):
        if batch_idx >= total:
            break
        filenames = [str(dataloader.dataset.samples[indices[0].item()]["image_name"])]

        observer = GradientObserver(enabled=True)
        adversarial = attacker.attack_batch(images, labels, observer=observer)

        save_adversarial_images(
            images=adversarial,
            output_dir=str(output_dir),
            prefix="adv",
            start_index=batch_idx,
            filenames=filenames,
        )

        # Per-step gradient features (averaged over 10 steps)
        summary = observer.summarize()
        record = {
            "sample_idx": batch_idx,
            "filename": filenames[0],
            "true_label": labels[0].item(),
        }
        # Flatten per-step records
        for step_idx, step_rec in enumerate(observer._records):
            for key, val in step_rec.items():
                if isinstance(val, (int, float)):
                    record[f"step{step_idx}_{key}"] = val
        # Add summary
        for key, val in summary.items():
            if isinstance(val, (int, float)):
                record[f"summary_{key}"] = val

        per_sample_records.append(record)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {batch_idx + 1}/{total} samples")

    # Evaluate transfer per sample
    print("Evaluating per-sample transfer...")
    transfer_results = evaluate_per_sample_transfer(
        output_dir,
        "data/image_name_to_class_id_and_name.json",
        total,
        eval_models,
    )

    # Merge transfer results
    for i, rec in enumerate(per_sample_records):
        for model_name in eval_models:
            if model_name in transfer_results and i < len(transfer_results[model_name]):
                rec[f"transfer_{model_name}"] = transfer_results[model_name][i]
            else:
                rec[f"transfer_{model_name}"] = 0.0
        # Overall transfer ASR (average of the requested models)
        rec["transfer_overall"] = np.mean([
            rec[f"transfer_{m}"] for m in eval_models
        ])

    # Correlation analysis
    print("\n=== Correlation: gradient features vs transfer success ===")
    scalar_features = [
        k for k in per_sample_records[0].keys()
        if k.startswith("summary_") and isinstance(per_sample_records[0][k], (int, float))
    ]
    scalar_features = [k for k in scalar_features if "std" not in k]  # skip std fields

    correlations = []
    for feat in scalar_features:
        feat_vals = np.array([r.get(feat, 0.0) for r in per_sample_records])
        transfer_vals = np.array([r["transfer_overall"] for r in per_sample_records])
        # Remove NaN/Inf
        valid = np.isfinite(feat_vals) & np.isfinite(transfer_vals)
        if valid.sum() < 5:
            continue
        feat_vals = feat_vals[valid]
        transfer_vals = transfer_vals[valid]
        if feat_vals.std() < 1e-12:
            continue
        corr = np.corrcoef(feat_vals, transfer_vals)[0, 1]
        correlations.append((feat, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"{'Feature':<50} {'Correlation':>10}")
    print("-" * 62)
    for feat, corr in correlations[:30]:
        print(f"{feat:<50} {corr:>10.4f}")

    # Step-level correlations
    print("\n=== Step-level correlations ===")
    step_features = [
        k for k in per_sample_records[0].keys()
        if any(k.startswith(f"step{s}_") for s in range(10))
        and isinstance(per_sample_records[0][k], (int, float))
    ]
    # Group by feature name (without step prefix)
    step_corrs = {}
    for feat in step_features:
        # Extract base name after stepX_
        parts = feat.split("_", 2)  # ["step0", "view_pairwise_cosine_mean"]
        if len(parts) < 3:
            continue
        base = parts[2]
        if base not in step_corrs:
            step_corrs[base] = []
        # Average correlation across all 10 steps
        feat_vals = np.array([r.get(feat, 0.0) for r in per_sample_records])
        transfer_vals = np.array([r["transfer_overall"] for r in per_sample_records])
        valid = np.isfinite(feat_vals) & np.isfinite(transfer_vals)
        if valid.sum() < 5 or feat_vals[valid].std() < 1e-12:
            continue
        corr = np.corrcoef(feat_vals[valid], transfer_vals[valid])[0, 1]
        step_corrs[base].append(corr)

    for base, corrs in sorted(step_corrs.items(), key=lambda x: abs(np.mean(x[1])), reverse=True):
        if len(corrs) >= 3:  # at least 3 steps have this feature
            print(f"{base:<45} mean_corr={np.mean(corrs):.4f} trend={'↑' if corrs[-1]>corrs[0] else '↓'}")

    # Save all records
    (output_dir / "per_sample_records.json").write_text(
        json.dumps(per_sample_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "correlations.json").write_text(
        json.dumps([{"feature": f, "correlation": c} for f, c in correlations], indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main()
