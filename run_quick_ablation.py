"""Fast ablation runner for gradient intervention hypotheses.

Tests each candidate on 20 samples with a fixed seed against the baseline.
Uses 3 quick black-box models (DeiT-B, Inc-v3, ResNet-101) plus white-box
for rapid signal detection.  Only candidates showing promise proceed to full
100-sample transfer eval.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from attack import PatchScoreAttacker
from nets import DEFAULT_MODEL_NAME, build_whitebox_model
from utils import DEVICE, load_data, save_adversarial_images


QUICK_EVAL_MODELS = [
    "deit_base_patch16_224",   # ViT black-box
    "inception_v3",             # CNN black-box
    "resnet101",                # CNN black-box
]


def build_quick_blackbox_model(model_name: str):
    """Lazy import to keep the ablation runner self-contained."""
    import timm
    from timm.data import resolve_data_config, create_transform

    model = timm.create_model(model_name, pretrained=True)
    model.to(DEVICE)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def evaluate_transfer_quick(adv_dir: Path, annotations_path: str, model_names: list[str]) -> dict:
    """Quick transfer eval on a small set of black-box models."""
    from transfer_eval import (
        collect_images,
        build_transfer_samples,
        pre_cache_tensors,
        TransferImageDataset,
        load_annotations,
    )

    annotations = load_annotations(Path(annotations_path))
    image_paths = collect_images(adv_dir, "adv")
    if not image_paths:
        return {}

    results = {}
    for model_name in model_names:
        try:
            model, transform = build_quick_blackbox_model(model_name)
        except Exception as exc:
            print(f"  skip {model_name}: {exc}")
            continue
        samples, skipped = build_transfer_samples(image_paths, annotations, "adv_")
        if not samples:
            continue
        images_cached, labels_cached = pre_cache_tensors(samples, transform, num_workers=4)
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(
            TensorDataset(images_cached, labels_cached),
            batch_size=64,
            shuffle=False,
            num_workers=0,
        )
        correct = 0
        total = 0
        with torch.inference_mode():
            for imgs, tgts in loader:
                imgs = imgs.to(DEVICE)
                tgts = tgts.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == tgts).sum().item()
                total += tgts.size(0)
        asr = 1.0 - correct / total if total > 0 else 0.0
        results[model_name] = asr
        print(f"  {model_name}: ASR={asr:.4f}")
    return results


def run_one_config(
    config: dict,
    num_samples: int,
    seed: int,
    output_base: Path,
) -> tuple[Path, dict]:
    """Run a single attack configuration, return output dir and metrics."""
    name = config["name"]
    output_dir = output_base / name
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataloader, num_classes = load_data(batch_size=8, num_workers=4, prefetch_factor=4)
    model = build_whitebox_model(num_classes=num_classes, model_name=DEFAULT_MODEL_NAME)

    attacker = PatchScoreAttacker(
        model=model,
        epsilon=16.0 / 255.0,
        steps=10,
        attack_method="original_score_postdrop_phase_pair",
        use_momentum=True,
        momentum_decay=config.get("mi_decay", 1.0),
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
        gradient_postprocess=config.get("gradient_postprocess", "mean"),
        gradient_smooth_sigma=config.get("gradient_smooth_sigma", 0.0),
        gradient_divisive_sigma=config.get("gradient_divisive_sigma", 0.0),
        gradient_clip_percentile=config.get("gradient_clip_percentile", 0.0),
        device=DEVICE,
    )

    total = min(num_samples, len(dataloader.dataset))
    attacked = 0
    saved_count = 0
    whitebox_correct = 0
    whitebox_total = 0

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

        adversarial = attacker.attack_batch(images, labels)
        saved = save_adversarial_images(
            images=adversarial,
            output_dir=str(output_dir),
            prefix="adv",
            start_index=saved_count,
            filenames=filenames,
        )
        saved_count += len(saved)

        # White-box evaluation
        with torch.inference_mode():
            wb_logits = model(adversarial.to(DEVICE))
            wb_preds = wb_logits.argmax(dim=1)
            whitebox_correct += (wb_preds == labels.to(DEVICE)).sum().item()
            whitebox_total += labels.size(0)

        attacked += images.size(0)

    whitebox_asr = 1.0 - whitebox_correct / whitebox_total if whitebox_total > 0 else 0.0
    print(f"  [{name}] White-box ASR: {whitebox_asr:.4f} ({whitebox_correct}/{whitebox_total})")

    # Save params
    params = {
        "name": name,
        "attack_method": "original_score_postdrop_phase_pair",
        "whitebox_model": DEFAULT_MODEL_NAME,
        "num_samples": total,
        "seed": seed,
        "epsilon": 16.0 / 255.0,
        "steps": 10,
        "gradient_postprocess": config.get("gradient_postprocess", "mean"),
        "gradient_smooth_sigma": config.get("gradient_smooth_sigma", 0.0),
        "gradient_divisive_sigma": config.get("gradient_divisive_sigma", 0.0),
        "gradient_clip_percentile": config.get("gradient_clip_percentile", 0.0),
        "mi_decay": config.get("mi_decay", 1.0),
        "whitebox_asr": whitebox_asr,
    }
    params.update(attacker.mainline_metadata())
    (output_dir / "attack_params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_dir, params


def main():
    parser = argparse.ArgumentParser(description="Quick ablation of gradient interventions")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-base", default="outputs/attack/quick_ablation")
    args = parser.parse_args()

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Candidate configurations
    configs = [
        {"name": "baseline_mean", "gradient_postprocess": "mean", "mi_decay": 1.0, "gradient_smooth_sigma": 0.0, "gradient_divisive_sigma": 0.0, "gradient_clip_percentile": 0.0},
        # Divisive normalization: reduce spatial concentration
        {"name": "divisive_20", "gradient_postprocess": "mean", "mi_decay": 1.0, "gradient_smooth_sigma": 0.0, "gradient_divisive_sigma": 2.0, "gradient_clip_percentile": 0.0},
        {"name": "divisive_40", "gradient_postprocess": "mean", "mi_decay": 1.0, "gradient_smooth_sigma": 0.0, "gradient_divisive_sigma": 4.0, "gradient_clip_percentile": 0.0},
        # Percentile clipping: reduce kurtosis
        {"name": "clip_005", "gradient_postprocess": "mean", "mi_decay": 1.0, "gradient_smooth_sigma": 0.0, "gradient_divisive_sigma": 0.0, "gradient_clip_percentile": 0.05},
        {"name": "clip_010", "gradient_postprocess": "mean", "mi_decay": 1.0, "gradient_smooth_sigma": 0.0, "gradient_divisive_sigma": 0.0, "gradient_clip_percentile": 0.10},
        # Combined: divisive + clip
        {"name": "div20_clip005", "gradient_postprocess": "mean", "mi_decay": 1.0, "gradient_smooth_sigma": 0.0, "gradient_divisive_sigma": 2.0, "gradient_clip_percentile": 0.05},
        {"name": "div40_clip010", "gradient_postprocess": "mean", "mi_decay": 1.0, "gradient_smooth_sigma": 0.0, "gradient_divisive_sigma": 4.0, "gradient_clip_percentile": 0.10},
    ]

    results = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"  gradient_postprocess={cfg.get('gradient_postprocess', 'mean')}")
        print(f"  mi_decay={cfg.get('mi_decay', 1.0)}")
        print(f"  gradient_smooth_sigma={cfg.get('gradient_smooth_sigma', 0.0)}")
        print(f"  gradient_divisive_sigma={cfg.get('gradient_divisive_sigma', 0.0)}")
        print(f"  gradient_clip_percentile={cfg.get('gradient_clip_percentile', 0.0)}")
        t0 = time.time()

        adv_dir, params = run_one_config(cfg, args.num_samples, args.seed, output_base)
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.0f}s")

        # Quick transfer eval
        print(f"  Transfer eval (quick):")
        transfer = evaluate_transfer_quick(
            adv_dir,
            "data/image_name_to_class_id_and_name.json",
            QUICK_EVAL_MODELS,
        )
        result = {
            "whitebox_asr": params["whitebox_asr"],
            "transfer": transfer,
        }
        # Compute quick averages
        vit_models = [m for m in transfer if m in ("deit_base_patch16_224",)]
        cnn_models = [m for m in transfer if m in ("inception_v3", "resnet101")]
        result["quick_vit_asr"] = sum(transfer[m] for m in vit_models) / len(vit_models) if vit_models else 0
        result["quick_cnn_asr"] = sum(transfer[m] for m in cnn_models) / len(cnn_models) if cnn_models else 0
        result["quick_overall_asr"] = sum(transfer.values()) / len(transfer) if transfer else 0
        results[name] = result

        print(f"  Quick overall: {result['quick_overall_asr']:.4f}")
        print(f"  Quick ViT: {result['quick_vit_asr']:.4f}")
        print(f"  Quick CNN: {result['quick_cnn_asr']:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Config':<25} {'WB ASR':>8} {'Quick Overall':>14} {'Quick ViT':>10} {'Quick CNN':>10}")
    print("-" * 67)
    baseline = results.get("baseline_mean", {})
    bl_overall = baseline.get("quick_overall_asr", 0)
    bl_vit = baseline.get("quick_vit_asr", 0)
    bl_cnn = baseline.get("quick_cnn_asr", 0)
    bl_wb = baseline.get("whitebox_asr", 0)

    for name, r in results.items():
        wb = r.get("whitebox_asr", 0)
        ov = r.get("quick_overall_asr", 0)
        vit = r.get("quick_vit_asr", 0)
        cnn = r.get("quick_cnn_asr", 0)
        d_ov = ov - bl_overall
        print(f"{name:<25} {wb:>8.4f} {ov:>14.4f} ({d_ov:+.4f}) {vit:>10.4f} {cnn:>10.4f}")

    # Save summary
    summary = {
        "baseline": baseline,
        "results": results,
        "configs": [
            {k: v for k, v in cfg.items()}
            for cfg in configs
        ],
    }
    (output_base / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSummary saved to {output_base / 'ablation_summary.json'}")


if __name__ == "__main__":
    print(f"Running on {DEVICE}")
    main()
