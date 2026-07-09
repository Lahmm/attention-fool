"""Connect per-pixel gradient properties to AE perturbation properties.

Runs the current best attack (traj copies=10 step=20 feature loss) on
N samples, captures intermediate gradient/momentum data, then computes
per-pixel correspondence between gradient structure and delta structure.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    ANNOTATIONS_PATH, IMAGE_DIR, create_attacker, parse_model_names,
)
from nets import build_vit_model
from utils import DEVICE, load_data, IMAGENET_MEAN, IMAGENET_STD


DEFAULT_OUTPUT = "outputs/analysis/gradient_perturbation_200_samples"


def _gray(x: torch.Tensor) -> torch.Tensor:
    w = x.new_tensor((0.2989, 0.5870, 0.1140)).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def _sobel(gray: torch.Tensor):
    kx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = gray.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kx)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), ky)
    return gx, gy, (gx.square() + gy.square()).sqrt()


def _fft_band_energy(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """x: [B, H, W] or [H, W] — grayscale image/perturbation."""
    if x.ndim == 2:
        x = x.unsqueeze(0)
    bsz, h, w = x.shape
    fy = torch.fft.fftfreq(h, device=x.device).view(h, 1)
    fx = torch.fft.fftfreq(w, device=x.device).view(1, w)
    radius = (fx.square() + fy.square()).sqrt().view(1, h, w)
    fft = torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho")
    power = fft.abs().square()  # [B, H, W]
    return {
        "low": power[:, radius[0] < 0.08].sum(dim=1),
        "mid": power[:, (radius[0] >= 0.08) & (radius[0] < 0.25)].sum(dim=1),
        "high": power[:, radius[0] >= 0.25].sum(dim=1),
    }


def collect_gradient_data(
    dataloader,
    model,
    attacker,
    max_samples: int,
    seed: int = 42,
) -> list[dict]:
    """Run attack and collect per-step gradient/momentum for each sample."""
    rows = []
    torch.manual_seed(seed)

    attacked = 0
    for images, labels, indices in tqdm(dataloader, desc="collecting gradients"):
        if attacked >= max_samples:
            break

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Manual attack loop with gradient capture
        clean_pixels = attacker._denormalize(images).detach()
        adv_pixels = clean_pixels.clone().detach()
        momentum = torch.zeros_like(adv_pixels)

        # Pre-compute clean feature target
        with torch.no_grad():
            outputs = model(attacker._normalize(clean_pixels), return_tokens=True)
            _, block_tokens = outputs
            layer_idx = attacker.feature_layer
            if layer_idx < 0:
                layer_idx = len(block_tokens) + layer_idx
            clean_feat = block_tokens[layer_idx][:, 1:, :].detach()

        step_grads = []
        step_momentums = []

        for step_idx in range(attacker.steps):
            grad_pixels = adv_pixels.detach().requires_grad_(True)

            # Compute gradient via attacker's pipeline
            if attacker.lowmid_dss_filter:
                grad, term_grads = attacker._attack_grad_terms(
                    grad_pixels, labels, clean_feat
                )
            else:
                grad = attacker._attack_grad(grad_pixels, labels, clean_feat)
                term_grads = None

            grad = attacker._smooth_grad(grad)
            if term_grads is not None:
                term_grads = tuple(attacker._smooth_grad(t) for t in term_grads)
            grad = attacker._apply_lowmid_dss_filter(grad, term_grads)
            grad = attacker._tune_lowmid_gradient(grad)
            grad = attacker._normalize_grad(grad)

            step_grads.append(grad.detach().cpu())

            if attacker.use_momentum:
                momentum = attacker.decay * momentum + grad
                update = momentum
            else:
                update = grad

            step_momentums.append(momentum.detach().cpu())

            with torch.no_grad():
                adv_pixels = adv_pixels + attacker.step_size * update.sign()
                delta_clamp = torch.clamp(adv_pixels - clean_pixels, -attacker.epsilon, attacker.epsilon)
                adv_pixels = torch.clamp(clean_pixels + delta_clamp, 0.0, 1.0).detach()

        # Final results
        final_delta = (adv_pixels - clean_pixels).detach().cpu()
        final_momentum = momentum.detach().cpu()

        # Per-sample storage
        for b in range(images.size(0)):
            if attacked >= max_samples:
                break

            # Compute per-pixel gradient sign consistency across steps
            sample_grads = torch.stack([g[b] for g in step_grads])  # [steps, C, H, W]
            grad_sign_consistency = sample_grads.sign().float().mean(dim=0)  # [C, H, W]
            grad_sign_consistency = grad_sign_consistency.abs() * 2 - 1  # rescale: 0.5→0, 1.0→1

            # Gradient spatial features
            grad_mag = sample_grads.abs().mean(dim=0)  # mean gradient magnitude
            grad_luma = _gray(grad_mag.unsqueeze(0))[0, 0]  # [H, W]
            _, _, grad_edge = _sobel(grad_luma.unsqueeze(0).unsqueeze(0))
            grad_edge = grad_edge[0, 0]  # [H, W]

            # FFT of gradient
            grad_fft = _fft_band_energy(grad_luma)
            grad_fft_total = grad_fft["low"] + grad_fft["mid"] + grad_fft["high"]

            # Perturbation features
            delta_b = final_delta[b]  # [C, H, W]
            delta_luma = _gray(delta_b.unsqueeze(0))[0, 0]  # [H, W]
            _, _, delta_edge = _sobel(delta_luma.unsqueeze(0).unsqueeze(0))
            delta_edge = delta_edge[0, 0]

            delta_fft = _fft_band_energy(delta_luma)
            delta_fft_total = delta_fft["low"] + delta_fft["mid"] + delta_fft["high"]

            # Per-pixel gradient-delta correspondence
            delta_b_abs = delta_b.abs()

            # Correlation: does high-gradient pixel = high-perturbation pixel?
            # grad_mag is [C,H,W], delta_b_abs is [C,H,W]
            grad_flat = grad_mag.flatten()
            delta_flat = delta_b_abs.flatten()
            pixel_corr = torch.corrcoef(torch.stack([grad_flat, delta_flat]))[0, 1].item()

            # Sobel edge alignment: is gradient edge where perturbation edge is?
            grad_edge_flat = grad_edge.flatten()
            delta_edge_flat = delta_edge.flatten()
            edge_align = F.cosine_similarity(
                grad_edge_flat.unsqueeze(0), delta_edge_flat.unsqueeze(0)
            ).item()

            # Sign consistency vs perturbation magnitude
            sign_cons_flat = grad_sign_consistency.flatten()
            delta_abs_flat = delta_b_abs.flatten()
            sign_cons_weighted = (sign_cons_flat.abs() * delta_abs_flat).sum() / delta_abs_flat.sum().clamp_min(1e-12)

            rows.append({
                "grad_low_ratio": float(grad_fft["low"] / grad_fft_total),
                "grad_mid_ratio": float(grad_fft["mid"] / grad_fft_total),
                "grad_high_ratio": float(grad_fft["high"] / grad_fft_total),
                "delta_low_ratio": float(delta_fft["low"] / delta_fft_total),
                "delta_mid_ratio": float(delta_fft["mid"] / delta_fft_total),
                "delta_high_ratio": float(delta_fft["high"] / delta_fft_total),
                "grad_delta_pixel_corr": pixel_corr,
                "grad_delta_edge_align": edge_align,
                "sign_consistency_weighted": float(sign_cons_weighted),
            })
            attacked += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data & build attacker (current best config)
    dataloader, num_classes = load_data(batch_size=args.batch_size, num_workers=4, prefetch_factor=2)
    model = build_vit_model(num_classes=num_classes, model_name="vit_base_patch16_224")

    attacker = create_attacker(
        model=model,
        epsilon=16.0 / 255.0,
        step_size=None,
        steps=20,
        dim=True,
        mi=True,
        mi_decay=1.0,
        guide_aug=True,
        guide_aug_methods=("feature_trajectory_dropout",),
        guide_aug_copies=10,
        guide_aug_strength=0.2,
        attack_loss="feature",
        feature_layer=-2,
        feature_scope="block",
        ti_sigma=0.0,
    )

    print(f"Collecting gradient data for {args.max_samples} samples...")
    rows = collect_gradient_data(dataloader, model, attacker, args.max_samples)

    # Summary statistics
    grad_mid_ratios = [r["grad_mid_ratio"] for r in rows]
    delta_mid_ratios = [r["delta_mid_ratio"] for r in rows]
    grad_delta_corrs = [r["grad_delta_pixel_corr"] for r in rows]
    edge_aligns = [r["grad_delta_edge_align"] for r in rows]
    sign_weights = [r["sign_consistency_weighted"] for r in rows]

    results = {
        "num_samples": len(rows),
        "config": "traj_copies10_step20_feature_layer-2",
        "gradient_fft": {
            "mean_low_ratio": float(np.mean([r["grad_low_ratio"] for r in rows])),
            "mean_mid_ratio": float(np.mean(grad_mid_ratios)),
            "mean_high_ratio": float(np.mean([r["grad_high_ratio"] for r in rows])),
        },
        "delta_fft": {
            "mean_low_ratio": float(np.mean([r["delta_low_ratio"] for r in rows])),
            "mean_mid_ratio": float(np.mean(delta_mid_ratios)),
            "mean_high_ratio": float(np.mean([r["delta_high_ratio"] for r in rows])),
        },
        "grad_delta_correspondence": {
            "pixel_corr_mean": float(np.mean(grad_delta_corrs)),
            "pixel_corr_std": float(np.std(grad_delta_corrs)),
            "edge_align_mean": float(np.mean(edge_aligns)),
            "edge_align_std": float(np.std(edge_aligns)),
            "sign_consistency_weighted_mean": float(np.mean(sign_weights)),
        },
        "cross_correlation": {
            "grad_mid_vs_delta_mid": float(np.corrcoef(grad_mid_ratios, delta_mid_ratios)[0, 1]),
            "grad_mid_vs_edge_align": float(np.corrcoef(grad_mid_ratios, edge_aligns)[0, 1]),
            "sign_cons_vs_pixel_corr": float(np.corrcoef(sign_weights, grad_delta_corrs)[0, 1]),
        },
    }

    # Save
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "per_sample.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Results ===")
    print(f"Gradient FFT: low={results['gradient_fft']['mean_low_ratio']:.4f}, "
          f"mid={results['gradient_fft']['mean_mid_ratio']:.4f}, "
          f"high={results['gradient_fft']['mean_high_ratio']:.4f}")
    print(f"Delta FFT:    low={results['delta_fft']['mean_low_ratio']:.4f}, "
          f"mid={results['delta_fft']['mean_mid_ratio']:.4f}, "
          f"high={results['delta_fft']['mean_high_ratio']:.4f}")
    print(f"Gradient-Delta pixel correlation: {results['grad_delta_correspondence']['pixel_corr_mean']:.4f} ± {results['grad_delta_correspondence']['pixel_corr_std']:.4f}")
    print(f"Gradient-Delta edge alignment:    {results['grad_delta_correspondence']['edge_align_mean']:.4f} ± {results['grad_delta_correspondence']['edge_align_std']:.4f}")
    print(f"Sign consistency weighted:        {results['grad_delta_correspondence']['sign_consistency_weighted_mean']:.4f}")
    print(f"Cross: grad_mid vs delta_mid:     {results['cross_correlation']['grad_mid_vs_delta_mid']:.4f}")
    print(f"Cross: grad_mid vs edge_align:    {results['cross_correlation']['grad_mid_vs_edge_align']:.4f}")
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
