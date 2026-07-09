# Current Attack Narrative & Bottleneck Analysis

## Best Configuration (as of 2026-07-09)

```
Command:
  python main.py --mode attack --max-attacked-samples 100 --steps 10 \
    --mi --mi-decay 1.0 --attack-loss logits --feature-layer 0 \
    --guide-aug --guide-aug-method token_patch_dropout \
    --guide-aug-copies 20 --guide-aug-strength 0.2 \
    --patch-dropout-ratio 0.3 --patch-dropout-score-mode low \
    --patch-dropout-fill-mode zero_noise \
    --patch-dropout-noise-mode opponent_channel_gaussian \
    --token-score-cls-noise --token-cls-noise-mode gaussian \
    --batch-size 32
```

Results (16/255, ViT-B/16 white-box, 100 ImageNet samples):
  avg=72.45%  ViT=77.00%  CNN=64.50%

Per-model:
  levit_256=71%  pit_b_224=76%  deit_base=77%  tnt_s=76%  convit_base=70%
  visformer_small=75%  cait_s24=92%
  inception_v3=66%  inception_v4=63%  inception_resnet_v2=61%  resnet101=69%
```

## Core Mechanism

### Single-copy flow (within one step, one of 20 copies)

```
adv_pixels
    │
patch_embed (W) → pos_embed → norm_pre → tokens at L=0 [CLS, 196 patches]
    │
    ├── Scoring branch (no gradient):
    │     CLS + gaussian_noise → cos(patch_i, CLS_jittered) → scores
    │     scores < median → candidates → random 30% → drop_mask
    │
    └── Forward branch (with gradient):
         apply drop_mask → zeroed tokens + opponent noise on survivors
         └── blocks[0:12] → norm → head → CE loss → backward
```

### Multi-copy aggregation

```
20 copies → 20 grads (different masks due to CLS jitter) → mean → MI sign → update
10 steps total
```

## Why It Works: The Outlier-Token Narrative

### 1. Darcet et al. (Register Tokens) Foundation

ViT self-attention produces high-L2-norm "outlier tokens" in background/smooth
regions. These tokens carry little classification information but dominate
softmax(QK^T/sqrt(d)) due to their magnitude. This is a structural property of
the softmax attention mechanism, not a learned preference of any specific model.

### 2. Our Empirical Validation

Computed on 50 ImageNet images at L=0 (patch_embed output):

| | Low-score patches (dropped) | High-score patches (kept) |
|---|---|---|
| Mean L2 norm | **13.55** | 11.90 |
| Ratio | **1.14×** | — |
| CLS cosine | 0.11 | 0.14 |

**50/50 images: low-score = high-norm. The patches we drop ARE the outlier tokens.**

### 3. Why Scoring at L=0

- L=0 CLS is a learned embedding vector shared across all images. It encodes a
  universal prior about "which directions matter" from the patch embedding weights.
- cos(patch, CLS₀) measures alignment with this universal prior. Low-score patches
  are those whose features were compressed by W's weak singular directions.
- Deep-layer CLS (L=-1, L=-2) carries image-specific semantics that overfit to
  the white-box model's attention strategy → less transferable.
- L=0 is the ONLY layer where patch tokens are diverse enough for noise to create
  meaningful score variation: mean pairwise patch cos=0.18 at L=0 vs 0.80 at L=-1.
  At deep layers, all patches converge to similar directions → CLS jitter shifts
  all scores in sync → no real dropout diversity.

### 4. Why Dropout (Zero) + Noise (Opponent)

**Dropout (15% of patches zeroed):**
Removes high-norm outlier tokens → self-attention redistributes to informative
patches → gradient forced to cover discriminative regions → perturbation covers
foreground, not just background noise.

**Opponent-channel noise on non-dropped patches:**
Per-pixel covariance C_opp = [[1, -0.25, -0.25], [-0.25, 1, -0.25], [-0.25, -0.25, 1]]
- Luminance variance -50%, chrominance variance +25%
- Suppresses CNN luminance-dominant filters (model-specific), amplifies
  color-opponent filters (universal across ConvNets)
- Noise generated in pixel space, projected to token space via W^T

**CLS Gaussian jitter for scoring:**
- Each of 20 copies jitters CLS₀ with independent Gaussian noise (σ=0.2×token_rms)
- Different CLS → different scores → different dropout masks per copy
- Creates gradient diversity: each copy's gradient sees a different dropout pattern

### 5. Why Transferable

The outlier token problem is a mathematical property of softmax attention.
All ViT architectures share this structural defect. Dropping these tokens
during the attack forces perturbations to work through the "real" information
pathways that are consistent across ViT variants.

The opponent-channel noise addresses CNNs specifically by exploiting the
universality of color-opponent features in ConvNet first layers.

## What Has Been Exhaustively Tested

### Dead Ends (confirmed harmful or zero gain)

| Direction | Result | Detail |
|-----------|--------|--------|
| Spatial correlation in noise | -23.6pp | Cross-patch smoothing overfits attention patterns |
| Deep-layer scoring (L>0) | -4.2pp | Deep patches too homogeneous, CLS semantics model-specific |
| TI smoothing | -22.5pp | Gradient smoothing destroys high-freq signal |
| grad_trim_ratio | ~-1pp | Removing extreme copy gradients hurts CNN |
| Noise strength > 0.2 | -7.8pp | Optimal SNR already found |
| Noise strength < 0.2 | -2.6pp | Insufficient gradient diversity |
| Per-pixel covariance optimization | 0% gain | W·W^T eff_rank=120 is the bottleneck, C only has 6 DoF |
| Token-only noise (no dropout) | -8.3pp | Dropout provides ~6pp of the total gain |
| Random dropout (no CLS scoring) | -1.6pp | CLS scoring provides modest but real gain |
| High-score dropout | -8pp | Dropping foreground always worse than background |
| DIM (Input Diversity) | OFF | Already tested in early experiments, not helpful here |

### Tuned to Optimum

| Parameter | Optimal value | Sensitivity |
|-----------|--------------|-------------|
| guide_aug_strength | 0.2 | ±0.1 causes significant drop |
| patch_dropout_ratio | 0.3 | 0.15 drops ~1.6pp, 0.5 not tested recently |
| patch_dropout_score_mode | low | high drops ~8pp |
| token_cls_noise_mode | gaussian | mahalanobis ~0.6pp worse |
| token_patch_dropout_layer | 0 | Any >0 catastrophic |
| CLS noise strength | 0.2 | Sweep 0.1-0.3 shows 0.2 is peak |

## Current Bottleneck: Analysis

### 1. The Patch Embedding Information Ceiling

The ViT-B/16 patch embedding W ∈ R^(768×768) has:
- Effective rank = 120/768 (only 15.6% of dimensions carry useful signal)
- σ_max²/σ_min² = 27717 (extreme spectral skew)
- 50% of power in top 45 singular vectors

This is a Shannon limit for gradient information flow. Whether noise is added
in pixel space or token space, the gradient ∂L/∂pixels = W^T · ∂L/∂tokens is
constrained to W^T's effective subspace (~120 dimensions).

### 2. ViT-CNN Tradeoff

CNN and ViT have conflicting noise requirements:
- CNN needs per-pixel, per-channel diversity (small receptive fields)
- ViT needs token-space diversity within W's 120-dim effective subspace

Opponent-channel noise partially bridges this (CNN +4.75pp) but the fundamental
tension remains. The ViT bottleneck (W^T eff_rank=120) cannot be broken by
changing the noise distribution — it requires changing the gradient path itself.

### 3. What Would Help

**Proven to work:**
- Increase epsilon: at 24/255, same method achieves 84.4% avg (87% ViT, 79.8% CNN)

**Potentially viable but untested:**
- Multi-white-box ensemble: two ViTs with complementary W matrices → combined
  effective rank could be 150-200 (theoretical, not implemented)
- Gradient path bypass: somehow inject gradient signal that doesn't go through W^T
- Token-space augmentation with gradient diversity loss: explicitly optimize for
  diverse gradients across copies rather than hoping noise creates diversity
- Adaptive CLS jitter strength per-copy or per-step: vary jitter magnitude as
  attack progresses (exploration vs exploitation)

**Likely not worth pursuing:**
- Any spatial structure in noise (proved harmful)
- Any form of gradient smoothing (proved harmful)
- Changing the noise distribution (6-dim covariance space is a dead end)
- Increasing copy count (gradient diversity already ~95%)
- Standard gradient post-processing (TI, NI, trim, agreement — all tested)

## Key Source Files

- `attack.py`: LMDSSAttacker class, _attack_loss_for_token_patch_dropout (~line 1411)
- `main.py`: CLI entry, create_attacker function
- `experiments/MATH_ANALYSIS.md`: Full mathematical derivation of opponent covariance
- `experiments/vit_gradient_spatial_analysis.py`: W·W^T spectral analysis

## Quick Start Commands

```bash
# Run attack
python main.py --mode attack --max-attacked-samples 100 --steps 10 \
  --mi --mi-decay 1.0 --attack-loss logits --feature-layer 0 \
  --guide-aug --guide-aug-method token_patch_dropout \
  --guide-aug-copies 20 --guide-aug-strength 0.2 \
  --patch-dropout-ratio 0.3 --patch-dropout-score-mode low \
  --patch-dropout-fill-mode zero_noise \
  --patch-dropout-noise-mode opponent_channel_gaussian \
  --token-score-cls-noise --token-cls-noise-mode gaussian \
  --output-dir outputs/attack/lmdss_ablation/<exp_name> --batch-size 32

# Run transfer eval
python transfer_eval.py \
  --image-dir outputs/attack/lmdss_ablation/<exp_name> \
  --prefix adv_ --exp-name <exp_name>
```
