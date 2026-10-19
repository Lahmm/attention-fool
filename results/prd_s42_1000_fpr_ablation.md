# PRD seed-42 FPR 13-target transfer results

Recorded 2026-10-19. This is a separate FPR-protocol evaluation of the same adversarial images used in [the seed-42 project-protocol run](prd_s42_1000_ablation.md). No attack was regenerated. Only `transfer_eval.py --protocol fpr` was run, with `--no-tf32`, batch size 32, and all 1000 images per source/cell.

ASR is `1 - adversarial accuracy` over all 1000 images with no filtering. The FPR target suite contains 7 Transformer targets (`vit_base_patch16_224`, `cait_s24_224`, `pit_b_224`, `visformer_small`, `swin_tiny_patch4_window7_224`, `deit_tiny_patch16_224`, `coat_tiny`) and 6 torchvision CNN targets (`resnet18`, `vgg16`, `densenet121`, `efficientnet_b0`, `mobilenet_v3_small`, `resnext50_32x4d`). Strict black-box excludes a target whose name matches the source: 12 targets for ViT-B/16 and CaiT-S24; 13 for PiT-T and distilled DeiT-B because the FPR suite has no exactly matching source target.

## Main four-source result (full PRD, FPR)

| Source | Overall ASR | Strict black-box | Transformer | CNN |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 88.02 | 87.06 | 91.17 | 84.35 |
| CaiT-S24 | 90.24 | 89.50 | 92.63 | 87.45 |
| PiT-T | 85.34 | 85.34 | 81.47 | 89.85 |
| DeiT-B distilled | 93.04 | 93.04 | 95.23 | 90.48 |
| **Macro mean** | **89.16** | **88.73** | **90.12** | **88.03** |

For context only, the historical 2026-10-12 FPR run (seed 20260907, implementation `bf092246`) reported the four-source means 89.12 / 88.69 / 90.14 / 87.92. The new seed-42 measurement is nearly identical but is a separate measurement, not a reproduction.

| Source | Historical Overall | New seed-42 Overall |
| --- | ---: | ---: |
| ViT-B/16 | 88.18 | 88.02 |
| CaiT-S24 | 90.08 | 90.24 |
| PiT-T | 85.12 | 85.34 |
| DeiT-B distilled | 93.08 | 93.04 |
| Macro | 89.12 | 89.16 |

## 2×2 factorial ASR (%, FPR)

| Cell | ViT-B | CaiT-S24 | PiT-T | DeiT-B | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` (off/off) | 64.50 | 78.24 | 70.00 | 88.18 | 75.23 |
| `drop` (on/off) | 72.92 | 81.75 | 72.27 | 85.03 | 77.99 |
| `noise` (off/on) | 79.32 | 85.14 | 83.99 | 94.46 | 85.73 |
| `full` (on/on) | 88.02 | 90.24 | 85.34 | 93.04 | 89.16 |

## Factorial effects (pp, FPR)

95% CIs use 10,000 paired bootstrap resamples over the same 1000 images. Macro effects average the four sources equally.

| Scope | Drop main effect | Noise main effect | Interaction | Full − base |
| --- | ---: | ---: | ---: | ---: |
| Macro | +3.10 [+2.65, +3.55] | +10.83 [+10.16, +11.52] | +0.67 [+0.09, +1.26] | +13.93 [+13.02, +14.87] |
| ViT-B | +8.56 [+7.58, +9.55] | +14.97 [+13.82, +16.10] | +0.28 [−1.03, +1.62] | +23.52 [+21.86, +25.22] |
| CaiT-S24 | +4.31 [+3.30, +5.37] | +7.69 [+6.82, +8.58] | +1.58 [+0.27, +2.92] | +12.00 [+10.62, +13.42] |
| PiT-T | +1.81 [+1.29, +2.32] | +13.53 [+12.52, +14.57] | −0.92 [−1.81, −0.03] | +15.34 [+14.15, +16.55] |
| DeiT-B distilled | −2.29 [−3.00, −1.62] | +7.14 [+6.40, +7.90] | +1.73 [+0.85, +2.65] | +4.85 [+3.97, +5.72] |

Selected pairwise differences:

| Scope | Drop − base | Noise − base | Full − drop | Full − noise |
| --- | ---: | ---: | ---: | ---: |
| Macro | +2.76 [+2.23, +3.30] | +10.50 [+9.80, +11.21] | +11.17 [+10.41, +11.96] | +3.43 [+2.90, +3.98] |
| ViT-B | +8.42 [+7.34, +9.52] | +14.82 [+13.43, +16.20] | +15.11 [+13.89, +16.35] | +8.70 [+7.47, +9.98] |
| CaiT-S24 | +3.52 [+2.35, +4.72] | +6.90 [+5.82, +8.02] | +8.48 [+7.38, +9.61] | +5.10 [+3.85, +6.40] |
| PiT-T | +2.27 [+1.50, +3.04] | +13.99 [+12.83, +15.17] | +13.07 [+12.01, +14.15] | +1.35 [+0.74, +1.95] |
| DeiT-B distilled | −3.15 [−4.13, −2.24] | +6.28 [+5.56, +7.01] | +8.01 [+7.03, +9.03] | −1.42 [−2.11, −0.75] |

## Observations

- The FPR protocol reproduces the project-protocol component pattern. Opponent noise is again the dominant mechanism (macro main effect +10.83 pp), while progressive drop contributes +3.10 pp on average.
- Progressive drop is strongly positive for ViT-B (+8.56 pp), positive for CaiT-S24 (+4.31 pp) and PiT-T (+1.81 pp), and negative for distilled DeiT-B (−2.29 pp).
- The full PRD macro result is the best of the four cells (89.16%), +3.43 pp above noise-only. For DeiT-B, noise-only is again higher than full (94.46% vs 93.04%, difference −1.42 pp), reproducing the source-specific exception seen under the project protocol.
- Paired McNemar tests against `base`: ViT-B improves significantly on 12/13 targets for drop and full, and 13/13 for noise; CaiT-S24 improves on 9/13 (drop) and 12/13 (noise, full); PiT-T improves on 8/13 (drop) and 13/13 (noise, full); DeiT-B drop is significantly worse on 9/13 targets, while noise and full improve on 12/13 and 10/13.

## Files

- `results/prd_s42_1000_fpr_summary.csv`: per-cell, per-source and macro FPR metrics.
- `results/prd_s42_1000_fpr_effects.csv`: FPR factorial effects and paired bootstrap CIs.
- `results/prd_s42_1000_fpr_per_target.csv`: per-target FPR ASR for all 16 cell/source combinations.
- `outputs/fpr_eval/prd_s42_{base,drop,noise,full}_{vit,cait,pit_t,deit_b}/transfer.csv` and `.json`: separate FPR evaluation outputs.
- `outputs/predictions_fpr/prd_s42_{base,drop,noise,full}_{vit,cait,pit_t,deit_b}.json`: per-sample FPR predictions.
- The existing project-protocol `transfer.csv`, `transfer.json`, and prediction files are untouched.
