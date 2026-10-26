# PRD seed-42 1000-image main run and 2×2 mechanism ablation

This document retains the historical project 14-target measurements and their original configurations. The current primary FPR13 result, including the 2026-10-23 PiT-T and distilled DeiT-B drop/full reruns, is recorded in [the FPR13 report](prd_s42_1000_fpr_ablation.md). These FPR reruns do not replace any 14-target measurements in this document.

Recorded 2026-10-19 on the `prd` branch. Implementation includes `7ce5b7a` (the `--no-progressive-drop` ablation switch; the default PRD path is unchanged). All runs use seed 42, 1000 images, `epsilon=16/255`, 10 steps, step size `epsilon/10`, 10 groups × 2 views, phase shifts `(4,4)/(8,8)/(12,12)`, 20-view raw gradient averaging, momentum 1, sign update, and L∞ projection. ASR is `1 - adversarial accuracy` over all 1000 images with no filtering.

## Setup

Sources are ViT-B/16 (`vit_base_patch16_224`), CaiT-S24 (`cait_s24_224`), PiT-T (`pit_ti_224`), and distilled DeiT-B (`deit_base_distilled_patch16_224`). Targets are the default 14-model project protocol (8 Transformer, 6 CNN). Batch sizes are fixed per source across all cells: ViT-B 48, CaiT-S24 16, PiT-T 96, DeiT-B 48. These sizes were chosen for the 10 GB RTX 3080; randomness is keyed by sample ID, step, group, view, and event, so batching does not change schedules or noise.

These recorded runs used the historical sample-keyed random streams. Subsequent executions use PyTorch's global RNG seeded to 42 at startup, with reproducibility scoped to a fixed batch layout and execution flow; this change does not alter the results recorded below.

The ablation is a complete 2×2 factorial:

| Cell | Progressive drop | Opponent noise | CLI |
| --- | --- | --- | --- |
| `base` | off | off | `--no-progressive-drop --opponent-noise-strength 0` |
| `drop` | on | off | `--opponent-noise-strength 0` |
| `noise` | off | on | `--no-progressive-drop` |
| `full` | on | on | defaults |

When a mechanism is on, each source uses its native checkpoints, drop ratios, and opponent strength: ViT-B and DeiT-B `block3,block10` with 10/10 drops and β=0.2; CaiT-S24 `block17,block23` with 2/28 drops and β=0.2; PiT-T `stage2_block1,stage3_block2,stage3_block3` with 4/2/5 drops and β=0.4. All cells retain phase pairing, 20-view gradient averaging, momentum, and projection.

## Main four-source result (full PRD)

| Source | Overall ASR | Strict black-box | Transformer | CNN |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 84.74 | 83.83 | 89.50 | 78.40 |
| CaiT-S24 | 87.71 | 86.83 | 90.91 | 83.43 |
| PiT-T | 76.69 | 76.69 | 78.77 | 73.90 |
| DeiT-B distilled | 90.49 | 90.49 | 93.97 | 85.85 |
| **Macro mean** | **84.91** | **84.46** | **88.29** | **80.40** |

## 2×2 factorial ASR (%)

| Cell | ViT-B | CaiT-S24 | PiT-T | DeiT-B | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` (off/off) | 56.67 | 71.99 | 55.29 | 83.80 | 66.94 |
| `drop` (on/off) | 66.40 | 74.62 | 57.85 | 79.04 | 69.48 |
| `noise` (off/on) | 75.18 | 81.89 | 75.30 | 92.36 | 81.18 |
| `full` (on/on) | 84.74 | 87.71 | 76.69 | 90.49 | 84.91 |

## Factorial effects (pp)

Effects are paired image-level differences over the same 1000 images; 95% CIs use 10,000 paired bootstrap resamples. Macro effects average the four sources equally.

| Scope | Drop main effect | Noise main effect | Interaction | Full − base |
| --- | ---: | ---: | ---: | ---: |
| Macro | +3.13 [+2.63, +3.64] | +14.84 [+14.01, +15.64] | +1.18 [+0.48, +1.92] | +17.97 [+16.94, +19.03] |
| ViT-B | +9.65 [+8.57, +10.76] | +18.42 [+17.22, +19.66] | −0.16 [−1.64, +1.34] | +28.07 [+26.31, +29.89] |
| CaiT-S24 | +4.23 [+3.15, +5.35] | +11.50 [+10.47, +12.56] | +3.18 [+1.69, +4.69] | +15.72 [+14.16, +17.34] |
| PiT-T | +1.97 [+1.38, +2.57] | +19.42 [+18.16, +20.70] | −1.17 [−2.24, −0.10] | +21.39 [+19.94, +22.92] |
| DeiT-B distilled | −3.31 [−4.14, −2.55] | +10.00 [+9.11, +10.91] | +2.89 [+1.81, +4.01] | +6.69 [+5.62, +7.74] |

Selected pairwise differences:

| Scope | Drop − base | Noise − base | Full − drop | Full − noise |
| --- | ---: | ---: | ---: | ---: |
| Macro | +2.54 [+1.95, +3.14] | +14.24 [+13.42, +15.08] | +15.43 [+14.50, +16.37] | +3.73 [+3.08, +4.39] |
| ViT-B | +9.73 [+8.56, +10.95] | +18.51 [+17.01, +20.01] | +18.34 [+17.01, +19.72] | +9.56 [+8.14, +11.05] |
| CaiT-S24 | +2.64 [+1.38, +3.94] | +9.91 [+8.66, +11.16] | +13.09 [+11.79, +14.41] | +5.81 [+4.44, +7.21] |
| PiT-T | +2.56 [+1.72, +3.41] | +20.01 [+18.64, +21.41] | +18.84 [+17.46, +20.22] | +1.39 [+0.64, +2.14] |
| DeiT-B distilled | −4.76 [−5.85, −3.74] | +8.56 [+7.67, +9.46] | +11.45 [+10.29, +12.67] | −1.86 [−2.74, −1.02] |

## Observations

- Opponent-channel noise is the dominant mechanism on every source: its macro main effect is +14.84 pp, versus +3.13 pp for progressive drop.
- Progressive drop is strongly positive for ViT-B (+9.65 pp main effect), positive for CaiT-S24 (+4.23 pp) and PiT-T (+1.97 pp), but negative for distilled DeiT-B (−3.31 pp). For DeiT-B, `noise` alone reaches 92.36%, 1.86 pp above `full`.
- The interaction is small on average (+1.18 pp), positive for CaiT-S24 and DeiT-B, approximately zero for ViT-B, and slightly negative for PiT-T. The full PRD macro result is nevertheless the best of the four cells.
- Paired McNemar tests against `base`: all ViT-B and PiT-T cells improve on 14/14 targets; CaiT-S24 improves on 7/14 (drop) and 13/14 (noise, full); DeiT-B drop is significantly worse on 11/14 targets while its noise and full cells improve on 13/14 and 10/14.

## Files

- `results/prd_s42_1000_summary.csv`: per-cell, per-source and macro ASR metrics.
- `results/prd_s42_1000_effects.csv`: factorial effects and paired bootstrap CIs.
- `results/prd_s42_1000_per_target.csv`: per-target ASR for all 16 cell/source combinations.
- Adversarial images, `attack_params.json`, `transfer.csv`, and `transfer.json` are under `outputs/attack/prd_s42_{base,drop,noise,full}_{vit,cait,pit_t,deit_b}/`.
- Per-sample predictions are under `outputs/predictions/`.
