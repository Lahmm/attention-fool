# PRD seed-42 FPR 13-target transfer and ablation results

Updated 2026-10-26 with the four reruns completed on 2026-10-23 (12:02:55–13:02:13 UTC), using code commit `2d2625b5588587b75da7f87694c0d52234b6faa7`. The current result combines these four updated cells—PiT-T `drop/full` and distilled DeiT-B `drop/full`—with the twelve unchanged cells measured on 2026-10-19. All sixteen cells use the same 1000 image IDs and labels. FPR13 is the primary transfer protocol; only FPR13 numerical records are retained in the current `results/` tree; earlier protocol tables can be retrieved from Git history.

## Setup and retained records

All attacks use epsilon=16/255, 10 steps, step size epsilon/10, momentum 1, 10 augmentation groups × 2 views, phase shifts (4,4)/(8,8)/(12,12), raw pixel-gradient averaging and projected sign updates. When enabled, each view injects opponent-channel noise on all initial local features before progressive checkpoint drops. When enabled, the two mechanisms use the following source configurations:

| Source | Checkpoints | Drop ratios | Local-token counts / drops | Opponent β | Generation batch size |
| --- | --- | --- | --- | ---: | ---: |
| ViT-B/16 | block3, block10 | 0.051020408163, 0.051020408163 | 196/196 → 10/10 | 0.2 | 48 |
| CaiT-S24 | block17, block23 | 0.010204081633, 0.142857142857 | 196/196 → 2/28 | 0.2 | 16 |
| PiT-T | stage2_block1, stage3_block2, stage3_block3 | 0.07142857142857142, 0.04081632653061224, 0.04081632653061224 | 196/49/49 → 14/2/2 | 0.4 | 32 for the rerun drop/full cells |
| DeiT-B distilled | block2, block9 | 0.051020408163, 0.051020408163 | 196/196 → 10/10 | 0.2 | 48 |

`base` disables both mechanisms; `drop` enables progressive drop with opponent strength 0; `noise` disables progressive drop and uses the source's opponent strength; `full` enables both. Checkpoints and drop ratios are inactive for `base/noise`, so their retained metadata may list the earlier checkpoint/drop settings. The retained PiT-T `base/noise` cells were generated with batch size 96; each run's exact saved parameters are archived below.

Evaluation uses `transfer_eval.py --protocol fpr --no-tf32`, batch size 32, and all 1000 images per target. The fixed suite is 7 Transformer and 6 CNN targets. ASR is `1 - adversarial accuracy`, without clean-correct filtering. Overall averages all 13 targets. Strict black-box excludes the exactly matching source target: 12 targets for ViT-B/16 and CaiT-S24, all 13 for PiT-T and distilled DeiT-B. The four-source macro mean weights sources equally.

The versioned directory [prd_s42_fpr_run_artifacts](prd_s42_fpr_run_artifacts/) retains `attack_params.json`, `transfer.csv`, and `transfer.json` for all 16 source/cell combinations. These lightweight records are available after a Git clone and directly record every per-target ASR and source/cell summary. Attack PNGs, per-image predicted class IDs, and logs remain server-local under ignored `outputs/`. [Provenance JSON](prd_s42_1000_fpr_provenance.json) records the original paths, hashes, code commit and per-run parameters.

## Full PRD ASR (%)

| Source | Overall ASR | Strict black-box | Transformer | CNN |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 88.02 | 87.06 | 91.17 | 84.35 |
| CaiT-S24 | 90.24 | 89.50 | 92.63 | 87.45 |
| PiT-T | 85.81 | 85.81 | 81.90 | 90.37 |
| DeiT-B distilled | 94.80 | 94.80 | 96.90 | 92.35 |
| Macro mean | 89.72 | 89.29 | 90.65 | 88.63 |

## Full PRD per-target ASR (%)

| Target | ViT-B/16 | CaiT-S24 | PiT-T | DeiT-B distilled |
| --- | ---: | ---: | ---: | ---: |
| vit_base_patch16_224 | 99.6 | 92.1 | 65.5 | 98.0 |
| cait_s24_224 | 95.0 | 99.1 | 82.2 | 99.3 |
| pit_b_224 | 88.7 | 90.6 | 75.4 | 96.3 |
| visformer_small | 87.9 | 90.5 | 81.4 | 95.8 |
| swin_tiny_patch4_window7_224 | 89.1 | 92.3 | 84.4 | 96.9 |
| deit_tiny_patch16_224 | 93.6 | 95.0 | 97.8 | 98.6 |
| coat_tiny | 84.3 | 88.8 | 86.6 | 93.4 |
| resnet18 | 83.6 | 87.7 | 91.9 | 91.2 |
| vgg16 | 83.0 | 88.9 | 91.7 | 91.7 |
| densenet121 | 85.3 | 89.1 | 90.4 | 93.2 |
| efficientnet_b0 | 89.6 | 91.2 | 94.6 | 95.8 |
| mobilenet_v3_small | 84.1 | 82.9 | 95.9 | 90.1 |
| resnext50_32x4d | 80.5 | 84.9 | 77.7 | 92.1 |

## 2×2 mechanism ablation: Overall ASR (%)

| Cell | ViT-B/16 | CaiT-S24 | PiT-T | DeiT-B distilled | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` (off/off) | 64.50 | 78.24 | 70.00 | 88.18 | 75.23 |
| `drop` (on/off) | 72.92 | 81.75 | 72.32 | 90.01 | 79.25 |
| `noise` (off/on) | 79.32 | 85.14 | 83.99 | 94.46 | 85.73 |
| `full` (on/on) | 88.02 | 90.24 | 85.81 | 94.80 | 89.72 |

## Factorial effects (percentage points)

The following values are point estimates obtained from the ASR tables. Drop main effect = ((drop − base) + (full − noise))/2; noise main effect = ((noise − base) + (full − drop))/2; interaction = full − drop − noise + base. All differences are in percentage points.

| Scope | Drop main effect | Noise main effect | Interaction | Full − base |
| --- | ---: | ---: | ---: | ---: |
| Macro mean | +4.00 | +10.48 | -0.03 | +14.49 |
| ViT-B/16 | +8.56 | +14.97 | +0.28 | +23.52 |
| CaiT-S24 | +4.31 | +7.69 | +1.58 | +12.00 |
| PiT-T | +2.07 | +13.74 | -0.51 | +15.81 |
| DeiT-B distilled | +1.08 | +5.53 | -1.48 | +6.62 |

Selected pairwise differences:

| Scope | Drop − base | Noise − base | Full − drop | Full − noise |
| --- | ---: | ---: | ---: | ---: |
| Macro mean | +4.02 | +10.50 | +10.47 | +3.99 |
| ViT-B/16 | +8.42 | +14.82 | +15.11 | +8.70 |
| CaiT-S24 | +3.52 | +6.90 | +8.48 | +5.10 |
| PiT-T | +2.32 | +13.99 | +13.48 | +1.82 |
| DeiT-B distilled | +1.82 | +6.28 | +4.79 | +0.34 |

## Observations

- Full PRD has the highest Overall ASR for every source. Its four-source mean is 89.72%, versus 85.73% for noise-only, a difference of +3.99 pp.
- Opponent noise has a macro main effect of +10.48 pp, and progressive drop has a macro main effect of +4.00 pp.
- PiT-T uses 14/2/2 drops and reaches 85.81%, +1.82 pp above noise-only.
- Distilled DeiT-B uses block2/block9 and reaches 94.80%, +0.34 pp above noise-only. Drop-only reaches 90.01%, +1.82 pp above the 88.18% base.

## Files and history

- [Summary CSV](prd_s42_1000_fpr_summary.csv): all 16 source/cell metrics and four macro rows; ASR values are percentages.
- [Per-target CSV](prd_s42_1000_fpr_per_target.csv): all 208 source/cell/target measurements; ASR values are fractions in [0,1].
- [Effects CSV](prd_s42_1000_fpr_effects.csv): 40 direct ASR effect estimates, in percentage points.
- [Provenance JSON](prd_s42_1000_fpr_provenance.json): measurement dates, seed convention, source paths and hashes, and parameters of every retained run.
- [Versioned run artifacts](prd_s42_fpr_run_artifacts/): actual attack parameters and original transfer CSV/JSON records for all 16 cells.
- Rerun outputs and logs: `outputs/reruns/prd_s42_adjusted_20261023T120255Z/`; corresponding images: `outputs/attack/prd_s42_adjusted_20261023T120255Z_{drop,full}_{pit_t,deit_b}/`.

The previous 2026-10-19 full-PRD FPR table had PiT-T 85.34% (4/2/5 drops), distilled DeiT-B 93.04% (block3/block10), and macro 89.16%. Those values describe the earlier configurations and remain in Git history and original outputs. Earlier project 14-target and seed-20260907 records can be retrieved from Git history; they are not part of this current result table.
