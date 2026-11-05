# GDI layer-selection evidence: standard deviation plus absolute mean

This historical comparison selects the largest population standard deviation plus absolute signed mean of clean local–global cosine scores. Select once per image, break ties by earliest native block and intervene at its input. Moments use float64 after raw float32 cosine. The current GDI localization rule combines direction variance ratio and positive fractional contraction.

Every source uses 1000 images, seed 42, epsilon 16/255, 10 steps, step size epsilon/10, momentum 1, 10 groups × 2 views, and the same phase shifts and opponent augmentation as its references. Strengths are 0.2 for ViT/CaiT/DeiT and 0.4 for PiT-T. Drop ratios are 20/196 for ViT/DeiT and 30/196 for CaiT/PiT-T, rounded on native grids. CaiT uses final pre-LayerNorm CLS.

FPR13 evaluates all adversarial samples with batch 32, no AMP, and TF32 disabled. ASR is 1 minus adversarial accuracy. Strict ASR excludes the exact source target. Differences are current minus reference in percentage points.

## Overall ASR

| Source | Variance | Second raw moment | Std + absolute mean | Difference vs variance | Difference vs second raw moment |
| --- | ---: | ---: | ---: | ---: | ---: |
| vit | 88.13% | 87.48% | 88.12% | -0.02 | +0.63 |
| cait | 83.22% | 88.10% | 87.78% | +4.56 | -0.32 |
| pit_t | 83.60% | 82.77% | 82.85% | -0.75 | +0.08 |
| deit_b | 94.38% | 94.96% | 95.03% | +0.65 | +0.07 |
| macro | 87.33% | 88.33% | 88.44% | +1.11 | +0.12 |

## Strict black-box ASR

| Source | Variance | Second raw moment | Std + absolute mean | Difference vs variance | Difference vs second raw moment |
| --- | ---: | ---: | ---: | ---: | ---: |
| vit | 87.15% | 86.44% | 87.12% | -0.03 | +0.68 |
| cait | 82.34% | 87.25% | 86.89% | +4.55 | -0.36 |
| pit_t | 83.60% | 82.77% | 82.85% | -0.75 | +0.08 |
| deit_b | 94.38% | 94.96% | 95.03% | +0.65 | +0.07 |
| macro | 86.87% | 87.86% | 87.98% | +1.11 | +0.12 |

## Records

`summary.csv` contains source-level ASRs and direct differences; `per_target.csv` contains all 52 target comparisons; `selection_counts.csv` contains every candidate checkpoint including zero counts. `runs/` stores the executed configuration and raw FPR13 results; both reference directories preserve the actual prior configurations. Provenance, source changes, CPU validation, and full preflight are retained. Adversarial images, predictions, and logs remain in the recorded server output directories.

## GDI archive maintenance (2026-11-05)

Archive directory names and report titles now use GDI. The executed JSON/CSV records, source patch, original run identifiers and source hashes retain their original bytes. The current iid main results are indexed in [the GDI iid archive](../gdi_iid_paired_s42_1000_fpr/README.md). Provenance records the directory migration, prior report hashes and refreshed complete retained-file manifest.
