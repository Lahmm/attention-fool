# GDI layer-selection evidence: layer-normalized standard deviation plus absolute mean

This historical comparison selects the largest sum of normalized population standard deviation and normalized absolute signed mean of clean local–global cosine scores. Each term is divided by its maximum across candidate blocks for that image; a zero maximum contributes zero. Select once per image, break ties by earliest native block and intervene at its input. Moments use float64 after raw float32 cosine. The current GDI localization rule combines direction variance ratio and positive fractional contraction.

Every source uses 1000 images, seed 42, epsilon 16/255, 10 steps, step size epsilon/10, momentum 1, 10 groups × 2 views, and the same phase shifts and opponent augmentation as its references. Strengths are 0.2 for ViT/CaiT/DeiT and 0.4 for PiT-T. Drop ratios are 20/196 for ViT/DeiT and 30/196 for CaiT/PiT-T, rounded on native grids. CaiT uses final pre-LayerNorm CLS.

FPR13 evaluates all adversarial samples with batch 32, no AMP, and TF32 disabled. ASR is 1 minus adversarial accuracy. Strict ASR excludes the exact source target. Differences are current minus reference in percentage points.

## Overall ASR

| Source | Variance | Second raw moment | Unnormalized std + absolute mean | Normalized std + absolute mean | Difference vs variance | Difference vs second raw moment | Difference vs unnormalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit | 88.13% | 87.48% | 88.12% | 88.15% | +0.02 | +0.66 | +0.03 |
| cait | 83.22% | 88.10% | 87.78% | 87.49% | +4.28 | -0.61 | -0.28 |
| pit_t | 83.60% | 82.77% | 82.85% | 83.06% | -0.54 | +0.29 | +0.21 |
| deit_b | 94.38% | 94.96% | 95.03% | 95.09% | +0.71 | +0.13 | +0.06 |
| macro | 87.33% | 88.33% | 88.44% | 88.45% | +1.12 | +0.12 | +0.00 |

## Strict black-box ASR

| Source | Variance | Second raw moment | Unnormalized std + absolute mean | Normalized std + absolute mean | Difference vs variance | Difference vs second raw moment | Difference vs unnormalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit | 87.15% | 86.44% | 87.12% | 87.16% | +0.01 | +0.72 | +0.03 |
| cait | 82.34% | 87.25% | 86.89% | 86.62% | +4.28 | -0.63 | -0.27 |
| pit_t | 83.60% | 82.77% | 82.85% | 83.06% | -0.54 | +0.29 | +0.21 |
| deit_b | 94.38% | 94.96% | 95.03% | 95.09% | +0.71 | +0.13 | +0.06 |
| macro | 86.87% | 87.86% | 87.98% | 87.98% | +1.12 | +0.13 | +0.01 |

## Records

`summary.csv` contains source-level ASRs and direct differences; `per_target.csv` contains all 52 target comparisons; `selection_counts.csv` contains every candidate checkpoint including zero counts. `runs/` stores the executed configuration and raw FPR13 results; all three reference directories preserve the actual prior configurations. Provenance, source changes, CPU validation, and full preflight are retained. Adversarial images, predictions, and logs remain in the recorded server output directories.

## GDI archive maintenance (2026-11-05)

Archive directory names and report titles now use GDI. The executed JSON/CSV records, source patch, original run identifiers and source hashes retain their original bytes. The current iid main results are indexed in [the GDI iid archive](../gdi_iid_paired_s42_1000_fpr/README.md). Provenance records the directory migration, prior report hashes and refreshed complete retained-file manifest.
