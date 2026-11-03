# Global-direction variance ratio: isolated FPR13 test

This experiment maximizes the population variance of normalized local features projected onto the unit global reference, divided by total local population variance per feature dimension. Each local vector and reference is L2 normalized separately in float64 from float32 model outputs. Zero total variance gives score zero. No fitted weights, fixed checkpoint labels, cross-image calibration or depth weighting enter selection. Select once per clean image, use earliest native block on ties and drop at that block input. Original raw float32 cosine variance is recorded separately. The repository mainline remains maximum variance; changes run only in the external snapshot.

Every source uses 1000 images, seed 42, epsilon 16/255, 10 steps, step size epsilon/10, momentum 1, 10 groups × 2 views, and the same phase shifts and opponent augmentation as its references. Strengths are 0.2 for ViT/CaiT/DeiT and 0.4 for PiT-T. Drop ratios are 20/196 for ViT/DeiT and 30/196 for CaiT/PiT-T, rounded on native grids. CaiT uses final pre-LayerNorm CLS.

FPR13 evaluates all adversarial samples with batch 32, no AMP, and TF32 disabled. ASR is 1 minus adversarial accuracy. Strict ASR excludes the exact source target. Differences are current minus reference in percentage points.

## Overall ASR

| Source | Variance | Second raw moment | Unnormalized std + absolute mean | Global-direction variance ratio | Difference vs variance | Difference vs second raw moment | Difference vs unnormalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit | 88.13% | 87.48% | 88.12% | 88.22% | +0.08 | +0.73 | +0.10 |
| cait | 83.22% | 88.10% | 87.78% | 83.88% | +0.67 | -4.22 | -3.89 |
| pit_t | 83.60% | 82.77% | 82.85% | 84.48% | +0.88 | +1.71 | +1.62 |
| deit_b | 94.38% | 94.96% | 95.03% | 94.65% | +0.26 | -0.32 | -0.38 |
| macro | 87.33% | 88.33% | 88.44% | 87.81% | +0.47 | -0.52 | -0.64 |

## Strict black-box ASR

| Source | Variance | Second raw moment | Unnormalized std + absolute mean | Global-direction variance ratio | Difference vs variance | Difference vs second raw moment | Difference vs unnormalized |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vit | 87.15% | 86.44% | 87.12% | 87.23% | +0.08 | +0.79 | +0.11 |
| cait | 82.34% | 87.25% | 86.89% | 83.01% | +0.67 | -4.24 | -3.88 |
| pit_t | 83.60% | 82.77% | 82.85% | 84.48% | +0.88 | +1.71 | +1.62 |
| deit_b | 94.38% | 94.96% | 95.03% | 94.65% | +0.26 | -0.32 | -0.38 |
| macro | 86.87% | 87.86% | 87.98% | 87.34% | +0.47 | -0.51 | -0.63 |

## Records

`summary.csv` contains source-level ASRs and direct differences; `per_target.csv` contains all 52 target comparisons; `selection_counts.csv` contains every candidate checkpoint including zero counts. `runs/` stores the executed configuration and raw FPR13 results; all three reference directories preserve the actual prior configurations. Provenance, source changes, CPU validation, and full preflight are retained. Adversarial images, predictions, and logs remain in the recorded server output directories.
