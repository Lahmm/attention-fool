# Second raw moment layer selection: isolated FPR13 test

This is an isolated experiment. The repository attack mainline continues to select the maximum raw-score population variance. Only an external source snapshot changes the selection statistic to the second raw moment: M2 = mean_i(s_i^2). The execution-only source changes are recorded in `source_changes.patch`; no experiment runner is added to the repository.

Each source attacks the same 1000 clean images with seed 42, epsilon 16/255, 10 steps, step size epsilon/10, momentum 1, and 10 groups × 2 views. Retain the baseline opponent augmentation: strength 0.2 for ViT/CaiT/DeiT and 0.4 for PiT-T. Total drop ratios are 20/196 for ViT/DeiT and 30/196 for CaiT/PiT-T, rounded on the selected native grid. CaiT uses the final CLS before final LayerNorm. All architecture references, random token masking, view pairing, gradient averaging, momentum, and projected updates are retained.

FPR13 evaluation uses all 1000 adversarial samples, batch 32, no AMP, and TF32 disabled. ASR = 1 − adversarial accuracy. Strict black-box excludes an exactly matching source target. Differences below are second raw moment minus the existing variance run, in percentage points.

| Source | Variance Overall | M2 Overall | Difference | Variance Strict | M2 Strict | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vit | 88.13% | 87.48% | -0.65 | 87.15% | 86.44% | -0.71 |
| cait | 83.22% | 88.10% | +4.88 | 82.34% | 87.25% | +4.91 |
| pit_t | 83.60% | 82.77% | -0.83 | 83.60% | 82.77% | -0.83 |
| deit_b | 94.38% | 94.96% | +0.58 | 94.38% | 94.96% | +0.58 |
| macro | 87.33% | 88.33% | +1.00 | 86.87% | 87.86% | +0.99 |

## Records

- `summary.csv`: source-level ASR and direct differences; four-source equal-weight macro mean.
- `per_target.csv`: all 52 source/target ASRs and direct differences.
- `selection_counts.csv`: candidate checkpoint counts, including zero counts, for both criteria.
- `runs/`: executed M2 attack parameters, selection records, FPR13 results, and execution metadata.
- `variance_references/`: the actual prior variance runs used in each comparison; CaiT uses the later pre-LayerNorm run.
- `provenance.json`, `preflight.json`, `source_changes.patch`: configuration, hashes, validation, external execution paths, and isolated source changes.

Preflight checks distinguish M2 from variance, verify earliest-block tie handling, reproduce native logits, execute 10 steps × 10 groups × 2 views, check one clean selection call per batch, and check the perturbation bound. Full runs verify 1000 image names, native drop counts, and every selected checkpoint against the previously calculated M2 choices. Every FPR13 ASR is independently recomputed from saved predictions.
