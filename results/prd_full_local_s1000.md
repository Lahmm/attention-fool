# Full-local PRD versus archived kept-only PRD

ASR = 1 - adversarial accuracy over all 1000 adversarial samples. Delta is in percentage points.
Baseline: results/prd_cross_arch_s1000.csv, kept-only version 98cb7cf1, seed 20260907.
Full-local run: `bf09224675093cd6a4fcccee2db10acae0eb5f4e`, seed `20260907`, completed on 2026-10-12. This is an additional measurement of the new noise policy; the kept-only mainline archive remains unchanged.

| Source | Metric | Kept-only ASR | Full-local ASR | Delta (pp) |
| --- | --- | ---: | ---: | ---: |
| vit_base_patch16_224 | overall | 85.20% | 85.06% | -0.14 |
| vit_base_patch16_224 | transformer | 90.45% | 89.96% | -0.49 |
| vit_base_patch16_224 | cnn | 78.20% | 78.52% | +0.32 |
| vit_base_patch16_224 | strict_black_box | 84.29% | 84.18% | -0.11 |
| cait_s24_224 | overall | 86.58% | 87.64% | +1.06 |
| cait_s24_224 | transformer | 90.06% | 90.89% | +0.82 |
| cait_s24_224 | cnn | 81.93% | 83.32% | +1.38 |
| cait_s24_224 | strict_black_box | 85.66% | 86.79% | +1.13 |
| pit_b_224 | overall | 84.36% | 84.34% | -0.02 |
| pit_b_224 | transformer | 89.91% | 89.55% | -0.36 |
| pit_b_224 | cnn | 76.95% | 77.38% | +0.43 |
| pit_b_224 | strict_black_box | 83.27% | 83.24% | -0.03 |
| visformer_small | overall | 80.65% | 81.18% | +0.53 |
| visformer_small | transformer | 83.19% | 83.16% | -0.02 |
| visformer_small | cnn | 77.27% | 78.53% | +1.27 |
| visformer_small | strict_black_box | 79.18% | 79.75% | +0.57 |
| four_source_mean | overall | 84.20% | 84.55% | +0.36 |
| four_source_mean | transformer | 88.40% | 88.39% | -0.01 |
| four_source_mean | cnn | 78.59% | 79.44% | +0.85 |
| four_source_mean | strict_black_box | 83.10% | 83.49% | +0.39 |

All 56 source-target evaluations used 1000 samples with zero skipped images. ASR includes every evaluated adversarial sample. Strict black-box averages exclude the source-matched target. Deltas use unrounded values.

Full-local raises the four-source Overall ASR by 0.36 percentage points and strict black-box ASR by 0.39 points. CNN ASR rises for all four sources, averaging +0.85 points; Transformer ASR averages -0.01 points. CaiT-S24 and Visformer-S improve Overall ASR, while ViT-B/16 and PiT-B stay close to their archived values. Across the 56 pairs, 31 improve, 4 are unchanged, and 21 decrease. These are the observations from the retained seed.

## Attack configuration

Both versions use the same 1000 sample IDs in the same order and seed `20260907`. The full-local policy adds opponent-channel projected noise to all initial local features, including positions selected for later checkpoint drops; prefix/CLS tokens remain unchanged.

| Source | Batch size | Checkpoints | Drop ratios | Opponent strength |
| --- | ---: | --- | --- | ---: |
| ViT-B/16 | 96 | block3,block10 | 0.051020408163,0.051020408163 | 0.2 |
| CaiT-S24 | 48 | block17,block23 | 0.010204081633,0.142857142857 | 0.2 |
| PiT-B | 96 | stage2_block1,stage3_block2,stage3_block3 | 0.02081165,0.03125,0.09375 | 0.4 |
| Visformer-S | 48 | stage2_block1,stage3_block1 | 0.209183673469,0.204081632653 | 0.4 |

Common settings: epsilon `16/255`, step size `1.6/255`, 10 steps, 10 groups with two views each, phase shifts `(4,4)/(8,8)/(12,12)`, MI decay `1.0`, Gaussian residual sigma `4.0` and alpha `0.75`. Transfer evaluation uses AMP and TF32 on the same 14 targets as the archive, with no clean-correct filtering.

Runtime: Python 3.10.19, PyTorch 2.10.0+cu128, timm 1.0.24, CUDA 12.8, NVIDIA RTX 4090. Model weights were moved intact from the global Hugging Face cache to `data/huggingface/hub`, with file SHA256 and symlink targets verified unchanged. This comparison uses archived kept-only measurements and records the new runtime independently.

## Archived files

- [Full-local transfer table](prd_full_local_cross_arch_s1000.csv): 56 source-target ASRs in the same schema as the kept-only table.
- [Per-target comparison](prd_full_local_vs_kept_only_per_target_s1000.csv): old/new ASR and percentage-point changes.
- [Summary comparison](prd_full_local_vs_kept_only_summary_s1000.csv): per-source and four-source means.
- [Run provenance](prd_full_local_run_artifacts/provenance.json), four attack parameter JSON files and four original transfer CSVs in `prd_full_local_run_artifacts/`.
- [Kept-only baseline](prd_cross_arch_s1000.csv): retained original measurements at `98cb7cf1`.

Server-side images, replay manifests, per-sample predictions and logs remain under `outputs/attack/prd1000_full_local_*_seed20260907/` and `outputs/runs/prd1000_full_local_seed20260907/`.

