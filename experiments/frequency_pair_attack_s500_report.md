# Frequency low/high view-pair attack: 500-sample report

> **历史结果归档。** 该分支的可执行脚本和专属测试已在项目精简时移除；正式 transfer CSV 与本报告继续保留。下文出现的脚本路径仅用于记录当时协议，不是当前入口。

## Scope and isolation

This experiment tests a replacement for the retained original/phase-shift view
pair. It does not add a module to the production attack and does not change the
defaults in `attack.py` or `main.py`. The historical implementation used an
isolated `FrequencyPairAttacker` subclass of `PatchScoreAttacker`.

The source model is ViT-B/16. All formal runs use the same 500 images, replay
seed `20260720`, 10 attack steps, `epsilon=16/255`, 10 groups x 2 views, final
layer patch score, high-tail candidate sampling, 29/196 dropped patches, and a
kept-only feature-noise strength of 0.2 at the initial RGB projection. Gaussian
gradient residual is disabled (`gaussian_alpha=0`) so that the Gaussian versus
opponent comparison changes only the kept-token feature-noise construction.

For an input `x`, the experimental views are

```text
low = FourierGaussianLowPass(x, sigma=2)
residual = x - low
low_view = low
high_view = clamp(low + 2 * residual, 0, 1)
```

The second view is deliberately a high-frequency-enhanced natural image rather
than the signed high-pass residual by itself. Each pair shares one patch-score
mask. The two per-view input gradients are merged by the same raw arithmetic
mean used by the retained mainline before MI accumulation.

The three same-seed formal runs have identical sample IDs. Every output has 500
images, and an exhaustive uint8 check found a maximum clean/adversarial pixel
difference of 16 for both frequency runs.

## Transfer ASR

ASR is `1 - top1 accuracy`. `ViT avg` covers the seven transformer transfer
targets and `CNN avg` covers the six CNN targets. The white-box ViT-B/16 result
is reported separately and is not included in the 13-target average.

| Attack views / kept-token noise | White-box | 13-target avg | ViT avg | CNN avg |
| --- | ---: | ---: | ---: | ---: |
| Frequency pair / feature IID Gaussian | 79.00 | 59.48 | 66.40 | 51.40 |
| Frequency pair / RGB opponent-projected | 76.60 | 60.63 | 65.60 | 54.83 |
| Original/phase pair / RGB opponent-projected | **89.60** | **76.91** | **81.83** | **71.17** |

All values are percentages. The mainline comparison in this table was rerun on
the same 500 images with seed `20260720` and `gaussian_alpha=0`. The previously
available seed-`20260716` mainline subset gives 76.94 average ASR, which is
almost identical and rules out seed choice as the explanation for the gap.

Per-target results:

| Target | Frequency Gaussian | Frequency opponent | Mainline opponent | Opponent - Gaussian | Frequency opponent - mainline |
| --- | ---: | ---: | ---: | ---: | ---: |
| LeViT-256 | 66.0 | 65.8 | 80.2 | -0.2 | -14.4 |
| PiT-B | 59.4 | 57.6 | 80.4 | -1.8 | -22.8 |
| DeiT-B | 64.0 | 63.8 | 81.6 | -0.2 | -17.8 |
| TNT-S | 67.0 | 66.0 | 83.2 | -1.0 | -17.2 |
| ConViT-B | 62.4 | 61.2 | 81.6 | -1.2 | -20.4 |
| Visformer-S | 60.0 | 60.6 | 79.0 | +0.6 | -18.4 |
| CaiT-S24 | 86.0 | 84.2 | 86.8 | -1.8 | -2.6 |
| Inception-v3 | 55.8 | 59.8 | 72.2 | +4.0 | -12.4 |
| Inception-v4 | 52.0 | 54.2 | 72.8 | +2.2 | -18.6 |
| Inception-ResNet-v2 | 49.6 | 52.8 | 70.4 | +3.2 | -17.6 |
| ResNet-101 | 52.6 | 55.2 | 74.4 | +2.6 | -19.2 |
| Inception-v3-adv | 53.0 | 56.4 | 71.0 | +3.4 | -14.6 |
| Inception-ResNet-v2-adv | 45.4 | 50.6 | 66.2 | +5.2 | -15.6 |

RGB opponent noise improves the frequency attack's overall mean by 1.15 points
and its CNN mean by 3.43 points, but lowers its transformer mean by 0.80 points.
It is therefore useful mainly for cross-family/CNN transfer in this view setup,
not as a uniform improvement. More importantly, the frequency replacement is
16.28 points below the same-seed mainline average, with nearly equal losses on
transformers (-16.23) and CNNs (-16.33). The current frequency pair should not
replace the mainline.

## Gradient diagnostics

| Diagnostic | Frequency Gaussian | Frequency opponent | Mainline opponent |
| --- | ---: | ---: | ---: |
| View cosine to final mean | 0.2816 | 0.2665 | 0.2430 |
| Sign agreement | 0.5914 | 0.5842 | 0.4795 |
| Effective rank | 18.48 | 18.77 | **19.69** |
| MI cumulative cosine | 0.5731 | 0.5848 | 0.5817 |
| Low/high paired gradient cosine | 0.0068 | 0.0069 | n/a |
| High/low gradient-norm ratio | 6.42 | 8.03 | n/a |

The low/high gradients are almost orthogonal, but the high-enhanced view has a
much larger gradient norm. Consequently, arithmetic averaging is nominally
equal per view but not equal in directional influence. The result also gives a
direct counterexample to the idea that a larger view-induced gradient change
or lower pair cosine necessarily raises transfer ASR: the frequency pair is
more visibly separated while transferring much worse. Mainline effective rank
is actually higher, so the frequency pair does not provide more usable
20-view gradient coverage under this diagnostic.

## View validity and perturbation spectrum

Before patch drop, ViT-B/16 accuracy on the 500 images is 99.8% for the original
view, 96.0% for the low-pass view, 99.8% for the high-enhanced view, and
99.6-100% for fixed 4/8/12-pixel phase views. With one shared score-guided mask
and no feature noise, accuracy is 99.2% for original-drop, 93.6% for low-drop,
98.6% for high-drop, and 94.8-96.4% for phase-drop. Thus, label preservation
alone does not explain the transfer gap: the phase views are not more accurate
than both frequency views after masking.

The final perturbations have the following mean statistics. Frequency energy
is computed after removing each channel's spatial mean, with low at <=0.125,
mid at (0.125, 0.25], and high at >0.25 cycles/pixel.

| Attack | RMS (uint8) | Boundary fraction | Total variation | Low energy | Mid energy | High energy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frequency Gaussian | 11.62 | 0.323 | 14.86 | 0.393 | 0.238 | 0.369 |
| Frequency opponent | 11.86 | 0.355 | 14.94 | 0.412 | 0.216 | 0.373 |
| Mainline opponent | 11.55 | 0.316 | **17.15** | 0.257 | 0.281 | **0.463** |

The frequency attack does not produce a more high-frequency final
perturbation. It has substantially more low-frequency energy and lower spatial
total variation than the mainline. Therefore the narrative "explicit texture
view -> higher-frequency perturbation -> better transfer" is contradicted by
this run. The mean per-image perturbation cosine is 0.398 between the two
frequency noise variants, but only 0.186/0.225 between frequency Gaussian /
frequency opponent and the mainline. The view replacement changes the attack
direction materially, but that change is not beneficial.

## Interpretation and next decision

This is a viable controlled experiment but not a viable mainline replacement
in its current form. The result supports three narrower conclusions:

1. Frequency decomposition supplies different gradients, but gradient
   difference is not equivalent to transferable gradient quality.
2. Raw averaging is poorly calibrated for the frequency pair because the two
   gradient norms differ by 6-8x. This is the clearest implementation-level
   bottleneck exposed by the diagnostics.
3. Opponent noise remains complementary for CNN transfer within the frequency
   setup, but it cannot recover the loss caused by replacing original/phase
   views.

If this thread is continued, the next controlled test should not tune ASR
blindly. First compare (a) per-view L1/L2-normalized gradient averaging, (b)
norm-matched low/high gradients, and (c) low/high frequency projections of the
*gradient* rather than strongly transforming the image. A four-cell
`original/phase` versus `low/high` x `raw mean` versus `norm-matched mean`
ablation on a smaller development split would determine whether the failure is
caused by frequency semantics or by merge imbalance. Until such evidence is
positive, the retained original/phase mainline should remain unchanged.

## Reproducibility artifacts

- Historical implementation and unit tests: removed after the failed branch was archived
- Frequency Gaussian outputs:
  `outputs/attack/vit_frequency_pair_gaussian_s500_seed20260720`
- Frequency opponent outputs:
  `outputs/attack/vit_frequency_pair_opponent_s500_seed20260720`
- Same-seed raw-mainline outputs:
  `outputs/attack/vit_mainline_raw_opponent_s500_seed20260720`
- Transfer CSV files use the corresponding names under `outputs/csv/`.
