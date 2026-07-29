# Fixed-mask routing experiment final report

Completed at `2026-07-29T10:39:38Z`. The full fixed-clean-mask experiment is
finished: 38 routing-calibration attacks, 114 off-diagonal calibration
evaluations, four H2 routing/gradient jobs, Grad-CAM Protocol A on all four
sources, and all 28 selector attacks. Each selector attack contains 500 images,
for 14,000 saved adversarial images. Transfer was evaluated on 13 target
models. The machine-readable completion marker is
`outputs/research/selector_suite_fixed_mask/pipeline_complete.json`.

## Frozen protocol

- The patch mask is computed once from the clean image and reused across every
  attack step, group, and view. A phase-shifted view shifts the same mask.
- Random routing samples its mask once from the clean-image candidate geometry
  and likewise reuses it everywhere.
- Global polarity is fixed to `high` for every architecture.
- Frozen layers are ViT `block6`, CaiT `block24_gap`, PiT `stage3_block2`, and
  Visformer `stage3_block2`.
- `token_score_cls_noise=true`; the 20-view opponent-channel-noise pipeline and
  all other attack settings are matched across selectors.

## Transfer ASR

The entries below are macro averages over the 13 targets. ASR uses only target
clean-correct images in its denominator.

| source | selected | random | selected-random (paired 95% CI) | no drop | final layer | Grad-CAM |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | 68.29 | 66.49 | +1.80 [+0.19, +3.36] | 74.55 | 62.93 | 67.13 |
| CaiT-S24 | 78.67 | 77.02 | +1.64 [+0.40, +2.92] | 81.98 | 77.97 | 78.40 |
| PiT-B | 70.91 | 70.82 | +0.08 [-1.69, +1.82] | 74.69 | 69.50 | 66.65 |
| Visformer-S | 54.85 | 60.25 | -5.40 [-7.07, -3.75] | 68.38 | 56.42 | 54.34 |
| four-source mean | 68.18 | 68.65 | -0.47 [-1.28, +0.34] | 74.90 | 66.71 | 66.63 |

The selected route beats random significantly for ViT and CaiT, is
indistinguishable from random for PiT, and is significantly worse for
Visformer. Across all four sources the selected-minus-random point estimate is
-0.47 percentage points and its confidence interval includes zero. The result
therefore does not support a universal patch-score-over-random claim.

No-drop is higher than selected on all four sources. The pooled paired gap is
-6.72 points with 95% CI [-7.67, -5.81]. Under this implementation, adding
patch drop to the opponent-noise attack lowers transfer ASR; selector quality
cannot repair that absolute drop penalty.

## Hypothesis audit

### H1: an architecture-dependent layer can outperform the final layer

Partially supported. Calibration selected a non-default routing location for
all architectures. On the 500-image attack set, selected-minus-final-layer was
+5.36 points for ViT, +0.69 for CaiT, +1.41 for PiT, and -1.57 for Visformer.
The pooled difference was +1.47 points with 95% CI [+0.69, +2.27]. This supports
layer calibration as a useful consideration, especially for ViT, but not as a
uniform per-architecture guarantee.

### H2: selected drop reorganizes routing into a more transferable gradient

Only locally supported. Relative to random, selected routing improves raw and
processed source/target gradient cosine and one-step held-out target response
for ViT and CaiT. It worsens those quantities for PiT and Visformer. Visformer
is the clearest counterexample: selected drop changes its global, kept-token,
and score-map representations more than random, yet has worse cross-model
gradient alignment, one-step target response, and full transfer ASR. Thus
representation disruption alone is not evidence of transferable routing.

### H3: the reorganization improves transfer ASR

Rejected as a general claim. It holds for ViT and CaiT under the tested fixed
mask pipeline, is null for PiT, and reverses for Visformer. The four-source
aggregate is not better than random, and no-drop is better for every source.

## Grad-CAM comparison

Protocol A compares both maps on the exact same selected-layer activation.
Patch-score versus Grad-CAM ReLU Spearman/top-half IoU are respectively
0.107/0.367 (ViT), 0.098/0.368 (CaiT), -0.214/0.306 (PiT), and 0.586/0.637
(Visformer). The two criteria are distinct, with greater partial overlap on
Visformer. Grad-CAM also changes substantially when its target label changes,
whereas patch-score remains label-free and gradient-independent.

Patch-score has a higher point ASR than Grad-CAM on every source, by +1.16,
+0.27, +4.25, and +0.51 points. At the predeclared 1-point noninferiority
margin, however, the confidence-bound test passes only for ViT and PiT; CaiT
and Visformer miss it narrowly. The defensible conclusion is criterion
distinction and a strong PiT result, not universal superiority or universal
noninferiority.

## Paper-level conclusion

The completed experiment does not establish the proposed universal mainline
story. There is a coherent ViT/CaiT sub-result: calibrated high-score routing
improves transferable-gradient diagnostics and transfer ASR relative to matched
random drop. PiT supplies no selector advantage and Visformer falsifies the
claim. More importantly, no-drop is strongest for every architecture. The
current evidence supports reporting architecture-dependent boundary behavior,
but it does not support presenting fixed high-score patch drop as a generally
effective component of the mainline attack.

Primary artifacts:

- `outputs/research/routing_calibration_fixed_mask/frozen_routing.json`
- `outputs/research/patch_score_routing_gradients_fixed_mask/*/summary.json`
- `outputs/research/patch_score_gradcam_selected_layer_fixed_mask/summary.json`
- `outputs/research/selector_suite_fixed_mask/summary.json`
- `outputs/research/selector_suite_fixed_mask/per_image.csv`
