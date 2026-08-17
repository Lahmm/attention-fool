# Archived exploration conclusions

This file records why removed experiment branches are not production code.
Exact metrics remain in the tracked formal summaries under `outputs/`.

| Direction | Formal artifact family | Conclusion |
| --- | --- | --- |
| Single-patch causality | `patch_score_budget64`, `patch_score_paired*` | Patch-score/occlusion rank correlation is near zero or negative. Patch-score is not a per-patch causal explainer. |
| Clean-output polarity | `patch_score_polarity_crossfit*` | Preferred clean-logit/CE polarity is architecture- and metric-dependent and does not select a transferable route. |
| Clean-output layer study | `patch_score_layers64` | Early/mid/final clean sensitivity is architecture-dependent and cannot select the production attack layer. |
| Source route response | `patch_score_route_response64` | Large source feature changes or low source gradient cosine are not evidence of cross-model transfer. |
| Dynamic selector suite | `routing_calibration`, `selector_suite` | Calibrated selected routes are close to matched random overall; this historical protocol is not the production default. |
| Fixed-mask selector suite | `routing_calibration_fixed_mask`, `selector_suite_fixed_mask` | Selected beats random only for ViT/CaiT, is null for PiT, reverses for Visformer, and loses to no-drop on every source. |
| Grad-CAM Protocol B | `patch_score_gradcam*` | Patch-score and Grad-CAM differ, but source deletion and protocol-B maps do not establish a universal selector winner. |
| Promotion-gated noise E4 | `patch_score_promotion_e4_gradients` | Promotion routing has no consistent four-architecture transferable-gradient advantage. |
| Semantic-equivalent gradient E5 | `semantic_equivalent_route_e5_gradient` | Forward semantic equivalence does not yield a uniformly superior single-source gradient aggregation rule. |
| Decoupled semantic gradient E6 | `semantic_gradient_consensus_e6` | Full iterative ASR deltas versus opponent-only are near zero or mixed, with negative target cases. |
| Route vulnerability E7 | `cross_arch_route_vulnerability_e7/discover` | The preregistered cross-architecture functional-route gate failed. |
| Semantic view weighting E8 | `semantic_conditioned_gradient_e8` | Semantic scores weakly rank individual views, while semantic weighting degrades the aggregate. |
| Marginal contribution E9 | `semantic_view_marginal_e9` | Semantic scores do not consistently predict leave-one-out aggregation contribution. |
| Residual covariance E10 | `semantic_residual_covariance_e10` | The preregistered covariance correction gate failed. |
| Frequency pair | `vit_frequency_pair_*`, frequency report | Opponent frequency pair reaches 60.63% versus 76.91% for the same-seed original/phase mainline. |
| Patch shuffle | `vit_patch_shuffle_*` | L2 matching improves shuffle transfer from 48.58% to 58.18%, still far below the original/phase mainline. |

The retained production story remains limited to patch-score-guided semantic
routing and RGB opponent-channel structured noise.  Original/phase views,
Gaussian residual, MI/NI/DIM/TI, and generic dropout attacks are supporting
mechanisms or controls.
