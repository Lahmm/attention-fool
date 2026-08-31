# Archived exploration conclusions

This file records why removed experiment branches are not production code.
Exact metrics remain in the tracked formal summaries under `outputs/`.

| Direction | Formal artifact family | Conclusion |
| --- | --- | --- |
| Promotion-gated noise E4 | `patch_score_promotion_e4_gradients` | Promotion routing has no consistent four-architecture transferable-gradient advantage. |
| Semantic-equivalent gradient E5 | `semantic_equivalent_route_e5_gradient` | Forward semantic equivalence does not yield a uniformly superior single-source gradient aggregation rule. |
| Decoupled semantic gradient E6 | `semantic_gradient_consensus_e6` | Full iterative ASR deltas versus opponent-only are near zero or mixed, with negative target cases. |
| Route vulnerability E7 | `cross_arch_route_vulnerability_e7/discover` | The preregistered cross-architecture functional-route gate failed. |
| Semantic view weighting E8 | `semantic_conditioned_gradient_e8` | Semantic scores weakly rank individual views, while semantic weighting degrades the aggregate. |
| Marginal contribution E9 | `semantic_view_marginal_e9` | Semantic scores do not consistently predict leave-one-out aggregation contribution. |
| Residual covariance E10 | `semantic_residual_covariance_e10` | The preregistered covariance correction gate failed. |

The retained production story remains limited to patch-score-guided semantic
routing and RGB opponent-channel structured noise.  Original/phase views,
Gaussian residual, MI/NI/DIM/TI, and generic dropout attacks are supporting
mechanisms or controls.
