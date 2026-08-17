# Result archive

This directory is the tracked index for the compact experiment archive.  The
artifact files themselves remain at their original `outputs/...` paths so that
old commands, reports, and provenance references do not change.  Selected
formal artifacts under `outputs/` are force-tracked even though generated
outputs are ignored by default.

## Production mainline

The production attack remains the dynamic-mask
`original_score_postdrop_phase_pair` pipeline.  Its retained report is
`experiments/mainline_data_aug_gaussian_story_s1000.md`; the evaluated transfer
records are the formal mainline CSV files under `outputs/csv/`.

Only the ViT Gaussian-residual comparison is part of the current written
paper-level conclusion.  CaiT, PiT, and Visformer Gaussian directories and CSV
records exist locally, but their provenance must be audited before they are
promoted into the paper narrative.

## Retained forward-semantic evidence

The following formal artifacts are retained together with executable code:

- `outputs/research/patch_score_layer_semantic_maturity/`
- `outputs/research/patch_score_promotion_e1_e2/`
- `outputs/research/patch_score_promotion_e3_mixing/`
- `outputs/research/semantic_equivalent_route_e5_forward/`
- `outputs/research/patch_score_same_activation64/summary.json`

They support three bounded conclusions:

1. all four architectures exhibit early-to-late semantic re-ranking;
2. phase-equivalent views can preserve global semantics while changing local
   routing;
3. cross-model commonality is functional and weakly image-specific, not a
   universal shared spatial mask.

The completed E5-forward CSV historically wrote the top-preserved pair value
to both `preserved_pair_distance` and `top_pair_distance`.  The retained code
keeps that behavior for artifact reproducibility; do not interpret the former
as an independently computed all-preserved-pair statistic.

## Archived negative or historical explorations

The executable code for the following directions has been removed from the
streamlined project.  Their formal summaries, protocol files, CSV records, and
existing reports remain tracked at their original output paths:

- clean-output causal, polarity, budget, layer, and route-response studies;
- dynamic and fixed-mask routing calibration and selector suites;
- Grad-CAM Protocol B, transfer, stability, and frozen-layer variants;
- promotion-gated gradient experiment E4;
- semantic-gradient experiments E5-gradient and E6-E10;
- frequency-pair and patch-shuffle attacks.

These artifacts are boundary evidence.  They must not be used to override the
current production mask policy or to claim transfer improvements that were not
observed.

Smoke outputs, adversarial PNG files, model caches, and large replay manifests
remain ignored and are not part of the compact Git archive.
