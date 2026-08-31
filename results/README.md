# Result archive

This directory indexes the retained experiment records. Selected formal
artifacts under `outputs/` are force-tracked even though generated outputs are
ignored by default.

## Production mainline

The production attack remains the dynamic-mask
`original_score_postdrop_phase_pair` pipeline.  Its retained report is
`experiments/mainline_data_aug_gaussian_story_s1000.md`; the evaluated transfer
records are the formal mainline CSV files under `outputs/csv/`.

Only the four 1000-image source-model runs and their Gaussian-residual
counterparts are retained. Smaller mainline subsets and alternate view/noise
attack records have been removed.

Only the ViT Gaussian-residual comparison is part of the current written
paper-level conclusion.  CaiT, PiT, and Visformer Gaussian directories and CSV
records exist locally, but their provenance must be audited before they are
promoted into the paper narrative.

## Archived negative or historical explorations

The executable code for the following directions has been removed from the
streamlined project.  Their formal summaries, protocol files, CSV records, and
existing reports remain tracked at their original output paths:

- promotion-gated gradient experiment E4;
- semantic-gradient experiments E5-gradient and E6-E10;

These artifacts are boundary evidence.  They must not be used to override the
current production mask policy or to claim transfer improvements that were not
observed.

Their smoke outputs and logs remain ignored and are not part of the compact
Git archive.
