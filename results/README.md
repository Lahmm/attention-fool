# Result archive

This directory indexes the retained experiment records. Selected formal
artifacts under `outputs/` are force-tracked even though generated outputs are
ignored by default.

## Promoted research mainline

The promoted research mainline is the progressive high-score attack represented
by `vit_progressive_patch_score_attack.py`, with the validated ViT-B/16 schedule
at checkpoint boundaries `(3, 7, 11)` and independent 5% local-token drops.
Its selector, checkpoint, and CLS-noise/gradient-residual transfer records are
retained under `outputs/csv/`.

The progressive implementation is currently ViT-only.  Migration into
`main.py` and adapter-backed CaiT, PiT, and Visformer support remain pending;
the existing `main.py` default must therefore be treated as a legacy executable
rather than the promoted research mainline.

## Historical cross-architecture baseline

The final-layer dynamic-mask `original_score_postdrop_phase_pair` pipeline is
retained as the four-source cross-architecture baseline.  Its report is
`experiments/mainline_data_aug_gaussian_story_s1000.md`.

The four 1000-image source-model runs and their Gaussian-residual counterparts
remain available as baseline CSV records. Smaller historical subsets and
alternate view/noise attack records have been removed.

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
promoted progressive mask policy or to claim transfer improvements that were
not observed.

Their smoke outputs and logs remain ignored and are not part of the compact
Git archive.
