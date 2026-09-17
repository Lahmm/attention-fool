# Result archive

This directory indexes the retained experiment records. Selected formal
artifacts under `outputs/` are force-tracked even though generated outputs are
ignored by default.

## Promoted research mainline

The promoted research mainline is the progressive high-score attack implemented
by `progressive_attack.py` and dispatched by `main.py`. ViT-B/16 uniformly uses
`block3,block10` with independent `0.051020408163` local-token drop ratios
(10 tokens at each checkpoint). No separate ViT reference schedule is retained.

The independent progressive implementation is integrated into `main.py`
through architecture adapters for ViT, CaiT, PiT, and Visformer. All four
selected defaults have completed 1000-image attack and 13-target transfer
evaluation. The implementation contract, selected configurations, screening
results, and auditable CSV paths are recorded in
`experiments/progressive_cross_arch_mainline_s1000.md`.

The `main.py` defaults are selected under `high`, `score-window-ratio=0.5`, and
`K>1`: ViT block3/10 with 10/10 drops, CaiT block17/23 with 2/28, PiT
stage2-block1/stage3-block2/stage3-block3 with 5/2/6, and Visformer
stage2-block1/stage3-block1 with 41/10.

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
