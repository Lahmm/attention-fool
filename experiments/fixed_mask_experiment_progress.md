# Fixed-mask routing experiment progress

Recorded at `2026-07-28T14:54:11Z`. The experiment is intentionally paused
after selector job 5 at the user's request.

## Completed

- 38/38 routing-calibration attacks and all 114 off-diagonal evaluations.
- Frozen global polarity: `high`.
- Frozen layers: ViT `block6`, CaiT `block24_gap`, PiT `stage3_block2`,
  Visformer `stage3_block2`.
- H2 fixed-mask gradient diagnostics for all four source models.
- Grad-CAM Protocol A at the newly frozen layers.
- Selector jobs 1-5, each with 500/500 saved adversarial images:
  `selected`, `opposite`, `deviation`, `random`, and `no_drop` for ViT.

The selector manifest SHA-256 is
`fa5c8a5251b45c051367ea9d9dc1385fddc22f728460da8b5f16ce0e5eb5b61e`.
The frozen-routing config SHA-256 is
`412d3c22a9dcb6697660618cb423ddb4750290b6ff3bcd63e4f7ead4b5f57114`.

## Resume point

- Next selector job: 6/28, ViT `final_layer`.
- Selector jobs 6-28 remain (23 jobs, 11,500 adversarial images).
- After all 28 jobs: run the 13-target paired transfer evaluation, compute
  `selected - random` bootstrap 95% confidence intervals and the Patch-score
  versus Grad-CAM comparison, then write the final H1-H3 report.

The machine-readable checkpoint is
`outputs/research/selector_suite_fixed_mask/progress_checkpoint.json`. The
primary attack orchestrator and automatic finalizer were both stopped; no
sixth-job output was created. Resume logic must validate and skip the five
completed jobs rather than overwrite them.
