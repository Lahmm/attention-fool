# New 1000-image configuration validation TODO

Date: 2026-09-15

Status: pending execution.

## Objective

Validate on 1000 images the highest-ASR configuration found for each of the
four source architectures in the fixed 192-image screen. These are screening
winners, not yet promoted defaults. Each configuration must be regenerated on
the full 1000-image set; results must not be extrapolated from earlier runs.

## Candidate configurations

| Source model | Checkpoints | Drop counts | Drop ratios | Score mode | Opponent strength | 192-image ASR |
|---|---|---:|---|---|---:|---:|
| ViT-B/16 | `block3,block10` | 10/10 | `0.051020408163,0.051020408163` | `cosine` | 0.2 | 86.26% |
| CaiT-S24 | `block17_gap,block23_gap` | 2/28 | `0.010204081633,0.142857142857` | `gap_projection` | 0.2 | 87.94% |
| PiT-B | `stage2_block1,stage3_block2,stage3_block3` | 5/2/6 | `0.02081165,0.03125,0.09375` | `cosine` | 0.4 | 87.42% |
| Visformer-S | `stage2_block1,stage3_block1` | 41/10 | `0.209183673469,0.204081632653` | `gap_projection` | 0.4 | 82.61% |

## Fixed protocol

- `attack-method=progressive`
- `progressive-patch-selector=high`
- `score-window-ratio=0.5`
- `score-global-noise-strength=0.2`
- 10 attack steps and 10 augmentation groups
- two phase-paired views
- `sample-offset=0`
- 1000 attacked samples in the same annotated-image order
- use the same campaign seed and record it in every artifact
- rebuild the three-/two-checkpoint schedule from current adversarial pixels at
  every step and augmentation group
- recompute global/local scores on the sequentially updated token state at each
  checkpoint
- sample from the score-high half and hard-zero selected local tokens
- permit the same spatial position to be selected again at later checkpoints
- evaluate all seven Transformer and six CNN targets
- define ASR as `1 - adversarial accuracy` over all evaluated adversarial
  samples, without clean-correct filtering

## Execution checklist

- [ ] Generate 1000 adversarial images for ViT-B/16.
- [ ] Run the complete transfer evaluation for ViT-B/16.
- [ ] Generate 1000 adversarial images for CaiT-S24.
- [ ] Run the complete transfer evaluation for CaiT-S24.
- [ ] Generate 1000 adversarial images for PiT-B.
- [ ] Run the complete transfer evaluation for PiT-B.
- [ ] Generate 1000 adversarial images for Visformer-S.
- [ ] Run the complete transfer evaluation for Visformer-S.
- [ ] Verify zero skipped images for every target and source configuration.
- [ ] Verify every attack directory contains exactly 1000 adversarial PNGs,
  `attack_params.json`, `gradient_diagnostics.json`, and
  `replay_manifest.json`.
- [ ] Record Overall, Transformer, CNN, and strict black-box ASR.
- [ ] Record all thirteen per-target ASRs.
- [ ] Compare each result with its corresponding previous 1000-image baseline.
- [ ] Audit configuration provenance before promoting any new default.

## Interpretation guardrails

- Treat the four runs as independent architecture validations; do not infer
  that a setting transfers between architecture adapters.
- The CaiT and Visformer candidates use GAP projection. ViT and PiT retain the
  CLS-compatible cosine score.
- The Visformer candidate combines two improvements already shown to be
  complementary at 192 images: GAP projection and opponent strength 0.4.
- Do not promote a 192-image lead unless it persists on this 1000-image
  validation under the same ASR definition.
