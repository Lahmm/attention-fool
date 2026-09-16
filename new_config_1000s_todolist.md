# New 1000-image configuration validation TODO

Date: 2026-09-15

Status: complete. All four attacks and thirteen-target transfer evaluations
finished with zero skipped images on 2026-09-16.

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

- [x] Generate 1000 adversarial images for ViT-B/16.
- [x] Run the complete transfer evaluation for ViT-B/16.
- [x] Generate 1000 adversarial images for CaiT-S24.
- [x] Run the complete transfer evaluation for CaiT-S24.
- [x] Generate 1000 adversarial images for PiT-B.
- [x] Run the complete transfer evaluation for PiT-B.
- [x] Generate 1000 adversarial images for Visformer-S.
- [x] Run the complete transfer evaluation for Visformer-S.
- [x] Verify zero skipped images for every target and source configuration.
- [x] Verify every attack directory contains exactly 1000 adversarial PNGs,
  `attack_params.json`, `gradient_diagnostics.json`, and
  `replay_manifest.json`.
- [x] Record Overall, Transformer, CNN, and strict black-box ASR.
- [x] Record all thirteen per-target ASRs.
- [x] Compare each result with its corresponding previous 1000-image baseline.
- [x] Audit configuration provenance before promoting any new default.

## Interpretation guardrails

- Treat the four runs as independent architecture validations; do not infer
  that a setting transfers between architecture adapters.
- The CaiT and Visformer candidates use GAP projection. ViT and PiT retain the
  CLS-compatible cosine score.
- The Visformer candidate combines two improvements already shown to be
  complementary at 192 images: GAP projection and opponent strength 0.4.
- Do not promote a 192-image lead unless it persists on this 1000-image
  validation under the same ASR definition.

## Completed 1000-image results

| Source | Overall | Transformer | CNN | Strict black-box | Previous Overall | Overall gain |
|---|---:|---:|---:|---:|---:|---:|
| ViT-B/16 | **84.35%** | 89.39% | 78.48% | **84.35%** | 80.28% | **+4.07pp** |
| CaiT-S24 | **86.20%** | 90.39% | **81.32%** | **85.22%** | 84.15% | **+2.05pp** |
| PiT-B | **84.88%** | **91.21%** | 77.50% | **83.73%** | 80.44% | **+4.45pp** |
| Visformer-S | 79.77% | 83.16% | 75.82% | 78.13% | 74.25% | **+5.52pp** |

Strict black-box excludes the target with the same architecture as the source.
ViT-B/16 itself is not in the target list, so its strict value equals Overall,
matching the established reporting convention.

All four 192-image winners retained substantial positive Overall gains at 1000
images. The corresponding strict gains versus the prior selected records are
+4.07pp for ViT, +2.20pp for CaiT, +4.87pp for PiT, and +6.01pp for
Visformer. This confirms that none of the gains is explained only by the
same-architecture target.

## Per-target transfer ASR

| Target | ViT source | CaiT source | PiT source | Visformer source |
|---|---:|---:|---:|---:|
| LeViT-256 | 86.50% | 88.50% | 87.20% | 87.20% |
| PiT-B/224 | 87.70% | 88.50% | 98.80% | 83.20% |
| DeiT-B/16 | 90.10% | 90.20% | 91.60% | 76.50% |
| TNT-S/16 | 90.20% | 90.00% | 90.90% | 84.90% |
| ConViT-B | 88.40% | 89.20% | 90.90% | 73.70% |
| Visformer-S | 87.90% | 88.30% | 90.80% | 99.40% |
| CaiT-S24 | 94.90% | 98.00% | 88.30% | 77.20% |
| Inception-v3 | 80.10% | 84.10% | 83.00% | 83.80% |
| Inception-v4 | 78.00% | 81.70% | 80.00% | 83.90% |
| Inception-ResNet-v2 | 78.30% | 81.40% | 78.50% | 76.90% |
| ResNet-101 | 82.70% | 83.90% | 81.20% | 82.90% |
| Inception-v3-adv | 78.30% | 80.80% | 75.10% | 72.50% |
| Inception-ResNet-v2-adv | 73.50% | 76.00% | 67.20% | 54.90% |

## Retained artifacts

Each attack directory contains 1000 PNGs and the three required metadata
files. The complete transfer records are:

- `outputs/csv/outputs_attack_newconfig1000_vit_b3_b10_c10_10_s1000_offset0_seed20260907.csv`
- `outputs/csv/outputs_attack_newconfig1000_cait_b17_b23_c02_28_projection_s1000_offset0_seed20260907.csv`
- `outputs/csv/outputs_attack_newconfig1000_pit_s2b1_s3b2_s3b3_c05_02_06_opp04_s1000_offset0_seed20260907.csv`
- `outputs/csv/outputs_attack_newconfig1000_vis_s2b1_s3b1_c41_10_projection_opp04_s1000_offset0_seed20260907.csv`

## CaiT GAP-projection schedule follow-ups

Two additional 1000-image CaiT runs isolate projection on the former K3
schedule and test a balanced K2 schedule. All use opponent strength 0.2,
`selector=high`, `w=0.5`, and the fixed protocol above.

| Checkpoints | Drop counts | Score | Overall | Transformer | CNN | Strict black-box |
|---|---:|---|---:|---:|---:|---:|
| block5/17/23 | 10/10/10 | cosine | 84.15% | 87.97% | 79.68% | 83.02% |
| block5/17/23 | 10/10/10 | gap_projection | **84.78%** | **88.81%** | **80.07%** | **83.68%** |
| block17/23 | 15/15 | cosine | 83.48% | 87.77% | 78.48% | 82.23% |
| block17/23 | 15/15 | gap_projection | **83.90%** | **87.83%** | **79.32%** | **82.70%** |
| block17/23 | 2/28 | gap_projection | **86.20%** | **90.39%** | **81.32%** | **85.22%** |

Projection improves the former 10/10/10 default by 0.63pp Overall and 0.67pp
strict. On 15/15 it improves Overall by 0.42pp and strict by 0.48pp, below the
0.5pp screening threshold. Redistributing the same K2 total budget from 15/15
to 2/28 is much more important: it adds 2.30pp Overall, 2.56pp Transformer,
2.00pp CNN, and 2.52pp strict. The full-set evidence therefore confirms both
that projection generalizes to CaiT and that CaiT's dominant schedule effect is
strong final-checkpoint concentration.

The added auditable records are:

- `outputs/csv/outputs_attack_caitproj1000_k3_b5_b17_b23_c10_10_10_s1000_offset0_seed20260907.csv`
- `outputs/csv/outputs_attack_caitproj1000_k2_b17_b23_c15_15_s1000_offset0_seed20260907.csv`
