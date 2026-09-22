# Progressive Route Disruption: 1000-image cross-architecture validation

## Research position

Patch-score analysis revealed that local-token semantic rankings reorganize substantially across depth. PRD turns that observation into a direct intervention: it repeatedly disrupts the evolving computation path with uniformly random local-token masks.

The attack contains exactly two paper mechanisms:

1. checkpoint-wise progressive random local-token hard zeroing;
2. kept-only RGB opponent-channel noise projected through the source model's initial RGB convolution and RMS-matched in feature space.

Phase pairing, raw multi-view gradient averaging, the Gaussian gradient residual, momentum, replay, and the projected pixel update are supporting implementation components.

## Progressive contract

For every attack step and augmentation group, PRD generates a fresh schedule. During both schedule construction and differentiable replay, checkpoints are traversed sequentially. Each mask is sampled uniformly from every local-token position on that checkpoint's native grid, applied immediately, and the modified state continues to the next checkpoint. The original view uses the schedule directly; the phase view uses a count-preserving spatial transformation of the same schedule.

| Source | Default checkpoints | Native grids | Drop counts | Opponent |
| --- | --- | --- | --- | ---: |
| ViT-B/16 | block3, block10 | 14×14, 14×14 | 10, 10 | 0.2 |
| CaiT-S24 | block17, block23 | 14×14, 14×14 | 2, 28 | 0.2 |
| PiT-B | stage2/block1, stage3/block2, stage3/block3 | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 |
| Visformer-S | stage2/block1, stage3/block1 | 14×14, 7×7 | 41, 10 | 0.4 |

The adapters expose only initial RGB-projection metadata, checkpoint traversal, token replacement/masking, and native forward completion. They do not expose local/global scoring features.

For cross-scale models, every selection retains its native grid. Phase transforms happen in image space and are mapped back with count-preserving top-k occupancy. Kept-only opponent noise uses the image-space union of all checkpoint masks and the true receptive-field geometry of the initial RGB projection.

## Formal transfer results

ASR is `1 - adversarial accuracy` over all 1000 evaluated adversarial samples. No target-clean-correct filtering is used. The target set contains eight Transformer and six CNN models. Overall is the complete 14-target mean; strict black-box excludes the source-matched target.

| Source | Overall ASR | Transformer avg | CNN avg | Strict black-box overall |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 85.20% | 90.45% | 78.20% | 84.29% |
| CaiT-S24 | 86.41% | 89.99% | 81.63% | 85.46% |
| PiT-B | 84.34% | 89.91% | 76.92% | 83.25% |
| Visformer-S | 80.66% | 83.19% | 77.28% | 79.19% |
| **Four-source mean** | **84.15%** | **88.39%** | **78.51%** | **83.05%** |

Every one of the 56 source-target evaluations used all 1000 adversarial samples and recorded zero skipped images.

### Per-target ASR

| Target | ViT source | CaiT source | PiT source | Visformer source |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 97.00% | 85.40% | 83.30% | 64.90% |
| LeViT-256 | 87.00% | 89.00% | 87.40% | 88.60% |
| PiT-B/224 | 88.00% | 88.40% | 98.50% | 87.10% |
| DeiT-B/16 | 90.10% | 90.70% | 91.30% | 80.20% |
| TNT-S/16 | 90.50% | 90.10% | 90.80% | 86.70% |
| ConViT-B | 88.20% | 89.60% | 89.70% | 77.70% |
| Visformer-S | 88.10% | 88.00% | 90.20% | 99.70% |
| CaiT-S24 | 94.70% | 98.70% | 88.10% | 80.60% |
| Inception-v3 | 80.50% | 84.70% | 82.20% | 85.40% |
| Inception-v4 | 77.70% | 82.50% | 79.80% | 85.00% |
| Inception-ResNet-v2 | 78.20% | 82.40% | 78.10% | 78.90% |
| ResNet-101 | 81.30% | 84.40% | 80.10% | 84.60% |
| Inception-v3-adv | 77.80% | 80.90% | 75.20% | 73.10% |
| Inception-ResNet-v2-adv | 73.70% | 74.90% | 66.10% | 56.70% |

## Auditable records

The completed runs are consolidated without changing their measurements into two PRD records:

- `results/prd_cross_arch_s1000.csv`: all 56 source-target ASRs, target families, source-match flags, and evaluated image counts;
- `results/prd_gradient_diagnostics_s1000.csv`: 20-view effective ranks and schedule/selection counts.

The four sources have effective ranks 19.51, 19.34, 19.46, and 18.39 respectively. Every formal run contains 1000 adversarial samples and records the expected 200/300 checkpoint selections per image for K2/K3 schedules.

## Verification boundary

- ViT uses only `block3,block10` with independent 10/196 drops.
- Every mask is uniformly random over all local-token positions.
- Original/phase schedules preserve per-checkpoint counts.
- Opponent noise is projected from RGB opponent directions and applied only outside the schedule's image-space drop union.
- Saved adversarial images remain within the `16/255` L-infinity budget.
- The implementation remains architecture-neutral across ViT, CaiT, PiT, and Visformer.
