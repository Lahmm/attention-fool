# Progressive cross-architecture mainline: implementation and 1000-image validation

Date: 2026-09-07

CaiT checkpoint follow-up: 2026-09-08

Code revisions used for the formal runs: initial matrix `cf4b9ada`, CaiT
checkpoint follow-up `c17d9ab5`

## Mainline definition

The production mainline is the independent `ProgressivePatchScoreAttacker` in
`progressive_attack.py`. It contains exactly the two paper mechanisms:

1. patch-score-guided progressive hard-zero routing at three checkpoints;
2. kept-only RGB opponent-channel random noise projected through the source
   model's initial RGB convolution and RMS-matched in feature space.

Score-global noise, phase pairs, Gaussian gradient residual, IID Gaussian
feature noise and random routing remain supporting controls. For ViT, the
canonical score-global setting is exposed under the old `score_cls_*` names as
a compatibility alias. The `*_active` metadata fields are derived booleans;
strengths remain numeric.

`progressive_attack.py` does not import, inherit or call `attack.py`.
`main.py` imports the legacy attacker only inside a non-progressive branch.
`vit_progressive_patch_score_attack.py` is now a thin CLI compatibility layer.

## Adapter contract and defaults

The attack owns selection, noise, phase pairing, replay, gradient aggregation,
MI/NI/TI, Gaussian residual and the projected pixel update. Each adapter owns
only the architecture-specific hidden-state traversal:

- begin from the initial RGB-projected state;
- advance to a registered checkpoint;
- expose local/global score features;
- apply a local-token mask;
- finish the native forward.

| Source | Default checkpoints | Global representation | Native grids | Drop counts |
| --- | --- | --- | --- | --- |
| ViT-B/16 | block3, block7, block11 | CLS | 14×14, 14×14, 14×14 | 10, 10, 10 |
| CaiT-S24 | block6, block18, block22 | GAP | 14×14, 14×14, 14×14 | 10, 10, 10 |
| PiT-B | stage1/block3, stage2/block5, stage3/block3 | CLS | 31×31, 16×16, 8×8 | 48, 13, 3 |
| Visformer-S | stage1/block4, stage2/block2, stage3/block3 | GAP | 28×28, 14×14, 7×7 | 39, 10, 2 |

For cross-scale models, every checkpoint mask retains its own grid. Phase
transforms happen in image space and are projected back with count-preserving
top-k occupancy. Kept-only opponent noise uses the image-space union of all
three masks, mapped through the true receptive fields of the initial RGB
projection.

## Completed verification gates

- Golden ViT parity: fixed-replay masks, every view gradient, raw mean,
  processed gradient and final adversarial pixels are bitwise identical to the
  pre-migration implementation. The golden hashes are retained in tests.
- Static independence: AST checks reject an `attack` import or parent class in
  `progressive_attack.py` and reject a top-level legacy import in `main.py`.
- Physical independence: moving `attack.py` outside the repository still
  permits imports and a complete progressive attack.
- Adapter parity: all four real timm models produce bitwise-equal native and
  no-mask resumed logits.
- Real gradients: all four adapters complete a two-view backward pass with
  finite gradients.
- Unit coverage: score-high/low/random sampling, replay, score noise metadata,
  opponent projection, Gaussian RMS matching, kept-only union, cross-grid
  phase count preservation, gradient aggregation and epsilon projection pass.
- Full-budget smoke: each source completes 8 images at 10 steps × 10 groups ×
  2 views. Every saved PNG has an observed maximum perturbation of 16/255.
- Small transfer: each smoke directory evaluates successfully on DeiT-B and
  ResNet-101.
- Legacy dispatch: the historical phase-pair branch completes a real one-image
  attack through `main.py`.
- Formal scale: four mainline sources and two ViT noise controls each complete
  1000 images. Every directory has 1000 adversarial PNGs, 1000 replay IDs,
  300 checkpoint selections per image and maximum saved-PNG L-infinity 16/255.
- Test suite: 59 tests pass; four optional/real tests are skipped in the default
  run, and the four-model real adapter test passes when explicitly enabled.

## Formal cross-architecture transfer results

ASR is `1 - adversarial accuracy` over all 1000 evaluated adversarial samples.
No target-clean-correct filtering is used. The default target set contains
seven Transformer and six CNN models.

| Source | Overall ASR | Transformer avg | CNN avg | Strict black-box overall* |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | **79.58%** | **84.94%** | **73.32%** | 79.58% |
| CaiT-S24 | 77.63% | 81.46% | 73.17% | 76.28% |
| PiT-B | 75.52% | 84.30% | 65.28% | 73.69% |
| Visformer-S | 71.46% | 76.83% | 65.20% | 69.12% |

`*` For CaiT, PiT and Visformer, strict black-box averages exclude the target
with the same architecture as the source. ViT-B/16 is not in the target list.

The per-target auditable records are:

- `outputs/csv/outputs_attack_progressive_mainline_vit_s1000_seed20260907.csv`
- `outputs/csv/outputs_attack_progressive_cait_b6_b18_b22_s1000_seed20260907.csv`
- `outputs/csv/outputs_attack_progressive_mainline_pit_s1000_seed20260907.csv`
- `outputs/csv/outputs_attack_progressive_mainline_visformer_s1000_seed20260907.csv`

## Controlled comparisons

### CaiT progressive checkpoint retuning

The initial cross-architecture matrix used CaiT checkpoints block6/14/22. A
same-seed 1000-image follow-up changed only the middle checkpoint to block18;
the high-score selector spelling is behaviorally identical to the earlier
`patch_score` alias.

| CaiT checkpoints | Overall | Transformer | CNN | Strict black-box overall |
| --- | ---: | ---: | ---: | ---: |
| block6/14/22 | 75.75% | 79.63% | 71.22% | 74.35% |
| block6/18/22 | **77.63%** | **81.46%** | **73.17%** | **76.28%** |

The selected block6/18/22 schedule improves Overall by 1.88pp, Transformer by
1.83pp, CNN by 1.95pp and strict black-box Overall by 1.93pp. All thirteen
individual targets improve. The original block6/14/22 record remains at
`outputs/csv/outputs_attack_progressive_mainline_cait_s1000_seed20260907.csv`.

Moving only the final checkpoint to block24 was a clear negative control on the
same 192-image screening subset: strict black-box Overall fell from 76.22% to
63.80%, while source CaiT ASR increased. This is retained as evidence that
terminal routing can overfit the source rather than improve transfer.

### Progressive routing versus historical final-layer routing

These same-seed 20260903 ViT experiments predate the generic adapter but are
covered by the bitwise golden parity gate.

| Routing | Overall | Transformer | CNN |
| --- | ---: | ---: | ---: |
| historical final layer | 78.45% | 83.56% | 72.48% |
| progressive block3/7/11 high | **79.75%** | **85.04%** | **73.58%** |

Progressive improves by 1.31pp Overall, 1.49pp Transformer and 1.10pp CNN.

### Patch score versus random routing

| Selector | Overall | Transformer | CNN |
| --- | ---: | ---: | ---: |
| random, block3/7/11 | 79.16% | 84.36% | 73.10% |
| score-high, block3/7/11 | **79.75%** | **85.04%** | **73.58%** |

With all other settings fixed, score-high improves by 0.59pp Overall, 0.69pp
Transformer and 0.48pp CNN.

### Score noise and Gaussian residual controls

| Score-global noise | Gradient residual | Overall | Transformer | CNN |
| --- | --- | ---: | ---: | ---: |
| on | on | **79.75%** | **85.04%** | **73.58%** |
| on | off | 79.17% | 84.41% | 73.05% |
| off | on | 79.36% | 84.89% | 72.92% |
| off | off | 79.08% | 84.21% | 73.10% |

These factors are retained as supporting improvements, not additional paper
mechanisms.

### Opponent noise versus IID Gaussian and noise-off

All rows use seed 20260907 and the same progressive masks, score noise, phase
pairs, gradient residual and optimization budget.

| Kept-only feature noise | Overall | Transformer | CNN |
| --- | ---: | ---: | ---: |
| none | 62.52% | 71.57% | 51.95% |
| IID Gaussian, RMS matched | 78.42% | **85.39%** | 70.28% |
| RGB opponent projection, RMS matched | **79.58%** | 84.94% | **73.32%** |

Opponent noise improves over noise-off by 17.05pp Overall, 13.37pp
Transformer and 21.37pp CNN. Against IID Gaussian it improves 1.16pp Overall
and 3.03pp CNN, while IID Gaussian is 0.44pp higher on the Transformer subset.
The supported claim is therefore stronger cross-family/CNN transfer and higher
overall ASR, not uniform superiority on every target family.

## Gradient complementarity diagnostics

The 1000-image optimization diagnostics show that feature noise makes the
actual 20-view gradient ensemble much less redundant:

| Noise | View cosine to mean | Sign agreement | Effective rank | MI cumulative cosine |
| --- | ---: | ---: | ---: | ---: |
| none | 0.4382 | 0.6383 | 10.45 | 0.5577 |
| IID Gaussian | 0.2671 | 0.5745 | 19.32 | 0.6037 |
| opponent | 0.2611 | 0.5671 | **19.54** | **0.6134** |

A fixed-replay 32-image diagnostic further found processed-gradient cosine of
only 0.034 between opponent and noise-off, 0.162 between opponent and IID
Gaussian, and 0.203 between score-high and random routing. Thus the routing and
noise controls contribute materially different directions rather than nearly
duplicating one another.

Direct cosine against clean DeiT-B and ResNet-101 gradients was near zero for
all methods, and one-step held-out loss changes did not predict the full
iterative ASR ordering. Those measurements are retained as a negative result:
they must not be used alone as evidence of transferability. The complementarity
claim rests on the combination of distinct source directions, higher ensemble
rank, and the matched 1000-image iterative transfer gains above.

## Conclusion

All implementation, isolation, adapter, regression, full-budget, formal
1000-image, transfer and control-variable gates in the migration plan are
complete. The independent progressive attack replaces the historical
final-layer route as the project mainline, while `attack.py` remains an
isolated legacy implementation.
