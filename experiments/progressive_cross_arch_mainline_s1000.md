# Progressive cross-architecture mainline: implementation and 1000-image validation

Initial formal matrix: 2026-09-07

Checkpoint, ratio, and selector follow-ups audited through: 2026-09-15

Code revisions used for the formal runs include: initial matrix `cf4b9ada`,
CaiT block6/18/22 follow-up `c17d9ab5`, block6/17/23 follow-up `3dd43259`,
block5/17/23 follow-up `3abd7c0b`, and the generalized-selector revision
`552f03b` used by the ViT extreme-high follow-up.

## Mainline definition

The production mainline is the independent `ProgressivePatchScoreAttacker` in
`progressive_attack.py`. It contains exactly the two paper mechanisms:

1. patch-score-guided progressive hard-zero routing at architecture-specific
   checkpoint schedules;
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
| ViT-B/16 | block3, block11 | CLS | 14×14, 14×14 | 10, 10 |
| CaiT-S24 | block5, block17, block23 | GAP | 14×14, 14×14, 14×14 | 10, 10, 10 |
| PiT-B | stage2/block1, stage3/block2, stage3/block3 | CLS | 16×16, 8×8, 8×8 | 5, 2, 6 |
| Visformer-S | stage2/block1, stage3/block1 | GAP | 14×14, 7×7 | 41, 10 |

For cross-scale models, every checkpoint mask retains its own grid. Phase
transforms happen in image space and are projected back with count-preserving
top-k occupancy. Kept-only opponent noise uses the image-space union of all
checkpoint masks, mapped through the true receptive fields of the initial RGB
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
  The current suite also covers arbitrary nonzero checkpoint counts, exact
  checkpoint/ratio cardinality, and extreme-high/extreme-low construction.
- Full-budget smoke: each source completes 8 images at 10 steps × 10 groups ×
  2 views. Every saved PNG has an observed maximum perturbation of 16/255.
- Small transfer: each smoke directory evaluates successfully on DeiT-B and
  ResNet-101.
- Legacy dispatch: the historical phase-pair branch completes a real one-image
  attack through `main.py`.
- Formal scale: the selected defaults for all four sources, the initial
  cross-architecture matrix, two ViT noise controls, and the retained
  full-scale checkpoint/selector follow-ups complete 1000 images. Every formal
  directory has 1000 adversarial PNGs; K2/K3 runs record 200/300 checkpoint
  selections per image and maximum saved-PNG L-infinity 16/255.
- Test suite: 68 tests pass; four optional/real tests are skipped in the default
  run, and the four-model real adapter test passes when explicitly enabled.

## Formal cross-architecture transfer results

ASR is `1 - adversarial accuracy` over all 1000 evaluated adversarial samples.
No target-clean-correct filtering is used. The default target set contains
seven Transformer and six CNN models.

| Source | Overall ASR | Transformer avg | CNN avg | Strict black-box overall* |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | **80.28%** | 85.43% | 74.28% | **80.28%** |
| CaiT-S24 | **84.15%** | **87.97%** | **79.68%** | **83.02%** |
| PiT-B | 80.44% | **88.87%** | 70.60% | 78.86% |
| Visformer-S | 74.25% | 78.99% | 68.73% | 72.13% |

`*` For CaiT, PiT and Visformer, strict black-box averages exclude the target
with the same architecture as the source. ViT-B/16 is not in the target list.

The per-target auditable records are:

- `outputs/csv/outputs_attack_scanD_vit_k2_budget20_confirm_s1000_offset0_seed20260907.csv`
- `outputs/csv/outputs_attack_progressive_cait_b5_b17_b23_s1000_seed20260907.csv`
- `outputs/csv/outputs_attack_progressive_pit_l2_s1000_seed20260907.csv`
- `outputs/csv/outputs_attack_scanD_visformer_k2eqr_s2b1_s3b1_confirm_s1000_offset0_seed20260907.csv`

The initial adapter defaults and the final selected defaults are both retained
for auditability:

| Source | Initial Overall | Selected Overall | Gain | Initial strict | Selected strict | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | 79.58% | **80.28%** | +0.71pp | 79.58% | **80.28%** | +0.71pp |
| CaiT-S24 | 75.75% | **84.15%** | +8.40pp | 74.35% | **83.02%** | +8.67pp |
| PiT-B | 75.52% | **80.44%** | +4.92pp | 73.69% | **78.86%** | +5.17pp |
| Visformer-S | 71.46% | **74.25%** | +2.79pp | 69.12% | **72.13%** | +3.01pp |

## Controlled comparisons

### CaiT progressive checkpoint retuning

The initial cross-architecture matrix used CaiT checkpoints block6/14/22.
Same-seed 1000-image follow-ups first moved the middle checkpoint to block18,
then identified block17, and finally moved the late checkpoint from block22 to
block23 and refined the early checkpoint from block6 to block5. Historical runs
that recorded the former `patch_score` selector used the same behavior as
`high`; the current progressive CLI retains only the explicit `high` spelling.

| CaiT checkpoints | Overall | Transformer | CNN | Strict black-box overall |
| --- | ---: | ---: | ---: | ---: |
| block16/20/24 | 63.87% | 70.03% | 56.68% | 61.11% |
| block6/14/22 | 75.75% | 79.63% | 71.22% | 74.35% |
| block6/18/22 | 77.63% | 81.46% | 73.17% | 76.28% |
| block6/17/22 | 81.44% | 85.37% | 76.85% | 80.21% |
| block6/17/23 | 83.81% | 87.70% | 79.27% | 82.68% |
| block5/17/23 | **84.15%** | **87.97%** | **79.68%** | **83.02%** |

Relative to block6/17/23, the selected block5/17/23 schedule improves Overall
by 0.34pp, Transformer by 0.27pp, CNN by 0.42pp and strict black-box Overall by
0.34pp. Nine of thirteen individual targets improve. Relative to the initial
block6/14/22 schedule, the full-run gains are 8.40pp Overall and 8.67pp strict
black-box Overall. The original record remains at
`outputs/csv/outputs_attack_progressive_mainline_cait_s1000_seed20260907.csv`.

Moving only the final checkpoint to block24 was a clear negative control on the
same 192-image screening subset: strict black-box Overall fell from 76.22% to
63.80%, while source CaiT ASR increased. This is retained as evidence that
terminal routing can overfit the source rather than improve transfer.

After block6/17/23 was established, the early checkpoint was screened on the
same 192-image offset-576 subset while block17/23 and all other settings were
fixed:

| Early checkpoint | Overall | Strict black-box overall |
| --- | ---: | ---: |
| block4 | 84.13% | 82.99% |
| **block5** | **85.02%** | **83.90%** |
| block6 | 84.13% | 82.90% |
| block7 | 83.81% | 82.73% |
| block8 | 82.25% | 81.08% |
| block10 | 83.21% | 81.99% |

This subset selected block5, which then retained a +0.34pp Overall advantage
over block6 in the full 1000-image verification.

### PiT ratio and checkpoint retuning

PiT pooling changes the local-token count from 31×31 to 16×16 and then 8×8.
The first screen therefore redistributed the per-stage drop budget instead of
forcing equal token counts. All five rows below use the initial
stage1/block3, stage2/block5, stage3/block3 checkpoints and the same 192-image
offset-0 subset.

| Ratio plan | Drop counts | Overall | Strict black-box overall |
| --- | ---: | ---: | ---: |
| A, early-heavy | 70/11/2 | 78.29% | 76.74% |
| B, uniform 5% | 48/13/3 | 77.56% | 75.87% |
| C, mild-late | 29/14/4 | 78.85% | 77.26% |
| **D, strong-late** | **20/8/6** | **79.17%** | **77.47%** |
| E, very-late | 11/6/7 | 78.81% | 77.13% |

On the independent offset-192 subset, D remained above C (74.08% versus
73.60% Overall; 72.22% versus 71.74% strict). Its 1000-image result was 76.58%
Overall, 85.76% Transformer, 65.87% CNN, and 74.73% strict, compared with
75.52%/84.30%/65.28%/73.69% for the initial uniform-ratio run.

With D's ratios fixed, a local layer screen on offset 384 gave:

| ID | Checkpoints | Overall | Strict black-box overall |
| --- | --- | ---: | ---: |
| P0 | s1b3 / s2b5 / s3b3 | 76.60% | 74.70% |
| P1 | s1b3 / s2b4 / s3b3 | 76.80% | 74.96% |
| **P2** | **s1b3 / s2b6 / s3b3** | **77.32%** | **75.48%** |
| P3 | s1b3 / s2b5 / s3b2 | 72.52% | 70.31% |
| P4 | s1b3 / s2b6 / s3b2 | 72.04% | 69.97% |

The subsequent topology screen expanded beyond one checkpoint per stage on
the offset-576 subset:

| ID | Checkpoints | Overall | Strict black-box overall |
| --- | --- | ---: | ---: |
| Q0 | s1b3 / s2b6 / s3b3 | 78.49% | 76.78% |
| Q1 | s2b1 / s2b6 / s3b3 | 79.73% | 78.12% |
| Q2 | s2b3 / s2b6 / s3b3 | 79.29% | 77.69% |
| Q3 | s1b3 / s3b1 / s3b3 | 80.65% | 79.08% |
| **Q4** | **s2b1 / s3b1 / s3b3** | **82.25%** | **80.86%** |
| Q5 | s2b3 / s3b1 / s3b3 | 79.81% | 78.21% |

Q4 was then refined on offset 768. The layer variants used the D-derived
ratios, which become native-grid counts 5/2/6 at these checkpoints:

| ID | Checkpoints | Overall | Strict black-box overall |
| --- | --- | ---: | ---: |
| L0 | s2b1 / s3b1 / s3b3 | 77.40% | 75.61% |
| L1 | s2b2 / s3b1 / s3b3 | 76.92% | 75.09% |
| **L2** | **s2b1 / s3b2 / s3b3** | **79.93%** | **78.26%** |
| L3 | s2b2 / s3b2 / s3b3 | 79.37% | 77.69% |

Ratio-only refinements around L0 did not exceed L2: R1 through R5 obtained
75.44%, 78.41%, 77.60%, 77.40%, and 75.44% Overall, respectively. Combining
the best ratio-only candidate R2 (counts 5/4/4) with L2 reduced Overall to
75.80% and strict Overall to 73.78%. The selected L2 therefore retains counts
5/2/6 and completed the full verification:

| PiT configuration, 1000 images | Overall | Transformer | CNN | Strict black-box overall |
| --- | ---: | ---: | ---: | ---: |
| Initial uniform | 75.52% | 84.30% | 65.28% | 73.69% |
| D ratios, initial layers | 76.58% | 85.76% | 65.87% | 74.73% |
| **L2 selected default** | **80.44%** | **88.87%** | **70.60%** | **78.86%** |

The auditable L2 record is
`outputs/csv/outputs_attack_progressive_pit_l2_s1000_seed20260907.csv`.

### Visformer checkpoint and ratio retuning

The initial six-way 192-image topology screen showed that moving all
checkpoints late was not beneficial:

| ID | Checkpoints | Overall | Strict black-box overall |
| --- | --- | ---: | ---: |
| V0 | s1b4 / s2b2 / s3b3 | 75.00% | 72.92% |
| V1 | s1b7 / s2b4 / s3b3 | 74.00% | 71.83% |
| V2 | s2b1 / s3b1 / s3b3 | 74.80% | 72.70% |
| V3 | s2b1 / s3b2 / s3b3 | 73.72% | 71.53% |
| V4 | s2b2 / s3b1 / s3b3 | 73.56% | 71.35% |
| V5 | s2b2 / s3b2 / s3b3 | 72.64% | 70.36% |

Keeping V0's layers and redistributing the nominal 15% ratio budget also
failed to beat the uniform R0 allocation:

| Ratio plan | Ratios | Overall | Strict black-box overall |
| --- | ---: | ---: | ---: |
| **R0** | **5% / 5% / 5%** | **75.00%** | **72.92%** |
| R1 | 7% / 4% / 4% | 74.60% | 72.48% |
| R2 | 6% / 6% / 3% | 73.92% | 71.74% |
| R3 | 4% / 7% / 4% | 73.92% | 71.74% |
| R4 | 4% / 5% / 6% | 74.84% | 72.74% |
| R5 | 3% / 4% / 8% | 74.24% | 72.09% |

R0 was therefore fixed while the early/middle/late layer neighborhoods and
their interactions were expanded. The most relevant local moves on the first
offset-0 screen were A1=s1b1/s2b2/s3b3 at 75.96% Overall, A2=s1b2/s2b2/s3b3
at 75.88%, and the early-final C1=s1b4/s2b2/s3b1 at 75.12%; the remaining
single-axis and late-only candidates ranged from 72.28% to 74.88%.

The interaction screen then produced:

| ID | Checkpoints | Overall | Strict black-box overall |
| --- | --- | ---: | ---: |
| I1 | s1b1 / s2b2 / s3b1 | 75.60% | 73.57% |
| I2 | s1b1 / s2b1 / s3b3 | 75.52% | 73.48% |
| **I3** | **s1b1 / s2b1 / s3b1** | **76.08%** | **74.09%** |
| I4 | s1b2 / s2b2 / s3b1 | 74.60% | 72.53% |
| I5 | s1b2 / s2b1 / s3b3 | 75.12% | 73.05% |
| I6 | s1b2 / s2b1 / s3b1 | 75.64% | 73.65% |

The offset-192 replication retained the ordering among the finalists: V0,
A1, A2, and I3 obtained 68.83%, 68.99%, 68.55%, and 69.15% Overall,
respectively; their strict scores were 66.23%, 66.45%, 65.97%, and 66.71%.
Both A1 and I3 were then run on all 1000 images:

| Visformer configuration, 1000 images | Overall | Transformer | CNN | Strict black-box overall |
| --- | ---: | ---: | ---: | ---: |
| Initial V0 | 71.46% | 76.83% | 65.20% | 69.12% |
| A1, s1b1/s2b2/s3b3 | 72.78% | **78.33%** | 66.30% | 70.54% |
| I3 former K3 default, s1b1/s2b1/s3b1 | 73.16% | 78.24% | 67.23% | 70.97% |
| **Current K2 default, s2b1/s3b1, 41/10** | **74.25%** | **78.99%** | **68.73%** | **72.13%** |

I3 won the original K3 comparison, while the later equal-ratio K2 follow-up
became the constrained-ASR default. The auditable full records are
`outputs/csv/outputs_attack_progressive_visformer_a1_s1b1_s2b2_s3b3_s1000_seed20260907.csv`
`outputs/csv/outputs_attack_progressive_visformer_i3_s1b1_s2b1_s3b1_s1000_seed20260907.csv`,
with the current K2 record at
`outputs/csv/outputs_attack_scanD_visformer_k2eqr_s2b1_s3b1_confirm_s1000_offset0_seed20260907.csv`.

### Progressive routing versus historical final-layer routing

These same-seed 20260903 ViT experiments predate the generic adapter but are
covered by the bitwise golden parity gate.

| Routing | Overall | Transformer | CNN |
| --- | ---: | ---: | ---: |
| historical final layer | 78.45% | 83.56% | 72.48% |
| progressive block3/7/11 high | **79.75%** | **85.04%** | **73.58%** |

Progressive improves by 1.31pp Overall, 1.49pp Transformer and 1.10pp CNN.

For completeness, the older cross-architecture final-layer records and the
then-selected progressive results are summarized below. Only the ViT pair above is
a same-seed controlled routing comparison. The other historical rows use seed
20260716 and raw-gradient final-layer runs, whereas the progressive rows use
seed 20260907 and the promoted optimization stack; their deltas are context,
not causal estimates of routing quality.

| Source | Historical final-layer Overall | Then-selected progressive Overall | Difference |
| --- | ---: | ---: | ---: |
| ViT-B/16, controlled pair | 78.45% | **79.75%** | +1.31pp |
| CaiT-S24, historical context | **87.00%** | 84.15% | -2.85pp |
| PiT-B, historical context | **82.78%** | 80.44% | -2.34pp |
| Visformer-S, historical context | **75.22%** | 73.16% | -2.06pp |

The historical provenance remains documented separately in
`experiments/mainline_data_aug_gaussian_story_s1000.md`.

### ViT progressive selector controls

| Selector | Overall | Transformer | CNN |
| --- | ---: | ---: | ---: |
| high-half random (`high`) | **79.75%** | **85.04%** | **73.58%** |
| low-half random (`low`) | 79.26% | 84.84% | 72.75% |
| all-token uniform (`random`) | 79.16% | 84.36% | 73.10% |

These three block3/7/11 runs use the same 1000 images and seed 20260903. With
all other settings fixed, high improves over random by 0.59pp Overall, 0.69pp
Transformer, and 0.48pp CNN. Low remains close on Transformer targets but is
0.83pp below high on CNN targets.

After the selector interface was generalized, an extreme-high follow-up was
run on block3/7/11 with seed 20260907 and compared with the formal high run at
the same seed:

| Selector, seed 20260907 | Overall | Transformer | CNN |
| --- | ---: | ---: | ---: |
| high-half random (`high`) | **79.58%** | **84.94%** | **73.32%** |
| exact top-score tail (`extreme-high`) | 77.82% | 83.11% | 71.63% |

Extreme-high is lower by 1.76pp Overall, 1.83pp Transformer, and 1.68pp CNN;
all 13 individual target ASRs decrease. This indicates that score guidance is
most useful together with stochastic coverage of the high-score half, rather
than repeatedly taking only each checkpoint's current top-ranked tail. Patch
scores are recomputed on the sequentially updated state at every checkpoint,
so this result does not imply that block3, block7, and block11 select identical
spatial positions. Extreme-low has implementation and smoke coverage but no
1000-image transfer result and is therefore not ranked here.

The extreme-high record is
`outputs/csv/outputs_attack_progressive_vit_extreme_high_c3711_s1000_seed20260907.csv`.

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
