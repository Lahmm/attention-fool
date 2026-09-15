# Progressive routing follow-up: 192-image exploration

Date: 2026-09-15

Status: 38-run base matrix complete; CaiT GAP-projection two-level replication
complete; remaining adaptive opponent-strength runs pending.

## Fixed protocol

All runs use the first 192 annotated images (`sample-offset=0`), seed 20260907,
10 attack steps, 10 augmentation groups, two phase-paired views, MI, Gaussian
residual sigma 4/alpha 0.75, `selector=high`, and
`score-window-ratio=0.5`. Schedules are rebuilt from current adversarial pixels
at every step/group; scores are recomputed on sequentially updated checkpoint
states; selected tokens are hard-zeroed and positions may be selected again at
later checkpoints. Transfer ASR is `1 - adversarial accuracy` over all samples
on the standard seven Transformer and six CNN targets, without clean-correct
filtering.

The primary selection metric is Overall ASR. Transformer, CNN, per-target and
strict black-box ASR are retained as diagnostics. Deltas below 0.5pp are treated
as screening ties, 0.5--1.0pp as weak leads, and at least 1.0pp as meaningful
screening leads. Runs from different offsets are not compared.

## Matrix

The executable matrix is `experiments/run_progressive_followup_192.sh`. It has
38 unique base runs:

- four same-revision architecture baselines;
- seven new CaiT late-heavy schedules: four K2 splits at block17/23 and three
  K3 splits at block5/17/23, all with total drop count 30;
- three alternative Visformer GAP score definitions at the current 41/10 K2
  schedule;
- sixteen opponent-strength runs completing a five-point
  `0.0/0.1/0.2/0.3/0.4` sweep on all four sources, with 0.2 shared by the
  baselines;
- four ViT late-checkpoint alternatives at block7--10, with block11 shared by
  the baseline and block3/10/10 fixed;
- four Visformer equal-ratio alternatives at approximately 10%, 15%, 25% and
  30%, with the current approximately 20% schedule shared by the baseline.

The three added GAP score modes preserve label-free, gradient-independent
routing:

- `gap_leave_one_out_cosine`: compare each token with the mean of all other
  tokens;
- `gap_projection`: rank by norm-sensitive projection onto the noisy GAP
  representation;
- `gap_channel_rms_cosine`: channel-RMS-normalize local/global features before
  cosine scoring.

The default `cosine` path retains golden replay, gradient and adversarial-output
parity.

## Adaptive follow-ups

After the base matrix:

1. replicate the best non-cosine Visformer score on CaiT K3 if it improves
   Overall by at least 0.8pp without a material strict-black-box reversal;
2. add two neighboring opponent strengths per model when the coarse winner
   improves on 0.2 by at least 0.5pp or lies at a scanned boundary;
3. test ViT block3/block6 only if block3/block7 is the late-layer boundary
   winner.

At most ten adaptive runs are added, for an overall ceiling of 48 runs. The
noise-off point is a control and cannot be promoted. GAP score variants remain
the same patch-score routing mechanism; no unrelated attack module is added.

## Required artifacts

Every attack directory must contain exactly 192 adversarial PNGs plus
`attack_params.json`, `gradient_diagnostics.json`, and `replay_manifest.json`.
Every run must complete transfer evaluation with zero skipped images and write
`outputs/csv/outputs_attack_<run-name>.csv`. The final report will include all
parameters, four aggregate ASRs, thirteen per-target ASRs, baseline deltas,
opponent-strength gradient diagnostics, and the selected adaptive follow-ups.

## CaiT GAP-projection replication

The Visformer `gap_projection` gain was replicated on CaiT with two controlled
192-image runs using the fixed protocol above:

| Schedule | Score | Overall | Transformer | CNN | Strict black-box |
|---|---|---:|---:|---:|---:|
| block5/17/23, 10/10/10 | cosine | 86.42% | 91.07% | 80.99% | 85.37% |
| block5/17/23, 10/10/10 | gap_projection | 87.30% | 91.67% | 82.20% | 86.37% |
| block17/23, 2/28 | cosine | 87.74% | 91.82% | 82.99% | 86.76% |
| block17/23, 2/28 | gap_projection | **87.94%** | **92.26%** | 82.90% | **86.98%** |

Changing only the score on the original 10/10/10 schedule improves Overall by
0.88pp and strict black-box ASR by 1.00pp, so the Visformer result does
generalize to the other GAP architecture. On the late-heavy 2/28 schedule the
gain is only 0.20pp Overall and 0.22pp strict, below the 0.5pp screening-tie
threshold. Thus GAP projection is supported as a cross-GAP score improvement,
but its benefit overlaps substantially with the late-heavy schedule rather than
adding independently. The 2/28 projection combination is the highest CaiT
Overall result in this 192-image campaign, but is not promoted without a larger
validation run.

## Visformer GAP-projection and opponent-noise combination

The independently favorable Visformer settings were combined at the retained
stage2/stage3 41/10 schedule:

| Score | Opponent strength | Overall | Transformer | CNN | Strict black-box |
|---|---:|---:|---:|---:|---:|
| cosine | 0.2 | 77.04% | 82.22% | 71.01% | 75.13% |
| gap_projection | 0.2 | 79.09% | 84.52% | 72.74% | 77.34% |
| cosine | 0.4 | 81.29% | 85.94% | 75.87% | 79.73% |
| gap_projection | 0.4 | **82.61%** | **87.13%** | **77.34%** | **81.16%** |

The combination improves Overall by 1.32pp and strict black-box ASR by 1.43pp
over cosine at strength 0.4. Relative to projection at strength 0.2, it improves
Overall by 3.53pp and strict ASR by 3.82pp. The gain therefore survives removal
of the Visformer self-target and shows that projection and stronger opponent
noise are complementary on this 192-image screen. The combination is the
highest observed Visformer setting in the campaign, pending larger-sample
validation.
