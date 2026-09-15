# Progressive routing follow-up: 192-image exploration

Date: 2026-09-15

Status: implementation complete; experiment matrix pending execution.

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
