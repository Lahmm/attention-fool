# Layer selection, drop ratio, and score window: scan results and discussion

Date: 2026-09-15

Code revision for every run in this document: `eb17588` (adds the
`--score-window-ratio` CLI parameter; default 0.5 is bitwise-identical to the
previous hard-coded behaviour, verified by matching replay-manifest event
digests and output PNG hashes).

Scope. Three per-model or global axes of the progressive patch-score mainline
were swept on all four white-box sources: ViT-B/16, CaiT-S24, PiT-B and
Visformer-S.

1. **Layer selection** — the number of checkpoints `K`, which checkpoints, and
   how much each drops.
2. **Drop ratio** — the per-checkpoint distribution and the total budget.
3. **Score window** — the fraction of the score ranking that the `high`
   selector samples drop positions from.

Two axes that were previously listed but are **not** covered here: opponent
noise strength and the phase/view budget. Both remain open.

---

## 1. Measurement conventions and noise calibration

Everything in this document is measured on a **fixed 192-image screening
subset at `--sample-offset 0`**, with seed 20260907, and promoted to 1000
images only where noted. The screening subset is chosen because it is exactly
the first 192 images of the full 1000-image run, so the two agree on sample
identity.

Three calibration facts constrain how large a delta may be believed. They were
measured, not assumed.

- **192-image screening is accurate to roughly ±0.5–0.8pp, and its error is
  two-sided.** ViT K=2 (block3,block11) measured +1.00pp at 192 images but
  only +0.57pp at 1000. CaiT K=1 (block23) measured +1.44pp at 192 but +2.25pp
  at 1000. Screening both over- and under-states the full-set effect, and at
  0.79pp it can even invert a ranking — see the CaiT K=2 row in section 2.1.
- **The seed-to-seed floor at 1000 images is ~0.17pp.** The identical
  configuration `(3,7,11) high` scored 79.75% at seed 20260903 and 79.58% at
  seed 20260907.
- **Screening offsets are not comparable.** The same Visformer-S
  configuration `s1b1,s2b2,s3b3` scored 75.96% at offset 0 and 68.99% at
  offset 192. All 192-image results quoted here are offset 0; historical runs
  at offsets 192/384/576 are therefore excluded from every comparison table.

An additional caveat specific to CaiT: the attack is **RNG-reproducible but
not float-reproducible**. Re-running the identical config and seed produced an
identical `replay_manifest.json` event digest (96000 events, same digest) but
different adversarial PNGs, moving ASR by ~0.08pp. ViT reproduced bitwise
under the same test. Sub-0.1pp deltas are treated as noise on all models.

ASR is `1 - adversarial accuracy` over all evaluated adversarial samples, with
no target-clean-correct filtering. The target set is 7 Transformers
(LeViT-256, PiT-B, DeiT-B, TNT-S, ConViT-B, Visformer-S, CaiT-S/24) and 6 CNNs
(Inception-v3/v4, Inception-ResNet-v2, ResNet-101, Inc-v3-adv,
IncRes-v2-adv). Because PiT-B, Visformer-S and CaiT-S/24 appear in the target
set, tables for those sources also report a **strict black-box** average that
excludes the same-architecture target. ViT-B/16 is not in the target set, so
its raw and strict averages are identical.

---

## 2. Layer selection

### 2.1 Checkpoint count

Each model's **total drop budget is held at its promoted default** while the
budget is redistributed over `K` checkpoints, so the ladder isolates
checkpoint count rather than budget. All rows: 192 images, offset 0.

| K | ViT-B/16 (total 30) | CaiT-S24 (total 30) | PiT-B (total 13) | Visformer-S (total 51) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 79.41 | **88.02** | 79.65 | 75.08 |
| 2 | **82.13** | 86.70 | 81.33 | **77.04** † |
| 3 | 81.13 | 86.58 | **83.05** | 76.20 |
| 4 | 80.13 | 83.65 | 79.93 | 73.28 |
| 6 | 79.93 | 80.33 | 80.09 | 75.08 |

† Equal-ratio split (41/10); the uniform-count split of the same checkpoint
pair collapses to 51.00 — see section 3.3.

The peak is at `K <= 3` for all four sources, and `K = 6` is clearly worse
everywhere. Recomputing with the strict black-box average leaves the ordering
unchanged — for example CaiT `K=1` 87.11 vs `K=6` 79.12, PiT `K=3` 81.68 vs
`K=6` 78.43 — so the pattern is not an artifact of source-model self-transfer.

**1000-image confirmation of the best non-`K=1` CaiT-S24 schedule.** The ladder
made `(block17,block23)` the best CaiT schedule with more than one checkpoint,
0.12pp *ahead* of the `K=3` baseline. On the full 1000 images the ranking
reverses:

| CaiT-S24 schedule | 192 images | 1000 images |
| --- | ---: | ---: |
| `K=1` block23_gap, 30 | 88.02 | **86.40** |
| `K=3` block5/17/23, 10/10/10 | 86.58 | **84.15** |
| `K=2` block17/23, 15/15 | 86.70 | **83.48** |

Screening put `K=2` 0.12pp above `K=3`; the full set puts it 0.67pp below, a
0.79pp swing that flips the sign. This leaves the adopted configuration
unchanged — `K=1` remains best by a wide margin — but it is the sharpest
illustration in this document of why a sub-1pp screening delta must not be
promoted without a full-set run.

**Caveat.** Every row holds the total budget constant, so the ladder measures
"spread the same budget thinner" rather than "add checkpoints that each keep a
full budget". The latter configuration (constant per-checkpoint budget with
growing `K`) has not been tested.

### 2.2 Checkpoint position

**ViT-B/16 early checkpoint** (K=2, late checkpoint fixed at block11, 15/15):

| early checkpoint | block1 | block2 | **block3** | block4 | block5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ASR | 78.61 | 78.81 | **82.13** | 79.61 | 80.49 |

block3 sits 1.6–3.5pp above either neighbour, so it is a genuine local
optimum rather than a screening fluctuation.

**CaiT-S24 single checkpoint** (K=1, 30 drops):

| checkpoint | block21 | block22 | **block23** | block24 |
| --- | ---: | ---: | ---: | ---: |
| ASR | 82.97 | 80.49 | **88.02** | **58.85** |

block23 is an extremely sharp peak and block24 collapses by 29pp. This
reproduces and sharpens the pre-existing block24 negative control (63.80% on a
different schedule) and supports the reading that **routing at the final layer
is destructive rather than merely unhelpful**.

**Combination structure.** CaiT K=2 improves monotonically as the early
checkpoint is removed: `(5,17)` 82.69 < `(5,23)` 85.46 < `(17,23)` 86.70. PiT
K=2 prefers keeping the stage-2 and stage-3 token: `(s2b1,s3b3)` 81.33 vs
`(s3b2,s3b3)` 74.32. Visformer K=2 (equal ratio) prefers the two later stages:
`(s2b1,s3b1)` 77.04 > `(s1b1,s3b1)` 75.96 > `(s1b1,s2b1)` 75.20. The common
theme is that the earliest checkpoint contributes least.

### 2.3 Best-measured per-model configurations

The unconstrained ASR-maximising configuration found for each source, all
confirmed on the full 1000 images:

| Source | Best-measured configuration | ASR | Former `K=3` default | ASR | Gain |
| --- | --- | ---: | --- | ---: | ---: |
| CaiT-S24 | **block23, 30** | **86.40** | block5/17/23, 10/10/10 | 84.15 | +2.25 |
| ViT-B/16 | **block3,block11, 10/10** | **80.28** | block3/7/11, 10/10/10 | 79.58 | +0.71 |
| Visformer-S | **stage2_block1,stage3_block1, 41/10** | **74.25** | stage1/2/3 block1, 39/10/2 | 73.16 | +1.09 |
| PiT-B | stage2_block1,stage3_block2,stage3_block3, 5/2/6 | 80.44 | identical | 80.44 | 0 |

Best-measured four-source mean is **80.34%**, against **79.33%** for the former
`K=3` defaults. CaiT's gain is largely a checkpoint-count effect (three points
to one); ViT's is a position and budget effect; Visformer's is a
stage-allocation effect.

The production mainline initially retained `K=3` uniformly to preserve a
three-point mechanism. That decision was subsequently superseded by the
constrained-ASR default selection below. The explicit ViT `(3,7,11)` schedule
remains the canonical mechanism-validation reference and is still supported.

### 2.4 Current constrained-ASR defaults

The production defaults maximise completed 1000-image Overall ASR subject to
`selector=high`, `score-window-ratio=0.5`, and `K>1`:

| Source | Current default | Overall ASR |
| --- | --- | ---: |
| ViT-B/16 | block3/block11, 10/10 | **80.28%** |
| CaiT-S24 | block5/17/23, 10/10/10 | **84.15%** |
| PiT-B | stage2_block1/stage3_block2/stage3_block3, 5/2/6 | **80.44%** |
| Visformer-S | stage2_block1/stage3_block1, 41/10 | **74.25%** |

The four-source mean is **79.78%**. CaiT retains `K=3` because its screening
`K=2` winner reversed on 1000 images (83.48% versus 84.15%). ViT's 10/10 result
is only 0.13pp above 15/15, below the measured 0.17pp seed floor, but it is the
highest completed result under the stated selection rule.

For window interpretation, the section 4 sweep ran on the best-measured layer
configurations. It applies directly to the PiT and Visformer defaults, but not
to CaiT's retained `K=3` schedule or ViT's newly reduced 10/10 budget. The
historical ViT `(3,7,11)` control still shows `high` ahead of `random` by
0.59pp (79.75 versus 79.16).

---

## 3. Drop ratio

### 3.1 Per-checkpoint distribution

ViT-B/16 and CaiT-S24 had never had their per-checkpoint distribution swept;
both always ran a uniform 0.05/0.05/0.05. The `K` ladder above subsumes that
axis for these two models: for both, removing a checkpoint beat redistributing
the same budget across three.

PiT-B and Visformer-S had already been swept across late-heavy and early-heavy
distributions in earlier work, on their own offsets. Those results are not
re-derived here.

### 3.2 Total budget

The total drop budget was previously pinned at 15% of the token grid. It is
not a sensitive parameter, and 15% is not its optimum.

| Model (schedule) | drops -> ASR |
| --- | --- |
| ViT-B/16 (block3,block11) | 20 -> **82.73** · 30 -> 82.13 · 40 -> 81.85 |
| CaiT-S24 (block23) | 20 -> 85.22 · 30 -> 88.02 · 40 -> **88.14** · 50 -> 86.70 |

ViT declines monotonically from 20 drops; CaiT plateaus over 30–40 and then
declines. Outside the extremes the spread is under 1pp, so the budget can be
treated as a weak parameter — with the important exception that no single
checkpoint may be pushed toward a 0.5 ratio (section 3.3).

The 20-drop ViT result (82.73) was screening-only in the original scan. Its
subsequent 1000-image confirmation obtained
**80.28%**, versus 80.15% for 30 drops. The +0.13pp full-set difference is below
the measured 0.17pp seed floor; 10/10 is nevertheless retained as the numeric
winner and current default.

### 3.3 Cross-scale models require equal-ratio allocation

Distributing the budget by **equal token counts** is wrong for models whose
checkpoints live on different grids. For Visformer-S the checkpoint pair
`(stage2_block1, stage3_block1)` has grids 196 and 49; an equal-count split
assigns 27/24, which is a 0.49 ratio at stage 3 — half of the final-stage
tokens.

| Visformer-S K=2 `(s2b1,s3b1)` | Counts | Ratios | ASR |
| --- | --- | --- | ---: |
| uniform token counts | 27 / 24 | 0.138 / **0.490** | **51.00** |
| equal ratio | 41 / 10 | 0.209 / 0.204 | **77.04** |

The 25pp gap is entirely a split artefact, not a property of the checkpoint
pair. All cross-scale `K` results in section 2.1 that are marked with a dagger
use the equal-ratio rule; the uniform-count rows for Visformer-S K=2 are the
51.00 and 50.72 entries and should be read as invalid configurations rather
than as evidence against `K=2`.

Applying the same rule to PiT-B leaves its conclusion unchanged (`K=3` still
optimal; best `K=2` falls from 81.33 to 80.17 under equal ratio).

---

## 4. Score window

### 4.1 The parameter and its two endpoints

The `high` selector previously sampled drop positions from a hard-coded top
half of the patch scores. `--score-window-ratio` (default 0.5) now exposes that
fraction. The sweep spans a spectrum whose endpoints are already-implemented
selectors:

- `w = 1.00` uses every token as a candidate, so the score ranking has no
  effect on the draw — it is **statistically identical to the `random`
  selector**.
- `w -> drop_count` reduces to a deterministic top-k — the **`extreme-high`**
  selector.

So the window sweep is a continuous measurement of how much score guidance is
worth, bracketed by two previously measured points.

The window is a **global** parameter: it must take the same value for all four
sources, and its objective is the four-source mean ASR.

### 4.2 Screening results

Sweep on each model's then-selected layer configuration, 192 images, offset 0.

| window | ViT-B/16 | CaiT-S24 | PiT-B | Visformer-S | mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 81.61 | 87.46 | **83.61** | 72.28 | 81.24 |
| 0.35 | 82.21 | 87.46 | 83.41 | 76.04 | 82.28 |
| 0.50 ★ | 82.13 | 88.10 | 83.17 | 77.08 | 82.62 |
| 0.70 | **82.49** | 88.46 | 82.37 | 78.77 | 83.02 |
| 1.00 | 81.61 | **89.14** | 82.33 | **80.93** | **83.50** |

★ current default. The `w = 0.50` ViT entry reproduces the section 2.1 `K=2`
row bitwise (82.13), confirming that the new parameter is inert at its default.

The four models disagree in direction. ViT-B/16 has an interior optimum at
0.70 and prefers guidance to no guidance by 0.52pp. CaiT-S24, PiT-B and
Visformer-S all trend the other way; Visformer-S is extreme, spanning 8.65pp
between its narrowest and widest window.

### 4.3 1000-image confirmation

Only the two endpoints were promoted.

| window | ViT-B/16 | CaiT-S24 | PiT-B | Visformer-S | mean |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | **80.15** | 86.40 | **80.44** | 74.25 | 80.31 |
| 1.00 | 80.08 | **86.78** | 79.95 | **77.69** | **81.13** |
| delta | **−0.07** | +0.38 | −0.49 | **+3.44** | **+0.82** |

Three findings.

**(a) Screening systematically overstated the window effect.** Three of four
per-model deltas shrank or reversed between the 192- and 1000-image
measurements: ViT +0.52 -> −0.07, CaiT +1.04 -> +0.38, PiT +0.84 -> −0.49.
Only Visformer-S held (+3.85 -> +3.44). The screening and full-set deltas
agree in sign for just one model.

**(b) Score guidance is worth nothing on the retuned ViT.** 80.15 versus 80.08
is 0.07pp — below the 0.17pp seed floor. This does not contradict the earlier
"score-high beats random by 0.59pp" result, which was measured on the
*previous* `(block3,7,11)` schedule. Retuning the checkpoints to
`(block3,block11)` removed the gain.

**(c) The mean advantage comes almost entirely from Visformer-S (+3.44).**
Visformer-S is both a GAP-based and a cross-scale model. The current
patch-score definition — cosine similarity between a local token and the
global token — is a saliency measure when the global token is a CLS token, but
for a GAP model it degenerates into similarity with the model's own mean
pooled representation, which need not track adversarial importance.

### 4.4 Window interpretation and current choice

Under the scan's original constraint (one window value shared by all four
sources, objective = four-source mean), the numeric optimum is **`w = 1.00`,
i.e. disabling score guidance**, worth +0.82pp over `w=0.50`. This conflicts
with the paper mechanism in which patch-score routing decides *where* to
perturb. The current production selection therefore explicitly fixes
`w=0.50`; redefining the GAP-family score remains a possible future direction.

---

## 5. Summary and confidence

| Finding | Confidence | Basis |
| --- | --- | --- |
| Per-model layer/budget specialisation reaches +1.01pp unconstrained | high | four models, 1000 images |
| `K <= 3` beats `K = 4` and `K = 6` | medium-high | 4/4 models, holds under strict black-box; all rows hold total budget constant, so constant-per-checkpoint `K >= 4` is untested |
| CaiT-S24 `K = 1` beats every `K >= 2` schedule on the full set | high | K=2 confirmed at 83.48 and K=3 at 84.15 versus 86.40 |
| CaiT-S24 optimum is the single checkpoint block23; final-layer routing is destructive | high | sharp peak plus 29pp collapse, 1000-image confirmation |
| ViT-B/16 optimum is `(block3, block11)`, skipping the middle | high | neighbour sweep plus 1000-image confirmation |
| 15% total budget is neither required nor optimal | medium-high | ViT `K=2` confirmed at 1000; CaiT swept only at `K=1` screening scale |
| Cross-scale sources must allocate by equal ratio | high | 25pp split artefact on Visformer-S |
| The window moves the four-source mean by at most 0.82pp | medium | 1000-image data at 0.50 and 1.00 only; intermediate points are screening-only |
| Score guidance is worthless on the 15/15 retuned ViT-B/16 | medium-high | 0.07pp at 1000 images, below the seed floor; 10/10 window interaction is untested |
| The window and layer-selection axes are coupled | high | the ViT routing gain present at `(3,7,11)` disappears at `(3,11)` |

On the schedules used by the window sweep, the best shared window is `w=1.00`
at **81.13%**, versus 80.31% at `w=0.50`. Those numbers are controlled window
evidence rather than current-default results because the ViT sweep used 15/15.

**Current production defaults:** `high`, `w=0.5`, `K>1`, with per-model
1000-image winners: ViT K2 10/10, CaiT K3 10/10/10, PiT K3 5/2/6, and
Visformer K2 41/10. Their four-source mean is **79.78%**.

---

## 6. Open items

Resolved by the current default decision: production uses the best completed
1000-image result under `high`, `w=0.5`, and `K>1`; checkpoint count is
architecture-specific rather than uniformly three.

Still open:

- Window and selector data for CaiT-S24 at its retained `K=3` schedule (none
  exists). The section 4 sweep used CaiT K=1 and does not apply to this default.

- Constant per-checkpoint budget with `K >= 4` (the complement of section 2.1).
- 1000-image confirmation of window values 0.25, 0.35 and 0.70.
- Opponent noise strength: only the on/off contrast (62.52% vs 79.58% on ViT)
  and one structural contrast (IID Gaussian 78.42% vs opponent 79.58%) exist.
  The magnitude has never been swept, and it is the largest unexplored
  interval in the attack.
- The window interaction for the new ViT 10/10 default; the completed window
  sweep used the same layers with 15/15 drops.
- The window and layer-selection axes are coupled (section 4.4), so any future
  change to the window would in principle require re-selecting layers under the
  new window, especially for Visformer-S. The converse also holds: because the
  window sweep ran on the best-measured rather than the retained schedules, it
  must be re-measured if layer positions are revisited.

---

## 7. Data locations

Screening and confirmation runs:

- `outputs/attack/scanB_*` — section 2.1 `K` ladder, section 3.3 ratios
- `outputs/attack/scanB2_*` — section 2.3 confirmations, equal-ratio retest
- `outputs/attack/scanB3_*` — section 2.1 CaiT-S24 `K=2` 1000-image confirmation
- `outputs/attack/scanD_*` — section 2.2 positions, section 3.2 budgets
- `outputs/attack/scanA_*` — section 4.2 window screening
- `outputs/attack/scanA2_*` — section 4.3 window confirmation

Transfer-eval records with per-target ASR are written automatically to
`outputs/csv/outputs_attack_<run-name>.csv`. Every attack directory contains
`attack_params.json`, `gradient_diagnostics.json` and `replay_manifest.json`.

## 8. Reproduction

The interpreter must be the `att-atk` conda environment; the default `python`
on PATH has no torch. CaiT-S24 exhausts the 24 GiB card at the default batch
size 96 and requires `--batch-size 48`. Batch size does not affect results:
replay seeds depend only on `(master_seed, sample_id, step, group, view,
event)` and the update uses `sign(momentum)`, so the `1/B` loss scaling
cancels.

```bash
PY=/root/miniconda3/envs/att-atk/bin/python

# Current CaiT-S24 default
$PY main.py --attack-method progressive --whitebox-model cait_s24_224 \
  --checkpoints block5_gap,block17_gap,block23_gap \
  --drop-ratios 0.05,0.05,0.05 --batch-size 48 \
  --max-attacked-samples 1000 --sample-offset 0 --seed 20260907 \
  --output-dir outputs/attack/<name>

# Current ViT-B/16 default
$PY main.py --attack-method progressive --whitebox-model vit_base_patch16_224 \
  --checkpoints block3,block11 --drop-ratios 0.051020408163,0.051020408163 \
  --batch-size 96 \
  --max-attacked-samples 1000 --sample-offset 0 --seed 20260907 \
  --output-dir outputs/attack/<name>

# Current PiT-B default
$PY main.py --attack-method progressive --whitebox-model pit_b_224 \
  --checkpoints stage2_block1,stage3_block2,stage3_block3 \
  --drop-ratios 0.02081165,0.03125,0.09375 --batch-size 96 \
  --max-attacked-samples 1000 --sample-offset 0 --seed 20260907 \
  --output-dir outputs/attack/<name>

# Current Visformer-S default
$PY main.py --attack-method progressive --whitebox-model visformer_small \
  --checkpoints stage2_block1,stage3_block1 --drop-ratios 0.209184,0.204082 \
  --batch-size 48 --max-attacked-samples 1000 --sample-offset 0 --seed 20260907 \
  --output-dir outputs/attack/<name>

# Window sweep (append to any of the above)
#   --score-window-ratio 0.25 | 0.35 | 0.50 | 0.70 | 1.00
```

Transfer evaluation is run afterwards with:

```bash
$PY transfer_eval.py --image-dir outputs/attack/<name> --amp --exp-name <name>
```
