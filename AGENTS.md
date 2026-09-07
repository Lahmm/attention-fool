# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.

## Current research mainline

- The paper mainline has exactly two core mechanisms: **patch-score-guided patch drop** and **RGB opponent-channel random noise**. Do not stack unrelated attack modules onto the main claim.
- The promoted research mainline is the progressive high-score attack currently implemented by `vit_progressive_patch_score_attack.py`. Its validated ViT-B/16 reference configuration uses checkpoint boundaries `(3, 7, 11)`, independently drops 5% of local tokens at each checkpoint, and permits the same position to be selected again later in the schedule.
- At every attack step and augmentation group, build a fresh three-checkpoint schedule from the current adversarial pixels. At each checkpoint, recompute global/local patch scores on the sequentially updated token state, sample from the score-high half, and hard-zero the selected local tokens. The original view uses that schedule and the phase view uses its spatially transformed counterpart. The default 10 steps × 10 groups produces 100 schedules and 300 checkpoint mask selections per image.
- Patch-score is a label-free, gradient-independent global/local representation routing coordinate for deciding **where** to perturb.
- Opponent-channel noise decides **how** to perturb kept evidence: sample luminance, red-green, and yellow-blue RGB directions, project through the initial RGB projection, and RMS-match in feature space.
- Validate complementarity with transferable-gradient diagnostics and transfer ASR defined as `1 - adversarial accuracy` over all evaluated adversarial samples; do not filter to a target-clean-correct subset.
- CLS score noise, phase pairs, raw multi-view mean, Gaussian residual, MI/NI/DIM/TI, and the `none`/pixel-drop/token-drop paths are supporting mechanisms or controlled ablations.
- The final-layer pixel-drop `original_score_postdrop_phase_pair` behavior is now a historical cross-architecture baseline, not the current research mainline.
- Migration into `main.py` must preserve support for ViT-B/16, CaiT-S24, PiT-B, and Visformer-S through architecture adapters. Do not encode ViT block assumptions as a nominally cross-architecture implementation.

## Retained executable scope

- Promoted mainline reference: `vit_progressive_patch_score_attack.py` (currently ViT-only and pending migration into `main.py`).
- Shared and legacy attack code: `main.py`, `attack.py`, `gradient_replay.py`, `transfer_eval.py`.
- Legacy attack methods: `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, `token_patch_dropout`.
- Progressive mainline selector: `high`; controlled selector ablations: `low`, `random`.
- Model adapters: `nets/`.

## Archived evidence

- Historical semantic-gradient E4-E10 results remain boundary evidence only.
- Do not claim Gaussian-residual results for models whose completed provenance has not been audited.
