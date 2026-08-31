# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.

## Current research mainline

- The paper mainline has exactly two core mechanisms: **patch-score-guided patch drop** and **RGB opponent-channel random noise**. Do not stack unrelated attack modules onto the main claim.
- Production uses the dynamic-mask `original_score_postdrop_phase_pair` behavior. At every attack step and augmentation group, recompute final-layer patch scores on current adversarial pixels and sample a fresh high-tail mask. Only the original/phase pair within that group shares the mask. The default 10 steps × 10 groups produces 100 mask selections per image.
- Patch-score is a label-free, gradient-independent global/local representation routing coordinate for deciding **where** to perturb.
- Opponent-channel noise decides **how** to perturb kept evidence: sample luminance, red-green, and yellow-blue RGB directions, project through the initial RGB projection, and RMS-match in feature space.
- Validate complementarity with transferable-gradient diagnostics and transfer ASR defined as `1 - adversarial accuracy` over all evaluated adversarial samples; do not filter to a target-clean-correct subset.
- Phase pairs, raw multi-view mean, Gaussian residual, MI/NI/DIM/TI, and the `none`/pixel-drop/token-drop paths are supporting mechanisms or controlled ablations.

## Retained executable scope

- Attack: `main.py`, `attack.py`, `gradient_replay.py`, `transfer_eval.py`.
- Attack methods: `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, `token_patch_dropout`.
- Mainline selectors: `patch_score`, `random`, `no_drop`.
- Model adapters: `nets/`.

## Archived evidence

- Historical semantic-gradient E4-E10 results remain boundary evidence only.
- Do not claim Gaussian-residual results for models whose completed provenance has not been audited.
