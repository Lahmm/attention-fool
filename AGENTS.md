# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.

## Current research mainline

- The paper mainline has exactly two core mechanisms: **patch-score-guided patch drop** and **RGB opponent-channel random noise**. Do not stack unrelated attack modules onto the main claim.
- The only retained attack is the progressive high-score attack implemented by `progressive_attack.py` and dispatched by `main.py`.
- The validated ViT-B/16 configuration uses checkpoint boundaries `block3` and `block10`, independently drops 10 of 196 local tokens at each checkpoint (`drop_ratio=0.051020408163`), and permits the same position to be selected again later in the schedule. This is the only ViT attack configuration that should be described as current or used by current executable examples and tests.
- At every attack step and augmentation group, build a fresh two-checkpoint ViT schedule from the current adversarial pixels. At each checkpoint, recompute global/local patch scores on the sequentially updated token state, sample from the score-high half, and hard-zero the selected local tokens. The original view uses that schedule and the phase view uses its spatially transformed counterpart. The default 10 steps × 10 groups produces 100 schedules and 200 checkpoint mask selections per ViT image.
- Patch-score is a label-free, gradient-independent global/local representation routing coordinate for deciding **where** to perturb.
- Opponent-channel noise decides **how** to perturb kept evidence: sample luminance, red-green, and yellow-blue RGB directions, project through the initial RGB projection, and RMS-match in feature space.
- Validate complementarity with transferable-gradient diagnostics and transfer ASR defined as `1 - adversarial accuracy` over all evaluated adversarial samples; do not filter to a target-clean-correct subset.
- Preserve support for ViT-B/16, CaiT-S24, PiT-B, and Visformer-S through architecture adapters. Do not encode ViT block assumptions as a nominally cross-architecture implementation.

## Retained executable scope

- Retain only the current progressive high-score attack behavior in `progressive_attack.py`, the minimal `main.py` execution path needed to run it, and the progressive portions of the four architecture adapters in `nets/`.
- Retain only utility, test, configuration, and documentation code that is directly required to execute or verify that progressive attack.
- The protected behavior is: fresh sequential checkpoint schedules from current adversarial pixels, the ViT `block3,block10` schedule, high-score-window sampling, local-token hard zeroing, transformed phase-pair schedules, kept-only RGB opponent-channel projected noise, and the projected iterative update used by the current progressive attack.
- Code is not protected merely because it is currently imported, exposed by the CLI, covered by a test, mentioned in an old report, or needed to reproduce a superseded experiment.

## Removable scope

- All non-progressive attack implementations and compatibility paths may be removed, including `attack.py`, `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, and `token_patch_dropout`.
- Legacy adapter APIs may be removed, including final-layer patch-score extraction, resumable legacy forwards, token hooks, and legacy checkpoint registries, provided the progressive adapter contract remains complete for all four supported architectures.
- Non-mainline selectors, score modes, augmentation modules, gradient post-processing variants, diagnostics, replay helpers, transfer-evaluation helpers, result-recording utilities, completed experiment runners, compatibility wrappers, historical tests, archived reports, and generated artifacts may be removed when they are not required by the retained progressive attack.
- Backward CLI compatibility and reproduction of historical experiments are not cleanup requirements. Remove stale parameters and metadata instead of preserving no-op or legacy options.
- Update or delete tests and documentation together with removed functionality so that the remaining repository describes only the retained progressive attack.
