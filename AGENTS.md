# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.

## Current research mainline

- The paper mainline is **Progressive Route Disruption (PRD)** and has exactly two core mechanisms: **checkpoint-wise progressive random token drop** and **RGB opponent-channel random noise**. Do not stack unrelated attack modules onto the main claim.
- The only retained attack is PRD, implemented by `progressive_attack.py` and dispatched by `main.py`.
- The validated ViT-B/16 configuration uses checkpoint boundaries `block3` and `block10`, independently drops 10 of 196 local tokens at each checkpoint (`drop_ratio=0.051020408163`), and permits the same position to be selected again later in the schedule. This is the only ViT attack configuration that should be described as current or used by current executable examples and tests.
- At every attack step and augmentation group, generate a fresh random checkpoint schedule. At each checkpoint, sample uniformly from all current local-token positions and hard-zero the selected tokens before continuing the same forward trajectory. The original view uses that schedule and the phase view uses its spatially transformed counterpart. The default 10 steps × 10 groups produces 100 schedules and 200 checkpoint mask selections per ViT image.
- Patch score belongs only to the motivating observation that token semantic rankings reorganize across model depth. It is not an attack selector, runtime dependency, adapter API, CLI option, or claimed source of PRD's transferability.
- Opponent-channel noise decides **how** to perturb kept evidence: sample luminance, red-green, and yellow-blue RGB directions, project through the initial RGB projection, and RMS-match in feature space.
- Validate complementarity with transferable-gradient diagnostics and transfer ASR defined as `1 - adversarial accuracy` over all evaluated adversarial samples; do not filter to a target-clean-correct subset.
- Preserve support for ViT-B/16, CaiT-S24, PiT-B, and Visformer-S through architecture adapters. Do not encode ViT block assumptions as a nominally cross-architecture implementation.

## Retained executable scope

- Retain only PRD's progressive random-drop behavior in `progressive_attack.py`, the minimal `main.py` execution path needed to run it, and the progressive portions of the four architecture adapters in `nets/`.
- Retain only utility, test, configuration, and documentation code that is directly required to execute or verify that progressive attack.
- The protected behavior is: fresh uniformly random checkpoint schedules, the ViT `block3,block10` schedule, local-token hard zeroing during sequential checkpoint traversal, transformed phase-pair schedules, kept-only RGB opponent-channel projected noise, and the projected iterative update used by PRD.
- Code is not protected merely because it is currently imported, exposed by the CLI, covered by a test, mentioned in an old report, or needed to reproduce a superseded experiment.

## Removed scope

- Do not restore non-progressive attack implementations or compatibility paths, including the deleted `attack.py`, `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, and `token_patch_dropout` paths.
- Do not restore selector or scoring paths, including high/low/extreme/rank-transition selection, score modes, global-score noise, final-layer patch-score extraction, token hooks, or legacy checkpoint registries. The adapters expose only the PRD traversal contract.
- Non-mainline augmentation modules, gradient post-processing variants, diagnostics, completed experiment runners, compatibility wrappers, historical tests, archived reports, and generated artifacts may be removed when they are not required by PRD.
- Backward CLI compatibility and reproduction of historical experiments are not cleanup requirements. Remove stale parameters and metadata instead of preserving no-op or legacy options.
- Update or delete tests and documentation together with removed functionality so that the remaining repository describes only the retained progressive attack.
