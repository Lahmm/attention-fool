# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.

## Current research mainline

- The paper mainline has exactly two core mechanisms: **patch-score-guided patch drop** and **RGB opponent-channel random noise**. Do not stack unrelated attack modules onto the main claim.
- Production uses the dynamic-mask `original_score_postdrop_phase_pair` behavior. At every attack step and augmentation group, recompute final-layer patch scores on current adversarial pixels and sample a fresh high-tail mask. Only the original/phase pair within that group shares the mask. The default 10 steps × 10 groups produces 100 mask selections per image.
- Patch-score is a label-free, gradient-independent global/local representation routing coordinate for deciding **where** to perturb. It is not a per-patch causal explainer.
- Opponent-channel noise decides **how** to perturb kept evidence: sample luminance, red-green, and yellow-blue RGB directions, project through the initial RGB projection, and RMS-match in feature space.
- Validate complementarity with transferable-gradient diagnostics and target-clean-correct transfer ASR. Clean source-logit suppression, clean deletion, and source feature drift are boundary diagnostics only.
- Phase pairs, raw multi-view mean, Gaussian residual, MI/NI/DIM/TI, and the `none`/pixel-drop/token-drop paths are supporting mechanisms or controlled ablations.

## Retained executable scope

- Attack: `main.py`, `attack.py`, `gradient_replay.py`, `transfer_eval.py`.
- Attack methods: `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, `token_patch_dropout`.
- Mainline selectors: `patch_score`, `random`, `no_drop`.
- Model adapters: `nets/`.
- Forward semantic evidence:
  - `experiments/patch_score_layer_semantic_maturity_experiment.py`
  - `experiments/patch_score_promotion_observation.py`
  - `experiments/patch_score_promotion_mixing_experiment.py`
  - `experiments/patch_score_same_activation_experiment.py`
  - `experiments/semantic_equivalent_route_experiment.py`
  - shared helpers in `experiments/semantic_forward_utils.py`

Do not reintroduce archived selector calibration, clean-output selection, semantic gradient weighting, frequency pair, or patch shuffle code without an explicit new user request and a mechanism-driven rationale.

## Semantic evidence boundaries

- Four architectures show early-to-late global/local semantic maturation and patch-rank reorganization.
- Late token mixing is a shared boundary involved in semantic promotion.
- Phase/noise views can preserve global semantics while changing local routing.
- Cross-model commonality is a functional routing pattern, not a universal spatial mask.
- Protocol A on the same logit-connected activation found patch-score versus Grad-CAM ReLU rank Spearman/top-half IoU of ViT 0.489/0.557, CaiT 0.011/0.359, PiT -0.148/0.320, and Visformer 0.757/0.738. This establishes distinct but partially overlapping criteria, not a selector winner.

## Archived evidence

- Removed experiment code is represented by formal artifacts and conclusions under `results/` and force-tracked `outputs/research/` paths.
- Historical fixed-clean-mask, causality, polarity, clean-layer, route-response, Grad-CAM Protocol B, semantic-gradient E4-E10, frequency-pair, and patch-shuffle results are boundary evidence only.
- The frequency-pair branch reached 59.48% average transfer ASR with feature Gaussian and 60.63% with opponent noise on 500 ViT-source images, versus 76.91% for the same-seed original/phase opponent mainline.
- Do not claim Gaussian-residual results for models whose completed provenance has not been audited.
