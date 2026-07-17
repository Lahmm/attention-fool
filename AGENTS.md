# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to remote repo

## Research direction memory

- The current mainline is a research method centered on two mechanisms: **patch-score-guided patch drop** and **RGB opponent-channel random noise**.
- The primary goal is to develop a self-consistent paper motivation and mechanism analysis around these two points, not to keep stacking attack modules or optimize ASR alone.
- Patch-score is the semantic routing mechanism: it answers **where** to perturb by comparing final-layer global and local patch features. Future work should analyze score meaning, high/low/random routing, layer choice, drop budget, location stability, feature response, and gradient response.
- RGB opponent-channel noise is the structured perturbation mechanism: it answers **how** to perturb kept evidence by sampling luminance, red-green, and yellow-blue directions in RGB and projecting them through the model's initial RGB projection. Future work should compare it with RGB Gaussian and feature IID Gaussian, isolate the three color directions, test injection position, kept-only versus other masks, and study RMS matching.
- The intended conceptual story is: patch-score determines where semantic evidence is erased; opponent-channel noise perturbs how the remaining evidence is represented. Their complementarity is the main method hypothesis.
- Phase pairs, multi-view gradient averaging, Gaussian gradient residual, MI/NI/DIM/TI, and transfer ASR are supporting mechanisms or validation tools. They should be held fixed or used as controlled ablations while the two core mechanisms are analyzed.
- ASR remains an important sanity check for transferability, but it is no longer the sole or primary optimization target for this project.
- When proposing future experiments, prefer mechanism-driven ablations and interpretable diagnostics over blind ASR tuning. Keep the motivation, method, ablations, and claims aligned; do not claim Gaussian-residual results for models that were not actually run.
