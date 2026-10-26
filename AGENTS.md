# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.
- When modifying code, do not add comments at the top of `.py` files.
- New executions use the fixed seed 42. Retain `--input-diversity-views-per-group` for ablations. Keep the FPR weight downloader, but remove experiment runner scripts.
- For experimental reporting, the user treats historical `randomness.py` seed-42 measurements and current PyTorch-global-seed-42 measurements as one seed-42 standard. RNG implementation differences alone do not require rerunning existing results; checkpoint/drop-parameter changes do.
- Discussions are read-only. Modify code or files only when the user explicitly requests those changes; a discussion of implementing an idea is not authorization to implement it.

## Current research mainline

- The paper mainline is **Progressive Route Disruption (PRD)** and has exactly two core mechanisms: **checkpoint-wise progressive random token drop** and **RGB opponent-channel random noise**. Do not stack unrelated attack modules onto the main claim.
- The only retained attack is PRD, implemented by `progressive_attack.py` and dispatched by `main.py`.
- The validated ViT-B/16 configuration uses checkpoint boundaries `block3` and `block10`, independently drops 10 of 196 local tokens at each checkpoint (`drop_ratio=0.051020408163`), and permits the same position to be selected again later in the schedule. This is the only ViT attack configuration that should be described as current or used by current executable examples and tests.
- At every attack step and augmentation group, generate a fresh random checkpoint schedule. At each checkpoint, sample uniformly from all current local-token positions and hard-zero the selected tokens before continuing the same forward trajectory. The original view uses that schedule and the phase view uses its spatially transformed counterpart. The default 10 steps × 10 groups produces 100 schedules and 200 checkpoint mask selections per ViT image.
- Patch score belongs only to the motivating observation that token semantic rankings reorganize across model depth. It is not an attack selector, runtime dependency, adapter API, CLI option, or claimed source of PRD's transferability.
- Opponent-channel noise perturbs **all initial local features**: sample luminance, red-green, and yellow-blue RGB directions, project through the initial RGB projection, and RMS-match in feature space. Add this noise once per differentiable view before checkpoint traversal, including positions selected for later drops. Keep prefix/CLS tokens unchanged. See `experiments/prd_method_contract.md` for the execution contract.
- The current primary transfer evaluation is seed-42, four-source, 1000-image FPR13. The updated PiT-T and distilled DeiT-B drop/full cells completed on 2026-10-23; the other twelve ablation cells retain their parameter-matched earlier measurements. Four-source full-PRD Overall ASR is 89.72%; strict black-box ASR is 89.29%. Current results and ablations are in `results/prd_s42_1000_fpr_{summary,per_target,effects}.csv` and `results/prd_s42_1000_fpr_ablation.md`; versioned parameters and raw transfer tables are in `results/prd_s42_fpr_run_artifacts/`, with provenance in `results/prd_s42_1000_fpr_provenance.json`. The 2026-10-12 project 14-target run at `bf092246` remains a historical record with its original values.
- Keep records and paper drafts focused on this single method and main experiment. Describe RGB opponent-channel noise on all initial local features followed by progressive checkpoint drops; do not introduce noise-policy version comparisons or a selection narrative.
- Evaluate transferability and component effects through transfer ASR defined as `1 - adversarial accuracy` over all evaluated adversarial samples; do not filter to a target-clean-correct subset. Gradient diagnostics, including effective rank, are internal experimental analysis tools and are not paper evidence for transferability.
- Current paper sources are ViT-B/16, CaiT-S24, PiT-T, and distilled DeiT-B. Preserve PiT-B as an additional source; remove Visformer source support while retaining it as an evaluation target. Do not encode ViT block assumptions as a nominally cross-architecture implementation.

## Paper-writing objective and voice

- Write a CVPR 2027 paper from the local `prd` branch. Use the current four-source, 1000-image, seed-42 FPR13 results and parameter-matched ablations as the primary transfer evidence; preserve the earlier project 14-target tables as historical records.
- Lead the motivation with the dynamic discovery: the same local token's cosine similarity to the global representation changes substantially across layers, and the locations of high-patch-score patches change with model depth. Show this process with patch-score visualizations of actual samples; keep Spearman statistics as supporting detail.
- Build the story around evolving local evidence, opponent-channel perturbation of initial local features, progressive random checkpoint disruption, and transfer ASR. Patch-score visualizations explain the discovery; checkpoint schedules drive the attack.
- Use vivid, direct, affirmative language. Explain what we observed, why it is interesting, and how PRD acts on that process. Keep the narrative focused and engaging, with numerical results and method descriptions faithful to the records.
- Avoid defensive clarifications about claims we do not make: causal patch importance, identified semantic causal paths, optimal random sampling, guaranteed superiority, and similar hypothetical objections. Do not repeatedly qualify the central discovery with unrelated limitations.
- ASR is the paper's measure of transferability. Keep effective rank, gradient diversity, and source-target gradient diagnostics in internal process records; do not plan them as main-text or supplementary evidence for transferability.
- Record implementation and provenance details as concise technical facts. Select paper content for its contribution to the story, method understanding, and ASR evaluation rather than exposing every diagnostic or code detail.
- Apply these conventions consistently to the blueprint, story documents, repository summaries, and subsequent paper drafts. If a requested wording or change is unclear, leave that part unchanged and check with the user.

## Persistent paper-writing conversation workflow

用户于 2026-10-12 明确规定：当前对话用于完整论文撰写；今后用户再次说明某个对话用于本项目论文撰写时，严格复用以下工作流。

- 依据 `experiments/cvpr2027_writing_blueprint.md` 的章节划分与写作顺序，逐部分推进整篇论文。
- 当用户指定某部分或章节时，每次只撰写一个自然段，细化到大章节或小章节中的具体段落；不要一次交付完整大章节、小章节或其全部结构。
- 每个自然段同时提供自然、地道的英文论文原文及忠实对应的中文翻译。
- 段落需要公式或伪代码时，在当前对话中以可阅读、正确渲染的形式呈现，不交付未渲染的公式源码或伪代码代码块。
- 段落需要图片或表格时，说明具体应呈现的内容及其在论证中的作用；依据已有材料区分可直接使用的证据与尚需准备的内容。
- 用户于 2026-10-14 补充：后续撰写中需要稍后插入图片时，在对应段落后提供 `todonotes` 的正文高亮待办框，使用 `\todo[inline,color=yellow!30,caption={...}]{...}`，写明待补图片的唯一标识及具体内容。待办须直接显示在编译后的 PDF 中，并纳入 `\listoftodos` 的统一清单，避免仅用 `% TODO` 注释记录。主文件导言区使用 `\usepackage[colorinlistoftodos]{todonotes}`；继续沿用已有配置，无需每段重复提供导言区设置。
- 交付当前段落后，与用户讨论、修改并确定细节，再推进下一段；用户确认的术语、措辞和细节在后续段落保持一致。
- 论文文字仅在对话中生成，由用户自行放入 LaTeX 编辑；不在本地生成论文文档。论文讨论保持只读，除非用户明确要求保存记忆或修改项目文件。

## Retained executable scope

- Retain only PRD's progressive random-drop behavior in `progressive_attack.py`, the minimal `main.py` execution path needed to run it, and the progressive adapters for the five retained source models in `nets/`.
- Retain only utility, test, configuration, and documentation code that is directly required to execute or verify that progressive attack.
- Pixel gradients are averaged across views and used directly for momentum accumulation and the projected update.
- The protected behavior is: fresh uniformly random checkpoint schedules, the ViT `block3,block10` schedule, local-token hard zeroing during sequential checkpoint traversal, transformed phase-pair schedules, RGB opponent-channel projected noise on all initial local features before traversal, and the projected iterative update used by PRD.
- Code is not protected merely because it is currently imported, exposed by the CLI, covered by a test, mentioned in an old report, or needed to reproduce a superseded experiment.

## Removed scope

- Do not restore non-progressive attack implementations or compatibility paths, including the deleted `attack.py`, `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, and `token_patch_dropout` paths.
- Do not restore selector or scoring paths, including high/low/extreme/rank-transition selection, score modes, global-score noise, final-layer patch-score extraction, token hooks, or legacy checkpoint registries. The adapters expose only the PRD traversal contract.
- Non-mainline augmentation modules, gradient post-processing variants, diagnostics, completed experiment runners, compatibility wrappers, historical tests, archived reports, and generated artifacts may be removed when they are not required by PRD.
- Backward CLI compatibility and reproduction of historical experiments are not cleanup requirements. Remove stale parameters and metadata instead of preserving no-op or legacy options.
- Update or delete tests and documentation together with removed functionality so that the remaining repository describes only the retained progressive attack.
