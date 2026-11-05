# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.
- When modifying code, do not add comments at the top of `.py` files.
- New executions use the fixed seed 42. Retain `--input-diversity-views-per-group` for ablations. Keep the FPR weight downloader, but remove experiment runner scripts.
- For experimental reporting, the user treats historical `randomness.py` seed-42 measurements and current PyTorch-global-seed-42 measurements as one seed-42 standard. RNG implementation differences alone do not require rerunning existing results; checkpoint/drop-parameter changes do.
- Keep `results/` exclusively for FPR13 ASR results and their parameters, ablations, direct ASR differences, and provenance. Do not restore non-FPR13 numerical snapshots there.
- Confidence intervals (置信区间), bootstrap/resampling error bars, statistical significance tests, McNemar tests, and p-values are not required for this project. Do not calculate, generate, store, or report them, and do not add or restore their analysis code, tools, tests, result files, metadata, or documentation. Evaluate and report PRD using ASR and direct ASR differences only.
- Discussions are read-only. Modify code or files only when the user explicitly requests those changes; a discussion of implementing an idea is not authorization to implement it.

## Current research mainline

- The paper focuses on per-image layer selection by direction variance ratio times positive fractional contraction and uniformly random local-token hard zeroing at the selected block input. PRD remains the repository working name; the final paper name is not decided. Treat these as one integrated design.
- The retained attack is implemented by `progressive_attack.py` and dispatched by `main.py`. The four paper sources are ViT-B/16, CaiT-S24, PiT-T and distilled DeiT-B. PiT-B and Visformer remain FPR13 evaluation targets; only the four dynamic paper sources are executable attack sources. Preserve all cached model weights, including legacy targets.
- Before each image's attack, run one clean forward without drops or feature noise. Capture native block local inputs and outputs. L2-normalize each local vector and global reference in float64 from float32 features. Divide output global-direction population variance by output local dispersion per native feature dimension, and multiply by max(0, 1 - output local dispersion / input local dispersion). Select the maximum product; break ties by earliest native block order. Drop at that block's INPUT. Keep the per-image boundary fixed throughout all steps and views. Zero input or output dispersion gives score zero; all-zero scores choose the earliest block. Reject nonfinite or zero-norm features. Preserve architecture-specific reference and pooling definitions in `experiments/prd_method_contract.md`.
- Patch score is a runtime dependency for layer selection and a motivating observation. At the selected boundary, positions are sampled uniformly without replacement from all local tokens; scores do not select dropped positions. Hard-zero all channels of selected local tokens, preserve prefix tokens and native grids, and continue the same forward.
- Since 2026-11-05, use independent identity-or-phase views. At each step/view and for each image, sample identity with probability 1/2 or a shift uniformly from the configured phase set with probability 1/2, and sample a fresh uniform mask directly on that view's native token grid. There is no paired mask or cross-view mask mapping. The retained CLI groups × views-per-group controls only the total budget (default 20 views); views-per-group=1 retains original-only ablation. Default 10 steps gives 200 independent mask selections per image, one selected boundary per view. Different boundaries in a batch are the union of different samples' choices.
- The paper uses i.i.d. RGB noise projected through the initial RGB projection and feature-RMS matched as ordinary random feature augmentation. It is not a separate mechanism, contribution, motivating story, or main-text noise subsection. Put its construction and parameters in supplementary implementation settings. Independent spatial-view sampling, gradient averaging, momentum and projection form the optimization procedure.
- The current code still defaults to opponent noise; `--noise-structure isotropic_rgb` explicitly enables i.i.d. RGB sampling. Documentation changes do not change this default. Projected features may be correlated. Add augmentation once to all initial local features per differentiable view, keep prefixes unchanged, and retain actual noise labels in measurements.
- Paper-source total drop ratios are 20/196 (ViT/DeiT) and 30/196 (CaiT/PiT-T); use the selected native grid for max(1, round(N*r)), recording actual counts. The only drop parameter is the per-image --drop-ratio; fixed-layer overrides and all fixed double/multiple-checkpoint attack paths have been removed.
- The evaluation is seed-42, four-source, 1000-image FPR13. Current and historical dynamic/opponent runs are indexed in experiments/prd_paper_story.md and experiments/server_handoff_inventory.md. The user retired the old fixed multi-checkpoint results and ablations and strong fixed-single-layer baselines on 2026-11-04; do not restore them. Final i.i.d. main results and matched controls must correspond to actual executed configurations.
- The sole mainline selector is direction variance ratio times positive fractional contraction. The completed paired-view opponent/1000-image FPR13 evidence is `results/prd_direction_contraction_s42_1000_fpr/` (Overall 89.1673%, strict 88.7234%); it was executed as an external experiment before promotion on 2026-11-04. Preserve original execution labels, parameters and source hashes. If user-authorized archive pruning changes a report, record the maintenance and prior report hashes and refresh the retained-file manifest. Maximum variance is now a historical comparison, not an executable selection alternative.
- Prioritize matched no-drop, per-image random-layer, historical maximum-variance-layer and current-selector ASR comparisons. Strong fixed-single-layer baselines have been retired. Present token-selection and fixed/resampled-mask results using their actual settings and direct differences. Keep observed selection frequencies faithful to the records.
- Evaluate transfer as 1 - adversarial accuracy over all adversarial samples, with no target-clean-correct filtering. Use ASR and direct ASR differences; gradient diagnostics remain internal. Keep `results/` exclusively for FPR13 evidence and provenance.

## Paper-writing objective and voice

- Write a CVPR 2027 paper from local `prd`. Follow `experiments/cvpr2027_writing_blueprint.md` and `experiments/prd_paper_story.md`, updated on 2026-11-04. Keep retained historical dynamic/opponent, current dynamic/opponent and final dynamic/i.i.d. configurations accurately identified.
- Lead with actual-sample cross-depth local/global alignment visualizations, then the direction variance ratio, positive local contraction and intervention-depth selection. Distinguish a token's changes across layers from token differences within a layer. Heatmap normalization is display-only; selection uses the direction variance ratio times positive fractional contraction, not maximum raw-score variance.
- Build the story around evolving local/global alignment, layer selection by direction variance ratio times positive fractional contraction, random token intervention at the selected block input, and transfer ASR. Dynamic means input-computed selection once before attack; resampling refers to token positions.
- Keep noise out of the abstract, introduction, contributions, conclusion and core-figure emphasis. Record i.i.d. random feature augmentation accurately in supplementary common settings. Do not preserve the former two-core-mechanism or multi-checkpoint narrative in current writing.
- Use vivid, direct, affirmative language. Let actual observations, method definitions and ASR comparisons advance the argument. Avoid repeated defensive clarifications about unclaimed causal importance, optimality or guarantees.
- ASR is the paper's transferability measure. Keep effective rank, gradient diversity and source-target gradient diagnostics internal. Put detailed architecture, augmentation and provenance information in concise implementation records or supplementary material.
- Apply these conventions to the blueprint, story, summaries and subsequent drafts. If requested wording is unclear, leave that part unchanged and check with the user.

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

- Retain the current single-boundary attack selected by direction variance ratio times positive fractional contraction, required source adapters, minimal `main.py` path, and utilities/tests/configuration/docs directly needed to execute or verify it.
- Protect once-per-clean-image direction variance ratio times positive fractional contraction selection, architecture-native references and input boundaries, fresh independent uniform masks per view, local-token hard zeroing, independent identity/phase input views, common initial projected feature augmentation, raw view-gradient averaging, momentum and projected updates.
- Only the current dynamic single-boundary attack is executable. Do not restore fixed-layer or fixed double/multiple-checkpoint paths, PiT-B fixed-source configuration, retired experiment runners, or the old Figure 2 script. Historical token-selection and dynamic-selector data may remain as evidence with actual settings.
- Code is not protected merely by existing imports, CLI exposure, tests, or old reports. Keep the FPR weight downloader and the view-count ablation option; do not restore completed experiment runners.

## Removed scope

- Do not restore unrelated attack implementations or compatibility paths, including deleted `attack.py`, `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, and `token_patch_dropout` paths.
- Do not restore high/low/extreme/rank-transition token selectors to the main attack, global-score noise, old score modes, hooks, or legacy checkpoint registries. Retain the normalized feature geometry required by the current selector and raw cosine variance as recorded diagnostic metadata. Isolated token-selection analyses remain evidence with their actual configuration, not additional attack mechanisms.
- Non-mainline modules, gradient post-processing variants, diagnostics, completed runners, compatibility wrappers and historical artifacts can be removed when explicitly requested and unnecessary for the current attack.
- Backward CLI compatibility and superseded-experiment reproduction are not cleanup requirements. Update tests and documentation with functionality changes so the current repository describes the retained attack accurately.
