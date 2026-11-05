# Project Instructions

- The assistant may use terminal commands in this repository except `rm` and commands that delete the Git repository.
- After each completed modification to the current repository contents, the assistant should initiate a `git push`.
- The assistant is allowed to use `git push` to push the change to the remote repository.
- When modifying code, do not add comments at the top of `.py` files.
- New executions use the fixed seed 42. Retain `--input-diversity-views-per-group` for ablations. Keep the FPR weight downloader, but remove experiment runner scripts.
- For experimental reporting, the user treats historical `randomness.py` seed-42 measurements and current PyTorch-global-seed-42 measurements as one seed-42 standard. RNG implementation differences alone do not require rerunning existing results; checkpoint/drop-parameter changes do.
- Keep `results/` exclusively for FPR13 ASR results and their parameters, ablations, direct ASR differences, and provenance. Do not restore non-FPR13 numerical snapshots there.
- Confidence intervals (置信区间), bootstrap/resampling error bars, statistical significance tests, McNemar tests, and p-values are not required for this project. Do not calculate, generate, store, or report them, and do not add or restore their analysis code, tools, tests, result files, metadata, or documentation. Evaluate and report GDI using ASR and direct ASR differences only.
- Discussions are read-only. Modify code or files only when the user explicitly requests those changes; a discussion of implementing an idea is not authorization to implement it.

## Current research mainline

- The method is GDI (Geometry-Guided Depth Intervention). The paper centers on stronger transfer attacks through effective layer-input intervention. Its two contributions are the discovery of layers that may play an important information-aggregation role and yield good transfer ASR when their inputs are disturbed, and a model-agnostic, sample-dependent computation that better locates effective intervention layers. The implementation selects a block per image using direction variance ratio times positive fractional contraction and uniformly hard-zeros local tokens at its input.
- The retained attack is implemented by `attack.py` as `GDIAttacker` and dispatched by `main.py`. The four paper sources are ViT-B/16, CaiT-S24, PiT-T and distilled DeiT-B. PiT-B and Visformer remain FPR13 evaluation targets; only the four dynamic paper sources are executable attack sources. Preserve all cached model weights, including legacy targets.
- Before each image's attack, run one clean forward without drops or feature noise. Capture native block local inputs and outputs. L2-normalize each local vector and global reference in float64 from float32 features. Divide output global-direction population variance by output local dispersion per native feature dimension, and multiply by max(0, 1 - output local dispersion / input local dispersion). Select the maximum product; break ties by earliest native block order. Drop at that block's INPUT. Keep the per-image boundary fixed throughout all steps and views. Zero input or output dispersion gives score zero; all-zero scores choose the earliest block. Reject nonfinite or zero-norm features. Preserve architecture-specific reference and pooling definitions in `experiments/gdi_method_contract.md`.
- Patch score is a runtime dependency for layer selection and a motivating observation. At the selected boundary, positions are sampled uniformly without replacement from all local tokens; scores do not select dropped positions. Hard-zero all channels of selected local tokens, preserve prefix tokens and native grids, and continue the same forward.
- Sample a fresh mask at every step/group. The original view uses it and the phase view uses its spatially transformed counterpart with the drop count preserved. Default 10 steps × 10 groups gives 100 original mask selections per image, one selected boundary per view. Different boundaries in a batch are the union of different samples' choices.
- Treat i.i.d. RGB noise projected through the initial RGB projection and feature-RMS matched, phase pairing, gradient averaging, momentum and projection as basic augmentation and transfer-attack settings. Describe them concisely in Attack Optimization and put construction and parameters in supplementary implementation settings.
- The CLI and attacker support only direct i.i.d. standard Gaussian RGB sampling before projection. Opponent noise is retired from execution and ablations: do not restore its selector, color basis, strength names or tests. Use --noise-strength and adapter noise_strength for beta; retain actual labels in historical records. Projected features may be correlated. Add augmentation once to all initial local features per differentiable view, keep prefixes unchanged, and retain actual noise labels in measurements.
- Paper-source total drop ratios are 20/196 (ViT/DeiT) and 30/196 (CaiT/PiT-T); use the selected native grid for max(1, round(N*r)), recording actual counts. The only drop parameter is the per-image --drop-ratio; fixed-layer overrides and all fixed double/multiple-checkpoint attack paths have been removed.
- The evaluation is seed-42, four-source, 1000-image FPR13. Completed i.i.d. mainline and retained historical opponent runs are indexed in experiments/gdi_paper_story.md and experiments/server_handoff_inventory.md. The user retired the old fixed multi-checkpoint results and ablations and strong fixed-single-layer baselines on 2026-11-04; do not restore them. Final i.i.d. main results and matched controls must correspond to actual executed configurations.
- The sole mainline selector is direction variance ratio times positive fractional contraction. The historical opponent/1000-image FPR13 evidence is `results/gdi_direction_contraction_s42_1000_fpr/` (Overall 89.1673%, strict 88.7234%); it was executed as an external experiment before promotion on 2026-11-04. Preserve original execution labels, parameters and source hashes. If user-authorized archive pruning changes a report, record the maintenance and prior report hashes and refresh the retained-file manifest. Maximum variance is now a historical comparison, not an executable selection alternative.
- Prioritize matched no-drop, per-image random-layer, historical maximum-variance-layer and current-selector ASR comparisons. Strong fixed-single-layer baselines have been retired. Present token-selection and fixed/resampled-mask results using their actual settings and direct differences. Keep observed selection frequencies faithful to the records.
- Evaluate transfer as 1 - adversarial accuracy over all adversarial samples, with no target-clean-correct filtering. Use ASR and direct ASR differences; gradient diagnostics remain internal. Keep `results/` exclusively for FPR13 evidence and provenance.

## Paper-writing objective and voice

- Write a CVPR 2027 paper following `experiments/cvpr2027_writing_blueprint.md` and `experiments/gdi_paper_story.md`, updated on 2026-11-05. Use GDI consistently in current documentation and result presentation.
- Focus on two contributions: (1) identifying layers that may play an important information-aggregation role and whose input disturbance yields good transfer attacks; (2) a model-agnostic, sample-dependent computation for locating layers that are more effective for transfer attacks. Keep transfer ASR at the center of both contributions.
- Use this story: intervention depth affects transfer ASR → effective layers and their representation characteristics → sample-dependent localization → random input intervention → transfer results. Use “layers that may play an information-aggregation role” for the observation, matching the user's wording.
- Preserve the existing Introduction / Related Work / Method / Experiments / Conclusion structure. Method retains 3.1 Preliminaries, 3.2 Depth-wise Local–Global Alignment, 3.3 Direction Variance and Contraction-Guided Layer Selection and Random Token Drop, and 3.4 Attack Optimization. Experiments retains 4.1 Experimental Setup, 4.2 Main Transfer Results, 4.3 Layer Selection, and 4.4 Token Dropping and Additional Analysis.
- Connect actual-sample cross-depth local/global observations to input-intervention ASR. Distinguish changes of one token across layers from differences among tokens within a layer. Heatmap normalization is display-only. Selection uses direction variance ratio times positive fractional contraction.
- Dynamic means selection computed once from the clean input; the boundary stays fixed across attack steps and views. Resampling refers to token positions.
- Treat noise, phase, gradient averaging, momentum and projection as basic augmentation and optimization settings. Keep their main-text description concise and place detailed construction in supplementary implementation settings.
- Use vivid, direct, affirmative language. Strictly prohibit defensive expressions about claims the paper has not made. Write only the actual research claims, observations, method definitions and supporting ASR comparisons; let each sentence advance the argument.
- Use ASR and direct ASR differences to discuss transferability and design effects. Keep gradient diagnostics internal. Present detailed architecture, augmentation and provenance in concise implementation records or supplementary material.
- Preserve actual historical augmentation labels, execution parameters, source hashes and original run identifiers. Current archive names and report titles use GDI; original execution records retain their measured identity. Record archive renames and report edits, previous hashes and refreshed retained-file manifests.
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
- Protect once-per-clean-image direction variance ratio times positive fractional contraction selection, architecture-native references and input boundaries, fresh uniform masks, local-token hard zeroing, spatially corresponding phase masks, common initial projected feature augmentation, raw view-gradient averaging, momentum and projected updates.
- Only the current dynamic single-boundary attack is executable. Do not restore fixed-layer or fixed double/multiple-checkpoint paths, PiT-B fixed-source configuration, retired experiment runners, or the old Figure 2 script. Historical token-selection and dynamic-selector data may remain as evidence with actual settings.
- Code is not protected merely by existing imports, CLI exposure, tests, or old reports. Keep the FPR weight downloader and the view-count ablation option; do not restore completed experiment runners.

## Removed scope

- Do not restore unrelated attack implementations or compatibility paths, including `original_score_postdrop_phase_pair`, `none`, `patch_dropout`, and `token_patch_dropout` paths.
- Do not restore high/low/extreme/rank-transition token selectors to the main attack, global-score noise, old score modes, hooks, or legacy checkpoint registries. Retain the normalized feature geometry required by the current selector and raw cosine variance as recorded diagnostic metadata. Isolated token-selection analyses remain evidence with their actual configuration, not additional attack mechanisms.
- Non-mainline modules, gradient post-processing variants, diagnostics, completed runners, compatibility wrappers and historical artifacts can be removed when explicitly requested and unnecessary for the current attack.
- Backward CLI compatibility and superseded-experiment reproduction are not cleanup requirements. Update tests and documentation with functionality changes so the current repository describes the retained attack accurately.
