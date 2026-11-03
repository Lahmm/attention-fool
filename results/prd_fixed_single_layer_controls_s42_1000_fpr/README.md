# Three fixed single-layer FPR13 controls

These are three independent new runs: CaiT-S24 block22, PiT-T stage3_block1, and PiT-T stage1_block1. Each attacks 1000 images with a single fixed native block input boundary. The repository maximum-variance mainline is unchanged. No dynamic score calculation or normalization is used in these attacks.

All cases use seed 42, epsilon 16/255, 10 steps, step size epsilon/10, momentum 1, and 10 groups × 2 views. Retain the original opponent augmentation, strength 0.2 for CaiT and 0.4 for PiT-T. The nominal drop ratio is 30/196; native rounding gives 30/196 at CaiT block22, 8/49 at PiT-T stage3_block1, and 112/729 at PiT-T stage1_block1. Random masks are resampled per step/group and phase views use spatially corresponding masks.

FPR13 evaluates all 1000 adversarial images with batch 32, no AMP, and TF32 disabled. ASR is 1 minus adversarial accuracy, without target-clean-correct filtering. Strict ASR excludes the exactly matching source target. There is no four-source macro average because these cases contain one CaiT and two PiT configurations.

| Case | Overall ASR | Strict ASR | Reference fixed checkpoint | Reference Strict | Difference vs fixed | Normalized dynamic Strict | Difference vs normalized |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| cait_block22 | 84.82% | 83.67% | block23 | 89.05% | -5.38 | 86.62% | -2.96 |
| pit_t_stage3_block1 | 82.95% | 82.95% | stage3_block3 | 85.08% | -2.14 | 83.06% | -0.12 |
| pit_t_stage1_block1 | 81.48% | 81.48% | stage3_block3 | 85.08% | -3.60 | 83.06% | -1.58 |

Differences are current minus reference, in percentage points. `summary.csv` includes both Overall and Strict comparisons. `per_target.csv` contains all 39 current source/case/target ASRs and direct differences. For PiT-T stage1_block1, the prior 1000-image run is also preserved and compared; this new run does not overwrite it.

`runs/` preserves the current parameters and FPR13 results. Reference folders preserve their actual configurations. `provenance.json` records hashes, execution settings, source snapshots, and runtime model configuration overrides. The external worker only sets fixed checkpoints, drop ratios, and dynamic_checkpoint=False before using the copied main execution path. Attack images, predictions, and logs remain in the recorded output directories.
