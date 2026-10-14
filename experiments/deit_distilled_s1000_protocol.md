# DeiT-B distilled supplementary PRD experiment

Requested supplementary source: `deit_base_distilled_patch16_224`.
This source reuses `ViTAdapter`, which separates local tokens using the
model's `num_prefix_tokens`. Both CLS and distillation tokens are preserved
by local-token noise and masking; the native `forward_head` averages the
two classification heads in evaluation mode.

Use `python experiments/run_prd_deit_fpr_targets.py` after caching
`timm/deit_base_distilled_patch16_224.fb_in1k`. Downloaded HF revision:
`ae7e500ba066fc77dcce2bf518f54a093b9f9c01`.

Settings use the current ViT PRD configuration: seed 20260907, 1000 images,
epsilon 16/255, step size 1.6/255, ten steps, ten original/phase pairs per
step, checkpoints `block3,block10`, drop ratios
`0.051020408163,0.051020408163` (ten of 196 local tokens per checkpoint),
opponent noise strength 0.2 and momentum decay 1.0. Local batch size is eight, matching the new FPR-target ViT run.

Evaluate the same thirteen FPR Table 2 targets with the existing FPR
protocol, FP32 and TF32 disabled. None is the distilled DeiT-B source,
so both overall and black-box averages include all thirteen targets.
Require sample IDs and labels to match the completed ViT predictions.

Outputs: `outputs/attack/prd_fpr_targets_deit_distilled_seed20260907/`
and `outputs/fpr_targets/deit_distilled_{attack.log,transfer.log,predictions.json,transfer_asr.csv}`.
Keep these supplementary results separate from the original four-source
main experiment and its result files.

Adapter validation on the real timm distilled architecture passed:
196 local tokens and two prefix tokens; uninterrupted checkpoint traversal
matches the native dual-head output; hard-zero masks preserve prefixes;
masked traversal yields finite, nonzero input gradients. The existing
seven transfer evaluation tests also pass.
