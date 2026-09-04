# CLS score noise × gradient postprocess：1000 图控制变量实验

## 目的与实验矩阵

本实验只控制两个因素，攻击的两个核心机制保持不变：动态 patch-score-guided
patch drop 与 kept-only RGB opponent-channel noise 始终开启。

- 因素 A：计算 patch score 时是否向当前 CLS token 加 Gaussian noise；开启强度为
  `0.2`，关闭时直接使用当前 CLS token。
- 因素 B：20-view raw mean 后、MI 累积前是否加入 Gaussian residual；开启配置为
  `sigma=4, alpha=0.75`，关闭配置为 `alpha=0`。

四个格子必须使用相同的 1000 张图、seed、白盒模型、攻击预算、动态 mask、phase
pair 和 opponent noise 配置。当前最高主线结果已经覆盖 `A=on, B=on`，其目录为
`outputs/attack/vit_final_drop15_s1000_seed20260903`，无需重跑。还需完成以下三组：

| 实验名 | CLS score noise | Gaussian residual | 状态 |
| --- | --- | --- | --- |
| `vit_final_drop15_s1000_seed20260903` | on | on | 已完成，Overall ASR 78.446% |
| `vit_clsnoise_on_gradres_off_s1000_seed20260903` | on | off | 待完成 |
| `vit_clsnoise_off_gradres_on_s1000_seed20260903` | off | on | 待完成 |
| `vit_clsnoise_off_gradres_off_s1000_seed20260903` | off | off | 待完成 |

旧的 seed `20260716` raw-mean 结果不能代替这里的 `A=on, B=off`，因为控制变量比较
还要求样本级随机流一致。`GradientReplay` 会按 sample、step、group、view 和 event
固定随机流，因此以下命令统一使用 seed `20260903`。

## 攻击命令

### A=on，B=off

```bash
python main.py \
  --attack-method original_score_postdrop_phase_pair \
  --whitebox-model vit_base_patch16_224 \
  --max-attacked-samples 1000 --sample-offset 0 \
  --epsilon 0.06274509803921569 --steps 10 \
  --seed 20260903 --mi --mi-decay 1.0 \
  --input-diversity-groups 10 --input-diversity-views-per-group 2 \
  --input-diversity-phase-shift-set '4,4;8,8;12,12' \
  --guide-aug-strength 0.2 \
  --patch-dropout-ratio 0.3 \
  --patch-dropout-score-mode high \
  --patch-dropout-sampling-mode random \
  --patch-dropout-score-quantile-jitter 0 \
  --patch-dropout-score-noise 0 \
  --token-score-cls-noise --token-score-cls-mode learned \
  --token-cls-noise-strength 0.2 \
  --post-dropout-phase-token-noise \
  --post-dropout-feature-noise-type opponent_projected \
  --post-dropout-feature-noise-strength 0.2 \
  --patch-score-layer final --patch-selector patch_score \
  --gaussian-sigma 4 --gaussian-alpha 0 \
  --batch-size 96 --num-workers 4 --prefetch-factor 4 \
  --output-dir outputs/attack/vit_clsnoise_on_gradres_off_s1000_seed20260903
```

### A=off，B=on

```bash
python main.py \
  --attack-method original_score_postdrop_phase_pair \
  --whitebox-model vit_base_patch16_224 \
  --max-attacked-samples 1000 --sample-offset 0 \
  --epsilon 0.06274509803921569 --steps 10 \
  --seed 20260903 --mi --mi-decay 1.0 \
  --input-diversity-groups 10 --input-diversity-views-per-group 2 \
  --input-diversity-phase-shift-set '4,4;8,8;12,12' \
  --guide-aug-strength 0.2 \
  --patch-dropout-ratio 0.3 \
  --patch-dropout-score-mode high \
  --patch-dropout-sampling-mode random \
  --patch-dropout-score-quantile-jitter 0 \
  --patch-dropout-score-noise 0 \
  --no-token-score-cls-noise --token-score-cls-mode learned \
  --token-cls-noise-strength 0.2 \
  --post-dropout-phase-token-noise \
  --post-dropout-feature-noise-type opponent_projected \
  --post-dropout-feature-noise-strength 0.2 \
  --patch-score-layer final --patch-selector patch_score \
  --gaussian-sigma 4 --gaussian-alpha 0.75 \
  --batch-size 96 --num-workers 4 --prefetch-factor 4 \
  --output-dir outputs/attack/vit_clsnoise_off_gradres_on_s1000_seed20260903
```

### A=off，B=off

```bash
python main.py \
  --attack-method original_score_postdrop_phase_pair \
  --whitebox-model vit_base_patch16_224 \
  --max-attacked-samples 1000 --sample-offset 0 \
  --epsilon 0.06274509803921569 --steps 10 \
  --seed 20260903 --mi --mi-decay 1.0 \
  --input-diversity-groups 10 --input-diversity-views-per-group 2 \
  --input-diversity-phase-shift-set '4,4;8,8;12,12' \
  --guide-aug-strength 0.2 \
  --patch-dropout-ratio 0.3 \
  --patch-dropout-score-mode high \
  --patch-dropout-sampling-mode random \
  --patch-dropout-score-quantile-jitter 0 \
  --patch-dropout-score-noise 0 \
  --no-token-score-cls-noise --token-score-cls-mode learned \
  --token-cls-noise-strength 0.2 \
  --post-dropout-phase-token-noise \
  --post-dropout-feature-noise-type opponent_projected \
  --post-dropout-feature-noise-strength 0.2 \
  --patch-score-layer final --patch-selector patch_score \
  --gaussian-sigma 4 --gaussian-alpha 0 \
  --batch-size 96 --num-workers 4 --prefetch-factor 4 \
  --output-dir outputs/attack/vit_clsnoise_off_gradres_off_s1000_seed20260903
```

## 迁移评估命令

三组攻击各自完成后运行对应命令。`--strict-model-loading` 用于保证 7 个 Transformer
和 6 个 CNN 全部实际完成，而不是静默跳过不可用模型。

```bash
python transfer_eval.py \
  --image-dir outputs/attack/vit_clsnoise_on_gradres_off_s1000_seed20260903 \
  --prefix adv_ --batch-size 64 --num-workers 4 --prefetch-factor 2 --amp \
  --strict-model-loading \
  --exp-name vit_clsnoise_on_gradres_off_s1000_seed20260903

python transfer_eval.py \
  --image-dir outputs/attack/vit_clsnoise_off_gradres_on_s1000_seed20260903 \
  --prefix adv_ --batch-size 64 --num-workers 4 --prefetch-factor 2 --amp \
  --strict-model-loading \
  --exp-name vit_clsnoise_off_gradres_on_s1000_seed20260903

python transfer_eval.py \
  --image-dir outputs/attack/vit_clsnoise_off_gradres_off_s1000_seed20260903 \
  --prefix adv_ --batch-size 64 --num-workers 4 --prefetch-factor 2 --amp \
  --strict-model-loading \
  --exp-name vit_clsnoise_off_gradres_off_s1000_seed20260903
```

## 完成判据与汇报

每个攻击目录必须有 1000 个 `adv_*` 文件，并保留 `attack_params.json`、
`gradient_diagnostics.json` 和 `replay_manifest.json`。迁移 CSV 中 13 个目标模型的
`total` 均应为 1000；ASR 统一按全部对抗样本计算：
`ASR = 1 - adversarial accuracy`。

记四格 Overall ASR 为 `Y11`、`Y10`、`Y01`、`Y00`（下标依次表示 A、B 的开关），
至少报告 `Y11-Y01`（在 residual 开启时 CLS noise 的收益）、`Y11-Y10`（在 CLS noise
开启时 residual 的收益）和交互项 `Y11-Y10-Y01+Y00`。同时报告 Transformer avg 与
CNN avg；不要筛选 target-clean-correct 子集。
