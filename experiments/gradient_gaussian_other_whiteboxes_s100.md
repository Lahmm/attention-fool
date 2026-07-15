# 其他白盒源模型上的 raw Gaussian gradient 100 样本测试

固定当前主线：`original_score_postdrop_phase_pair`、final score、初始 RGB opponent-projected feature noise、10 steps、ε=16/255、MI、10 groups × 2 views = 20 actual views、seed=20260715。仅把白盒源模型换为 CaiT、PiT、Visformer，并在 20-view raw mean 后使用 `gaussian_blend_s40_a075`（σ=4、α=0.75），再进入 MI。每组 100 样本，完整评估 13 个本地缓存模型，均 `skipped=0`。

## 汇总

| Source | Baseline Overall | Gaussian Overall | Δ Overall | Baseline ViT | Gaussian ViT | Δ ViT | Baseline CNN | Gaussian CNN | Δ CNN | Gaussian WB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cait_s24_224 | 86.31% | 87.62% | +1.31pp | 90.14% | 91.57% | +1.43pp | 81.83% | 83.00% | +1.17pp | 96.00% |
| pit_b_224 | 81.62% | 81.92% | +0.31pp | 89.29% | 89.71% | +0.43pp | 72.67% | 72.83% | +0.17pp | 96.00% |
| visformer_small | 72.69% | 73.62% | +0.92pp | 79.43% | 80.29% | +0.86pp | 64.83% | 65.83% | +1.00pp | 100.00% |

## 逐模型 Gaussian ASR

| Source | levit_256 | pit_b_224 | deit_base_patch16_224 | tnt_s_patch16_224 | convit_base | visformer_small | cait_s24_224 | inception_v3 | inception_v4 | inception_resnet_v2 | resnet101 | inception_v3_adv | inception_resnet_v2_adv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cait_s24_224 | 89% | 92% | 91% | 91% | 92% | 90% | 96% | 85% | 83% | 83% | 88% | 80% | 79% |
| pit_b_224 | 83% | 96% | 89% | 92% | 88% | 92% | 88% | 77% | 79% | 74% | 80% | 68% | 59% |
| visformer_small | 80% | 84% | 75% | 77% | 67% | 100% | 79% | 74% | 73% | 65% | 80% | 61% | 42% |

## 梯度诊断

| Source | View→final cosine | Sign agreement | Effective rank | MI cosine |
|---|---:|---:|---:|---:|
| cait_s24_224 | 0.2730 | 0.4900 | 19.28 | 0.6181 |
| pit_b_224 | 0.3447 | 0.5099 | 18.40 | 0.6112 |
| visformer_small | 0.4240 | 0.5418 | 16.17 | 0.5955 |

## 结论

- CaiT：Overall 87.62%，相对 raw mean +1.31pp；ViT +1.43pp，CNN +1.17pp。
- PiT：Overall 81.92%，相对 raw mean +0.31pp；ViT +0.43pp，CNN +0.17pp。
- Visformer：Overall 73.62%，相对 raw mean +0.92pp；ViT +0.86pp，CNN +1.00pp。
- 三个源模型均出现正向变化，但都没有达到 Overall +3pp，因此 Gaussian blend 仍是有希望的候选，不足以成为已验证的统一主线升级。
- 三组均保持 20 views、原始 epsilon 投影路径和 100 样本预算；输出目录分别为 `outputs/attack/gradient_gaussian_s40a075_{source}_mainline_s100_seed20260715`。
