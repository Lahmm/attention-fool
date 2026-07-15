# PiT / Visformer 噪声类型与注入位置消融（100 样本）

本报告记录 2026-07-15 完成的噪声类型与注入位置消融。每组固定 100 个样本、`original_score_postdrop_phase_pair`、10 steps、ε=16/255、MI、10 groups × 2 views = 20 actual views、post-dropout strength=0.2，并使用 `seed=20260715` 回放 mask/phase/noise 随机序列。迁移评估覆盖本地缓存中的全部 13 个模型，`skipped=0`。

`none` 是同一随机回放下的无新增 post-dropout feature noise 控制；`opponent_projected` 是当前初始 RGB 投影噪声主线。`avg_vit`/`avg_cnn` 沿用评估脚本定义，白盒列单独取 source model 的 ASR。

## 结果总览

| Source | Noise | Position | Overall | ViT avg | CNN avg | Whitebox | Δ Overall vs none | Δ Overall vs opponent |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pit_b_224 | feature_iid | final | 72.23% | 84.43% | 58.00% | 99.00% | +0.00pp | -9.38pp |
| pit_b_224 | feature_iid | initial | 79.62% | 89.57% | 68.00% | 98.00% | +7.38pp | -2.00pp |
| pit_b_224 | feature_iid | pre_last_downsample | 72.31% | 85.86% | 56.50% | 99.00% | +0.08pp | -9.31pp |
| pit_b_224 | feature_lowpass | final | 72.15% | 84.43% | 57.83% | 99.00% | -0.08pp | -9.46pp |
| pit_b_224 | feature_lowpass | initial | 78.85% | 89.43% | 66.50% | 97.00% | +6.62pp | -2.77pp |
| pit_b_224 | feature_lowpass | pre_last_downsample | 72.92% | 85.57% | 58.17% | 99.00% | +0.69pp | -8.69pp |
| pit_b_224 | feature_rademacher | final | 72.15% | 84.43% | 57.83% | 99.00% | -0.08pp | -9.46pp |
| pit_b_224 | feature_rademacher | initial | 79.62% | 89.14% | 68.50% | 98.00% | +7.38pp | -2.00pp |
| pit_b_224 | feature_rademacher | pre_last_downsample | 71.54% | 83.00% | 58.17% | 99.00% | -0.69pp | -10.08pp |
| pit_b_224 | none | initial | 72.23% | 84.43% | 58.00% | 99.00% | +0.00pp | -9.38pp |
| pit_b_224 | opponent_projected | initial | 81.62% | 89.29% | 72.67% | 96.00% | +9.38pp | +0.00pp |
| pit_b_224 | pixel_iid | initial | 81.38% | 89.57% | 71.83% | 97.00% | +9.15pp | -0.23pp |
| pit_b_224 | pixel_opponent | initial | 81.08% | 89.43% | 71.33% | 96.00% | +8.85pp | -0.54pp |
| visformer_small | feature_iid | final | 63.77% | 69.14% | 57.50% | 100.00% | -0.92pp | -8.38pp |
| visformer_small | feature_iid | initial | 69.77% | 76.86% | 61.50% | 100.00% | +5.08pp | -2.38pp |
| visformer_small | feature_iid | pre_last_downsample | 64.38% | 70.57% | 57.17% | 100.00% | -0.31pp | -7.77pp |
| visformer_small | feature_lowpass | final | 64.31% | 70.29% | 57.33% | 100.00% | -0.38pp | -7.85pp |
| visformer_small | feature_lowpass | initial | 74.92% | 82.14% | 66.50% | 100.00% | +10.23pp | +2.77pp |
| visformer_small | feature_lowpass | pre_last_downsample | 64.62% | 72.00% | 56.00% | 100.00% | -0.08pp | -7.54pp |
| visformer_small | feature_rademacher | final | 64.31% | 71.14% | 56.33% | 100.00% | -0.38pp | -7.85pp |
| visformer_small | feature_rademacher | initial | 69.15% | 76.57% | 60.50% | 99.00% | +4.46pp | -3.00pp |
| visformer_small | feature_rademacher | pre_last_downsample | 64.08% | 70.14% | 57.00% | 100.00% | -0.62pp | -8.08pp |
| visformer_small | none | initial | 64.69% | 71.14% | 57.17% | 100.00% | +0.00pp | -7.46pp |
| visformer_small | opponent_projected | initial | 72.15% | 77.71% | 65.67% | 99.00% | +7.46pp | +0.00pp |
| visformer_small | pixel_iid | initial | 75.31% | 80.29% | 69.50% | 100.00% | +10.62pp | +3.15pp |
| visformer_small | pixel_opponent | initial | 74.62% | 79.29% | 69.17% | 100.00% | +9.92pp | +2.46pp |

## 结论

- PiT 类型筛选的最高 Overall 是初始位置 `opponent_projected`：81.62%；`pixel_iid` 为 81.38%，`pixel_opponent` 为 81.08%。相对于同种子 `none`，分别提升 +9.39、+9.15、+8.85pp。
- Visformer 类型筛选最高的是初始位置 `pixel_iid`：75.31%；其次为 `feature_lowpass` 74.92% 和 `pixel_opponent` 74.62%。相对于 `none` 分别提升 +10.62、+10.23、+9.93pp。
- 位置筛选中，新增特征噪声放在初始 RGB 投影处远好于后层：PiT 三类噪声的初始 Overall 为 78.85–79.62%，而 pre-last/final 仅 71.54–72.92%；Visformer 初始为 69.15–74.92%，后层为 64.08–64.62%。
- 后层注入没有带来稳定迁移收益。原因符合计算图：final 噪声位于大部分网络之后，且噪声本身与输入独立，不能像初始噪声一样改变后续特征提取路径；pre-last 只影响最后一段网络，作用有限。
- 当前最有效的已测配置仍是：PiT 白盒 + 初始 `opponent_projected`，Overall 81.62%、ViT 89.29%、CNN 72.67%、白盒 96%。Visformer 白盒的最佳类型筛选配置是初始 `pixel_iid`，Overall 75.31%、ViT 80.29%、CNN 69.50%、白盒 100%。
- PiT 没有新的类型超过当前 `opponent_projected` 主线；但 Visformer 的初始 `pixel_iid` 相对其 `opponent_projected` 主线提升 +3.15pp（75.31% vs 72.15%），满足 3–5pp 的 Overall 区间。这个收益是 source-specific 的，不能直接宣称为统一主线改进；初始 RGB 投影噪声有效，但后层位置不是推进方向。

## 逐模型 ASR（全部 13 个模型）

| Config | levit_256 | pit_b_224 | deit_base_patch16_224 | tnt_s_patch16_224 | convit_base | visformer_small | cait_s24_224 | inception_v3 | inception_v4 | inception_resnet_v2 | resnet101 | inception_v3_adv | inception_resnet_v2_adv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pit_b_224/feature_iid/final | 83% | 99% | 75% | 84% | 84% | 88% | 78% | 63% | 65% | 62% | 69% | 53% | 36% |
| pit_b_224/feature_iid/initial | 88% | 98% | 87% | 89% | 87% | 92% | 86% | 77% | 75% | 69% | 73% | 62% | 52% |
| pit_b_224/feature_iid/pre_last_downsample | 82% | 99% | 81% | 86% | 84% | 92% | 77% | 62% | 67% | 60% | 63% | 52% | 35% |
| pit_b_224/feature_lowpass/final | 83% | 99% | 75% | 84% | 84% | 88% | 78% | 63% | 66% | 60% | 69% | 53% | 36% |
| pit_b_224/feature_lowpass/initial | 88% | 97% | 85% | 90% | 88% | 93% | 85% | 74% | 69% | 71% | 74% | 63% | 48% |
| pit_b_224/feature_lowpass/pre_last_downsample | 84% | 99% | 82% | 84% | 84% | 89% | 77% | 67% | 65% | 58% | 66% | 55% | 38% |
| pit_b_224/feature_rademacher/final | 83% | 99% | 75% | 84% | 84% | 88% | 78% | 63% | 65% | 61% | 69% | 53% | 36% |
| pit_b_224/feature_rademacher/initial | 88% | 98% | 85% | 89% | 90% | 90% | 84% | 73% | 72% | 72% | 80% | 62% | 52% |
| pit_b_224/feature_rademacher/pre_last_downsample | 83% | 99% | 74% | 83% | 81% | 88% | 73% | 66% | 66% | 56% | 70% | 54% | 37% |
| pit_b_224/none/initial | 83% | 99% | 75% | 84% | 84% | 88% | 78% | 63% | 65% | 62% | 69% | 53% | 36% |
| pit_b_224/opponent_projected/initial | 84% | 96% | 88% | 92% | 87% | 90% | 88% | 79% | 76% | 75% | 79% | 67% | 60% |
| pit_b_224/pixel_iid/initial | 87% | 97% | 89% | 92% | 87% | 89% | 86% | 78% | 75% | 74% | 80% | 65% | 59% |
| pit_b_224/pixel_opponent/initial | 88% | 96% | 89% | 89% | 86% | 92% | 86% | 79% | 75% | 74% | 76% | 66% | 58% |
| visformer_small/feature_iid/final | 67% | 74% | 61% | 68% | 54% | 100% | 60% | 63% | 67% | 59% | 68% | 55% | 33% |
| visformer_small/feature_iid/initial | 75% | 78% | 69% | 75% | 67% | 100% | 74% | 72% | 73% | 60% | 73% | 55% | 36% |
| visformer_small/feature_iid/pre_last_downsample | 71% | 72% | 64% | 71% | 56% | 100% | 60% | 65% | 64% | 59% | 69% | 55% | 31% |
| visformer_small/feature_lowpass/final | 70% | 72% | 61% | 67% | 58% | 100% | 64% | 66% | 67% | 58% | 65% | 56% | 32% |
| visformer_small/feature_lowpass/initial | 81% | 84% | 76% | 81% | 73% | 100% | 80% | 76% | 76% | 66% | 77% | 63% | 41% |
| visformer_small/feature_lowpass/pre_last_downsample | 75% | 76% | 64% | 71% | 56% | 100% | 62% | 64% | 67% | 56% | 68% | 52% | 29% |
| visformer_small/feature_rademacher/final | 71% | 73% | 65% | 69% | 55% | 100% | 65% | 63% | 66% | 59% | 64% | 53% | 33% |
| visformer_small/feature_rademacher/initial | 76% | 82% | 68% | 76% | 64% | 99% | 71% | 70% | 71% | 58% | 73% | 56% | 35% |
| visformer_small/feature_rademacher/pre_last_downsample | 72% | 71% | 61% | 69% | 58% | 100% | 60% | 64% | 65% | 61% | 66% | 55% | 31% |
| visformer_small/none/initial | 70% | 77% | 64% | 66% | 59% | 100% | 62% | 65% | 64% | 59% | 64% | 56% | 35% |
| visformer_small/opponent_projected/initial | 80% | 80% | 70% | 75% | 62% | 99% | 78% | 77% | 75% | 65% | 73% | 62% | 42% |
| visformer_small/pixel_iid/initial | 81% | 81% | 73% | 80% | 68% | 100% | 79% | 81% | 80% | 72% | 79% | 60% | 45% |
| visformer_small/pixel_opponent/initial | 80% | 79% | 75% | 78% | 65% | 100% | 78% | 78% | 79% | 71% | 78% | 65% | 44% |

## 梯度诊断

以下来自各攻击目录的 `gradient_diagnostics.json`：view 与最终梯度 cosine、sign agreement、effective rank、MI 累积方向与当前梯度 cosine。

| Config | View cosine | Sign agreement | Effective rank | MI cosine |
|---|---:|---:|---:|---:|
| pit_b_224/feature_iid/final | 0.4583 | 0.5511 | 13.12 | 0.5587 |
| pit_b_224/feature_iid/initial | 0.3641 | 0.5183 | 17.74 | 0.5920 |
| pit_b_224/feature_iid/pre_last_downsample | 0.4561 | 0.5506 | 13.29 | 0.5596 |
| pit_b_224/feature_lowpass/final | 0.4584 | 0.5511 | 13.11 | 0.5587 |
| pit_b_224/feature_lowpass/initial | 0.3537 | 0.5152 | 17.94 | 0.5898 |
| pit_b_224/feature_lowpass/pre_last_downsample | 0.4465 | 0.5471 | 13.66 | 0.5612 |
| pit_b_224/feature_rademacher/final | 0.4587 | 0.5512 | 13.10 | 0.5586 |
| pit_b_224/feature_rademacher/initial | 0.3622 | 0.5176 | 17.77 | 0.5907 |
| pit_b_224/feature_rademacher/pre_last_downsample | 0.4571 | 0.5506 | 13.23 | 0.5585 |
| pit_b_224/none/initial | 0.4586 | 0.5511 | 13.11 | 0.5587 |
| pit_b_224/opponent_projected/initial | 0.3470 | 0.5106 | 18.40 | 0.6075 |
| pit_b_224/pixel_iid/initial | 0.3368 | 0.5080 | 18.61 | 0.6068 |
| pit_b_224/pixel_opponent/initial | 0.3420 | 0.5092 | 18.52 | 0.6060 |
| visformer_small/feature_iid/final | 0.5371 | 0.5761 | 10.73 | 0.5772 |
| visformer_small/feature_iid/initial | 0.4236 | 0.5443 | 16.14 | 0.5890 |
| visformer_small/feature_iid/pre_last_downsample | 0.5306 | 0.5738 | 11.05 | 0.5793 |
| visformer_small/feature_lowpass/final | 0.5360 | 0.5759 | 10.79 | 0.5788 |
| visformer_small/feature_lowpass/initial | 0.3707 | 0.5229 | 18.02 | 0.5641 |
| visformer_small/feature_lowpass/pre_last_downsample | 0.5191 | 0.5695 | 11.62 | 0.5766 |
| visformer_small/feature_rademacher/final | 0.5357 | 0.5761 | 10.76 | 0.5789 |
| visformer_small/feature_rademacher/initial | 0.4267 | 0.5449 | 16.07 | 0.5881 |
| visformer_small/feature_rademacher/pre_last_downsample | 0.5322 | 0.5738 | 11.03 | 0.5774 |
| visformer_small/none/initial | 0.5375 | 0.5761 | 10.67 | 0.5792 |
| visformer_small/opponent_projected/initial | 0.4233 | 0.5424 | 16.21 | 0.5883 |
| visformer_small/pixel_iid/initial | 0.4132 | 0.5343 | 16.83 | 0.5812 |
| visformer_small/pixel_opponent/initial | 0.4166 | 0.5351 | 16.68 | 0.5794 |

## 可复现性与限制

- 所有攻击目录均包含 `attack_params.json`，记录噪声类型、位置、strength、20-view 配置及 seed；迁移 CSV 的 `attack_params` 也包含这些字段。
- 所有模型均从 `data/huggingface/hub` 本地缓存加载，完整评估日志显示 13 个模型 `skipped=0`。
- initial 位置使用首个 RGB projection 的真实卷积 receptive-field mask；后层位置使用 image mask 到对应 token grid 的空间映射。后续若要把位置差异做成最终方法，仍应进一步做架构特定的严格 receptive-field 映射验证。
