# 梯度后处理 100 样本消融

## 设置

- 白盒：`vit_base_patch16_224`。
- 100 samples，10 steps，epsilon `16/255`，MI，10 groups × 2 views = 20 actual views。
- 固定 seed：`20260710`；L12 score、原图 score mask、post-drop phase pair、kept-only opponent-channel noise 均保持主线设置。
- Overall/ViT/CNN 只统计 11 个可由 timm 加载的标准黑盒；所有 11 个模型均成功加载并完成评估。本地缺失权重由 timm 自动下载。白盒 ViT 单独列出，不计入迁移平均。
- 历史主线基线：Overall `79.73%`，ViT `82.57%`，CNN `74.75%`；白盒基线另行在同一 100 样本目录评估为 `88.00%`。

## 汇总

| 配置 | lambda | Overall ASR | Δ Overall | ViT 平均 | Δ ViT | CNN 平均 | Δ CNN | 白盒 ASR | Δ 白盒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean（固定 seed 对照） | 0.2 | 80.00% | +0.27pp | 83.71% | +1.14pp | 73.50% | -1.25pp | 91.00% | +3.00pp |
| view_l2_mean | 0.2 | 79.27% | -0.45pp | 83.00% | +0.43pp | 72.75% | -2.00pp | 90.00% | +2.00pp |
| sign_consensus | 0.2 | 73.55% | -6.18pp | 76.86% | -5.71pp | 67.75% | -7.00pp | 84.00% | -4.00pp |
| sign_consensus_transport | 0.1 | 73.73% | -6.00pp | 77.00% | -5.57pp | 68.00% | -6.75pp | 83.00% | -5.00pp |
| sign_consensus_transport | 0.2 | 74.00% | -5.73pp | 77.57% | -5.00pp | 67.75% | -7.00pp | 83.00% | -5.00pp |
| sign_consensus_transport | 0.3 | 73.55% | -6.18pp | 76.71% | -5.71pp | 68.00% | -6.75pp | 83.00% | -5.00pp |

## 逐模型 ASR

| 配置 | LeViT | PiT-B | DeiT-B | TNT-S | ConViT | Visformer | CaiT | Inc-v3 | Inc-v4 | IncRes-v2 | ResNet-101 | 白盒 ViT-B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 83% | 82% | 83% | 84% | 83% | 81% | 90% | 74% | 76% | 69% | 75% | 91% |
| view_l2_mean | 82% | 81% | 82% | 83% | 85% | 78% | 90% | 75% | 71% | 71% | 74% | 90% |
| sign_consensus | 77% | 71% | 77% | 78% | 75% | 75% | 85% | 72% | 68% | 61% | 70% | 84% |
| transport λ=0.1 | 75% | 73% | 78% | 78% | 75% | 74% | 86% | 72% | 67% | 61% | 72% | 83% |
| transport λ=0.2 | 75% | 75% | 78% | 78% | 77% | 74% | 86% | 70% | 67% | 64% | 70% | 83% |
| transport λ=0.3 | 75% | 71% | 78% | 76% | 77% | 74% | 86% | 71% | 67% | 64% | 70% | 83% |

## 梯度诊断

数值是 70 个 gradient batches（100 samples、batch size 16、10 steps）上的 batch 均值。`view_cosine_to_final` 是每个 view 与最终聚合梯度的 cosine；`sign_agreement` 是相对最终梯度的逐坐标符号一致率；effective rank 使用 20 个 view 梯度的 Gram 矩阵谱熵；MI cosine 是累积 MI 方向与当前归一化梯度的 cosine。

| 配置 | 每 view→最终 cosine | sign agreement | effective rank | MI 累积→当前 cosine |
|---|---:|---:|---:|---:|
| mean | 0.2433 | 0.4796 | 19.6576 | 0.5186 |
| view_l2_mean | 0.2751 | 0.4848 | 19.6621 | 0.5265 |
| sign_consensus | 0.1526 | 0.5249 | 19.6663 | 0.5378 |
| transport λ=0.1 | 0.1531 | 0.5050 | 19.6679 | 0.5377 |
| transport λ=0.2 | 0.1528 | 0.5051 | 19.6662 | 0.5380 |
| transport λ=0.3 | 0.1529 | 0.5051 | 19.6639 | 0.5379 |

## 选择结论

没有候选相对历史主线达到 Overall `+3pp`，也没有达到更严格的 `+5pp` 标准；同时 sign/transport 候选在 ViT 和 CNN 上均明显下降。因此保留 `mean` 主线，结论为：当前梯度后处理未验证出稳定收益。固定 seed 的 `mean` 对照仅比历史主线高 `0.27pp`，不构成梯度增强。

各配置的对抗样本目录、`attack_params.json` 和 `gradient_diagnostics.json` 位于：

- `outputs/attack/gradient_postprocess_mean_s100`
- `outputs/attack/gradient_postprocess_l2mean_s100`
- `outputs/attack/gradient_postprocess_sign_s100`
- `outputs/attack/gradient_postprocess_transport_l01_s100`
- `outputs/attack/gradient_postprocess_transport_l02_s100`
- `outputs/attack/gradient_postprocess_transport_l03_s100`

对应迁移 CSV 位于 `outputs/csv/outputs_attack_gradient_postprocess_*_s100.csv`，其中 `attack_params` 包含 `gradient_postprocess` 和 `gradient_consensus_lambda`。

## 后续 ViT 定向梯度分析

后续分析只用 7 个黑盒 ViT 作为梯度处理的主要筛选指标；CNN 仅作为最终不大幅损伤的约束。所有迁移评估仍包含完整的 11 个可加载黑盒模型（7 个 ViT、4 个 CNN），并单独评估白盒 ViT-B/16。攻击约束固定为 raw mean 主线：10 steps、20 views、`epsilon=16/255`，不重新启用 `_normalize_grad`。

### 跨尺度协方差输运 + 弱高斯

对每个样本，把 20 个 view 梯度分成 Fourier 低频 `l_v` 与高频 `h_v`。用跨 view 协方差得到由低频变化支持的高频方向 `C_hl l`，再与原始平均梯度的高频分量插值，最后加入弱 Gaussian blur：

`g' = low_mean + (1-x) high_mean + x C_hl l + 0.75 GaussianBlur(g, sigma=4)`。

其中当前最佳候选为 cutoff `0.50`、输运比例 `x=0.50`。它没有改变 score、mask、noise、phase pair、view 数、MI 或投影。

三组 100 样本 seed 的结果如下，增量均相对于同 seed、同一随机轨迹的 raw mean：

| seed | Overall | Δ Overall | 黑盒 ViT 平均 | Δ ViT | CNN 平均 | Δ CNN | 白盒 ViT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260713 | 82.45% | +1.64pp | 85.71% | +2.29pp | 76.75% | +0.50pp | 93% |
| 20260714 | 82.36% | +1.09pp | 85.14% | +0.29pp | 77.50% | +2.50pp | 93% |
| 20260715 | 83.00% | +1.82pp | 86.14% | +1.86pp | 77.50% | +1.75pp | 93% |
| 三 seed 平均增量 | — | **+1.52pp** | — | **+1.48pp** | — | **+1.58pp** | **+0.67pp** |

三 seed 的黑盒 ViT 逐模型平均（候选相对 raw mean）为：

| 模型 | raw mean | 跨尺度+高斯 | Δ |
|---|---:|---:|---:|
| LeViT-256 | 81.67% | 84.33% | +2.67pp |
| PiT-B | 83.67% | 84.00% | +0.33pp |
| DeiT-B | 84.00% | 85.33% | +1.33pp |
| TNT-S | 85.33% | 85.67% | +0.33pp |
| ConViT | 84.33% | 85.00% | +0.67pp |

## 最后一次下采样前的 score：PiT / Visformer

为检验 `8×8/7×7` 最终网格是否过粗，新增了一个只改变 score 提取位置的消融：

- PiT：在最后一个 `transformers[2]` 的 pooling 前，用 `transformers[1]` 输出的
  `16×16` 局部特征和 CLS token 计算 score，source 为
  `transformers[1]+pre_pool`；
- Visformer：在最后的 `patch_embed3` 前，用 stage2 输出的 `14×14` 局部特征和
  GAP pseudo-CLS 计算 score，source 为 `stage2+pre_patch_embed3+gap`。

最终分类 logits、pixel dropout、phase pair、kept-only opponent-channel noise、20
views、MI、epsilon 和步长均未改变。由于 score 网格改变，drop 数量按相同约 15% 的
空间比例重新取整：PiT 为 `38/256`，Visformer 为 `29/196`。

两组均为 100 样本、seed `20260715`，并使用完整 13 个可加载黑盒模型（7 个 ViT、4
个标准 CNN、2 个 adversarial CNN）；表中 ViT/CNN 均包含白盒源模型，便于与当前
transfer_eval CSV 保持一致。

| 白盒源模型 | score 网格 | Overall | Δ Overall | ViT 平均 | Δ ViT | CNN 平均 | Δ CNN | 白盒 ASR | Δ 白盒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PiT final（原主线） | 8×8 | 81.85% | — | 90.14% | — | 72.17% | — | 97% | — |
| PiT pre-last-downsample | 16×16 | 81.54% | -0.31pp | 88.71% | -1.43pp | 73.17% | +1.00pp | 96% | -1pp |
| Visformer final（原主线） | 7×7 | 72.69% | — | 79.43% | — | 64.83% | — | 99% | — |
| Visformer pre-last-downsample | 14×14 | 74.08% | +1.38pp | 78.29% | -1.14pp | 69.17% | +4.33pp | 100% | +1pp |

逐模型结果如下：

| 模型 | PiT final | PiT pre | Δ | Visformer final | Visformer pre | Δ |
|---|---:|---:|---:|---:|---:|---:|
| LeViT-256 | 86% | 86% | 0pp | 81% | 80% | -1pp |
| PiT-B | 97% | 96% | -1pp | 82% | 74% | -8pp |
| DeiT-B | 90% | 88% | -2pp | 73% | 70% | -3pp |
| TNT-S | 91% | 88% | -3pp | 80% | 81% | +1pp |
| ConViT | 87% | 88% | +1pp | 67% | 66% | -1pp |
| Visformer-S | 91% | 89% | -2pp | 99% | 100% | +1pp |
| CaiT-S/24 | 89% | 86% | -3pp | 74% | 77% | +3pp |
| Inception-v3 | 79% | 78% | -1pp | 73% | 80% | +7pp |
| Inception-v4 | 73% | 80% | +7pp | 76% | 81% | +5pp |
| Inception-ResNet-v2 | 73% | 76% | +3pp | 65% | 69% | +4pp |
| ResNet-101 | 78% | 80% | +2pp | 73% | 79% | +6pp |
| Adv Inception-v3 | 69% | 69% | 0pp | 60% | 63% | +3pp |
| Adv Inception-ResNet-v2 | 61% | 56% | -5pp | 42% | 43% | +1pp |

梯度诊断（70 个 gradient batches）为：

| 白盒源模型 | score 层 | view→final cosine | sign agreement | effective rank | MI→current cosine |
|---|---|---:|---:|---:|---:|
| PiT | final | 0.3464 | 0.5106 | 18.4123 | 0.6049 |
| PiT | pre-last-downsample | 0.3065 | 0.5030 | 19.2054 | 0.5955 |
| Visformer | final | 0.4278 | 0.5433 | 16.1419 | 0.5893 |
| Visformer | pre-last-downsample | 0.3813 | 0.5200 | 17.6171 | 0.5657 |

### 结论

这个实验没有支持“最后一次下采样前 score 能稳定提升 ViT 迁移”的假设：PiT 的
ViT 平均下降 `1.43pp`，Visformer 下降 `1.14pp`。Visformer 的 Overall 增长
`1.38pp` 几乎全部来自 CNN，CNN 平均增长 `4.33pp`；这说明更高空间分辨率的 score
更偏向局部/卷积共享方向，而不是 ViT 跨模型共享方向。PiT 也呈现同样趋势，只是
Overall 的 CNN 增益不足以抵消 ViT 损失。

因此，最后一次下采样前的 score 可以作为解释性和 mask 粒度消融，但当前不应替换
最终层 score 主线，也没有达到 Overall `+3–5pp` 或 ViT 定向提升目标。

## PiT / Visformer kept-only feature noise 消融

为检验迁移性差是否由 `kept-only opponent-channel Gaussian feature noise` 引起，
固定 final score、原始 mask、phase pair、20 views、MI、seed `20260715`、10 steps、
epsilon `16/255` 和完整的 13 模型评估，只改变 post-dropout feature noise strength。
score 阶段的 CLS noise、pixel mask、phase 和其他 `guide_aug_strength=0.2` 分支均保持不变。

新增参数 `post_dropout_feature_noise_strength`，`0.20` 是当前主线，`0` 表示只关闭
kept-only feature noise，不关闭其他随机增强。四档均生成独立 100 样本对抗样本并完成
13 模型评估。

### 汇总结果

| 白盒源 | strength | Overall | ViT 平均 | CNN 平均 | 白盒 ASR |
|---|---:|---:|---:|---:|---:|
| PiT-B | 0.00 | 72.00% | 84.29% | 57.67% | 99% |
| PiT-B | 0.05 | 74.85% | 85.43% | 62.50% | 98% |
| PiT-B | 0.10 | 78.54% | 87.71% | 67.83% | 96% |
| PiT-B | 0.20 | 81.85% | 90.14% | 72.17% | 97% |
| Visformer-S | 0.00 | 63.62% | 69.29% | 57.00% | 100% |
| Visformer-S | 0.05 | 64.23% | 70.43% | 57.00% | 100% |
| Visformer-S | 0.10 | 67.54% | 73.57% | 60.50% | 100% |
| Visformer-S | 0.20 | 72.85% | 79.57% | 65.00% | 99% |

相对于同一白盒源的无 feature noise 配置，strength `0.20` 带来：

| 白盒源 | Δ Overall | Δ ViT | Δ CNN | Δ 白盒 |
|---|---:|---:|---:|---:|
| PiT-B | +9.85pp | +5.86pp | +14.50pp | -2pp |
| Visformer-S | +9.23pp | +10.29pp | +8.00pp | -1pp |

### 逐模型变化：strength 0.00 → 0.20

| 模型 | PiT Δ | Visformer Δ |
|---|---:|---:|
| LeViT-256 | +3pp | +13pp |
| PiT-B | -2pp | +13pp |
| DeiT-B | +13pp | +12pp |
| TNT-S | +7pp | +12pp |
| ConViT | +6pp | +11pp |
| Visformer-S | 0pp | -1pp |
| CaiT-S/24 | +14pp | +12pp |
| Inception-v3 | +14pp | +8pp |
| Inception-v4 | +3pp | +13pp |
| Inception-ResNet-v2 | +15pp | +2pp |
| ResNet-101 | +10pp | +7pp |
| Adv Inception-v3 | +19pp | +6pp |
| Adv Inception-ResNet-v2 | +26pp | +12pp |

### 梯度诊断

| 白盒 | strength | view→final cosine | sign agreement | effective rank | MI→current cosine |
|---|---:|---:|---:|---:|---:|
| PiT | 0.00 | 0.4584 | 0.5512 | 13.0616 | 0.5553 |
| PiT | 0.05 | 0.4140 | 0.5383 | 15.5035 | 0.5763 |
| PiT | 0.10 | 0.3804 | 0.5256 | 17.1441 | 0.5885 |
| PiT | 0.20 | 0.3464 | 0.5106 | 18.4117 | 0.6049 |
| Visformer | 0.00 | 0.5369 | 0.5764 | 10.6612 | 0.5786 |
| Visformer | 0.05 | 0.5136 | 0.5707 | 12.0283 | 0.5820 |
| Visformer | 0.10 | 0.4795 | 0.5603 | 13.8538 | 0.5853 |
| Visformer | 0.20 | 0.4278 | 0.5433 | 16.1432 | 0.5893 |

### 结论

当前结果否定了“PiT/Visformer 的迁移性差主要是因为 feature noise 有害”这一假设。
在两个白盒源上，feature noise strength 从 0 增加到 0.20 都带来近似单调的 Overall、
ViT 和 CNN 提升；关闭噪声会显著损害黑盒迁移，尤其是 Visformer 的 ViT 迁移下降
超过 `10pp`。代价是白盒 ASR 小幅下降 `1–2pp`，即该噪声牺牲少量源模型拟合，换取
大量跨模型迁移收益。

诊断上，strength 增大使 view-to-final cosine 和 sign agreement 下降、effective rank
上升，同时 MI 累积方向与当前梯度的 cosine 上升。这说明该噪声不是简单引入有害的
源模型偏置，而是在当前主线中提供了更多 view 梯度多样性，并改善了 MI 方向的累积。
因此后续不应沿“去除或减弱噪声”寻找 ViT 迁移提升；若继续优化，应研究如何保留这种
多样性，同时减少其对白盒 ASR 的损失。

实验目录为 `outputs/attack/{pit_b_224,visformer_small}_noise_{s000,s005,s010,s020}_s100_seed20260715`，
对应 CSV 为 `outputs/csv/outputs_attack_{pit_b_224,visformer_small}_noise_*_s100_seed20260715.csv`。
| Visformer-S | 80.33% | 84.33% | +4.00pp |
| CaiT-S24 | 90.00% | 91.00% | +1.00pp |

在同一 30 样本 screen 中，直接的 view-PC、view/covariance transport、CCA 及 Fourier phase consensus 均未形成稳定正向信号。围绕候选做的局部检查也显示：cutoff `0.35` 在 30 样本为 ViT `+1.90pp`，但 100 样本只有 `+1.00pp`；增大输运比例或改变 Gaussian 强度没有改善。因此目前证据支持“低频结构支持的部分高频方向 + 温和空间平滑”是有效的 ViT 共享成分，但收益约为 `+1.5pp`，尚未达到 Overall `+3–5pp` 或黑盒 ViT `90%` 的目标。

相关目录：

- `outputs/attack/raw_cross_scale_confirm_s100_seed20260713`
- `outputs/attack/raw_cross_scale_confirm_s100_seed20260714`
- `outputs/attack/raw_cross_scale_confirm_s100_seed20260715`
- `outputs/attack/raw_cross_scale_c35_confirm_s100_seed20260713`
- `outputs/attack/raw_view_subspace_screen_s30_seed20260713`

当前结论：不把该候选切为攻击主线；下一步若继续，应分析为什么只有部分 ViT（尤其 LeViT、Visformer）受益，并做按 ViT 共享性的梯度诊断，而不是继续无约束增加后处理算子。目标阈值仍未满足。

## 进一步 raw 梯度验证

### 高频与 raw scale

在当前 raw 主线重新验证旧的 `freq_high` 线索后，直接去掉高频并没有改善 ViT：30 样本中 `frequency_remove_high` 为 ViT `-1.90pp`。随机频段删除为 `-6.19pp`，说明破坏频率结构会明显伤害迁移。对高频衰减后的结果回标原始 L1/L2 范数也没有恢复收益，L1 最好只达到 ViT `+0.48pp`。

进一步只衰减高频能量高分位样本的自适应版本同样没有正向信号，最佳为 ViT `+0.95pp`。因此旧实验中“高频有害”的相关性不能直接转化为全局高频删除；在 raw 主线下，梯度的整体幅值和频率组成存在耦合。

### raw temporal + 跨尺度协方差 + Gaussian

将每个样本的当前 raw 梯度相对 EMA 的 scale 权重与跨尺度协方差输运、`sigma=4` 弱高斯组合，是 30 样本中最强的 ViT 定向候选。`power=0.45` 的 100 样本三 seed 结果为：

| seed | Overall | Δ Overall | 黑盒 ViT | Δ ViT | CNN | Δ CNN | 白盒 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260713 | 82.18% | +1.36pp | 85.86% | +2.43pp | 75.75% | -0.50pp | 93% |
| 20260714 | 81.64% | +0.36pp | 85.14% | +0.29pp | 75.50% | +0.50pp | 91% |
| 20260715 | 82.82% | +1.64pp | 85.71% | +1.43pp | 77.75% | +2.00pp | 93% |
| 三 seed 平均增量 | — | **+1.12pp** | — | **+1.38pp** | — | **+0.67pp** | **0pp** |

因此它没有超过不加 temporal 的跨尺度+Gaussian 三 seed 平均 ViT `+1.48pp`，也没有达到 90%。全局把当前 raw 梯度乘以 `0.25/0.5/2/4` 在 sign-MI 更新下 ASR 完全不变，说明常数 scale 不改变最终符号轨迹。

### Patch 能量输运回标

对 ViT 14×14 patch 能量输运做 L1/L2 raw scale 回标后，30 样本最佳仍为 ViT `-0.95pp`；较强 L1 版本 Overall `-8.48pp` 且 CNN `-13.33pp`。这排除了“patch 输运仅因 raw scale 失控而失败”的解释。

本轮新增探针均只作用于梯度后处理，未改变 data augmentation、20 views、steps、epsilon、MI 或投影逻辑。新增实验目录包括：

- `outputs/attack/raw_frequency_screen_s30_seed20260713`
- `outputs/attack/raw_frequency_rescaled_screen_s30_seed20260713`
- `outputs/attack/raw_cross_scale_residual_screen_s30_seed20260713`
- `outputs/attack/raw_temporal_cross_scale_p045_confirm_s100_seed20260713`
- `outputs/attack/raw_temporal_cross_scale_p045_confirm_s100_seed20260714`
- `outputs/attack/raw_temporal_cross_scale_p045_confirm_s100_seed20260715`
- `outputs/attack/raw_patch_energy_rescaled_screen_s30_seed20260713`
- `outputs/attack/raw_global_scale_screen_s30_seed20260713`

当前证据仍支持跨尺度协方差支持的高频方向加弱 Gaussian 是最有效的数学后处理，但稳定收益约为 `+1.5pp`，距离 Overall `+3–5pp` 和黑盒 ViT `90%` 仍有差距；不应把任一候选宣称为已达标主线。

### 低频 view-sign 共识 + 弱 Gaussian 筛选

针对“ViT 共享方向可能集中在低频结构”的假设，新增了只对 Fourier 低频分量做逐 view 符号共识的探针；高频仍保留 raw mean，最后使用 `sigma=4, a=0.75` 的弱 Gaussian。该探针未改变 20 views、score、mask、noise、phase、MI 或投影。

首轮目录因并发评估竞态导致基线只包含 14 个已评估样本，已废弃；以下是单进程、30/30 完整重跑的结果：

| 探针 | Overall | Δ Overall | 黑盒 ViT | Δ ViT | CNN | Δ CNN | 白盒 ViT |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw mean baseline | 84.55% | — | 85.71% | — | 82.50% | — | 93.33% |
| low consensus c50, x0.25 | 85.45% | +0.91pp | 87.14% | +1.43pp | 82.50% | 0pp | 93.33% |
| low consensus c35, x0.25 | 84.85% | +0.30pp | 86.19% | +0.48pp | 82.50% | 0pp | 93.33% |
| low consensus c50, x0.50 | 81.82% | -2.73pp | 83.33% | -2.38pp | 79.17% | -3.33pp | 90.00% |

因此低频共识在小样本 screen 中有弱正向信号，但远低于 Overall `+3–5pp` 目标；共识权重过大还会同时损害 ViT 与 CNN。该方向暂不进入 100 样本确认，后续应优先做 ViT 逐模型梯度共享性诊断，而不是继续增大 consensus 权重。有效重跑目录为 `outputs/attack/raw_low_consensus_screen_s30_seed20260713_rerun`。

### ViT 共享性置信度门控跨尺度 + Gaussian

逐样本 7 个黑盒 ViT 分析中，raw 幅值、空间熵、group agreement 与低高频占比共同表现出正向共享性相关性。为避免把这些观察性变量直接当成坐标因果开关，新增门控探针：仅对 batch 内共享性置信度最高的样本应用已知的 `cross-scale c=0.50, x=0.50 + sigma=4, a=0.75`，其余样本保持 raw mean。

30 样本完整 11 黑盒筛选结果如下：

| 选择比例 | Overall | Δ Overall | 黑盒 ViT | Δ ViT | CNN | Δ CNN | 白盒 ViT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| raw mean | 84.55% | — | 85.71% | — | 82.50% | — | 93.33% |
| top 25% | 84.55% | 0pp | 85.71% | 0pp | 82.50% | 0pp | 93.33% |
| top 50% | 83.94% | -0.61pp | 85.71% | 0pp | 80.83% | -1.67pp | 93.33% |
| top 75% | 84.85% | +0.30pp | 86.67% | +0.95pp | 81.67% | -0.83pp | 93.33% |

门控验证了相关性与因果增益之间的差距：top 75% 只产生弱正向，且低于 Overall `+3–5pp` 目标；top 50% 已损伤 CNN。该方向不进行 100 样本确认，也不改变当前主线。目录为 `outputs/attack/confidence_cross_scale_screen_s30_seed20260713`。

### 早期 raw Gaussian 的轨迹窗口

30 样本窗口筛选显示，`sigma=4, a=0.75` 的 raw Gaussian 只作用于前 5 步时，黑盒 ViT `+1.90pp`、Overall `+1.21pp`，而只作用于第 3–8 步时 ViT `-0.95pp`；这提示后期梯度可能更容易包含源模型特异方向。随后进行 100 样本、3 seed、完整 11 黑盒确认：

| seed | Overall | Δ Overall | 黑盒 ViT | Δ ViT | CNN | Δ CNN | 白盒 | Δ 白盒 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260713 | 82.55% | +1.73pp | 85.57% | +2.14pp | 77.25% | +1.00pp | 93% | +1pp |
| 20260714 | 81.73% | +0.45pp | 85.14% | +0.29pp | 75.75% | +0.75pp | 92% | 0pp |
| 20260715 | 82.18% | +1.09pp | 84.86% | +0.71pp | 77.50% | +1.75pp | 93% | 0pp |
| 三 seed 平均增量 | — | **+1.09pp** | — | **+1.05pp** | — | **+1.17pp** | — | **+0.33pp** |

该方向具有一定轨迹解释力，但 seed 间波动较大，未达到 Overall `+3–5pp` 或黑盒 ViT `90%`；不进入主线。结果目录为：

- `outputs/attack/raw_gaussian_early_s100_seed20260713`
- `outputs/attack/raw_gaussian_early_s100_seed20260714`
- `outputs/attack/raw_gaussian_early_s100_seed20260715`

将同样的 step-window 施加到当前稳定候选 `cross-scale + Gaussian` 后，30 样本结果为：前 5 步 Overall/ViT/CNN `-0.30/+0.95/-2.50pp`，第 3–8 步 `+0.61/+1.43/-0.83pp`，后 5 步 `+0.30/0/+0.83pp`。它们都没有超过全程跨尺度候选，也没有达到 Overall `+3–5pp`；因此 Gaussian 的早期轨迹效应不能与跨尺度输运简单叠加。

相关目录：`outputs/attack/cross_scale_gaussian_window_screen_s30_seed20260713`。

对前 5 步 Gaussian 的强度/尺度进行有限探索后，`a=1.00` 和 `a=1.50` 均只有 Overall `+0.30pp`（ViT 分别 `+0.48pp/0pp`），而 sigma 从 4 增至 6 使 Overall/ViT/CNN 变为 `-1.21/-0.48/-2.50pp`。因此早期收益不是简单的“更强平滑”或“更大空间尺度”，停止继续扩展 Gaussian 参数。目录为 `outputs/attack/raw_gaussian_early_strength_screen_s30_seed20260713`。

### 正交 Gaussian 残差

为区分 Gaussian 的方向作用与其沿原梯度投影的幅值混合，测试了
`g' = g + a(B(g)-proj_g B(g))`。全程 sigma=4、a=0.75 时 Overall/ViT/CNN 为 `0/+0.48/-0.83pp`；a=1.50 为 `0/+1.43/-2.50pp`；前 5 步 a=0.75 为 `-0.30/+0.48/-1.67pp`。正交残差没有超过原始 Gaussian 或跨尺度候选，说明 Gaussian 的正向效应不能简化为一个独立的正交共享方向。目录为 `outputs/attack/orthogonal_gaussian_screen_s30_seed20260713`。

### 自适应跨步轨迹创新

新增 raw-scale trajectory probe：维护每个样本的历史 raw 梯度 EMA，计算当前梯度相对历史方向的正交 innovation，只在当前/历史 cosine 较低时按比例抑制该 innovation。30 样本结果为：

| strength | Overall Δ | ViT Δ | CNN Δ | 白盒 Δ |
|---:|---:|---:|---:|---:|
| 0.25 | -0.61pp | -0.48pp | -0.83pp | 0pp |
| 0.50 | -1.21pp | -1.43pp | -0.83pp | 0pp |
| 0.75 | -0.91pp | -1.43pp | 0pp | -3.33pp |

因此跨步不一致的方向不能简单视为有害源模型成分；攻击迁移需要保留部分轨迹创新。该方向不进入大样本确认。目录为 `outputs/attack/adaptive_trajectory_screen_s30_seed20260713`。

### Geometric median view aggregation

新增 Weiszfeld 几何中位数聚合，以检验少数源模型特异 view 是否拉偏 raw mean。结果显示 view 离群点不是主要问题：raw strength `0.50/1.00` 的 ViT 变化为 `-0.48/-2.38pp`；L1 scale-preserving 版本为 `0/-1.43pp`，对应 CNN 变化为 `-4.17/-2.50pp`。因此迁移需要保留 view 多样性，鲁棒地压制离群 view 反而损害共享方向。目录为 `outputs/attack/geometric_median_screen_s30_seed20260713`。

### ViT patch embedding metric 预条件

进一步使用白盒 ViT 的 patch embedding 权重构造 `W^T W`，对每个 16×16 patch 的像素梯度做特征度量预条件，再回到像素空间。该操作仍只改变聚合梯度，不改变 view、score、mask、noise、MI 或投影。

30 样本结果显示该白盒几何不能直接代表跨 ViT 共享方向：raw strength `0.25/0.50/0.75` 的 ViT 变化为 `-0.95/-2.38/-3.81pp`；L1 scale-preserving 版本为 `-0.95/-0.48/-2.86pp`。最佳 scaled `0.25` 的 Overall/CNN/白盒变化为 `-0.30/+0.83/-3.33pp`。因此不进行 100 样本确认。目录为 `outputs/attack/patch_embedding_metric_screen_s30_seed20260713`。

### 跨 ViT 黑盒梯度的颜色边际

对 7 个黑盒 ViT 的输入梯度做独立分析，发现绿色通道的平均梯度能量约为 `1.2–1.3×`，蓝色约为 `0.7–0.85×`，红色接近平均。将这一统计转成固定的逐通道 gain，并作用于白盒 raw 梯度后，30 样本中所有正向/反向强度的 ViT 变化均为 `0pp`（最差也只有 CNN `-0.83pp`）。因此黑盒 ViT 的颜色边际统计不足以提供可用的共享 sign 方向；需要更高阶的跨模型空间方向信息。目录为 `outputs/attack/cross_vit_channel_screen_s30_seed20260713`。

### 黑盒 ViT CE gradient 与 L12 攻击 sign 的对齐诊断

为验证是否可以直接利用黑盒 ViT 的输入 CE gradient，比较了 30 个样本上 7 个黑盒 ViT 的 gradient sign 与白盒 L12 攻击第 10 步 sign：在 clean 输入点，单模型 agreement 约 `0.50`，7 模型 consensus 与白盒约 `0.507`；在同一批最终对抗样本点，单模型约 `0.50`，consensus 约 `0.504`。黑盒模型彼此的 CE sign consensus agreement 约 `0.66`，但它与当前 L12 攻击方向几乎独立。

这说明黑盒 ViT 的 CE gradient consensus 不能作为当前 L12 score 梯度的后处理共享方向；直接注入它会改变攻击目标来源，因此不纳入本主线。

### 黑盒 DeiT 的同类 L12 score 梯度对齐诊断

为避免 CE 梯度与 L12 score 梯度目标不同造成误判，进一步在同一攻击接口上比较了
源 `vit_base_patch16_224` 与黑盒 `deit_base_patch16_224` 的 L12 score view 梯度。
两者均为 12-block、CLS+patch token 的标准 ViT 接口，因此没有使用架构近似；两边
均固定 20 views、相同 sample IDs、相同 `GradientReplay(seed=20260713)` 事件序列，
只在 clean pixel 上运行一步以捕获 20 个独立 view 梯度。该实验是方向诊断，不是 ASR
候选攻击。

30 样本结果如下：

| 对齐指标 | 源 ViT vs DeiT |
|---|---:|
| 对应 view raw gradient cosine | `0.0193` |
| 20-view raw mean gradient cosine | `0.0622` |
| 对应 view sign agreement | `0.4334` |
| 对应 view低频分量 cosine | `0.0269` |

对应 view 的 sign agreement 甚至低于随机符号基线附近的 `0.5`，而不是一个可以稳定
注入的共享方向。由于这次比较使用的是与攻击完全相同的 L12 score、score mask、
phase-pair、kept-only noise 和 view 数，结果说明问题不只是 CE 目标不匹配：即使改成
同类 L12 score，源 ViT 与 DeiT 的像素梯度仍然高度模型特异。因而不能把黑盒 L12
梯度直接做 ensemble 后作为当前白盒梯度后处理；这条路径不再继续扩展到结构不兼容
的模型，避免把架构接口差异或模型特异方向误当成可迁移信息。

该诊断进一步支持当前的下一步边界：若要继续提升黑盒 ViT，后处理必须提取不依赖
单个模型坐标的跨架构空间统计/算子（例如对源梯度施加可证伪的多尺度不变变换），
而不是直接寻找另一个模型的梯度方向。当前已验证的最佳候选仍为三 seed 平均
`+1.52pp` Overall 的 cross-scale Gaussian，尚未达到 Overall `+3–5pp` 或黑盒 ViT
约 `90%` 的目标。

### patch-scale 与 cross-scale 组合补充

由于单纯像素 Gaussian 的最佳效果来自多尺度平滑，而 ViT 还具有 `16×16` patch
token 结构，新增了两个只处理聚合梯度的可证伪方向。

第一，`patch_gaussian_s40_a075_p100` 在 raw mean 上叠加 sigma=4 的 Gaussian 和
16×16 patch 均值；它不是替换原始梯度。30 样本筛选中该方向为 Overall `+1.52pp`、
黑盒 ViT `+1.43pp`、CNN `+1.67pp`，因此进入 100 样本确认。但在 seed `20260716`
的完整 11 黑盒评估中：

| 配置 | Overall | Δ Overall | 黑盒 ViT | Δ ViT | CNN | Δ CNN | 白盒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw mean | 81.91% | — | 84.43% | — | 77.50% | — | 91.00% |
| patch + Gaussian | 81.91% | 0pp | 85.43% | +1.00pp | 75.75% | -1.75pp | 91.00% |

逐模型变化为：LeViT `+4pp`、PiT `-3pp`、DeiT `+2pp`、TNT `+1pp`、ConViT
`+1pp`、Visformer `+2pp`、CaiT `0pp`；CNN 四模型分别为 `-2/-3/-1/-1pp`。
这说明 patch-scale 结构只能帮助部分模型，不能稳定增加 Overall，且会牺牲 CNN。

第二，测试了把 patch 均值叠加到已经验证的 cross-scale Gaussian 上。30 样本完整
11 黑盒筛选结果为：

| patch strength | Δ Overall | Δ ViT | Δ CNN | Δ 白盒 |
|---:|---:|---:|---:|---:|
| 0.25 | +0.30pp | +0.48pp | 0pp | -3.33pp |
| 0.50 | -1.21pp | 0pp | -3.33pp | 0pp |
| 1.00 | +0.30pp | +1.43pp | -1.67pp | 0pp |

patch DC 与 cross-scale transport 没有表现出可加性，因此不进行 100 样本确认。

### 频率级 cross-scale coherence gating

全局 cross-scale transport 对所有高频坐标使用同一替换比例。为检验是否是低相干
高频被误替换，新增 `cross_scale_coherent_gaussian`：先估计每个 Fourier 坐标的
低/高尺度跨-view coherence，只在超过阈值的坐标使用 transport，其余保留原始
高频，再加入相同的 weak Gaussian。

30 样本完整 11 黑盒结果如下：

| coherence threshold | Δ Overall | Δ ViT | Δ CNN | Δ 白盒 |
|---:|---:|---:|---:|---:|
| 0.00 | -1.21pp | -0.95pp | -1.67pp | -3.33pp |
| 0.25 | +0.30pp | +0.48pp | 0pp | -3.33pp |
| 0.50 | +0.30pp | 0pp | +0.83pp | -3.33pp |

没有一个阈值接近 Overall `+3pp` 或 ViT `+3pp`。因此当前 cross-scale Gaussian 的
收益不是简单来自“挑出高相干频率”；过窄的频率门控会丢掉对迁移有用的多样性。

截至目前，patch-scale 组合和 coherence gating 均被筛除；稳定最佳仍是三 seed
平均 Overall `+1.52pp` 的全局 cross-scale Gaussian。代码探针已通过单元测试并推送，
但没有任何新候选满足切换主线的 3–5pp Overall 标准。

### 跨样本共享梯度原型

静态 Fourier/patch 变换没有达到目标后，进一步测试了“迁移共享方向也许在不同图像
之间稳定”的假设。新增 probe 在每个攻击 step 维护一个跨样本梯度原型：第一个
batch 使用 leave-one-out 均值，后续 batch 使用同一 step 的 EMA；原型先按当前样本
raw 梯度的 L2 norm 缩放，再以弱权重注入。该操作只发生在 view 聚合之后，不增加
forward、view、data-aug 或攻击预算。

30 样本完整 11 黑盒筛选结果如下：

| 原型范围/强度 | Δ Overall | Δ ViT | Δ CNN | Δ 白盒 |
|---|---:|---:|---:|---:|
| low-pass, a=0.05 | -2.12pp | -0.95pp | -4.17pp | -3.33pp |
| low-pass, a=0.10 | -1.82pp | -1.90pp | -1.67pp | -3.33pp |
| low-pass, a=0.20 | -3.03pp | -2.86pp | -3.33pp | -3.33pp |
| raw, a=0.05 | -0.30pp | -0.95pp | +0.83pp | -3.33pp |
| raw, a=0.10 | -1.52pp | -1.43pp | -1.67pp | -3.33pp |

跨样本原型没有产生正向 ViT 迁移，且低频原型对 CNN 伤害明显。该结果说明当前
有效方向主要依赖单个样本的类别/空间结构；把不同图像的梯度平均或 EMA 化会丢失
这些必要信息。因此后续不再把跨样本 universal direction 作为主线候选，下一轮若
继续，应只在同一样本的 view 梯度内部做保留样本结构的非线性重构。

### patch-wise view L2 方向重构

最后测试了同一样本内部、以 ViT patch 为粒度的 view 聚合。与整图级
`view_l2_mean` 不同，该 probe 对每个 `16×16` patch 独立归一化 20 个 view 的方向，
求局部方向均值，再恢复该 patch 的 raw mean 能量，并与 raw mean 按强度融合；不改变
view 数或其他攻击路径。

30 样本完整 11 黑盒筛选：

| strength | Δ Overall | Δ ViT | Δ CNN | Δ 白盒 |
|---:|---:|---:|---:|---:|
| 0.10 | +0.91pp | +1.43pp | 0pp | 0pp |
| 0.25 | -1.52pp | -2.86pp | +0.83pp | -3.33pp |
| 0.50 | -1.52pp | -2.38pp | 0pp | -3.33pp |
| 0.75 | -1.21pp | -1.90pp | 0pp | -3.33pp |

由于 `a=0.10` 是唯一通过小样本守门的候选，使用新 seed `20260717` 做 100 样本
完整确认：

| 配置 | Overall | Δ Overall | 黑盒 ViT | Δ ViT | CNN | Δ CNN | 白盒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw mean | 81.73% | — | 84.43% | — | 77.00% | — | 92.00% |
| patch-wise L2 a=0.10 | 82.00% | +0.27pp | 85.00% | +0.57pp | 76.75% | -0.25pp | 92.00% |

配对 bootstrap 95% CI 为 Overall `[-0.73,+1.27]pp`。30 样本的 +1.43pp ViT 信号未能
稳定复现，说明 view 的有效共享信息不仅是 patch 局部方向，还依赖 raw 梯度在 patch
之间的幅值耦合。

截至本轮收尾，没有候选达到 Overall `+3pp` 与黑盒 ViT 约 `90%` 的目标。当前最好的
稳定候选仍为 cross-scale Gaussian（三 seed 平均 Overall `+1.52pp`、ViT `+1.48pp`），
但不切换默认 raw mean 主线。
