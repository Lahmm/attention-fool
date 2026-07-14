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
