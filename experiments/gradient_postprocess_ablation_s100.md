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
