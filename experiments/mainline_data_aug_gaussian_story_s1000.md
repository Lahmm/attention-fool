# Patch 路由攻击：1000 样本主线结果

## 结果范围

本文只记录 2026-07-16 完成的五组 1000 样本实验，不包含早期 30、50、100 或
500 样本筛选结果。五组实验使用相同的 1000 张 ImageNet 验证图像和 seed
`20260716`。

历史结果包含四组 raw-mean 攻击和一组 ViT Gaussian gradient residual 攻击。
清理后的代码对四个白盒默认启用 Gaussian residual；该默认值不代表 CaiT、PiT、
Visformer 已经完成新的 Gaussian 1000 样本实验，使用 `--gaussian-alpha 0` 可复现
本文 raw-mean 配置。

## 攻击方法

每个迭代步对当前对抗图像执行 10 个 group。每个 group：

1. 从白盒模型最终语义层提取 global/local features，以余弦相似度计算 patch score。
2. 取 high-score half 为候选集，从中随机选择约占全部 patch 15% 的位置。
3. 将离散 mask 映射到 pixel space 并把对应区域置零。
4. 使用同一 mask 生成 original 与 phase-shifted 两个 view；phase 从 `(4,4)`、
   `(8,8)`、`(12,12)` 中采样。
5. 在 initial RGB projection 输出处向 kept tokens 注入 RMS-matched
   opponent-projected RGB 噪声，dropped tokens 保持为零。
6. 对 10 groups × 2 views 的 20 个像素梯度直接求 raw mean。

raw 聚合梯度为：

\[
g_t=\frac{1}{20}\sum_{k=1}^{10}(g^A_{t,k}+g^B_{t,k}).
\]

Gaussian 实验在 raw mean 后、MI 累积前使用：

\[
g'_t=g_t+0.75\,\mathcal G_{\sigma=4}(g_t).
\]

Gaussian 分量不做额外归一化，也不替换原始梯度。随后执行 MI-FGSM：

\[
m_t=m_{t-1}+g'_t,
\qquad
x_{t+1}=\Pi_{B_\infty(x,16/255)}
\left(x_t+\frac{16/255}{10}\operatorname{sign}(m_t)\right).
\]

## 实验设置

| 项目 | 设置 |
| --- | --- |
| 样本数 | 1000 images |
| seed | 20260716 |
| 白盒模型 | ViT-B/16、CaiT-S24、PiT-B、Visformer-S |
| 攻击 | MI-FGSM，decay=1.0 |
| 扰动预算 | L∞，epsilon=16/255 |
| 迭代 | 10 steps，step size=epsilon/10 |
| view 预算 | 10 groups × 2 views = 20 views/step |
| patch drop | high-score half 中随机抽取，约占全部 patch 15% |
| feature noise | initial RGB `opponent_projected`，strength=0.2，kept-only |
| 迁移评估 | 7 个 Transformer + 6 个 CNN |

不同白盒的 score 网格和实际 drop 数量为：

| 白盒源模型 | score 来源 | 网格 | 每组实际 drop |
| --- | --- | ---: | ---: |
| ViT-B/16 | `blocks[11]` CLS/patch | 14×14 | 29/196 |
| CaiT-S24 | `blocks[23]` + class-attention CLS | 14×14 | 29/196 |
| PiT-B | `transformers[2].blocks[3]` | 8×8 | 10/64 |
| Visformer-S | `stage3[3]` + GAP pseudo-CLS | 7×7 | 7/49 |

## 1000 样本结果

| 白盒源模型 | 梯度方法 | Overall ASR | Transformer avg | CNN avg |
| --- | --- | ---: | ---: | ---: |
| ViT-B/16 | raw mean | 77.12% | 81.90% | 71.53% |
| ViT-B/16 | Gaussian residual | **78.08%** | **83.24%** | **72.05%** |
| CaiT-S24 | raw mean | **87.00%** | **90.47%** | **82.95%** |
| PiT-B | raw mean | **82.78%** | **90.10%** | **74.23%** |
| Visformer-S | raw mean | **75.22%** | **80.54%** | **69.02%** |

ViT 的 Gaussian residual 相对相同样本和 seed 的 raw mean 提升 0.96pp Overall、
1.34pp Transformer 和 0.52pp CNN。它是稳定但有限的增量，结构化 patch drop、
phase pair 和 kept-only RGB noise 仍是攻击主体。

CaiT、PiT、Visformer 位于 Transformer 目标集合内。排除源模型自身结果后，严格
黑盒指标为：

| 白盒源模型 | 严格黑盒 Overall | 严格黑盒 Transformer avg | CNN avg |
| --- | ---: | ---: | ---: |
| CaiT-S24 raw | 86.13% | 89.30% | 82.95% |
| PiT-B raw | 81.59% | 88.95% | 74.23% |
| Visformer-S raw | 73.18% | 77.35% | 69.02% |

## 保留的结果文件

- [ViT-B/16 raw](../outputs/csv/outputs_attack_vit_base_patch16_224_opponent_projected_mainline_s1000_seed20260716.csv)
- [ViT-B/16 Gaussian residual](../outputs/csv/outputs_attack_vit_base_patch16_224_opponent_projected_gaussian_s40a075_mainline_s1000_seed20260716.csv)
- [CaiT-S24 raw](../outputs/csv/outputs_attack_cait_s24_224_opponent_projected_mainline_s1000_seed20260716.csv)
- [PiT-B raw](../outputs/csv/outputs_attack_pit_b_224_opponent_projected_mainline_s1000_seed20260716.csv)
- [Visformer-S raw](../outputs/csv/outputs_attack_visformer_small_opponent_projected_mainline_s1000_seed20260716.csv)

每个对应攻击目录保留 1000 张 adversarial images、`attack_params.json`、
`gradient_diagnostics.json` 和 `replay_manifest.json`。普通 feature-space IID Gaussian
噪声是当前代码保留的对照能力，但尚无保留的 1000 样本结果，因此本文不报告其 ASR。
