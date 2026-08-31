# Patch 路由攻击+RGB对手噪声

## 当前研究定位

文章主线围绕两个具有明确职责分工的机制建立 motivation：

1. **Patch-score-guided patch drop：决定扰动位置。** 普通 patch dropout 只对空间块做随机删除，无法区分哪些局部区域正在承担当前类别判断。patch-score 用最终语义层的 global/local 关系建立一个模型感知的路由器，从高分候选区域中选择性删除 patch，目标是破坏模型依赖的判别证据。
2. **RGB opponent-channel random noise：决定保留证据如何被扰动。** 普通 IID Gaussian 没有颜色结构，也没有经过模型输入投影。opponent-channel 噪声在亮度、红绿对抗、黄蓝对抗坐标中采样，再经首层 RGB projection 映射到初始特征，目标是在保留 token 上破坏颜色表达与局部表征的稳定性。

后续实验优先验证这两个机制是否真的改变了语义证据、特征响应和梯度结构。phase pair、20-view gradient mean、Gaussian residual、MI-FGSM 和 ASR 都是支撑机制或验证指标，不应成为文章贡献叙事的中心。

## 攻击方法

每个迭代步对当前对抗图像执行 10 个 group。每个 group：

1. 从白盒模型最终语义层提取 global/local features，以余弦相似度计算 patch score；该分数承担语义路由功能。
2. 取高 patch score（定义为大于中位数的score）为候选集，从中随机选择约占全部 patch 15% 的位置。
3. 将离散 mask 映射到 pixel space 并把对应区域置零。
4. 使用同一 mask 生成 original 与 phase-shifted 两个 view；phase 从 `(4,4)`、
   `(8,8)`、`(12,12)` 中采样。
5. 在 initial RGB projection 输出处向 kept tokens 注入 RMS-matched
   opponent-projected RGB 噪声；该噪声由亮度、红绿对抗、黄蓝对抗三个 RGB 方向构成，并通过真实首层 RGB projection 映射到特征空间。
6. 对 10 groups × 2 views 的 20 个像素梯度直接求 raw mean。

其中第 1—3 步构成“语义选择性 patch drop”，第 5 步构成“颜色结构随机扰动”。文章应分别分析它们的作用，再分析二者组合后的互补性。

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

本文及对应 CSV 统一采用 `ASR = 1 - adversarial accuracy`。分母为送入目标模型的全部
1000 张对抗样本，不按目标模型在 clean 图上的预测结果筛选子集。

不同白盒的 score 网格和实际 drop 数量为：

| 白盒源模型 | score 来源 | 网格 | 每组实际 drop |
| --- | --- | ---: | ---: |
| ViT-B/16 | `blocks[11]` CLS/patch | 14×14 | 29/196 |
| CaiT-S24 | `blocks[23]` + class-attention CLS | 14×14 | 29/196 |
| PiT-B | `transformers[2].blocks[3]` | 8×8 | 10/64 |
| Visformer-S | `stage3[3]` + GAP pseudo-CLS | 7×7 | 7/49 |

## 1000 样本结果：迁移性验证而非后续唯一目标

下面结果用于说明当前实现具备迁移攻击能力。后续更重要的结果包括 score 路由有效性、颜色噪声结构差异、
特征/梯度响应变化以及跨架构一致性。

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

每个对应攻击目录保留 1000 张 adversarial images、`attack_params.json`、
`gradient_diagnostics.json` 和 `replay_manifest.json`。普通 feature-space IID Gaussian
噪声是当前代码保留的对照能力，但尚无保留的 1000 样本结果，因此本文不报告其 ASR。
