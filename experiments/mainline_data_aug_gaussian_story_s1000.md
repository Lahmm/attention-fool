# 从 Patch 路由干预到共享梯度增强：主攻击方法与 1000 样本结果

## 1. 汇报摘要

本项目关注单白盒视觉 Transformer 攻击中的一个核心问题：常规迭代攻击容易沿着源模型固定的 patch 划分、token 路由和局部高响应方向过拟合，因而白盒攻击很强，但跨模型迁移受限。

我们的解决思路分为两层：

1. **在前向路径上制造结构化数据增强**：利用深层 Patch Score 找到与全局表示联系最强的一半 patch，再从中随机丢弃约 15%；对同一个 mask 构造原始视图和 patch-grid phase-shift 视图，并只在保留区域注入 RGB opponent-channel 投影噪声。固定每步 20 个实际 view，对这些不同路由下的梯度做期望聚合。
2. **在梯度路径上增强空间共享方向**：对 ViT-B/16 白盒的 20-view raw mean 梯度加入弱 Gaussian 平滑残差：

   \[
   g'_t=g_t+0.75\,\mathcal G_{\sigma=4}(g_t),
   \]

   其中原始梯度完整保留，Gaussian 分量只强化空间连续、低频占比更高的方向。处理后的梯度进入原始 MI 累积和 sign update，不改变步长、迭代次数或 \(L_\infty\) 约束。

在 1000 个样本、10 steps、\(\epsilon=16/255\)、20 actual views 和 13 个评估模型上，最终汇报配置为：

- ViT-B/16 白盒：data augmentation + Gaussian blend；
- CaiT-S24、PiT-B、Visformer-S 白盒：相同 data augmentation + raw gradient mean，不使用 Gaussian blend。

ViT-B/16 使用 Gaussian blend 后，Overall ASR 从 77.12% 提升到 **78.08%**，Transformer 平均从 81.90% 提升到 **83.24%**，CNN 平均从 71.53% 提升到 **72.05%**。Gaussian 带来了稳定但有限的改善；主线最主要的增益仍来自 Patch Score drop、phase pair 和 kept-only opponent noise 组成的数据增强。

---

## 2. 问题与动机

### 2.1 迁移攻击的瓶颈不是白盒损失不够大，而是方向过度依赖源模型

对抗迁移要求扰动不仅能提高源模型损失，还要命中其他模型共享的判别方向。ViT 的输入首先被划分为固定 patch，再通过全局 token 和多层 self-attention 完成信息聚合。因此，同一张图像的攻击梯度会同时依赖：

- 源模型的 patch embedding 和 patch-grid 对齐；
- 深层全局 token 对局部 patch 的路由方式；
- 少数高响应区域形成的源模型特异梯度峰值；
- 当前迭代中随机增强路径产生的梯度尺度和频率组成。

如果每一步都在同一条固定前向路径上优化，MI-FGSM 会持续积累这条源模型路径最容易利用的方向。这样的方向可以提高白盒 ASR，却未必在 DeiT、CaiT、PiT、Visformer 或 CNN 中成立。

我们的目标不是简单增加 view 数或攻击步数，而是在固定 20-view 预算中构造不同的 token 路由和 patch embedding Jacobian，使最终梯度更接近“跨路由仍然有效”的方向。

### 2.2 为什么使用 Patch Score

项目受到深层 Patch Score 和 lazy aggregation 现象的启发。对深层局部表示 \(p_i\) 与全局表示 \(c\)，定义：

\[
s_i=\cos(p_i,\tilde c),
\qquad \tilde c=c+\xi_{cls}.
\]

其中 \(\xi_{cls}\) 是只用于评分分支的弱随机扰动。深层 score 描述了当前图像中局部 patch 与全局语义路由的联系强度，因此比尚未经过注意力聚合的输入层 token 几何更适合决定干预位置。

但这里必须明确一个解释边界：

> **high-score patch 不等于背景，也不等于可安全删除的无用区域。**

我们不把 Patch Score 当作语义分割器，而把它当作源模型路由强度的可观测量。drop 的目的不是删除“背景”，而是随机打断源模型当前最依赖的部分全局路由，迫使不同 view 从剩余证据中产生梯度。

### 2.3 为什么 drop 高分候选，而不是确定性删除最高分 patch

每个 group 先取 score 最高的一半 patch 作为候选集，再从候选集中随机选择约占全部 patch 15% 的位置进行 pixel-space dropout。这样设计有三个原因：

1. **干预有效性**：候选区域与全局表示联系较强，drop 能真实改变 token 路由，而不是只产生无关的小扰动。
2. **保留可攻击性**：只 drop 约 15%，其余图像证据仍然存在，不会把前向过程变成与原图任务无关的严重破坏。
3. **路由覆盖**：不同 group 随机选择不同子集，相当于对多个可能的路由缺失状态做 EOT。确定性 top-k drop 只会形成另一条固定路径，仍然容易过拟合。

评分分支被 detach；score 和离散 mask 只决定增强路径，不直接把评分计算图混入攻击梯度。drop 发生在 pixel space，因而同一个方法可以适配 ViT、CaiT、PiT 和 Visformer，而不要求它们具有完全相同的 token 流程。

---

## 3. 从实验中得到的关键发现

### 3.1 固定预算下，patch-grid Jacobian 多样性比单纯增加独立 mask 更重要

早期主线在每一步使用 20 个独立 mask view。随后将相同预算重新分配为 `10 groups × 2 views`：每个 group 内两个 view 共享同一 drop mask，view A 使用 drop 后图像，view B 在 drop 后再做 phase shift。

100 样本对照中，这一 phase-pair plain mean 将 Overall 从 70.0% 提高到 77.6%，Transformer 平均提高 6.6pp，CNN 平均提高 9.5pp。它说明：

- phase shift 改变了像素邻域进入 patch embedding 的组合方式；
- 两个 view 对应不同 patch-grid Jacobian；
- 在固定 20-view 预算下，这种 Jacobian 多样性比额外采样 10 个独立 mask 更有价值；
- CNN 的增益甚至更大，表明 phase-pair 得到的不只是 ViT 内部 token 技巧，而是更通用的像素梯度。

进一步加入 pair-difference、cross-patch transport 或 kept-token rotation 都没有超过 plain pair mean。最稳妥的聚合仍然是直接平均原始与 shifted view 的梯度。

### 3.2 score、drop 和 phase 的顺序决定了可解释性与效果

当前流程固定为：

```text
当前对抗图像
  → 在原始 patch grid 上计算深层 Patch Score 和 mask
  → pixel-space patch dropout
  → 构造原始 / phase-shifted 两个视图
```

phase shift 不参与 score 计算，因此 score 始终对应模型原生 patch grid；它只改变 drop 之后的 forward Jacobian。该顺序既保留了 Patch Score 作为原始路由诊断量的含义，也利用了 phase diversity。

采用 original-score/post-dropout phase pair 并加入 kept-only opponent noise 后，早期 100 样本 Overall 从上一代的 77.64% 提高到 79.73%。这部分收益来自 score/mask 语义一致性、post-dropout phase Jacobian 和噪声范围三者的组合，不能单独归因于某一个组件。

### 3.3 dropped 区域必须保持为零，噪声只应作用于 kept 区域

drop 负责制造“这条路由不可用”的反事实。如果又向 dropped token 写入噪声，就会重新打开一条人工路径，破坏 dropout 的含义。因此当前主线将 drop 区域保持为零，只在保留区域注入噪声。

噪声先在 RGB opponent-channel 基中采样，再通过每个源模型真实的首层 RGB projection 映射到 feature/token 空间，并按当前 token RMS 匹配尺度。该噪声具有较弱的亮度方差和较强的两个色度方向方差，其作用是：

- 在不恢复 dropped 路由的情况下扰动 kept feature；
- 增加不同 group/view 的局部通道响应多样性；
- 缓解攻击只适配单一源模型通道基的问题；
- 为 CNN 迁移提供补充，但不假设 CNN 不使用背景信息。

早期关闭 kept-only noise 的版本 Overall 只有 66.82%，开启后达到 79.73%。这个差异同时受到当时完整流程配置影响，但足以说明 noise scope 是主线中不可忽略的变量。后续位置/类型消融也表明，初始 RGB projection 位置的 `opponent_projected` 对 PiT 最有效；把噪声移动到后层通常没有收益。

### 3.4 raw gradient scale 应保留到 MI，而不是每一步重新归一化

旧实现会在每一步将聚合梯度做全局归一化，这会删除不同 step 的绝对尺度信息。取消 `_normalize_grad` 后，攻击仍然使用 sign update，并继续投影到 \(L_\infty\) 球，所以不会破坏攻击约束；变化只发生在 MI 对各步梯度的相对加权上：

\[
m_t=m_{t-1}+g_t.
\]

100 样本对照中，raw gradient 将白盒 ViT ASR 提高约 3pp，同时黑盒 Transformer 不下降、CNN 约提高 1pp。因此当前主线保留 raw-scale 梯度。

这也给出了一个重要启示：梯度的绝对幅值、空间位置和频率结构是耦合的。把每个 view 单独 L2 归一化、做 sign consensus、硬删除高频/高幅值坐标，都会破坏这种自然比例。

### 3.5 广泛梯度搜索后，Gaussian residual 是唯一稳定但有限的正向方向

项目先后测试了约 200 类/组梯度探针，包括幅值裁剪与幂变换、view L2 mean、sign consensus、PCA/GLS 聚合、频带删除与 Wiener shrink、小波处理、局部能量均衡、phase-difference transport、跨步符号持久性和 MI 方向投影等。

主要失败规律是：

- 高频中同时存在源模型特异噪声和必要的边缘/细节方向，硬删除会同时损害二者；
- 大幅值坐标中也包含有效攻击信号，裁剪峰值不能稳定提高迁移；
- view 间符号或频谱相位的一致性不等价于跨模型一致性；
- 强行标准化平滑分量的幅值会消除 Gaussian 的收益；
- 更强的低通、频带替换或局部能量重分配通常偏向 CNN，并损害 Transformer。

最终较稳定的候选是保留原始梯度、只加入平滑残差。ViT-focused 的三个 100 样本 seed 上，`sigma=4, alpha=0.75` 平均提高 Transformer ASR 1.24pp、Overall 1.12pp、CNN 0.92pp。它没有达到预设的 3pp Transformer 增益目标，但方向一致，因而进入 ViT-B/16 的 1000 样本验证。

对其他三个白盒，100 样本 Gaussian 增益分别为：CaiT Overall +1.31pp、PiT +0.31pp、Visformer +0.92pp，均未达到 3pp 选择阈值。因此本报告的最终组合只对 ViT 使用 Gaussian，其他源模型保留 raw mean。

---

## 4. 完整方法

### 4.1 每一步的结构化增强

设干净图像为 \(x\)，第 \(t\) 步对抗图像为 \(x_t\)。每一步执行 10 个 group。对第 \(k\) 个 group：

1. 从 \(x_t\) 的最终语义层提取全局表示 \(c_{t,k}\) 和局部表示 \(p_{t,k,i}\)。
2. 使用带弱 CLS jitter 的余弦相似度计算 score：

   \[
   s_{t,k,i}=\cos(p_{t,k,i},c_{t,k}+\xi_{t,k}).
   \]

3. 取 score 最高的一半 patch 为候选集 \(H_{t,k}\)，再随机抽取约占全部 patch 15% 的位置形成 mask \(M_{t,k}\)。
4. 将离散 patch mask 最近邻上采样到 224×224，得到 pixel mask，并执行：

   \[
   x^{drop}_{t,k}=x_t\odot(1-M_{t,k}).
   \]

5. 从位移集合 \(\{(4,4),(8,8),(12,12)\}\) 中采样 phase，以 reflect padding 完成空间位移，构造共享 mask 的两个 view：

   \[
   v^A_{t,k}=x^{drop}_{t,k},
   \qquad
   v^B_{t,k}=T_{\phi_{t,k}}(x^{drop}_{t,k}).
   \]

6. 分别将两个 view 送入源模型，在初始 RGB projection 后只对 kept feature 加入 RMS 匹配的 opponent-projected noise，计算交叉熵损失和像素梯度 \(g^A_{t,k},g^B_{t,k}\)。

每一步严格使用：

\[
10\ \text{groups}\times2\ \text{views}=20\ \text{actual views}.
\]

raw 聚合梯度为：

\[
g_t=\frac{1}{20}\sum_{k=1}^{10}\left(g^A_{t,k}+g^B_{t,k}\right).
\]

实现上等价于将 20 个独立 view 梯度 stack 后直接求 mean；不做逐 view L2 normalization 或 sign voting。

### 4.2 Gaussian blend

ViT-B/16 白盒使用：

\[
\bar g_t=\mathcal G_{\sigma=4}(g_t),
\qquad
g'_t=g_t+0.75\bar g_t.
\]

其中 \(\mathcal G\) 是按 RGB 通道独立执行的二维 Gaussian 卷积，使用 reflect padding。该操作位于 20-view mean 之后、MI 累积之前。

它不是把梯度替换成模糊梯度，也不是传统 TI 中只使用平滑方向。残差形式保留了 \(g_t\) 的全部细节和自然尺度，同时让在邻域内方向一致的坐标获得额外权重。直观上：

- 孤立、快速变化的源模型特异峰值主要只保留原始权重；
- 空间连续的边缘、区域和低频方向同时存在于 \(g_t\) 与 \(\bar g_t\) 中，因此被增强；
- 不归一化 Gaussian 分量，使其强度仍由当前 raw gradient 的真实局部结构决定；
- 最终仍经过 MI 和 sign，Gaussian 改变的是坐标进入历史方向的相对权重，而不是攻击预算。

CaiT、PiT 和 Visformer 使用 \(g'_t=g_t\)。

### 4.3 MI-FGSM 更新与攻击约束

所有源模型均使用：

\[
m_t=m_{t-1}+g'_t,
\]

\[
x_{t+1}=\Pi_{B_\infty(x,\epsilon)}
\left(x_t+\frac{\epsilon}{10}\operatorname{sign}(m_t)\right),
\qquad \epsilon=\frac{16}{255}.
\]

同时将像素裁剪到 \([0,1]\)。因此 data augmentation、raw-scale MI 和 Gaussian blend 都不改变 10 steps、step size、epsilon 或投影逻辑。

### 4.4 四种源模型的 score 与 drop 网格

| 白盒源模型 | 最终 score 来源 | score 网格 | 每个 group 实际 drop | Gaussian |
|---|---|---:|---:|---|
| ViT-B/16 | `blocks[11]` CLS/patch | 14×14 | 29/196 = 14.80% | `sigma=4, alpha=0.75` |
| CaiT-S24 | `blocks[23]` patch + 最终 class-attention CLS | 14×14 | 29/196 = 14.80% | 无，raw mean |
| PiT-B | `transformers[2].blocks[3]` CLS/patch | 8×8 | 10/64 = 15.63% | 无，raw mean |
| Visformer-S | `stage3[3]` local feature + GAP pseudo-CLS | 7×7 | 7/49 = 14.29% | 无，raw mean |

不同架构使用各自最终语义网格产生 score，但 mask 统一映射回 pixel space。这样保留各模型自身的全局/局部关系，同时保持统一的约 15% 干预预算。

---

## 5. 方法原理：三层去源模型过拟合

整条攻击可以理解为三个互补层次：

```text
Patch Score stochastic drop
  └─ 改变源模型依赖的 token / feature 路由

Original / shifted phase pair
  └─ 改变像素到 patch embedding 的局部划分与 Jacobian

Raw mean + Gaussian residual + MI
  └─ 保留自然梯度尺度，并增强空间连续的共享方向
```

第一层解决“模型在看哪些局部证据”，第二层解决“像素怎样被分块并映射为特征”，第三层解决“不同增强路径的梯度怎样进入跨步累积”。

如果只做 drop，梯度仍可能过拟合固定 patch grid；如果只做 phase shift，模型仍可能沿原始强路由优化；如果用强低通替换原始梯度，又会丢失必要细节。当前方法的关键不是某一个组件单独工作，而是：

> 用 stochastic drop 和 phase pair 扩展前向路径分布，再以保守的 residual Gaussian 方式增强这些路径中空间上更连续的梯度成分。

---

## 6. 1000 样本实验设置

共同设置如下：

| 项目 | 设置 |
|---|---|
| 数据量 | 1000 images |
| 随机种子 | 20260716 |
| 攻击 | MI-FGSM, decay=1.0 |
| 扰动预算 | \(L_\infty\), \(\epsilon=16/255\) |
| 迭代 | 10 steps，step size=\(\epsilon/10\) |
| 增强预算 | 10 groups × 2 views = 20 views/step |
| drop | high-score half 中随机抽取，目标约 15% 全部 patch |
| phase set | (4,4), (8,8), (12,12) |
| feature noise | initial RGB `opponent_projected`, strength=0.2, kept-only |
| 梯度尺度 | raw scale，不执行 `_normalize_grad` |
| 评估模型 | 7 个 Transformer + 6 个 CNN，共 13 个 |

Transformer 目标：LeViT-256、PiT-B、DeiT-B/16、TNT-S、ConViT-B、Visformer-S、CaiT-S24。

CNN 目标：Inception-v3、Inception-v4、Inception-ResNet-v2、ResNet-101、Adv Inception-v3、Adv Inception-ResNet-v2。

---

## 7. 1000 样本结果

### 7.1 四个白盒源模型汇总

下表的 Overall/Transformer/CNN 沿用项目评估脚本定义。需要注意：CaiT、PiT、Visformer 本身位于 7 个 Transformer 目标集合中，所以对应行的 Overall 和 Transformer avg 包含一次源模型自身结果；ViT-B/16 不在 13 模型目标集合中，因此其结果是严格的 13 黑盒迁移结果。

| 白盒源模型 | 梯度方法 | Overall ASR | Transformer avg | CNN avg | 目标集合中的源模型 ASR |
|---|---|---:|---:|---:|---:|
| ViT-B/16 | Gaussian blend | **78.08%** | **83.24%** | **72.05%** | 不在目标集合 |
| CaiT-S24 | raw mean | **87.00%** | **90.47%** | **82.95%** | 97.50% |
| PiT-B | raw mean | **82.78%** | **90.10%** | **74.23%** | 97.00% |
| Visformer-S | raw mean | **75.22%** | **80.54%** | **69.02%** | 99.70% |

为避免源模型自身结果抬高迁移均值，排除 source target 后的严格黑盒指标如下：

| 白盒源模型 | 严格黑盒 Overall | 严格黑盒 Transformer avg | CNN avg |
|---|---:|---:|---:|
| ViT-B/16 + Gaussian | 78.08% | 83.24% | 72.05% |
| CaiT-S24 + raw mean | **86.13%** | **89.30%** | **82.95%** |
| PiT-B + raw mean | 81.59% | 88.95% | 74.23% |
| Visformer-S + raw mean | 73.18% | 77.35% | 69.02% |

### 7.2 ViT-B/16 上 Gaussian blend 的直接贡献

该对照使用相同 1000 样本、seed 和完整 data augmentation，只改变 20-view mean 后是否加入 Gaussian residual。

| 指标 | raw mean | Gaussian blend | 变化 |
|---|---:|---:|---:|
| Overall ASR | 77.12% | **78.08%** | **+0.96pp** |
| Transformer avg | 81.90% | **83.24%** | **+1.34pp** |
| CNN avg | 71.53% | **72.05%** | **+0.52pp** |

Gaussian 在 13 个模型中的 12 个上不下降，主要改善 LeViT（+1.9pp）、PiT（+1.4pp）、TNT（+1.6pp）、ConViT（+1.4pp）、Visformer（+2.0pp）、Inception-ResNet-v2（+1.1pp）、ResNet-101（+1.0pp）和 Adv Inception-ResNet-v2（+1.2pp）；唯一明显下降的是 Inception-v4（-1.0pp）。

因此 1000 样本结果确认了 Gaussian 的正向趋势，但也确认其上限：它没有达到 3pp 的 Transformer 提升目标，不能被描述为大幅改进。更准确的结论是，Gaussian residual 是当前已测梯度后处理里最稳定的弱增强，而不是主线成功的主要来源。

### 7.3 逐目标模型 ASR

| 白盒 / 梯度 | LeViT | PiT | DeiT | TNT | ConViT | Visformer | CaiT | Inc-v3 | Inc-v4 | IncRes-v2 | ResNet-101 | Adv Inc-v3 | Adv IncRes-v2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-B/16 / Gaussian | 82.4% | 81.0% | 83.4% | 84.1% | 82.1% | 81.4% | 88.3% | 75.7% | 72.5% | 72.7% | 73.2% | 71.7% | 66.5% |
| CaiT-S24 / raw | 89.3% | 88.2% | 90.0% | 90.7% | 89.6% | 88.0% | 97.5% | 85.1% | 84.1% | 84.2% | 84.7% | 83.0% | 76.6% |
| PiT-B / raw | 87.5% | 97.0% | 90.2% | 90.4% | 88.6% | 91.0% | 86.0% | 80.4% | 78.1% | 76.3% | 79.6% | 69.8% | 61.2% |
| Visformer-S / raw | 83.2% | 82.4% | 73.8% | 80.0% | 69.8% | 99.7% | 74.9% | 78.0% | 77.8% | 70.8% | 76.6% | 64.6% | 46.3% |

### 7.4 梯度多样性诊断

| 白盒 / 梯度 | View→final cosine | Sign agreement | Effective rank | MI 与当前梯度 cosine |
|---|---:|---:|---:|---:|
| ViT-B/16 / Gaussian | 0.2419 | 0.4790 | 19.69 | 0.5938 |
| CaiT-S24 / raw | 0.2717 | 0.4893 | 19.31 | 0.6068 |
| PiT-B / raw | 0.3487 | 0.5106 | 18.39 | 0.6092 |
| Visformer-S / raw | 0.4300 | 0.5433 | 16.08 | 0.5844 |

这些数据说明 20 个 view 并不是高度重复的梯度。ViT 的 view cosine 和 sign agreement 最低、effective rank 接近 20，说明其增强路径多样性最高，但简单平均时也更容易发生抵消；这一现象与在 ViT 上补充 Gaussian residual 的动机一致，并可能解释它为何获得相对更多收益。该解释仍是由诊断量支持的推断，不是独立因果证明。Visformer 的 view 更一致，但方向更集中，Gaussian 在此前 100 样本中只带来有限增益，因此最终保留 raw mean。

---

## 8. 如何解读最终结果

### 8.1 data augmentation 是主贡献，Gaussian 是增量贡献

phase-pair 在早期固定预算对照中带来约 7.6pp Overall 提升，original-score/post-dropout 顺序和 kept-only noise 又将 100 样本主线推到 79.73%。相比之下，Gaussian 在 ViT 的 1000 样本上贡献 0.96pp Overall 和 1.34pp Transformer。

因此汇报时应把方法主次表述为：

> 主方法是 Patch-Score-conditioned stochastic dropout、post-dropout phase pair 和 kept-only opponent-projected noise 形成的结构化 data augmentation；Gaussian blend 是对 ViT 源梯度的保守共享方向增强。

### 8.2 源模型选择仍然决定迁移上限

CaiT raw mean 的严格黑盒 Overall 为 86.13%，明显高于 ViT Gaussian 的 78.08%；PiT 的严格黑盒 Transformer 平均也达到 88.95%。这说明当前上限主要受源模型本身的决策边界与目标模型共享程度影响，梯度后处理不能完全弥补源模型差异。

Visformer 对自身达到 99.7%，但严格黑盒 Transformer 平均只有 77.35%，是典型的源模型过拟合：白盒很强并不意味着迁移方向共享。这个现象也支持项目最初的动机——攻击优化不能只看源模型损失或白盒 ASR。

### 8.3 当前可以支持和不能支持的结论

可以支持：

- Patch Score 能作为随机路由干预的有效条件变量；
- 固定 20-view 预算下，original/shifted phase pair 明显优于只增加 mask 样本；
- dropped 区域保持为零、只对 kept feature 加噪是必要的机制边界；
- raw-scale MI 不破坏约束，并保留有用的跨步梯度尺度；
- Gaussian residual 对 ViT 迁移有可复现但有限的增益。

不能支持：

- high-score patch 就是背景或无用区域；
- 低频总是有益、高频总是有害；
- view 间一致的梯度一定是跨模型共享方向；
- Gaussian blend 已经带来 3–5pp 的大幅提升；
- 四种白盒都应统一使用 Gaussian。

---

## 9. 最终结论

本项目得到的核心故事不是“删除某些 patch 就能提高攻击”，而是：

> 深层 Patch Score 暴露了源模型当前的全局—局部路由关系。对高关联候选做轻量随机 dropout，可以在保留图像主要证据的同时生成多个路由反事实；共享 mask 的 original/phase-shift pair 进一步改变 patch embedding Jacobian；kept-only opponent noise 扩展保留证据的通道响应。对这些增强路径求 raw gradient expectation，再以 MI 累积，就能得到比单一路径更可迁移的像素方向。ViT 上加入 Gaussian residual，可以在不删除必要细节的前提下进一步强化空间连续的共享成分。

1000 样本结果表明，这套 data augmentation 能稳定适配四种白盒结构；其中 CaiT raw mean 获得最高严格黑盒 Overall 86.13%，PiT raw mean 获得 88.95% 的严格黑盒 Transformer 平均。ViT 的 Gaussian blend 将 Transformer 平均从 81.90% 提高到 83.24%，证明梯度平滑残差方向有效，但提升仍只有 1.34pp。

因此当前最准确的项目结论是：**结构化 data augmentation 已构成稳定主线；Gaussian blend 是 ViT 上当前最有效的增量梯度方法，但尚未解决跨 ViT 迁移提升不足 3pp 的瓶颈。**

---

## 10. 结果文件

- ViT-B/16 + Gaussian：[迁移结果 CSV](../outputs/csv/outputs_attack_vit_base_patch16_224_opponent_projected_gaussian_s40a075_mainline_s1000_seed20260716.csv)
- CaiT-S24 + raw mean：[迁移结果 CSV](../outputs/csv/outputs_attack_cait_s24_224_opponent_projected_mainline_s1000_seed20260716.csv)
- PiT-B + raw mean：[迁移结果 CSV](../outputs/csv/outputs_attack_pit_b_224_opponent_projected_mainline_s1000_seed20260716.csv)
- Visformer-S + raw mean：[迁移结果 CSV](../outputs/csv/outputs_attack_visformer_small_opponent_projected_mainline_s1000_seed20260716.csv)
