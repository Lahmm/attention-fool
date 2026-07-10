# 方案一 Novelty Analysis — arXiv Literature Review

**Date**: 2026-07-10 | **Method**: 10×2 Phase-Shifted Patch Embedding Pairs + CLS-Score Masking + Pair-Mean Gradient

---

## 1. 最相近的已有工作

### 1.1 图像级空间变换攻击（与方案一最接近的范式）

| 论文 | 会议 | 方法 | 与方案一的关系 |
|------|------|------|---------------|
| **BSR** (Wang et al.) | CVPR 2024 | 图像分块 → 随机打乱 + 旋转 → 多 view 梯度平均 | 同为空间变换多 view，但 BSR 在**图像空间**操作整块，方案一在**patch embedding 对齐层面**操作像素偏移 |
| **ETIM** (Yang et al.) | ACM 2025 | 多尺度平移图像 → 梯度平均（替代 TI kernel） | 同样用像素平移增加梯度多样性，但 ETIM 是**后处理梯度平滑**，方案一是**前向传播内**的 view 配对 |
| **Input Transpose** (Wan et al.) | arXiv 2025.03 | 图像转置/1°旋转 → 大幅提升 ASR | 同样是极低成本的输入变换，但仅做整图 transpose，不涉及 patch grid 对齐 |
| **PEAS** (Avraham et al.) | arXiv 2024.10 | 搜索感知等价变换 → 选择最优对抗样本 | 使用像素偏移作为候选变换，但**选择而非平均**；不做 view 配对 |
| **SIA** (IEEE 2024) | IEEE 2024 | 逐块随机变换（保持结构）→ 增加输入多样性 | 类似思路（局部空间变换增加 diversity），但变换粒度是图像块而非 patch grid |

### 1.2 ViT Token/Patch 级别攻击

| 论文 | 会议 | 方法 | 与方案一的关系 |
|------|------|------|---------------|
| **ATT** (Ming et al.) | NeurIPS 2024 | Self-Paced PatchOut + token 梯度平滑 | ViT patch 级多样性，但通过**丢弃 patch** 而非变换输入 |
| **PNA+PatchOut** (Naseer et al.) | AAAI 2022 | 跳过 attention 梯度 + 随机丢弃 patch token | 方案一的 score/mask 机制可视为 PatchOut 的 CLS-score 条件版本 |
| **TESSER** (Guesmi et al.) | arXiv 2025.05 | token 梯度重加权 + 频谱正则化 | ViT token 级梯度调制，但不涉及输入变换或 view 配对 |
| **GNS-HFA** (ICLR 2024) | ICLR 2024 | 梯度归一化缩放 + 高频自适应 | ViT 频率域分析，但方向相反（利用而非抑制高频） |
| **SGM** (Wang et al.) | ICLR 2022/2024 | skip connection 梯度衰减 | 梯度路径调制，与方案一的 Jacobian diversity 互补 |

### 1.3 输入多样性/增强攻击

| 论文 | 方法 | 与方案一的关系 |
|------|------|---------------|
| **DIM** (Xie et al.) | CVPR 2019 | 随机 resize+padding → 输入多样性 | 最早的系统性输入多样性工作，但基于 CNN，使用单一全局变换 |
| **FAUG** (Wang et al.) | arXiv 2024.07 | 中间特征注入噪声 → 梯度多样性 | 在**特征空间**而非输入空间增加多样性 |
| **MFI** (IEEE TIFS 2024) | 频域混合输入 → 稳定梯度方向 | 在**频域**混合不同图像的成分 |

---

## 2. Novelty Assessment

### 2.1 方案一中有新颖性的部分

**A. Budget-Allocation Framing（中等新颖性）**

将固定 20-view 预算显式建模为「mask diversity」与「Jacobian diversity」之间的资源分配问题，这在已有文献中**未见明确讨论**。已有工作要么单纯增加 view 数（DIM, BSR），要么在每个 view 内做变换但不改变 view 结构。方案一提出的 trade-off 框架——将 20 个独立 mask sample 换成 10 组双 phase 对——是新颖的问题表述。

**B. Phase-Shifted Patch-Embedding + CLS-Score Masking 的组合（低-中等新颖性）**

虽然像素偏移（TIM, ETIM）和 patch 评分掩码（PatchOut, ATT）各自有成熟工作，但将它们组合在**每个 view 独立运行完整 score→mask→L0 injection 流程**的架构中未见报道。关键在于：

- ETIM 的平移仅用于后处理梯度平滑（kernel convolution），不参与前向传播
- BSR 的变换在图像空间操作，不改变 ViT 的 patch embedding 对齐
- 方案一的 phase shift 改变了 `patch_embed(pixels)` 与 `patch_embed(shifted_pixels)` 之间的对应关系，产生了不同的 `W^T` Jacobian 方向

**C. 实证发现：Jacobian Diversity > Mask Diversity（高等新颖性）**

方案一的核心发现——将一半的 independent mask samples 换成 phase-shifted pair 后总体 ASR 从 70.0% 提升到 77.6%——是**反直觉的（counterintuitive）**。已有文献的主流假设是更多的独立随机 mask sample 带来更好的多样性（因此 guide_aug_copies 越大越好）。方案一证明在固定预算下，Jacobian diversity 的边际收益可以超过 mask diversity 的边际损失。

### 2.2 新颖性较弱的方面

**A. 像素偏移/平移本身（低新颖性）**

TIM (CVPR 2019) 已经提出用高斯核卷积梯度来近似平移不变性。ETIM (2025) 进一步用多个离散平移代替高斯卷积。方案一的 phase shift (4,4; 8,8; 12,12) 本质上是一种离散平移，在操作层面与 ETIM 高度相似。

**B. Pair-Mean 梯度聚合（低新颖性）**

方案一的 plain pair mean 与 20-view flat mean 在数学上等价（当所有 view 对称时）。只有当 phase shift set 不为对称时（如只使用偏向一侧的偏移），pair-mean 才有所不同。

**C. CLS-Cosine Score Masking（低新颖性）**

方案一的 score/mask 机制（L12 detached CLS-patch cosine → median threshold → random sampling）是已有方法的变体（PatchOut, ATT/SPPO）。虽然具体实现不同，但核心思想（基于 token 重要性选择性保留/丢弃 patch）已被广泛探索。

---

## 3. 与最相关工作的详细对比

### 3.1 vs BSR (CVPR 2024)

```
BSR:    图像 → 分块 → 随机打乱+旋转 → 完整ViT前向 → 梯度
方案一:  原始像素 + 偏移像素 → 各自 patch_embed → 各自 L12 score → 
         各自 mask → 各自 L0 injection → 各自 CE loss → pair-mean 梯度
```

**关键区别**: BSR 变换的是**图像空间的内容排列**（re-shuffle blocks），方案一变换的是**patch embedding 的对齐方式**（phase shift）。BSR 改变了"哪个图像内容去哪"，方案一改变了"哪个像素进入哪个 patch token"。这两者在数学上不等价：BSR 等价于同时改变输入 + 位置编码，方案一仅改变输入→token 的投影映射。

### 3.2 vs ETIM (ACM 2025)

```
ETIM:   原图 → 5个不同平移幅度 → 每个做完整前向 → 梯度平均 → MI → sign
方案一:  原图 + 偏移图 → 各自 patch_embed → L12 score → mask → L0注入 → CE → pair-mean
```

**关键区别**: ETIM 的平移视图**共享相同的模型前向路径**（标准 forward），方案一的每个 view **运行完整的 score→mask→L0 injection 流程**。方案一的 view 内部有随机性（mask sampling, CLS jitter），而 ETIM 的平移视图是确定性的。

### 3.3 vs ATT (NeurIPS 2024)

ATT 通过 Self-Paced PatchOut 增加 patch 级多样性，但所有 view 共享同一个输入。方案一通过改变输入来增加 patch embedding Jacobian 的多样性。两者可以互补。

---

## 4. 总体评估

| 维度 | 评价 |
|------|------|
| **问题框架** | ⭐⭐⭐ 新颖 — 将 view budget 建模为 mask/Jacobian diversity trade-off |
| **方法论** | ⭐⭐ 增量 — 组合已知组件（phase shift + score masking + pair mean） |
| **实证发现** | ⭐⭐⭐⭐ 反直觉 — Jacobian diversity > mask diversity 是一个有力的发现 |
| **与最相关工作的区分度** | ⭐⭐⭐ 可区分 — 与 BSR/ETIM/ATT 有明确的差异点 |
| **潜在影响力** | ⭐⭐⭐ 中等 — 可能改变社区对 view-budget 分配的认知 |

### 可行性评估

作为一篇 transfer adversarial attack 论文，**方案一+方案二的对比已经足够支撑一篇 workshop paper 或中等级别的 conference paper**（如 ECCV, BMVC, WACV）。核心故事线是：

> "在固定计算预算下，patch-embedding Jacobian 的多样性（通过 phase-shifted views）比独立的 random mask samples 更有价值。我们将此发现形式化为 mask-diversity vs Jacobian-diversity 的 trade-off 问题，并通过严格的预算控制实验验证。"

需要加强的部分：
1. **理论分析**: 梯度有效秩（effective rank）、梯度协方差谱分析，定量证明 Jacobian diversity 确实扩大了梯度子空间
2. **更多 baseline**: 与 BSR、ETIM、ATT 等方法的直接实验对比
3. **更多模型**: 在多於一个 white-box 模型上验证（如 ViT-S, ViT-L, Swin）
4. **Phase shift 的消融**: 不同 shift magnitude、方向、数量的影响
5. **Why CNN 受益最大**: 解释为什么 CNN ASR 提升幅度（+9.5%）甚至超过 ViT（+6.6%）

---

## 5. 参考文献速查

| 论文 | ID | 关键方法 |
|------|-----|---------|
| BSR | CVPR 2024 | Block Shuffle + Rotation |
| ETIM | ACM 2025 | Multi-scale Translation Gradient Averaging |
| ATT | NeurIPS 2024 | Self-Paced PatchOut + Token Gradient Smoothing |
| TESSER | arXiv 2505.19613 | Feature-Sensitive Gradient Scaling + SSR |
| GNS-HFA | ICLR 2024 | Gradient Normalization Scaling + High-Freq Adaptation |
| Input Transpose | arXiv 2503.00932 | Image Transpose / 1° Rotation |
| PEAS | arXiv 2410.15409 | Perceptual Equivalence Search over Shifts |
| SIA | IEEE 2024 | Structure Invariant Block-wise Transformation |
| FAUG | arXiv 2407.06714 | Feature Noise Injection for Gradient Diversity |
| MFI | IEEE TIFS 2024 | Mixed-Frequency Input Fusion |
| PNA+PatchOut | AAAI 2022 | Skip Attention Gradients + Random Patch Drop |
| SGM | ICLR 2022/2024 | Skip Connection Gradient Decay |
