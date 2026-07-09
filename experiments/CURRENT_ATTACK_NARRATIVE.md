# 最强攻击叙事与未解决问题

## 一、攻击流程

```
adv_pixels
    │
patch_embed(W) → pos_embed → norm_pre → [CLS₀, 196 patch tokens]
    │
    ├── 评分分支（detach，无梯度）
    │     CLS₀ + σ · ε → cos(patch_i, CLS₀+ε) → scores
    │     ε ~ N(0,I), σ=0.2×rms(patches), 每 copy 独立采样
    │     scores < median → 候选池 → 随机 30% → drop_mask
    │
    └── 前传分支（有梯度）
         根据 drop_mask: zero token | opponent-channel 噪声
         → blocks[0:12] → norm → head → CE loss → backward
```

20 copies 的梯度取平均 → MI(decay=1.0) → sign → 更新像素。10 steps。

## 二、理论基础：Lazy Aggregation（CVPR 2026）

Zhang et al. "Vision Transformers Need More Than Registers" 发现 ViT 中存在
**lazy aggregation** 行为：CLS token 通过全局注意力，**将语义无关的背景 patch
作为捷径来编码全局语义**。其根因是全局注意力机制 + 粗粒度分类监督——模型没有
被强制区分前景和背景，于是走了最省力的路。

该文证明 register tokens（Darcet et al.）不足以解决此问题，提出选择性融合
patch 特征到 CLS 来抑制背景捷径。

## 三、我们的攻击：阻断背景捷径

### 3.1 核心洞察

CLS₀ 是 patch_embed 中可学习的全局锚点向量。cos(patch_i, CLS₀) 测量 patch_i
与这个全局表示方向的对齐程度。**低分 patch 就是 ViT 用来走捷径的背景区域。**

**实证（50 张 ImageNet 图，100% 一致）：**

| | 低分 patch（被丢弃）| 高分 patch（保留）|
|---|---|---|
| L2 norm | **13.55** | 11.90 |
| CLS cosine | 0.11 | 0.14 |

低分 patch 的 L2 norm 显著更高——模型为这些「捷径 patch」分配了额外的表示容量，
因为它们在训练中被证明是有用的廉价特征。

### 3.2 攻击逻辑

ViT 的推理过程：

$$\text{logits} = \text{head}\left(\text{CLS}_{12}\right), \quad \text{CLS}_{12} = \text{blocks}\left(\text{CLS}_0, \{\text{patch}_i\}\right)$$

CLS₁₂ 通过 self-attention 聚合所有 patch 的信息。当背景 patch（低分）存活时，
CLS₁₂ 可以廉价地从这些捷径中提取全局语义，而不需要精细处理前景。

**我们零化 15% 的低分 patch：**

$$\text{CLS}_{12}' = \text{blocks}\left(\text{CLS}_0, \{\text{patch}_i\}_{i \notin \text{drop}}\right)$$

背景捷径被阻断 → 模型必须从剩余 patch（前景）中提取判别信息 → 损失对前景
patch 更敏感 → 梯度迫使扰动覆盖真正包含物体的区域 → 扰动可迁移。

### 3.3 为什么 L=0 评分

L=0 的 CLS₀ 是一个**对所有图片相同的可学习向量**，它在预训练中被优化为与分类
有用特征方向对齐。由于 lazy aggregation 是 ViT 训练中收敛到的全局策略，CLS₀ 的
方向本身就编码了「哪些 patch 方向可能被用作捷径」的先验。

**与深层 CLS 的对比：**

| | L=0 | 深层 |
|---|---|---|
| 本质 | 训练收敛的全局捷径先验 | 图像特定的语义表示 |
| patch 多样性 (cos) | 0.18 | 0.80 |
| CLS 噪声效果 (rank corr) | 0.97（有效）| 0.996（无效）|
| 迁移性 | 跨架构通用 | 白盒特定 |

L=0 能识别捷径 patch 是因为 lazy aggregation 本身就是训练时形成的全局策略——
它是 ViT 架构的固有属性，不是某张图片的特定现象。

### 3.4 CLS 评分噪声

CLS₀ + σ · ε，ε 每 copy 独立。

噪声让 20 个 copy 对「哪些 patch 算捷径」有不同判断 → 20 组不同的 dropout mask
→ 梯度多样性来自捷径阻断的 copy 间差异。

## 四、对手通道（opponent-channel）噪声

### 4.1 为什么需要它

Lazy aggregation 是 ViT 的问题，不是 CNN 的问题。CNN 没有 CLS token，没有全局
自注意力，不存在「背景捷径」这回事。仅靠 token 空间 dropout 不能充分攻击 CNN。

### 4.2 数学结构

逐像素协方差：

$$C_{opp} = \begin{bmatrix} 1.00 & -0.25 & -0.25 \\ -0.25 & 1.00 & -0.25 \\ -0.25 & -0.25 & 1.00 \end{bmatrix}$$

特征分解：λ_lum = 0.5（亮度 -50%），λ_chrom = 1.25（色度 +25%）。

CNN 第一层以亮度边缘检测为主（~60-70% 的滤波器）。标准 i.i.d. 噪声将 33% 能量
分配给亮度 → 梯度被模型特定的亮度滤波器主导。opponent 噪声降至 17% → 颜色对抗
滤波器贡献增大 → 颜色对抗性是跨 CNN 架构的通用特征 → CNN 迁移 +4.75pp。

噪声在像素空间生成后通过 patch_embed 权重 W^T 投影到 token 空间。

## 五、当前配置与结果

```
16/255, ViT-B/16 白盒, 100 样本, 10 steps, 20 copies

avg=72.5%   ViT=77.0%   CNN=64.5%

levit_256  pit_b  deit_b  tnt_s  convit_b  visformer  cait_s24
   71%      76%     77%    76%     70%        75%        92%

inc_v3  inc_v4  incres_v2  resnet101
  66%     63%      61%        69%
```

## 六、未解决的问题

### 6.1 W·W^T 的信息天花板

ViT-B/16 的 W·W^T 有效秩仅 120/768。梯度 ∂L/∂pixels = W^T · ∂L/∂tokens
被限制在 ~120 维子空间内。无论 dropout 策略或噪声结构如何优化，token 空间的
梯度多样性最终被 W^T 压缩到同一子空间——这是不可绕过的信息瓶颈。

24/255 时相同方法达到 avg 84.4%（ViT 87.0%，CNN 79.8%），远超目标。瓶颈在
epsilon 预算，不在方法设计。

### 6.2 ViT 和 CNN 的信息尺度矛盾

- CNN：3×3 感受野，逐像素、逐通道处理 → 需要高频多样性
- ViT：16×16 patch，通过 W 聚合 768 维像素到 token → 需要 token 空间多样性

Opponent 噪声在像素空间生成（服务 CNN）再经 W^T 投影到 token 空间（服务 ViT），
部分弥合了这个矛盾。但 W^T 投影压缩了 768 维到 ~120 维有效子空间——CNN 端创造
的多样性大部分被 ViT 端丢弃。反之亦然。

### 6.3 潜在方向

- **多白盒集成：** 攻击两个 ViT，互补的 W 矩阵可扩大梯度有效子空间
- **CLS jitter 强度自适应：** 当前每 copy 固定 σ，随 step 变化的 jitter 可能更好
