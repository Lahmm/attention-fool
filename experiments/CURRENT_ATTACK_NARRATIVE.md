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

## 二、数学支撑

### 2.1 为什么低分 patch 是应该被丢弃的

Patch embedding W ∈ R^(768×768)，有效秩 ~120。W 的奇异值极度倾斜（σ_max²/σ_min² ≈ 27717）。

- 高 σ 方向：W 放大，编码分类有用特征（边缘、纹理、颜色对比）
- 低 σ 方向：W 压缩，背景/平滑区域的信号被压制

**在 L=0，cos(patch_i, CLS₀) 测量的是 patch_i 在 W 的「高 σ 方向」上的投影分量。** CLS₀ 作为一个可学习的全局锚点向量，在预训练中被优化为对准这些高信息方向。低分 patch 在高 σ 方向上投影弱 → 被 W 压缩 → 在 token 空间表现出异常高的 L2 norm。

**实证：** 50 张图上，低分 patch 的 L2 norm 是高分 patch 的 1.14 倍（50/50 一致）。

### 2.2 为什么高 norm 是问题

Self-attention 的 softmax 赋权：

$$\alpha_{ij} = \frac{\exp(Q_i \cdot K_j)}{\sum_k \exp(Q_i \cdot K_k)}$$

高 norm 的 K_j 使指数项 $\exp(Q_i \cdot K_j)$ 即使在语义方向不匹配时也占据主导。softmax 是指数函数，对输入的量级敏感——这不是训练偏好，是数学必然。Darcet et al. 将此定义为「outlier token 劫持注意力」。

### 2.3 为什么零化它们有效

零化 dropout 的 token 后，softmax 权重重新分配：

$$\alpha_{ij}' = \frac{\exp(Q_i \cdot K_j)}{\sum_{k \notin \text{drop}} \exp(Q_i \cdot K_k)}$$

分母去掉高 norm 项 → 真正有判别信息的 token 获得更多注意力权重 → 损失对这些 token 更敏感 → 梯度迫使扰动覆盖真正重要的图像区域 → 扰动可迁移。

### 2.4 为什么 L=0 评分而不是深层

| | L=0 | 深层 (L=-1) |
|---|---|---|
| CLS 含义 | 全局可学习锚点（所有图相同） | 图像特定语义表示 |
| patch 间平均 cos | 0.18（多样化） | 0.80（近乎共线） |
| CLS 噪声的 rank 扰动 | 0.97（有效重排） | 0.996（几乎不变） |
| 评分本质 | 通用底层先验 | 白盒注意力偏好 |
| 迁移性 | 可迁移 | 过拟合白盒 |

L=0 评分基于 W 学的通用先验——所有 ViT 的 patch_embed 学到类似的底层特征 → low-score 命中的是「架构层面的信息贫乏区」而非「图像特定的背景」→ 所有 ViT 共享这个结构 → 可迁移。

深层 CLS 经过 12 层自注意力，已携带 vit_base 特定的语义偏好。深层 patch 因自注意力收敛而同质化（cos=0.80），CLS 噪声无法有效分化它们。

### 2.5 对手通道（opponent-channel）噪声

逐像素通道协方差：

$$C_{opp} = \begin{bmatrix} 1.00 & -0.25 & -0.25 \\ -0.25 & 1.00 & -0.25 \\ -0.25 & -0.25 & 1.00 \end{bmatrix}$$

特征分解：λ_lum=0.5（亮度 -50%），λ_rg=λ_yb=1.25（色度 +25%）。

CNN 第一层滤波器以亮度边缘检测为主（~60-70%）。标准 i.i.d. 噪声 33% 方差在亮度方向 → 梯度被亮度滤波器主导 → 模型特定。Opponent 噪声降至 17% → 颜色对抗滤波器贡献增加 → 颜色对抗性是所有 CNN 的共性 → +4.75pp CNN 迁移。

噪声在**像素空间**生成（具备 C_opp 协方差结构），通过 W^T 投影到 token 空间后注入。

### 2.6 CLS 评分噪声

CLS₀ + σ · ε 中 ε 在每个 copy 独立采样。噪声让 20 个 copy 对「低分边界在哪里」有不同判断 → 20 组不同的 dropout mask → 梯度多样性来自 dropout 集合的差异，而非仅来自 token 数值的差异。

## 三、当前配置与结果

```
16/255, ViT-B/16 白盒, 100 样本, 10 steps

avg=72.5%   ViT=77.0%   CNN=64.5%

levit_256  pit_b  deit_b  tnt_s  convit_b  visformer  cait_s24
   71%      76%     77%    76%     70%        75%        92%

inc_v3  inc_v4  incres_v2  resnet101
  66%     63%      61%        69%
```

## 四、未解决的问题

### 4.1 W·W^T 的信息天花板

W·W^T 的有效秩仅 120/768。无论噪声在像素空间还是 token 空间，梯度 ∂L/∂pixels = W^T · ∂L/∂tokens 最终被 W^T 压缩到这 120 维子空间内。这是不可绕过的信息瓶颈——当前所有优化都在这个子空间内部微调，没有触及瓶颈本身。

**这意味着：** 在 16/255 下，无论怎样优化噪声结构、评分策略、dropout 比例，ViT ASR 的理论上限都在 80% 附近（24/255 时同一方法达到 87%，证明瓶颈是 epsilon 而非方法设计）。

### 4.2 ViT 和 CNN 的结构性矛盾

- CNN 需要逐像素、多通道高频多样性（3×3 感受野，每通道独立滤波）
- ViT 被 W·W^T 的 120 维有效子空间限制，逐像素多样性的大部分被 patch_embed 平均掉

Opponent-channel 噪声部分弥合了这个矛盾（CNN +4.75pp），但 ViT 端仍然受限。这个矛盾不是噪声设计问题——CNN 和 ViT 处理信息的尺度根本不同。

### 4.3 可尝试但未验证的方向

- **多白盒集成：** 同时攻击两个 ViT（如 vit_base + deit_base）。两者的 W 矩阵不同，互补的有效子空间可扩大梯度覆盖范围
- **Patch 评分和 dropout 比率自适应：** 当前 CLS jitter 强度在各 copy 和各 step 间固定
