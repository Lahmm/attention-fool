# Patch-Score–Conditioned Stochastic Token Masking

## 0. 主线定调

本项目的主攻击定义为：

> **Patch-Score–Conditioned Stochastic Token Masking for Transferable ViT Attacks**
>
> 基于深层 CLS–patch 相似度的随机 token 路由干预攻击。

攻击不是显式的前景分割，也不声称每个 high-score patch 都是背景。我们利用 ViT
全局 CLS 路由的结构性偏置，在测试时对 CLS 最强关联的 patch 进行随机干预；再对
多种 token 路由和 patch-grid phase 下的梯度做 EOT/MI 聚合，优化跨路由仍然稳定的
像素扰动。

当前主线是**方案一：10 groups × 2 views = 20 actual views**。每个 group 使用一个
原始 view 和一个 shifted/phase view；两个 view 各自独立运行完整的 L12 score、CLS
jitter、high-score mask、L0 injection 和 12-layer forward，两个梯度先 pair mean，
10 个 group 再做最终 mean。方案一不是未来候选，而是当前默认攻击协议。

## 1. 攻击流程

```text
adv_pixels
    │
    ├── group 1..10
    │     │
    │     ├── view A: 原始 pixels
    │     │      → patch_embed + pos_embed + norm_pre
    │     │      → L12 score + CLS jitter + high-score mask
    │     │      → L0 injection + full forward → g_A
    │     │
    │     └── view B: differentiable phase-shifted pixels
    │            → patch_embed + pos_embed + norm_pre
    │            → L12 score + CLS jitter + high-score mask
    │            → L0 injection + full forward → g_B
    │
    └── g_group = (g_A + g_B) / 2

10 个 group gradient 的 mean → MI(decay=1.0) → sign → 更新像素。
实际 model views/forwards = 10 × 2 = 20。
默认 10 steps，L_inf budget=16/255。
```

重要实现边界：

- L12 只用于产生评分和 mask，不在 L12 hidden state 中删除 token。
- 所有 mask 最终作用于 L0，前向分支重新运行完整 12 层网络。
- 评分分支 detached；CLS jitter 不直接进入最终 forward 的 CLS。
- `token_score_patch_noise=False` 时，评分 patch 不加噪；前向未删除 patch 仍可加入
  `opponent_channel_gaussian`。
- 方案一的 view B 只使用一个 phase shift；同一个 view 内不展开额外 phase。总实际
  view 数严格为 20，不能按 group 数少报。
- `high/low` 是相对于当前图像 score median 的候选方向，不等同于 foreground/background
  标签。

### 1.1 Phase-pair 的作用

view A 与 view B 看到的是同一张图像的不同 patch-grid 对齐：

```math
g_A \sim W^T\frac{\partial L_A}{\partial z_A},
\qquad
g_B \sim T_s^T W^T\frac{\partial L_B}{\partial z_B}.
```

相同的 ViT patch embedding 在不同输入 phase 下会把不同邻域组合成 patch。pair mean
因此在固定 20-view 预算内同时保留原始/shifted 的 score-mask sample 和两种
patch-grid 对应的像素梯度路径。它不在一个 view 内展开多个 phase，也不增加 EOT 数量。

## 2. 与现有文献的关系

Shi, Yu and Yang, *Vision Transformers Need More Than Registers*（CVPR 2026）提出
Patch Score，即 patch 与 CLS 全局表示的相似度，并将 ViT 的背景主导路由解释为
lazy aggregation：在全局注意力和粗粒度语义监督下，背景 patch 可能吸收并承载全局
语义捷径。

论文给本项目提供的是**问题动机和可观测量**，不是攻击因果的直接证明：

1. 深层 CLS–patch 相似度可以诊断 ViT 的全局 token routing；
2. high-score 区域与背景主导现象相关，且限制全局依赖或选择性聚合会改变该现象；
3. 我们把这个诊断量转化为测试时的 token-routing intervention，而不是训练
   selective aggregation。

本项目的对应假设是：

```text
深层 CLS 路由偏置
    + CLS jitter 导致的 score ranking 变化
    + 随机 token masking
    + 多 copy 梯度聚合
    → 更丰富且更不依赖单一源模型路由的攻击梯度
```

因此主张应写成“Patch-Score-conditioned stochastic routing intervention”，不应
写成“我们准确识别并删除了背景区域”。只有在 bbox/Point-in-Box/AUROC 验证完成后，
才能进一步讨论 score 与语义前景的对应关系。

## 3. 为什么使用 L12 评分

L0 score 使用尚未经过 self-attention 的输入 token 几何；L12 score 则是图像相关的
深层 CLS–patch 关系，与论文的 Patch Score 定义更接近。L12 评分分支为：

```math
x^{(12)} = B_{12}(x^{(0)}),
\qquad s_i = \cos\left(x^{(12)}_{patch,i},
                         x^{(12)}_{CLS}+\sigma\epsilon\right).
```

L12 score 不意味着在 L12 删除 patch。我们采用 detached deep-score / early-injection
设计：深层关系负责决定 mask，L0 注入负责让干预影响整个后续计算图并保留像素梯度。

## 4. CLS jitter 的作用

CLS jitter 只存在于评分分支，但它会经过离散 mask 影响前向梯度：

```text
CLS jitter
  → cosine score 改变
  → patch 是否跨越 median 改变
  → high/low candidate set 改变
  → L0 drop mask 改变
  → forward loss 和 gradient 改变
```

因此 jitter 的直接作用不是改变 logits，而是增加不同 copy 的 token-routing diversity。
当前结果显示，无 jitter 时 L12-high/L12-low 差距很小；有 jitter 时差距扩大。这说明
真正的机制候选是 **Patch Score 与 stochastic mask/EOT 的交互**，而不是纯 Patch Score
排序本身。

## 5. Opponent-channel noise

token dropout 负责 ViT 路由干预；opponent-channel noise 是独立的跨架构梯度多样化
模块，不应被描述成“CNN 没有背景捷径”。CNN 同样可能利用背景相关性，只是没有 CLS
全局 token 路由。

像素通道协方差为：

```math
C_{opp}=
\begin{bmatrix}
1.00&-0.25&-0.25\\
-0.25&1.00&-0.25\\
-0.25&-0.25&1.00
\end{bmatrix}.
```

其亮度方向方差为 0.5，两个色度方向方差为 1.25。当前证据支持它作为 CNN transfer
的经验性增强，但“第一层滤波器比例”和“色度特征更通用”等解释仍需独立滤波器统计
和梯度频谱实验验证。

## 6. 当前结果（100 images）

共同配置：ViT-B/16 white-box、16/255、10 steps、20 copies、约 15% token masking、
opponent-channel noise；overall 为 7 个 ViT 与 4 个标准 CNN 的平均 ASR。

| 评分配置 | Overall | ViT | CNN |
|---|---:|---:|---:|
| **方案一：L12 high + phase pair 10×2** | **77.64%** | **80.86%** | **72.00%** |
| L0 learned CLS, low | 70.73% | 75.29% | 62.75% |
| L0 Gaussian CLS, low | 69.64% | 74.14% | 61.75% |
| L0 random 15% | 68.91% | 73.57% | 60.75% |
| 真正 L12, high + CLS jitter | 70.00% | 74.29% | 62.50% |
| 真正 L12, low + CLS jitter | 64.91% | 69.14% | 57.50% |
| 真正 L12, high，无 score noise | 68.45% | 73.29% | 60.00% |
| 真正 L12, low，无 score noise | 67.55% | 71.71% | 60.25% |

方案一逐模型结果：

| 模型 | ASR |
|---|---:|
| LeViT-256 | 76% |
| PiT-B | 82% |
| DeiT-B | 82% |
| TNT-S | 81% |
| ConViT-B | 79% |
| Visformer-S | 79% |
| CaiT-S/24 | 87% |
| Inception-v3 | 71% |
| Inception-v4 | 71% |
| Inception-ResNet-v2 | 73% |
| ResNet-101 | 73% |

相对于真正 L12-high + CLS jitter 的 70.00% 基线，方案一提升：

```text
Overall: +7.64pp
ViT:     +6.57pp
CNN:     +9.50pp
```

逐模型增益为：LeViT `+10pp`、PiT-B `+11pp`、DeiT-B `+10pp`、TNT-S `+10pp`、
ConViT-B `+11pp`、Visformer-S `+11pp`、CaiT-S/24 `-1pp`、Inception-v3 `+11pp`、
Inception-v4 `+6pp`、Inception-ResNet-v2 `+14pp`、ResNet-101 `+14pp`。

结论边界：

- 方案一是当前最高 ASR 和默认主线；它相对单 phase L12-high 的提升来自 phase-pair
  输入路径与 routing/mask diversity 的组合，不能归因于 score mask 单独变化。
- 无 jitter 时 high/low 仅差 0.91pp；不能声称纯 Patch Score 本身造成主要增益。
- L0 learned 比 Gaussian CLS 高约 1.09pp，说明 learned direction 有弱但可测的作用，
  但不能把 L0 CLS 宣称为已验证的背景语义先验。
- CaiT-S/24 是唯一未提升的标准 ViT（88%→87%），可能已接近该预算下的 ASR 上限，
  但需要更多样本和 paired bootstrap 才能下结论。
- 100 张图的差异尚未做 paired bootstrap/significance，不报告“显著优于”。

## 7. 必须保留的消融和诊断

后续论文表格至少保留：

1. L12-high / L12-low / random mask；
2. CLS jitter on/off；
3. learned CLS / Gaussian CLS；
4. token mask、opponent noise、EOT copies 的逐项消融；
5. mask Jaccard、median crossing rate、copy-to-copy gradient cosine；
6. 有 bbox 时报告 Point-in-Box/AUROC，验证 score 与语义区域的关系。

最关键的因果诊断是：固定 mask 只改变 score query，或固定 score query 只改变 mask
抽样；如果 ASR 差异随 mask flip rate 和 gradient cosine 的变化同步，才能把 CLS jitter
解释为 routing diversity，而不是未控制的随机种子效应。

## 8. 下一阶段：单白盒继续推高 ASR

约束保持不变：不 ensemble 多个 white-box。优化重点是以下两条主线。

### 8.1 Data augmentation

- 以方案一 `10 groups × 2 views` 作为固定预算 baseline；任何新增强必须保持实际
  view 数 `<=20`。
- 在 phase-pair 内比较固定 shift、shift set 和 pair composition，但不在单个 view 内
  展开隐藏 phase。
- kept-patch 的新噪声只能写入 `drop_mask=False`；dropped patch 保持 zero。
- 不把 adaptive schedule、平滑或额外 hidden forward 当作默认方向；先测试结构化
  phase/Jacobian 变化是否带来可复现的 transfer 增益。

### 8.2 新梯度方法（不增加 white-box 数量）

目标不是简单增加 copies，而是改变多路由梯度的聚合规则。候选方向：

- **routing-consensus gradient**：按 mask flip/route stability 对 copy 梯度加权；
- **conflict-aware gradient projection**：保留跨 view 一致方向，投影掉只在单一路由出现
  的冲突分量；
- **sign-space robust aggregation**：在 sign 或 quantized gradient 空间做 coordinate-wise
  median/trimmed vote，避免少数 mask 主导 MI；
- **two-band gradient transport**：分别聚合 ViT token-有效子空间与 CNN 高频/色度子空间，
  再做受控旋转，而不是无约束相加；
- **cross-step routing memory**：记录历史 mask/gradient agreement，抑制只在当前 step
  偶然出现的方向。

每个新方法必须与方案一的 pair mean + MI 对比，并报告 source loss、per-view/group
gradient cosine、mask diversity、gradient effective rank、ViT ASR、CNN ASR 和
overall ASR。下一阶段的成功标准是：在单一 ViT-B/16 white-box 下，超过方案一的
`77.64%` overall ASR，而不是通过增加 white-box 模型数量或实际 view 数获得收益。
