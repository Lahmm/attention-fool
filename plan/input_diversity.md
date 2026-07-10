# Input Diversity Plan

## Hard Budget

所有增强实验按**实际 model view/forward 数量**计数，而不是按外层 group 或 wrapper
数量计数。总 view 数必须满足：

```text
total_views <= 20
```

因此：

```text
20 groups × 1 view = 20 views
10 groups × 2 views = 20 views
5 groups × 4 views  = 20 views
```

不能把一个包含多个内部 view 的 group 报告成一个增强样本。

## 方案一：10 Groups × 2 Views

### 目标

在不超过 20 个实际增强 view 的前提下，同时获得：

1. 不同 patch-embedding Jacobian 的像素梯度；
2. 多个 CLS/mask 路由下的 EOT 梯度；
3. 与当前 20-copy 单 view baseline 可公平比较的总计算预算。

### View 分配

每个 group 包含两个真实 forward view：

```text
view A: 原始 patch embedding / 原始 patch-grid phase
view B: shifted/phase patch embedding
```

建议的 view B 使用小幅整数像素 phase shift，例如：

```text
shift ∈ {(4, 4), (8, 8), (12, 12)}
```

具体 phase 集合需要在正式实验前固定，不能在同一个 view 内再展开多个隐藏 phase，
否则实际 view 数会超出预算。

### 每个 view 的攻击流程

每个 view 独立执行完整的攻击前向流程：

```text
input view
  -> patch_embed + pos_embed + norm_pre
  -> detached L12 CLS/patch scoring
  -> CLS jitter（如启用）
  -> Patch Score mask
  -> L0 token injection
  -> full 12-block forward
  -> CE loss
  -> pixel gradient
```

view A 与 view B 不共享 forward logits，也不把 B 的 patch embedding 隐藏在 A 的
token noise 中。这样每一个 view 都能明确计入 20-view budget。

### 梯度聚合

同一个 group 的两个 view 先做 pair mean：

```math
g_{group} = \frac{1}{2}(g_A + g_B).
```

10 个 group 再做最终 EOT 平均：

```math
\bar g = \frac{1}{10}\sum_{j=1}^{10} g_{group,j}
       = \frac{1}{20}\sum_{j=1}^{10}(g_{A,j}+g_{B,j}).
```

随后沿用当前 MI 更新：

```text
20-view gradient mean
  -> MI decay=1.0
  -> sign
  -> L_inf constrained pixel update
```

如果后续使用 pair-aware gradient weighting，必须先报告 plain mean 结果作为基线；
不能通过不等权聚合隐式改变 view budget 或比较口径。

## 与当前 baseline 的公平对照

### Baseline A：20 Groups × 1 View

```text
20 groups
每个 group 一个原始 patch-grid view
总 view 数 = 20
```

该配置保留 20 个独立的 CLS jitter/mask sample，拥有更高的 routing-mask diversity，
但每个 view 只有一个 patch embedding Jacobian。

### 方案一：10 Groups × 2 Views

```text
10 groups
每个 group 一个原始 view + 一个 shifted view
总 view 数 = 20
```

该配置将一半的 group 数换成第二种 patch embedding Jacobian。它可能扩大像素梯度
的联合子空间，但独立的 CLS/mask routing sample 从 20 个降为 10 个。

两者必须固定以下条件：

```text
white-box model
epsilon / step size / steps
L12 score layer
score noise setting
mask ratio and high/low policy
token noise mode
MI setting
实际总 view 数 = 20
```

## 主要假设

方案一检验的假设是：

> 在固定 20 个实际 view 的预算下，10 组双 view 的 Jacobian diversity 带来的收益，
> 是否超过 20 组单 view 的 routing-mask diversity 损失。

如果两个 view 使用不同的 patch-grid phase，则单个 group 的梯度可能包含不同的
像素到 token 映射方向：

```math
g_{A,j} \sim (W^{(0)})^T\frac{\partial L_A}{\partial z_A},
\qquad
g_{B,j} \sim (W^{(s_j)})^T\frac{\partial L_B}{\partial z_B}.
```

方案一不保证突破 patch embedding 的信息上限；它只是用 20 个真实 view 在两种
多样性之间重新分配预算。是否扩大有效梯度子空间必须通过梯度谱和 transfer ASR
验证，不能仅凭 token-space noise 的协方差推断。

## 实验矩阵

第一轮只比较以下两项，避免同时改变多个因素：

| Branch | Groups | Views/group | Total views | View B | 聚合 |
|---|---:|---:|---:|---|---|
| `single_view_20` | 20 | 1 | 20 | none | mean over 20 |
| `phase_pair_10x2` | 10 | 2 | 20 | shifted phase | pair mean, then group mean |

第二轮再分别加入：

```text
CLS jitter on/off
high-score / low-score mask
opponent-channel noise on/off
phase shift magnitude
```

每个实验仍需保存完整 attack parameters，并在输出目录和 CSV 中显式记录：

```text
num_groups
views_per_group
total_views
phase_shift
pair_aggregation
```

## 必须报告的指标

```text
overall ASR
ViT ASR
CNN ASR
source-model CE
per-view gradient cosine
within-group A/B gradient cosine
gradient covariance effective rank
mask Jaccard / mask flip rate
wall-clock time
peak memory
```

重点观察：

1. `phase_pair_10x2` 是否提高 ViT transfer；
2. CNN transfer 是否因减少独立 mask sample 而下降；
3. 梯度 effective rank 是否上升；
4. pair mean 是否明显降低 view-to-view gradient variance；
5. 总 view 数固定为 20 时，收益是否仍然存在。

## 预算纪律

- 一个真实模型 forward 视为一个 view。
- 在一个 view 内计算多个 phase embedding，按多个 view 计数。
- 不允许用“10 groups”掩盖每组 2 个以上的实际 view。
- 不允许为了提高 ASR 偷增 view 数、白盒模型数量或隐藏额外的 phase forward。
- 所有新梯度聚合方法先与 plain 20-view mean 比较，再讨论方法收益。
