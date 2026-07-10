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

## Agent Implementation Contract

本节是方案一的实现约束。Coding agent 必须按此契约实现，不得用“等价”重构偷偷改变
view 数或比较预算。

### Scope and files

第一版只修改以下边界：

```text
main.py       CLI 参数、attack_params 序列化、参数传递
attack.py     phase view 生成、group/view 调度、梯度日志
tests/        参数校验、phase shift、view budget 的单元测试
```

不要修改 transfer model、黑盒模型列表、默认 white-box 模型或现有
`token_patch_dropout` 的默认语义。旧配置在没有新开关时必须产生相同 forward 数和
相同 loss 路径。

### New explicit configuration

不要复用 `guide_aug_copies` 表示 group 数。新增显式参数，建议命名为：

```text
--input-diversity-groups              default=20
--input-diversity-views-per-group     default=1
--input-diversity-phase-shift         default="0,0"
--input-diversity-phase-shift-set     optional, e.g. "4,4;8,8;12,12"
--input-diversity-pair-aggregation     choices=[mean]  default=mean
```

当启用方案一时，使用：

```text
input_diversity_groups = 10
input_diversity_views_per_group = 2
total_views = 20
```

参数校验必须在构造 attacker 前完成：

```python
if groups <= 0:
    raise ValueError
if views_per_group <= 0:
    raise ValueError
if groups * views_per_group > 20:
    raise ValueError("actual input-diversity views must be <= 20")
if pair_aggregation != "mean":
    raise NotImplementedError
```

不允许把 `groups * views_per_group` 只写入日志而不用于实际调度；日志中的
`total_views` 必须等于实际调用 model forward 的次数。

### Exact group/view lifecycle

每个 attack step、每个 image batch 都按以下顺序执行：

```python
for group_idx in range(num_groups):
    phase = choose_one_phase(group_idx)
    for view_idx in range(views_per_group):
        view_pixels = make_view(
            pixels,
            group_idx=group_idx,
            view_idx=view_idx,
            phase=phase if view_idx == 1 else (0, 0),
        )
        loss = token_patch_dropout_loss(view_pixels, labels)
        grad = autograd(loss, pixels)
        record_view(group_idx, view_idx, grad)

    group_grad = (grad[group_idx, 0] + grad[group_idx, 1]) / 2
final_grad = mean(group_grad for group_grad in groups)
```

当 `views_per_group=1` 时，view A 是唯一 view；当 `views_per_group=2` 时，view 0
必须是原始输入，view 1 必须是一个 phase-shifted 输入。view 0 和 view 1 都必须
独立执行完整的 `token_patch_dropout_loss`，包括：

```text
patch embedding
L12 detached score
CLS score noise（若启用）
score mask sampling
L0 injection
full 12-block forward
CE loss
```

不得复用 view 0 的 L12 token、score、mask 或 logits 给 view 1。不得在 view 0 内部
额外计算多个 phase token。

### Phase shift implementation

phase shift 必须发生在送入模型前的 pixel tensor 上；不能直接修改模型的
`patch_embed` 权重，也不能伪造 token tensor 作为第二 view。

要求：

```text
输入 shape: [B, 3, H, W]
输出 shape: [B, 3, H, W]
shift: integer (dx, dy)
```

边界处理第一版固定为 differentiable reflect padding + crop。禁止 circular wrap，
因为它会把图像右边缘连接到左边缘，产生非自然内容。实现必须满足：

```python
shift == (0, 0) -> pixels 等值返回，不复制并改变 dtype/device
```

正负 shift 的方向必须在单元测试中明确：对一个单点图案检查输出位置；不能只凭
函数名猜方向。phase shift 只对 view 1 应用，view 0 不得经过隐式 resize/crop。

### Randomness and reproducibility

每个 `(attack_step, batch_index, group_idx, view_idx)` 必须有可追踪的随机状态。至少
记录：

```text
base_seed
group_idx
view_idx
phase_shift
```

如果 phase 从 `phase_shift_set` 随机选取，每个 group 只能选一个 phase；不能为同一
group 的 view 1 再采样多个 phase。CLS jitter、mask sampling、token noise 仍按每个
view 独立采样。

调试模式下必须支持固定 seed，使相同输入、相同参数产生相同 view phase、mask 和 loss。

### Mask and noise semantics

方案一不改变当前 token patch dropout 的语义：

```text
L12 score branch: detached
score mask: 每个 view 独立生成
dropped patch: zero_noise fill 时置零
kept patch: 按现有 patch_dropout_noise_mode 加噪
```

phase shift 只是改变 view 的输入和其 L12 score；它不是额外的 score noise，也不能
把 phase residual 藏进 kept-patch token noise。若未来实现 kept-patch phase residual，
必须作为新的真实 view 计数，除非明确取消第二个 forward 并重新定义预算。

### Gradient aggregation contract

第一版只允许等权聚合：

```python
group_grad = sum(view_grads) / views_per_group
final_grad = sum(group_grads) / num_groups
```

对 `10 × 2` 而言，这与 20 个 view 的等权 mean 数值等价，但必须保留 group/view
维度的日志，以便计算 within-group A/B gradient cosine。不能在第一版加入 loss weighting、
gradient clipping、top-k view selection 或 conflict projection；这些属于后续新梯度
方法，必须使用同一 view budget 单独做 ablation。

MI、gradient normalization、TI、sign update 的执行位置保持现有实现不变：先完成
全部 20 个 view 的聚合，再进入现有的 normalization/momentum/sign 流程。不得对每个
group 单独更新像素。

### Logging and metadata

输出的 attack parameter JSON 和 transfer CSV 必须包含：

```text
input_diversity_enabled
input_diversity_groups
input_diversity_views_per_group
input_diversity_total_views
input_diversity_phase_shift_set
input_diversity_pair_aggregation
actual_forward_view_count
```

`actual_forward_view_count` 必须由调度器递增，而不是由配置推导后直接填写。攻击结束
时断言：

```python
actual_forward_view_count == groups * views_per_group
actual_forward_view_count <= 20
```

### Required tests

至少新增以下测试：

1. `20 × 1`、`10 × 2`、`5 × 4` 均通过 budget validation，`11 × 2` 必须失败。
2. `phase_shift=(0,0)` 输出与输入数值等值，shape/dtype/device 保持一致。
3. 非零 phase shift 的单点图案位置符合约定，且输出尺寸不变。
4. `10 × 2` 恰好调用 20 次 view loss/forward；不能调用 21 次或更多。
5. 每个 group 恰好有一个原始 view 和一个 shifted view。
6. view 0/view 1 的 score、mask、CLS jitter 随机状态相互独立。
7. pair mean 后再 group mean 与 20-view plain mean 数值一致（浮点容差内）。
8. 未启用 input diversity 时，旧 token patch dropout 流程和参数默认值不变。

### Acceptance criteria

方案一只有在以下条件全部满足后才能进入 ASR 长跑：

```text
代码能通过全部 required tests
实际 view 数严格为 20
没有额外 white-box model
没有隐藏 phase forward
view 0/view 1 都执行完整 score -> mask -> L0 injection -> full forward
CSV/JSON 能复现 groups/views/phase/aggregation
single_view_20 与 phase_pair_10x2 使用完全相同的 epsilon、steps 和 MI
```

第一轮只比较：

```text
single_view_20
phase_pair_10x2
```

在这两个 branch 的 source loss、ViT/CNN/overall ASR、gradient cosine、effective rank
和 wall-clock 数据齐全前，不得继续叠加新的 gradient aggregation 方法。
