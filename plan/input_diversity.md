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

### Implementation Status

方案一目前**尚未实现**，当前代码仍使用原有的 `guide_aug_copies` 单 view 调度。
本节是待实现的工程方案；后续 coding agent 必须先完成方案一，再运行方案二、三、四
的对照实验。方案一不能只作为文档中的理论 baseline，必须有真实的：

```text
10 groups
每组 2 个独立 model views
实际 forward 总数 = 20
pair mean -> group mean -> 现有 MI/sign update
```

方案一的实现完成前，任何方案二/三/四的 ASR 结果都不能宣称是在完整 input-diversity
预算协议下取得的改进。

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

## 方案二、三、四：实现与优先级

方案一、方案二、方案三和方案四均属于**待实现方案**。其中方案一是必须先完成的
固定 20-view 双 view 基线；方案二、三、四必须在方案一完成并通过验收后再实现或
进行 ASR 比较。

以下方案都必须复用当前主线的 score/mask 流程，不得改变 score 或 mask 的定义。

```text
L12 detached CLS/patch feature
  -> 可选 CLS score jitter
  -> cosine Patch Score
  -> 当前 median candidate policy
  -> 当前 random mask sampling
  -> 得到 drop_mask
  -> 仅对 kept patch（drop_mask=False）做新注入
  -> dropped patch 保持 zero
  -> full 12-block forward
```

硬约束：

- 不修改 `feature_layer=12`、cosine score、median threshold、high/low candidate
  定义、`patch_dropout_ratio` 或 mask sampling 逻辑。
- 所有新 noise/residual/rotation 只能写入 `drop_mask == False` 的 patch。
- `drop_mask == True` 的 patch 在 `zero_noise` 主线中必须严格为零，不能加噪、回填或
  接收 donor token。
- 方案二和方案四使用 `20 groups × 1 view = 20 views`。
- 方案三使用 `10 groups × 2 views = 20 views`。
- 不允许在单个 view 内计算多个隐藏 phase、多个 donor forward 或额外 white-box。

### 方案二：Cross-Patch Counterfactual Transport

#### 目标

在不增加 view 数的情况下，把 kept patch 的 token residual 从其它空间位置搬运过来，
改变 ViT 的 patch-content / spatial-position / CLS-routing 对应关系。它不是 iid
Gaussian noise，也不改变 score/mask 产生过程。

#### 配置

```text
groups = 20
views_per_group = 1
total_views = 20
transport_mode ∈ {rotate180, mirror_x, checkerboard}
transport_alpha ∈ {0.10, 0.20, 0.30, 0.50}
transport_scope = kept_only
```

第一版固定一个 permutation `pi`，不能对同一个 view 计算多个 permutation。推荐先做
`rotate180`，即对 14×14 patch grid 做 180° index rotation。

#### Token 计算

评分和 mask 完成后，令 `z` 为 L0 patch tokens，`K = ~drop_mask`：

```python
transport_residual = z[:, pi, :] - z
transported = z + alpha * transport_residual
patch_tokens = torch.where(
    K.unsqueeze(-1),
    transported,
    torch.zeros_like(z),
)
```

注意：上式中的 `z[:, pi, :]` 来自同一个 L0 forward 的 token tensor，不是另一个
model view。它不会增加 view 数。只有 `K` 位置可以得到 transported token；dropped
位置必须是 zero。

建议对 residual 做 per-image RMS matching，避免仅因为幅度变大而取得虚假收益：

```python
residual_rms = rms(transport_residual[K])
base_rms = rms(z[K])
transport_residual = transport_residual * (base_rms / residual_rms.clamp_min(eps))
```

RMS matching 只用于 kept patch；不得用 dropped patch 统计量。

#### 预期机制

```text
kept token i receives residual from pi(i)
  -> Q/K/V sees altered content-position correspondence
  -> CLS attention routing changes
  -> pixel gradient includes cross-patch dependency
```

它不保证扩大单个 patch 的 `W^T` 行空间，但可以改变 ViT 的全局 token routing，且不
牺牲 20-view 的 mask diversity。

#### 必须对照

```text
baseline: kept patch + current opponent-channel noise
transport: kept patch + cross-patch residual transport
hybrid: kept patch + transport + current opponent-channel noise
```

`hybrid` 仍只能对 kept patch 加 opponent noise；不得给 dropped patch 加任何 noise。

### 方案三：Pair-Difference Gradient

#### 目标

方案三不新增 token noise，而是重新利用方案一的两个真实 view 的梯度差异，提取
phase-sensitive gradient。score 和 mask 在两个 view 中仍各自按当前原流程计算。

#### 配置

```text
groups = 10
views_per_group = 2
total_views = 20
view A = original input phase
view B = one shifted input phase
pair_aggregation = difference_mix
lambda_difference ∈ {0.05, 0.10, 0.20}
```

每个 view 都独立完成：

```text
L12 score -> CLS jitter -> current mask -> kept-only noise -> full forward -> gradient
```

不允许复用 A 的 score、mask 或 token noise 给 B。

#### 梯度计算

对 group `j` 的两个 view 得到 `g_A,j` 和 `g_B,j`。先计算：

```math
g^+_j = \frac{g_{A,j}+g_{B,j}}{2},
\qquad
g^-_j = g_{A,j}-g_{B,j}.
```

对 `g^-` 做 per-image L2 normalization 后混合：

```math
g_j = g^+_j
    + \lambda\frac{g^-_j}{\|g^-_j\|_2+\epsilon}.
```

最终只对 10 个 `g_j` 做 group mean，然后进入现有 MI/normalization/sign 更新。不得
对每个 group 单独更新像素。

#### 对照顺序

```text
10×2 plain pair mean
10×2 pair-difference, lambda=0.05
10×2 pair-difference, lambda=0.10
10×2 pair-difference, lambda=0.20
```

`lambda=0` 必须与方案一的 plain pair mean 数值一致（浮点误差范围内）。

#### 预期机制

```text
g+ : 两个 embedding view 的共同 transfer direction
g- : 对 embedding phase 敏感的差异 direction
g  : 稳定方向 + 受控 phase-sensitive direction
```

这是 gradient aggregation 变体，不增加 view 数，也不应与 adaptive schedule、TI
smoothing 或额外 forward 混合测试。

### 方案四：Kept-Token Orthogonal Residual

#### 目标

对 kept patch 做固定 channel-space orthogonal residual rotation，改变 ViT 的 token
channel direction，同时保持 token residual 的幅度可控。它不修改 score/mask，也不增加
view 数。

#### 配置

```text
groups = 20
views_per_group = 1
total_views = 20
rotation_mode ∈ {pair_swap, hadamard_block}
rotation_alpha ∈ {0.05, 0.10, 0.20, 0.30}
rotation_scope = kept_only
```

推荐先实现 `pair_swap`：在固定 channel pairs `(0,1), (2,3), ...` 上做交换，并对
交换后的 residual 使用固定符号 pattern。第二版再实现 block-Hadamard rotation。

#### Token 计算

令 `z_K` 为 kept patch token，`mu_K` 为 kept patch 的 per-image channel mean，`R` 为
固定正交矩阵：

```python
centered = z - mu_kept
rotation_residual = centered @ (R.T - I)
rotated = z + alpha * rotation_residual
patch_tokens = torch.where(
    kept_mask.unsqueeze(-1),
    rotated,
    torch.zeros_like(z),
)
```

`mu_kept` 只能由 kept patch 计算；当 kept patch 数不足时必须回退为原始 token，不得
用 dropped patch 填充统计量。`R` 在一个实验分支中固定，不能每个 hidden sub-view
重新采样。

#### 噪声边界

方案四的 `rotation_residual` 只能添加到 kept patch：

```text
kept patch: z + alpha * rotation_residual
dropped patch: exactly zero
```

如果与 opponent noise 做 hybrid，顺序固定为：

```text
z_kept = z + alpha * rotation_residual
z_kept = z_kept + opponent_noise
z_dropped = 0
```

不得对完整 patch tensor 先加 rotation/noise，再通过后处理把 dropped patch 清零而不
记录该中间注入；实现上应直接使用 kept mask，保证实验日志和 autograd 路径清晰。

#### 预期机制

```text
fixed orthogonal channel residual
  -> 改变 Q/K/V 投影看到的 channel direction
  -> 改变 ViT attention routing
  -> 产生不同但不完全随机的 kept-token gradient
```

该方案不声称突破 `W^T` rank ceiling；它的价值是低成本地改变 ViT-specific token
geometry，同时保持 20 个独立 mask sample。

## 三个方案的优先级

最终优先级按“只看 ASR、严格 20-view budget、单白盒”确定：

### P0：方案二 Cross-Patch Counterfactual Transport

理由：

- 保留 `20 groups × 1 view` 的完整 mask/CLS-jitter diversity；
- 不增加真实 view 数；
- 直接干预 ViT 的跨 patch routing；
- 不需要修改 score/mask 主线；
- `rotate180` 版本实现和调试成本低。

首轮只跑：

```text
alpha = 0.10, 0.20, 0.30
mode = rotate180
```

### P1：方案四 Kept-Token Orthogonal Residual

理由：

- 同样保持 `20 groups × 1 view`；
- 不牺牲 mask diversity；
- 实现成本低，适合快速筛选；
- 直接改变 ViT token channel geometry。

首轮只跑：

```text
rotation_mode = pair_swap
alpha = 0.05, 0.10, 0.20
```

它不如方案二直接改变空间 routing，因此排在 P0，但不应被跳过。

### P2：方案三 Pair-Difference Gradient

理由：

- 能利用方案一的第二 view，而不是简单 pair mean；
- 是新的梯度聚合方法，不增加 view 数；
- 但它把独立 mask sample 从 20 降为 10，方差风险更高；
- 需要先有方案一的 `10×2 plain pair mean` 作为稳定基线。

首轮只跑：

```text
lambda = 0.05, 0.10
```

## 统一实验顺序与验收

所有方案严格按以下顺序执行：

```text
1. 复现当前 baseline，确认 actual_forward_view_count=20
2. 运行方案二，不叠加方案三/四
3. 运行方案四，不叠加方案二/三
4. 运行方案一 plain pair mean
5. 运行方案三 pair-difference
6. 只对单项最优方案做 hybrid ablation
```

每个 branch 必须报告：

```text
overall / ViT / CNN ASR
source CE
per-view gradient cosine
within-group gradient cosine（方案三必须）
gradient effective rank
mask Jaccard / flip rate
actual_forward_view_count
wall-clock / peak memory
```

在所有方案中，score 和 mask 的实现必须保持完全一致；唯一允许改变的是：

```text
方案二：kept-only cross-patch residual
方案三：view grouping 和 gradient aggregation
方案四：kept-only channel rotation residual
```

## 方案一增量对照：Original-Score / Post-Dropout Phase Shift

### 定义

这是方案一的增量对照，不替换方案一，也不改变方案一的主线定义。新方案检验：

> 如果 score 和 drop mask 始终来自原图，phase shift 只发生在 dropout 之后，
> 仍然能否获得方案一的 ASR 增益？

流程顺序固定为：

```text
original pixels
  -> detached L12 Patch Score
  -> CLS jitter（如启用）
  -> original-image high-score mask
  -> pixel-space patch dropout
  -> view A: dropped image
  -> view B: phase_shift(dropped image)
  -> forward loss / gradient
```

预算仍然是：

```text
10 groups × 2 views = 20 actual views
```

### 与方案一的差别

| 项目 | 方案一主线 | 增量对照 |
|---|---|---|
| score 来源 | 每个 view 自己计算 | 始终使用原图计算 |
| mask 数量 | 20 个 view 各自生成 mask | 每个 group 一个原图 mask，共 10 个 |
| dropout 位置 | L0 token | pixel-space patch |
| phase shift 位置 | score/mask 之前 | dropout 之后 |
| view B 的 score | 重新计算 | 不重新计算 |
| 文献 Patch Score 对齐 | phase-conditioned | 原图 Patch Score 对齐更严格 |

### 为什么必须使用 pixel-space dropout

“dropout 后再对图像做 shift”要求 dropout 发生在图像空间：

```python
image_mask = patch_mask_to_image_mask(drop_mask, patch_size=16)
dropped_pixels = pixels * (1.0 - image_mask)
view_a = dropped_pixels
view_b = apply_phase_shift(dropped_pixels, dx, dy)
```

不能先执行 L0 token zero 再声称对图像做 shift；token 已经经过 patch embedding，无法
无损还原为可平移的 pixel image。因此该分支与方案一的 L0 token dropout 是不同的
intervention，必须作为独立实验记录。

### Score 与 mask 约束

每个 group 只在原图上计算一次 score/mask：

```text
score_image = original pixels
score_layer = 12
score_cls = original L12 CLS (+ optional CLS jitter)
score_patch = original L12 patches
candidate/mask = current median + current high/low + current random sampling
```

view A 和 view B 共享该 group 的原图 mask。不得对 shifted image 重新计算 score，
不得为 view B 重新采样 candidate mask，否则就退化回方案一主线。

### Phase shift 约束

只允许对已经 dropout 的图像执行一次 phase shift：

```text
view A = dropped_pixels
view B = differentiable_reflect_shift(dropped_pixels, phase)
```

phase 从现有集合中每个 group 选择一个：

```text
(4,4), (8,8), (12,12)
```

同一个 view 内不得展开多个 phase。`view B` 的 mask 也可以同步平移：

```python
mask_b = apply_phase_shift(image_mask, dx, dy)
```

如果实现同步 mask，必须保证 view B 的 dropped/kept 判定来自同一个原图 mask 的
空间变换，而不是重新 score 或重新 sample。

### Kept-patch noise 约束

第一轮建议先关闭 token-level opponent noise，隔离 post-drop phase shift 的效果：

```text
pixel dropout: enabled
token opponent noise: disabled
```

第二轮才加入 opponent noise，并严格限制为 kept patch：

```text
view A:
    dropped image regions = zero
    kept token positions = original token + opponent noise

view B:
    shifted dropped image regions = zero
    shifted kept token positions = original token + opponent noise
```

任何新 noise 只能写入 `drop_mask=False` 的 kept 区域；dropped 区域必须保持 zero。不能
先对完整 token tensor 加噪、再事后清零 dropped token 后忽略该中间注入。

### 梯度聚合

每个 group 的两个 view 都是独立 forward，但共享原图 score/mask：

```math
g_{group} = \frac{1}{2}(g_{A}+g_{B}),
\qquad
\bar g = \frac{1}{10}\sum_{j=1}^{10}g_{group,j}.
```

最终继续使用现有 MI、normalization 和 sign update。不得对每个 group 单独更新像素。

### 该对照回答的问题

```text
方案一：phase shift 是否通过改变 score/mask + forward 同时获益？
增量对照：固定原图 score/mask 后，phase shift 只改变 dropout 后的 forward，是否仍有效？
```

结果解释：

- 若增量对照接近方案一，phase/Jacobian 变化可能是主要收益来源；
- 若增量对照显著低于方案一，方案一的收益很可能依赖 shifted view 重新计算 score/mask
  后产生的 routing diversity；
- 若 pixel-space dropout 本身改变结果，必须单独报告 `view A only`，不能把收益全部
  归因于 phase shift。

### 实验分支与优先级

方案一增量对照的优先级低于当前方案一主线，但高于方案二/三/四的混合实验：

```text
P0: plan1_phase_pair_10x2_mean（当前主线）
P0.5: original_score_pixel_drop_then_phase_pair
P1: 方案二 Cross-Patch Transport
P2: 方案四 Kept-Token Rotation
P3: 方案三 Pair-Difference Gradient
```

第一轮只跑：

```text
original score + pixel dropout + view A/B pair mean
token opponent noise = off
total views = 20
```

第二轮再比较：

```text
token opponent noise = on, kept-only
mask synchronization for view B = on/off
phase shift = (4,4), (8,8), (12,12)
```

所有结果必须与方案一使用相同的 epsilon、steps、white-box、L12 score policy 和
20-view budget，并单独记录：

```text
dropout_space = pixel
score_source = original
phase_order = after_dropout
mask_shared_within_group = true
token_noise_scope = kept_only
actual_forward_view_count = 20
```
