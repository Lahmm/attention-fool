# 梯度幅值、频率与联合组成实验

## 结论先行

在当前 `original_score_postdrop_phase_pair` 主线、20 actual views、L12 score、现有 mask/phase/noise、MI、10 steps 和 epsilon `16/255` 全部不变的条件下，尚未找到满足目标的梯度处理方法。

最好的候选是弱 Gaussian 梯度叠加：

```text
g' = g + 0.25 * GaussianBlur(g, sigma=1)
```

它在三个 100 样本 seed 上的平均变化为：

| 指标 | mean 主线 | Gaussian blend | 变化 |
|---|---:|---:|---:|
| Overall，11 黑盒 | 80.76% | 81.15% | +0.39pp |
| ViT 平均 | 83.48% | 83.95% | +0.48pp |
| CNN 平均 | 76.00% | 76.25% | +0.25pp |
| 白盒 ViT-B/16 | 89.67% | 90.33% | +0.67pp |

它远低于要求的 ViT `+3–5pp`，且第三个 seed 的 Overall 为 `-0.36pp`。因此主线仍应保留 `gradient_postprocess=mean`。

## 幅值信息的具体表现

现有 `_normalize_grad` 会消除每一步的全局幅值，只保留坐标之间的相对幅值。相对幅值通过 MI 累积进入后续 step 的方向，因此最终 sign update 仍然会受到幅值分布影响。

观察结果显示：

- `agg_abs_mean` 对早期 ViT 迁移有较强正相关，step 2 的 discovery/validation 相关约为 `+0.57/+0.56`。但它是全局尺度，随后会被 mean-absolute normalization 消除，不能直接通过整体放大获得收益。
- `spatial_top5_energy` 与 ViT 迁移呈负相关，说明过度集中的大能量峰值通常不利。
- 但删除最大 5% 坐标在 30 样本上使 ViT 下降约 `3.81pp`，CNN 下降约 `5.83pp`；删除最低 20% 坐标也使 ViT 下降约 `2.38pp`。所以“大幅值有害”或“小幅值无效”都不成立：高幅值方向本身包含有效攻击信号，不能简单裁剪。
- 幅值幂变换 `sign(g)|g|^p` 在 `p=1.25/1.5` 和 `p=0.75/0.90` 均未提升 ViT。连续压缩或放大都改变了源模型特异的方向比例。
- 坐标 Wiener 信号/噪声缩放、group norm 均衡、group reliability 软权重也没有产生 ViT 增益。

因此当前最可信的幅值结论是：有效信息不在某一个绝对幅值区间，而在“幅值与空间/频率方向共同形成的相对结构”中；只按幅值排序或全局幂变换无法利用它。

## 频率信息的具体表现

之前的观察分析发现 `freq_high` 与 Overall 迁移负相关，且 step 0–4 对 ViT 的负相关最明显，step 2 discovery/validation 约为 `-0.56/-0.27`。

因果探针结果表明：

- 全程删除高频：首次 100 样本 probe 为 Overall `+0.82pp`，但 ViT 仅 `+0.14pp`，三 seed 确认后不稳定。
- 高频增益减半、早期 step 删除高频、只在高频做 Wiener 抑制，均未产生稳定 ViT 增益。
- 低频叠加在修正参数解析后反而降低 ViT，之前看似正向的低频结果是 cutoff/strength 传反造成的无效结果，已作废。
- 对高频 Wiener 残差做反向放大也降低 ViT，说明“Wiener 抑制提升 CNN”不能简单反向解释为“残差帮助 ViT”。

所以“低频有益、高频有害”只是一阶统计描述，不足以作为逐频带因果规则。高频内部同时包含有害和必要的细节方向，硬删除会损害 ViT 的跨模型共享方向。

## 幅值与频率的关系

本轮共筛选 185 个 30 样本候选，覆盖：

- 幅值分位删除、极值裁剪、幅值幂放大/压缩；
- 坐标和 group Wiener/可靠性/幅值均衡；
- 保留原始幅值的跨-view 符号可靠性 boost/gate；
- 跨-view 中心化梯度的 PCA 主方向传输；
- 20-view Gram 矩阵广义最小方差（GLS）组合；
- 固定高频增益、频率 Wiener、频谱幅值幂变换；
- 高频共享/残差成分放大；
- patch/local spatial energy equalization；
- 时间窗口频率处理；
- 低频叠加、Gaussian 平滑叠加及 feature-conditioned Gaussian；
- 频谱复数系数的 circular phase consensus。
- 低频—高频跨尺度协方差替换、投影和 canonical correlation；
- 跨步符号持久性增益；
- Haar 小波 MAD/universal-threshold 高频细节收缩；
- 局部 divisive normalization、软百分位幅值裁剪及跨尺度/Gaussian 组合。
- Tikhonov/Laplacian proximal 低通预条件。
- MI 累积之后、最终 sign update 之前的 Gaussian/Laplacian/高频预条件。
- 低幅值×高频占比条件下的收缩、低频增益、Gaussian、低频-only 和幅值 equalize。
- phase-pair A/B difference transport 及 pair-difference Wiener 噪声估计。
- 基于低幅值、高频占比和跨-group 不一致性的风险分数自适应 Gaussian 组成重构。
- 高频不确定能量向低频方向的 L2 守恒传输；
- log 幅值包络扩散、PCGrad 式 view 冲突投影；
- ViT 14×14 patch-mean 子空间投影及 patch 内 L2 能量传输；
- 仅对高频 Fourier 幅值做幂变换。

关键对照是：

- 全频段 Fourier Wiener 会提高部分 CNN，但降低 ViT；
- Fourier 幅值压缩和放大也偏向 CNN，不能改善 ViT；
- 按 ViT patch 或局部空间能量均衡会提高源白盒的部分结果，却损害黑盒 ViT；
- 未归一化的弱 Gaussian 叠加是唯一在 100 样本上出现正向 Overall/ViT 的组合；把平滑分量强行归一化后，收益消失。

另外，只有对全部样本施加原始 Gaussian 弱叠加才出现弱正向；按低空间熵或高频能量选择样本后再平滑，30 样本上没有稳定收益。说明当前特征更像迁移性的描述变量，不能直接作为样本级因果开关。

这说明幅值与频率不是可以独立相加的两个“好/坏开关”。有效影响来自原始梯度中自然幅值比例和空间频率相位共同形成的方向；改变其中一个而不保持另一个的比例，会破坏迁移。

## 风险分数驱动的组成重构

观察分析中，`group_to_rest_cosine_mean` 与迁移性在 discovery/validation 中分别为 `+0.355/+0.510`。因此新增了一个不使用黑盒标签的风险分数：低 raw amplitude、高 high-frequency fraction、以及低跨-group agreement 分别转换为 batch 内 percentile rank，再取几何平均；风险分数只分配一个弱 Gaussian residual，原始梯度和其全部符号均保留。

六个 30 样本候选结果如下：

| 探针 | Overall Δ | ViT Δ | CNN Δ | 白盒 Δ |
|---|---:|---:|---:|---:|
| risk Gaussian，低幅值×高频，a=0.25 | 0.00pp | -0.95pp | +1.67pp | +3.33pp |
| risk Gaussian，低幅值×高频，a=0.50 | -0.91pp | -2.38pp | +1.67pp | 0.00pp |
| risk Gaussian，再加入低 group agreement，a=0.25 | -0.61pp | -2.38pp | +2.50pp | -3.33pp |
| risk Gaussian，再加入低 group agreement，a=0.50 | -1.52pp | -2.38pp | 0.00pp | 0.00pp |
| risk Gaussian，高频×低 group agreement，a=0.25 | -1.21pp | -1.43pp | -0.83pp | 0.00pp |
| risk Gaussian，高频×低 group agreement，a=0.50 | -1.52pp | -1.90pp | -0.83pp | 0.00pp |

结果否定了“先找出低幅值高频困难样本，再对其施加平滑即可修复迁移”的简单因果解释。该联合特征更像样本难度和源方向质量的标志；把平滑预算集中到它们身上没有改善 ViT，加入 group 不一致性后反而进一步下降。因此这些候选不进入 100 样本确认。

## 幅值×频率交互分析

对已有 100 样本逐样本记录做联合分箱后，发现一个比边际相关更强的规律：低原始幅值且高频能量占比高的样本迁移率最低。这里的幅值是攻击前归一化之前的 `agg_abs_mean`，频率是 `freq_high`；该分析是观察性证据，不等价于因果证据。

Overall 迁移率的三分位联合均值如下，列为低/中/高频占比，行为低/中/高幅值：

| 幅值 \ 高频占比 | 低 | 中 | 高 |
|---|---:|---:|---:|
| 低 | 0.920 | 0.716 | 0.321 |
| 中 | 0.947 | 0.947 | 0.758 |
| 高 | 0.972 | 0.993 | 0.875 |

“低幅值×高频”子集的 discovery Overall 为 `0.455`，其余样本为 `0.941`；validation 中分别为 `0.554` 和 `0.909`。ViT 与 CNN 均保持同方向差异。因此幅值的重要表现不是全局放大，而是它改变了高频组成对最终方向的相对影响：当总幅值低且高频占比高时，梯度更像源模型特异的弱方向。

但条件反事实没有修复这一子集：

| 条件处理 | Overall Δ | ViT Δ | CNN Δ |
|---|---:|---:|---:|
| 高频收缩，a=0.25 | 0.00pp | 0.00pp | 0.00pp |
| 高频收缩，a=0.50 | -0.91pp | -0.95pp | -0.83pp |
| 低频增益，a=0.25 | -1.82pp | -1.90pp | -1.67pp |
| 条件 Gaussian，a=0.25 | 0.00pp | +0.48pp | -0.83pp |
| 条件低频-only，a=0.25 | -2.73pp | -2.86pp | -2.50pp |
| 条件低频-only，a=0.50 | -2.73pp | -2.86pp | -2.50pp |

这说明交互项主要是样本难度/方向质量的标志，而不是可以通过删除高频直接修复的“无效幅值”。两个条件 `low_equalize` 配置因模型权重服务连续返回 HTTP 503，未产生黑盒 ASR，不纳入结论。

## 频谱相位共识补充

为区分“频率幅值”与“跨 view 的频谱相位一致性”，新增了相位共识 probe。对每个频率坐标先取 20 个 view 的复数 FFT，保留平均幅值，并对单位复数相位做 circular mean；随后将共识频谱的整体幅度缩放回原始 view-mean 的幅度。最终与原始平均频谱按 `a025/a050/a075/a100` 混合。该处理只发生在 view 聚合之后，20 views、mask、phase、noise、MI 和更新规则均未改变。

30 样本结果如下；基线是同一 replay manifest 下的 `mean`，Overall 为 11 个黑盒平均，ViT/CNN 为对应架构平均：

| 相位共识强度 | Overall Δ | ViT Δ | CNN Δ | 白盒 Δ | Overall bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| 0.25 | -2.42pp | -3.33pp | -0.83pp | 0.00pp | [-5.76, 0.00]pp |
| 0.50 | -4.85pp | -4.76pp | -5.00pp | 0.00pp | [-9.09, -1.21]pp |
| 0.75 | -4.55pp | -4.76pp | -4.17pp | -3.33pp | [-9.09, -0.91]pp |
| 1.00 | -5.76pp | -4.76pp | -7.50pp | -3.33pp | [-10.30, -1.82]pp |

相位的一致性并不等价于迁移性的一致性。随着共识强度增加，黑盒 ViT 和 CNN 同时下降，且下降具有单调趋势；因此“保留跨 view 共同频谱相位”应从候选方向中排除。它还进一步支持了前面的判断：增强 view 之间的相位差并非纯噪声，其中包含有助于跨架构迁移的方向多样性。

## 跨尺度、时间与幅值联合补充

为直接检验“高频中只有与低频耦合的部分有用”这一假设，使用 20 个 view 的低频—高频联合协方差构造高频 transport 方向，并测试了替换高频、只投影高频残差、跨尺度 canonical correlation 三类变换：

| 探针 | Overall Δ | ViT Δ | CNN Δ | 白盒 Δ |
|---|---:|---:|---:|---:|
| cross-scale replace，c=0.50，a=0.25 | 0.00pp | +0.48pp | -0.83pp | +3.33pp |
| cross-scale replace，c=0.50，a=0.50 | -2.42pp | -3.33pp | -0.83pp | 0.00pp |
| cross-scale project，c=0.50，a=0.50 | -0.30pp | -1.43pp | +1.67pp | 0.00pp |
| cross-scale canonical，c=0.50，a=0.25 | -0.61pp | -2.38pp | +2.50pp | 0.00pp |

弱替换只在白盒和少量黑盒 ViT 上给出离散正向，Overall 没有变化；增强强度后 ViT 下降。因此“低频可预测的高频就是迁移方向”没有得到支持。

跨步符号持久性也未能成为有效性判据：全频增益的 Overall/ViT/CNN 变化为 `-0.30/-0.95/+0.83pp`，仅高频持久性为 `-2.42/-1.90/-3.33pp`。高频的跨步稳定性不等于跨架构迁移性。

Haar 小波的 MAD 阈值实验进一步区分了高频幅值：即使只收缩小幅值 detail，ViT 仍下降 `1.43–2.38pp`；阈值 0.50 时 CNN 下降 `5.00pp`。这说明有害的不是“低幅值高频”这一整体，而是幅值、空间位置和模型方向共同形成的结构。

直接抑制局部幅值峰值同样失败：divisive normalization 的最佳变化为 Overall/ViT/CNN `-0.91/-0.95/-0.83pp`，软百分位裁剪最佳为 `-0.61/-0.48/-0.83pp`。跨尺度替换再叠加弱 Gaussian 后为 `-0.30/-1.43/+1.67pp`，没有出现幅值与频率的协同增益。

Tikhonov/Laplacian proximal 低通的四个强度中，最佳结果为 `lambda=1.0`：Overall/ViT/CNN/白盒变化为 `+0.61/0.00/+1.67/+3.33pp`，ViT 没有提升；其余强度的 ViT 变化为 `-1.90` 至 `-2.86pp`。因此把 Gaussian 低频叠加改写成更强的二阶平滑正则，并没有解决 ViT 瓶颈。

最后增加了一个独立的处理位置：MI 累积状态保持原样，只对当前用于 sign update 的 momentum 做预条件。结果为：

| 后动量处理 | Overall Δ | ViT Δ | CNN Δ | 白盒 Δ |
|---|---:|---:|---:|---:|
| Gaussian，sigma=1，a=0.25 | 0.00pp | 0.00pp | 0.00pp | +3.33pp |
| Gaussian，sigma=1，a=0.50 | -2.12pp | -2.86pp | -0.83pp | 0.00pp |
| Laplacian，lambda=0.50 | -1.82pp | -3.33pp | +0.83pp | 0.00pp |
| Laplacian，lambda=1.00 | 0.00pp | -0.48pp | +0.83pp | +3.33pp |
| 高频收缩，a=0.25 | -0.61pp | -1.90pp | +1.67pp | 0.00pp |

因此高频危害既不是简单的 view 聚合错误，也不是 MI 累积后才出现的 sign 噪声；它与源模型梯度的空间结构共同决定最终迁移方向。

## Phase-pair difference 补充

主线的 20 个 view 实际上是 10 个共享 dropout mask 的 A/B phase pair。令每个 pair 的均值为 `m=(g_A+g_B)/2`，差分为 `d=(g_A-g_B)/2`，新增实验直接测试了 `m±αd`，以及把 `d²` 作为相位噪声的 Wiener shrink：

| pair 处理 | Overall Δ | ViT Δ | CNN Δ | 白盒 Δ |
|---|---:|---:|---:|---:|
| `m + 0.25d` | -3.33pp | -4.29pp | -1.67pp | 0.00pp |
| `m + 0.50d` | -3.94pp | -3.81pp | -4.17pp | +3.33pp |
| `m - 0.25d` | -0.30pp | -1.43pp | +1.67pp | 0.00pp |
| `m - 0.50d` | -2.12pp | -0.95pp | -4.17pp | 0.00pp |
| `m - 0.25 proj_d(m)` | -1.21pp | -3.33pp | +2.50pp | 0.00pp |
| `m - 0.50 proj_d(m)` | -2.12pp | -3.33pp | 0.00pp | 0.00pp |
| 全频 pair-Wiener，floor=0.25 | -1.52pp | -0.95pp | -2.50pp | 0.00pp |
| 全频 pair-Wiener，floor=0.50 | +0.30pp | 0.00pp | +0.83pp | +3.33pp |
| 高频 pair-Wiener，floor=0.25 | -0.91pp | -1.43pp | 0.00pp | 0.00pp |
| 高频 pair-Wiener，floor=0.50 | -1.52pp | -1.43pp | -1.67pp | 0.00pp |

进一步测试了数学上更保守的正交化：只从 `m` 中删除沿 `d` 的投影，即 `m' = m - α proj_d(m)`，不触碰与 phase difference 正交的方向。α=0.25/0.50 都使 ViT 下降 3.33pp，Overall 分别下降 1.21/2.12pp；这说明问题不只是把差分方向直接加回或减去，连“仅删除差分对齐分量”也会损失 ViT 所需的有效方向。phase difference 确实包含源模型特异成分，但它同时与有效方向纠缠；A/B 均值仍是当前最稳妥的 phase 处理。仅用 `d²` 做统计 shrink 只能改善 CNN，不能提升 ViT。

## Gaussian 三 seed 结果

候选为 `gaussian_blend_s10_a25`，每个 seed 都使用同 seed baseline 的完全一致 replay manifest。

| seed | Overall Δ | ViT Δ | CNN Δ | 白盒 Δ | Overall bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| 20260713 | +1.18pp | +0.86pp | +1.75pp | +1.00pp | [0.00, 2.45]pp |
| 20260714 | +0.36pp | +0.43pp | +0.25pp | -1.00pp | [-1.00, 1.73]pp |
| 20260715 | -0.36pp | +0.14pp | -1.25pp | +2.00pp | [-1.18, 0.45]pp |

它不满足 Overall `+3–5pp` 或 ViT `+3–5pp`，也没有稳定 CNN 保持条件。因此不进入主线。

## 轨迹级处理补充

为检验“当前梯度与 MI 累积方向的平行分量是否应放大”，增加了有状态的 trajectory probe。它只在当前梯度上做平行/正交分解，攻击内部的 MI 累积实现不变。

- `momentum_trajectory_align_a25`：降低 ViT，说明历史方向不能直接作为可靠方向投影。
- `momentum_trajectory_align_a50`：接近 mean，但没有正收益。
- `momentum_trajectory_parallel_boost_a25/a50`：没有产生 ViT 正收益。

这排除了简单的“沿 MI 方向加权”解释：MI 方向包含源模型特异的历史成分，直接增强会放大过拟合，而不是增强迁移性。

## 失败原因

当前失败不是因为梯度处理接口或随机性失控：

- 同一 seed 的 baseline/candidate replay 事件完全一致；
- 每次攻击严格保持 20 views；
- 所有候选只在 view 聚合后、MI/normalize/sign update 前处理梯度；
- 55 个单元测试覆盖了 probe 数值、形状、时序窗口、幅值符号、Fourier 分解、相位共识、跨尺度协方差/canonical、小波、幅值峰值、Laplacian、后动量方向、条件交互、phase-pair 处理、风险分数自适应组成、能量传输、冲突投影和 patch 子空间处理；
- 当前主线和候选均评估项目配置的 11 个黑盒及单独白盒 ViT。

失败的主要原因是可观测梯度特征与可迁移方向不是一一对应关系：

1. 相关性描述的是样本难度和方向质量，删除/放大该特征会同时破坏其中的有效子方向。
2. 20 个增强 view 的差异不是纯噪声；ViT 需要其中一部分非一致性来保持跨架构多样性。
3. `_normalize_grad` 和 MI 把绝对尺度变成相对坐标结构；单独调整 norm、频带或概率权重不能直接增加有效 sign trajectory。
4. 当前白盒是单一 ViT-B/16，任何 patch 对齐或强平滑都容易提高源模型而损害其他 ViT。
5. MI 历史方向同时包含有效和源模型特异成分，不能仅通过当前梯度与历史方向的 cosine 做放大。

最后测试的 `sign_reliability` 使用 `abs(mean_v sign(g_v))` 作为坐标可靠性，对原始均值做 boost 或 gate。boost 只产生少量离散模型变化，gate 在多个 ViT 上下降，因此没有进入 100 样本确认。

风险分数候选也没有进入 100 样本确认。100 样本确认的门槛是小样本中至少出现正向 ViT/Overall 且 CNN 不下降的信号；最佳风险候选的 ViT 已为 `-0.95pp`，其余候选为 `-1.43` 至 `-2.38pp`，没有达到该门槛。相比之下，弱 Gaussian 全量叠加虽然只在小样本显示弱正向，并且后来三 seed 的平均 Overall 只有 `+0.39pp`，仍是唯一值得做大样本复核的候选。

## 后续数学组成探针

为避免把“高频有害”误解为“高频能量可以安全搬走”，又测试了几类保持原始约束的组成变换：

| 方向 | 最佳 30 样本结果 Overall / ViT / CNN | 结论 |
|---|---:|---|
| 不确定高频能量 L2 传输到低频 | `-0.61 / -1.43 / +0.83pp` | 高频 residual 不是可直接搬运的无效能量 |
| log 幅值包络扩散 | `0.00 / -0.48 / +0.83pp` | 幅值尖峰不能简单视为有害 |
| PCGrad view 冲突投影 | `+0.30 / -0.48 / +1.67pp` | 减少负内积未提升 ViT；view 多样性并非纯冲突噪声 |
| 14×14 patch-mean 投影 | `-0.91 / -0.95 / -0.83pp` | ViT patch 子空间不是可直接增强的迁移子空间 |
| patch 内 L2 能量传输 | `-1.21 / -0.48 / -2.50pp` | 局部 residual 能量转移损害 CNN；强度增大时整体灾难性下降 |
| 高频 Fourier 幅值幂变换 | `-0.30 / -0.48 / 0.00pp` | 高频幅值压缩/放大均未产生正向 ViT |

其中最极端的 patch 内能量传输在 strength=`0.50/0.75` 时分别使 Overall 下降 `10.30/45.15pp`，直接证明“把高频或 patch residual 的能量集中到低频/patch DC”会破坏 sign trajectory，而不是增强迁移。

PCA 主方向传输同样没有通过筛选。最大跨-view 方差方向并不是迁移方向；把它与均值相加会把增强 view 的多样性误当成有效共享信号。

GLS 组合在小 ridge 下显著降低多个 ViT，较大 ridge 只逐渐恢复到 mean。因而“最小 view 方差”也不是迁移共享子空间；迁移需要保留一部分 view 多样性，不能简单优化 view 间稳定性。

## 下一步方向

如果继续推进，最值得保留的方向只有构造性的弱低频叠加（Gaussian blend），而不是继续删除或重排单个幅值/频率成分。后续应围绕它做有限的数学化预条件实验，例如以 Laplacian/Tikhonov 正则的低通解替代固定 Gaussian，并严格采用多 seed、Overall 主指标和 CNN guard；不应再把“幅值峰值”“小幅值高频”“跨 view/跨步共识”直接当作无效梯度。

在取得至少一个候选达到 ViT `+3pp`、Overall 同时达到 `+3pp` 且 CNN 不下降超过 `1pp` 之前，不应更新攻击主线。
