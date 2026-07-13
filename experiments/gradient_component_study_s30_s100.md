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

本轮共筛选 111 个 30 样本候选，覆盖：

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

关键对照是：

- 全频段 Fourier Wiener 会提高部分 CNN，但降低 ViT；
- Fourier 幅值压缩和放大也偏向 CNN，不能改善 ViT；
- 按 ViT patch 或局部空间能量均衡会提高源白盒的部分结果，却损害黑盒 ViT；
- 未归一化的弱 Gaussian 叠加是唯一在 100 样本上出现正向 Overall/ViT 的组合；把平滑分量强行归一化后，收益消失。

另外，只有对全部样本施加原始 Gaussian 弱叠加才出现弱正向；按低空间熵或高频能量选择样本后再平滑，30 样本上没有稳定收益。说明当前特征更像迁移性的描述变量，不能直接作为样本级因果开关。

这说明幅值与频率不是可以独立相加的两个“好/坏开关”。有效影响来自原始梯度中自然幅值比例和空间频率相位共同形成的方向；改变其中一个而不保持另一个的比例，会破坏迁移。

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
- 44 个单元测试覆盖了 probe 数值、形状、时序窗口、幅值符号、Fourier 分解、相位共识、跨尺度协方差/canonical、小波和幅值峰值处理；
- 当前主线和候选均评估项目配置的 11 个黑盒及单独白盒 ViT。

失败的主要原因是可观测梯度特征与可迁移方向不是一一对应关系：

1. 相关性描述的是样本难度和方向质量，删除/放大该特征会同时破坏其中的有效子方向。
2. 20 个增强 view 的差异不是纯噪声；ViT 需要其中一部分非一致性来保持跨架构多样性。
3. `_normalize_grad` 和 MI 把绝对尺度变成相对坐标结构；单独调整 norm、频带或概率权重不能直接增加有效 sign trajectory。
4. 当前白盒是单一 ViT-B/16，任何 patch 对齐或强平滑都容易提高源模型而损害其他 ViT。
5. MI 历史方向同时包含有效和源模型特异成分，不能仅通过当前梯度与历史方向的 cosine 做放大。

最后测试的 `sign_reliability` 使用 `abs(mean_v sign(g_v))` 作为坐标可靠性，对原始均值做 boost 或 gate。boost 只产生少量离散模型变化，gate 在多个 ViT 上下降，因此没有进入 100 样本确认。

PCA 主方向传输同样没有通过筛选。最大跨-view 方差方向并不是迁移方向；把它与均值相加会把增强 view 的多样性误当成有效共享信号。

GLS 组合在小 ridge 下显著降低多个 ViT，较大 ridge 只逐渐恢复到 mean。因而“最小 view 方差”也不是迁移共享子空间；迁移需要保留一部分 view 多样性，不能简单优化 view 间稳定性。

## 下一步方向

如果继续推进，最值得保留的方向只有构造性的弱低频叠加（Gaussian blend），而不是继续删除或重排单个幅值/频率成分。后续应围绕它做有限的数学化预条件实验，例如以 Laplacian/Tikhonov 正则的低通解替代固定 Gaussian，并严格采用多 seed、Overall 主指标和 CNN guard；不应再把“幅值峰值”“小幅值高频”“跨 view/跨步共识”直接当作无效梯度。

在取得至少一个候选达到 ViT `+3pp`、Overall 同时达到 `+3pp` 且 CNN 不下降超过 `1pp` 之前，不应更新攻击主线。
