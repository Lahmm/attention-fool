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

本轮共筛选 69 个 30 样本候选，覆盖：

- 幅值分位删除、极值裁剪、幅值幂放大/压缩；
- 坐标和 group Wiener/可靠性/幅值均衡；
- 固定高频增益、频率 Wiener、频谱幅值幂变换；
- 高频共享/残差成分放大；
- patch/local spatial energy equalization；
- 时间窗口频率处理；
- 低频叠加、Gaussian 平滑叠加及 feature-conditioned Gaussian。

关键对照是：

- 全频段 Fourier Wiener 会提高部分 CNN，但降低 ViT；
- Fourier 幅值压缩和放大也偏向 CNN，不能改善 ViT；
- 按 ViT patch 或局部空间能量均衡会提高源白盒的部分结果，却损害黑盒 ViT；
- 未归一化的弱 Gaussian 叠加是唯一在 100 样本上出现正向 Overall/ViT 的组合；把平滑分量强行归一化后，收益消失。

另外，只有对全部样本施加原始 Gaussian 弱叠加才出现弱正向；按低空间熵或高频能量选择样本后再平滑，30 样本上没有稳定收益。说明当前特征更像迁移性的描述变量，不能直接作为样本级因果开关。

这说明幅值与频率不是可以独立相加的两个“好/坏开关”。有效影响来自原始梯度中自然幅值比例和空间频率相位共同形成的方向；改变其中一个而不保持另一个的比例，会破坏迁移。

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
- 31 个单元测试覆盖了 probe 数值、形状、时序窗口、幅值符号和 Fourier 分解；
- 当前主线和候选均评估项目配置的 11 个黑盒及单独白盒 ViT。

失败的主要原因是可观测梯度特征与可迁移方向不是一一对应关系：

1. 相关性描述的是样本难度和方向质量，删除/放大该特征会同时破坏其中的有效子方向。
2. 20 个增强 view 的差异不是纯噪声；ViT 需要其中一部分非一致性来保持跨架构多样性。
3. `_normalize_grad` 和 MI 把绝对尺度变成相对坐标结构；单独调整 norm、频带或概率权重不能直接增加有效 sign trajectory。
4. 当前白盒是单一 ViT-B/16，任何 patch 对齐或强平滑都容易提高源模型而损害其他 ViT。
5. MI 历史方向同时包含有效和源模型特异成分，不能仅通过当前梯度与历史方向的 cosine 做放大。

## 下一步方向

如果继续推进，最值得做的不是继续遍历静态幅值/频率阈值，而是建立“方向轨迹级”的可迁移性约束：在不引入其他模型、不改变 data augmentation 的前提下，利用当前梯度与历史 MI 方向的关系，构造只改变当前梯度分解的正交/平行投影，并在早期 step 与后期 step 使用不同的幅值预算。重点应验证：

- 当前梯度中与 MI 累积方向平行的分量是否应放大；
- 与 MI 方向正交但跨 view 稳定的分量是否应保留；
- 频率处理是否应作用于“相对历史方向的残差”，而不是作用于原始梯度的径向频带；
- 这种轨迹处理能否同时避免源 ViT 特异化和 CNN 下降。

在取得至少一个候选达到 ViT `+3pp`、Overall 同时达到 `+3pp` 且 CNN 不下降超过 `1pp` 之前，不应更新攻击主线。
