# PRD 方法与执行契约

2026-10-29 按用户确认更新执行方式：四个主要源模型在攻击前，按每张干净图像自身的原始 patch-score 层内方差定位一个 checkpoint，然后执行随机 token drop 和初始 RGB opponent-channel noise。以下约定适用于新执行；已有 FPR13 结果保留其实际固定 checkpoint 参数。

## 源与随机设置

四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，额外保留固定边界的 PiT-B。所有新执行在入口设置 PyTorch 全局 seed 42；固定 batch 布局和执行流程时可复现采样。历史 `randomness.py` seed 42 与当前 PyTorch seed 42 仍按用户约定视为统一报告标准；改变 checkpoint 策略后的测量需要重新执行。

| Source | 观察范围 | 单样本总 drop 比例 | Opponent strength |
| --- | --- | ---: | ---: |
| ViT-B/16 | 全部 12 个 block | 20/196 | 0.2 |
| CaiT-S24 | 全部 24 个局部 token block | 30/196 | 0.2 |
| PiT-T | 三个 stage 的全部 2+6+4 个 block | 14/196 + 2/49 + 2/49 = 30/196 | 0.4 |
| DeiT-B distilled | 全部 12 个 block | 20/196 | 0.2 |
| PiT-B（额外保留） | 固定 stage2_block1、stage3_block2、stage3_block3 | 原 drop ratios 0.02081165、0.03125、0.09375 | 0.4 |

## 逐样本定位

1. 每张干净图像只在完整攻击开始前做一次无 drop、无特征噪声的完整前向。逐个保存完整 block 输出的局部 token，不额外应用最终 `model.norm`。
2. ViT 和 distilled DeiT 使用最终 `model.norm` 后的分类 CLS 作为所有层共享的参照。DeiT 排除 CLS 和 distillation 两个 prefix。CaiT 使用全部局部 block、class-attention 和最终归一化之后的 CLS。PiT 使用各 stage 末端的原始 CLS 作为该 stage 的固定参照，最后一个 stage 使用最终归一化后的 CLS。
3. 对每张图、每个 block，以 float32 计算局部 token 与参照的余弦相似度，eps=1e-8，限制数值到 [-1,1]；以 float64 计算原始分数的总体方差，分母为该层原生局部 token 数。局部表示与评分保留在模型设备上，最终将各层方差和有效性标记一次性传回 CPU 后选层。定位不使用 min–max，不跨图平均。非有限值或零范数表征会报错。
4. 每张图选择方差最大的 block；并列选择前向顺序最早者。drop 放在峰值 block 的输入处，即前一 block 输出后。该选择在所有攻击步和视图中固定。
5. `block0` 是第一个 block 的输入；PiT 的 `stageN_block0` 是该 stage pooling 和 stage norm 之后、第一个 block 的输入。跨 stage 的边界不挪到 pooling 之前。
6. 同批样本可以使用不同边界。每个视图保持完整 batch，按模型原生顺序遍历本批选择的边界；每个边界只清零选择它的样本，其余样本的 mask 为零。所有样本共同经过 pooling、归一化和后续 block，token 数与网格保持原生结构。损失为完整 batch 的均值。

总比例按原配置各 drop ratio 求和并保留 12 位小数。所选边界处从全部当前局部 token 均匀、无放回抽样 `max(1, round(N*r))` 个位置。PiT-T 的 N 为 729、196 或 49；实际 drop 数与实际比例逐样本记录。PiT-B 额外源保留原固定三边界。

## 攻击执行顺序

1. 定位结束后，每个 step/group 根据已知网格和 drop 比例构建新 mask schedule，不为生成 mask 执行额外前向，也不从 GPU 回读 mask 计数。固定源的网格由其 patch embedding 和 pooling 几何计算。动态四源每张图仍只有选定的一个边界；整批 schedule 包含本批边界的并集，并记录各边界的有效样本。位置始终均匀随机抽样，分数不选择被清零的 token。
2. 每个可微视图只构建一次初始状态。在全部初始局部位置加独立采样的 opponent noise，包括未来会被清零的位置。prefix 不变。
3. 到达选定边界，清零所选局部 token 的全部通道，再继续同一次前向。原生网格和 token 数保持不变。
4. 默认原始／平移两个视图。平移通过整批索引实现反射填充裁剪的相同映射，mask 同步映射到图像空间、平移后按原生网格 occupancy top-k 保持 drop 数量。未选择当前边界的样本在两个视图中都保持全零 mask。同组两视图使用相同模型边界。相位在 CPU 上通过 PyTorch 全局随机函数采样，避免 GPU 标量回读。
5. 默认 10 groups×2 views。像素梯度直接平均，以系数 1 累积动量，再做 sign 更新、相对干净输入的 L∞ 投影和 [0,1] 截断。默认 10 steps、epsilon=16/255、步长 epsilon/steps。动态四源每图产生 100 条 schedule、100 次 checkpoint mask selection。

`--input-diversity-views-per-group` 保留 1、2 两种消融设置。单视图不采样平移；默认平移集合为 (4,4)/(8,8)/(12,12)，每步每图视图不超过 20。关闭 progressive drop 时跳过方差定位。显式 ViT 固定边界分析参数和 Figure 2 固定版本素材入口用于复现原配置，普通四源入口默认动态定位。

## 噪声

亮度、红绿、黄蓝方向为 `(1,1,1)/√3`、`(1,-1,0)/√2`、`(1,1,-2)/√6`，采样方差为 `0.5,1.25,1.25`。每个初始局部位置按 RGB 投影核采样，使用真实投影权重 W 映射：

`η = W ξ_RGB`，`η_hat = β RMS(H0)/RMS(η) · η`，`H0_noise = H0 + η_hat`。

RMS 在单样本全部初始局部位置和通道上计算。特征 RMS 与投影权重固定 detach。噪声每个可微视图只注入一次。

## 记录与评估

动态 `attack_params.json` 保存定位定义、总比例、并列规则，以及逐样本图像名、峰值层、checkpoint、峰值方差、网格、实际 drop 数和实际比例。保存跨 batch 的 checkpoint 频次；不把最后一个子组的固定参数当作整批结果。

迁移评估仍以 FPR13 为主。ASR 为全部对抗样本上的 `1 - adversarial accuracy`。`results/` 仅保存 FPR13 数值和参数；已有固定 checkpoint 结果不会作为动态版本结果重标。
