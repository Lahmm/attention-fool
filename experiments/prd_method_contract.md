# PRD 方法与执行契约

2026-11-04：用户确认方向方差比 × 正收缩比例为唯一主线。每张干净图像观察各原生 block 的局部输入／输出，以输出方向方差比乘正收缩比例定位最高分 block，在其输入处随机 token drop。论文核心为统计选层与随机干预；共同增强的构造放入补充设置。代码仍默认 opponent，显式 `--noise-structure isotropic_rgb` 使用 i.i.d. RGB 采样。已完成千图结果保持实际 opponent 配置标签。

## 源与随机设置

四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，额外保留固定边界的 PiT-B。所有新执行在入口设置 PyTorch 全局 seed 42；固定 batch 布局和执行流程时可复现采样。历史 `randomness.py` seed 42 与当前 PyTorch seed 42 仍按用户约定视为统一报告标准；改变 checkpoint 策略、drop 参数或实际增强分布后的报告需要对应配置的测量。

| Source | 观察范围 | 单样本总 drop 比例 | 特征增强强度 β |
| --- | --- | ---: | ---: |
| ViT-B/16 | 全部 12 个 block | 20/196 | 0.2 |
| CaiT-S24 | 全部 24 个局部 token block | 30/196 | 0.2 |
| PiT-T | 三个 stage 的全部 2+6+4 个 block | 14/196 + 2/49 + 2/49 = 30/196 | 0.4 |
| DeiT-B distilled | 全部 12 个 block | 20/196 | 0.2 |
| PiT-B（额外保留） | 固定 stage2_block1、stage3_block2、stage3_block3 | 原 drop ratios 0.02081165、0.03125、0.09375 | 0.4 |

## 逐样本定位

1. 唯一主线为方向方差比 × 正收缩比例。每张干净图像在攻击前做一次无 drop、无特征增强的完整前向，采集每个原生 block 的局部输入和完整输出。输入位于该 block 前，不额外应用最终 LayerNorm。
2. ViT/DeiT 使用最终 `model.norm` 后的分类 CLS，DeiT 排除两个 prefix；CaiT 使用全部 class-attention 后、最终 LayerNorm 前 CLS；PiT 使用各 stage 末端原始 CLS，最后 stage 使用最终归一化 CLS。PiT 的 block 输入在原生 pooling 和 stage norm 之后，收缩不包含 pooling。
3. 对 float32 模型特征转 float64 后逐局部向量与全局参考分别 L2 归一化。输入、输出各自围绕本层局部中心的平均平方距离为输入、输出分散度。输出沿全局方向投影的总体方差，除以输出分散度／原生特征维数，得到方向比；再乘 `max(0, 1 - 输出分散度 / 输入分散度)`。正的小分母不截断。输入或输出分散度为零时分数为零；非有限值与零范数特征报错。仅使用当前图像，不跨图校准、不使用固定层标签、拟合系数或深度权重。
4. 选择乘积最大者，并列和全零分数均选择原生顺序最早者。drop 在得分最高 block 的输入处实施。每图只定位一次，全部步与视图保持该边界。原始 float32 cosine（eps1e-8、clamp[-1,1]）的 float64 总体方差作为所选层诊断量记录，不作为另一选层策略。
5. `block0` 是第一个 block 的输入；PiT 的 `stageN_block0` 是该 stage pooling 和 stage norm 之后、第一个 block 的输入。跨 stage 的边界不挪到 pooling 之前。
6. 同批样本可以使用不同边界。每个视图保持完整 batch，按模型原生顺序遍历本批选择的边界；每个边界只清零选择它的样本，其余样本的 mask 为零。所有样本共同经过 pooling、归一化和后续 block，token 数与网格保持原生结构。损失为完整 batch 的均值。

总比例按原配置各 drop ratio 求和并保留 12 位小数。所选边界处从全部当前局部 token 均匀、无放回抽样 `max(1, round(N*r))` 个位置。PiT-T 的 N 为 729、196 或 49；实际 drop 数与实际比例逐样本记录。PiT-B 额外源保留原固定三边界。

## 攻击执行顺序

1. 定位结束后，每个 step/group 根据已知网格和 drop 比例构建新 mask schedule，不为生成 mask 执行额外前向，也不从 GPU 回读 mask 计数。固定源的网格由其 patch embedding 和 pooling 几何计算。动态四源每张图仍只有选定的一个边界；整批 schedule 包含本批边界的并集，并记录各边界的有效样本。位置始终均匀随机抽样，分数不选择被清零的 token。
2. 每个可微视图只构建一次初始状态。在全部初始局部位置加该视图独立采样的投影特征增强，包括未来会被清零的位置。prefix 不变。最终论文采用显式 isotropic_rgb 设置；旧运行按记录中的噪声配置解释。
3. 到达选定边界，清零所选局部 token 的全部通道，再继续同一次前向。原生网格和 token 数保持不变。
4. 默认原始／平移两个视图。平移通过整批索引实现反射填充裁剪的相同映射，mask 同步映射到图像空间、平移后按原生网格 occupancy top-k 保持 drop 数量。未选择当前边界的样本在两个视图中都保持全零 mask。同组两视图使用相同模型边界。相位在 CPU 上通过 PyTorch 全局随机函数采样，避免 GPU 标量回读。
5. 默认 10 groups×2 views。像素梯度直接平均，以系数 1 累积动量，再做 sign 更新、相对干净输入的 L∞ 投影和 [0,1] 截断。默认 10 steps、epsilon=16/255、步长 epsilon/steps。动态四源每图产生 100 条 schedule、100 次 checkpoint mask selection。

`--input-diversity-views-per-group` 保留 1、2 两种消融设置。单视图不采样平移；默认平移集合为 (4,4)/(8,8)/(12,12)，每步每图视图不超过 20。关闭 progressive drop 时跳过选层定位。显式 ViT 固定边界分析参数和 Figure 2 固定版本素材入口用于复现原配置，普通四源入口默认动态定位。

## 共同随机特征增强（补充材料设置）

论文采用 `--noise-structure isotropic_rgb`：三个正交 RGB 方向的独立高斯系数方差均为 1，等价于投影前 RGB 空间的 i.i.d. 标准高斯采样。使用真实初始投影权重 W 映射并 RMS 匹配：

`η = W ξ_RGB`，`η_hat = β RMS(H0)/RMS(η) · η`，`H0_aug = H0 + η_hat`。

RMS 在单样本全部初始局部位置和通道上计算。特征 RMS 与投影权重固定 detach。每个可微视图注入一次，覆盖全部初始局部位置，prefix 不变。投影后的特征可以相关，i.i.d. 描述投影前采样。

代码当前默认 `opponent` 保持方差 `0.5,1.25,1.25`，对应亮度、红绿、黄蓝正交基；历史记录保留真实设置。此配置差异只作为实现与实验来源事实记录，噪声不再单列为论文机制、贡献或正文方法小节。

## 记录与评估

动态 `attack_params.json` 保存定位定义、总比例、并列规则，以及逐样本图像名、峰值层、checkpoint、原始余弦方差、方向比、最终乘积分数、收缩比例、输入和输出分散度、网格、实际 drop 数和实际比例。CaiT 额外保存 `patch_score_reference`，标明 class-attention 后、最终 LayerNorm 前的 CLS 参照。保存跨 batch 的 checkpoint 频次；不把最后一个子组的固定参数当作整批结果。

迁移评估仍以 FPR13 为主。ASR 为全部对抗样本上的 `1 - adversarial accuracy`。`results/` 仅保存 FPR13 数值和参数；已有固定 checkpoint/opponent 结果保持原标签；当前乘积选层四源 opponent 评估已完成，归档为 `results/prd_direction_contraction_s42_1000_fpr/`；历史最大方差评估也保留实际配置，索引见 [故事记录](prd_paper_story.md)。最终动态 i.i.d. 结果需对应实际配置，不从旧表重标。
