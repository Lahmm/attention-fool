# PRD 方法与执行契约

更新日期：2026-11-05。唯一可执行攻击为方向方差比 × 正收缩比例选层与单边界随机 token hard-zero。固定单层、固定双／多 checkpoint 路径和覆盖入口已移除。四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled；PiT-B 与 Visformer 保留为评估目标。全部模型权重保留。

## 源与随机设置

| Source | 观察范围 | 总 drop 比例 | 当前有效默认比例 | β |
| --- | --- | ---: | ---: | ---: |
| ViT-B/16 | 12 blocks | 20/196 | 0.102040816326 | 0.2 |
| CaiT-S24 | 24 个局部 blocks | 30/196 | 0.15306122449 | 0.2 |
| PiT-T | 三 stage 的 2+6+4 blocks | 30/196 | 0.15306122449 | 0.4 |
| DeiT-B distilled | 12 blocks | 20/196 | 0.102040816326 | 0.2 |

直接保存原动态运行对历史分项比例求和后的有效值，不改变已执行预算。新 CLI 只有单个 `--drop-ratio`，保留 12 位小数，满足 `0 < r <= 0.5`。所选边界的数量为 `max(1, round(N*r))`，ViT/CaiT/DeiT 为 20/30/20，PiT-T 原生 N=729/196/49 时为 112/30/8。

所有新执行在入口设置 PyTorch 全局 seed 42；固定 batch 布局和流程时可复现采样。历史 randomness.py seed 42 与当前 PyTorch seed 42 按用户约定统一报告，不因 RNG 实现变化要求补跑。改变 checkpoint、drop 参数或实际增强分布需要对应配置的测量。

## 逐图定位与架构定义

1. 每张干净图像在攻击前做一次无 drop、无特征增强的完整前向，采集所有候选 block 的原生局部输入与完整输出，不额外施加最终 LayerNorm。
2. ViT/DeiT 的参考为最终 `model.norm` 后的分类 CLS；DeiT 排除 CLS 和 distillation 两个 prefix。CaiT 使用全部 class-attention 后、最终 LayerNorm 前 CLS。PiT 使用各 stage 末端原始 CLS，最后 stage 使用最终归一化 CLS。
3. 将 float32 特征转为 float64，逐局部向量与参考分别 L2 归一化。局部特征各自减去输入／输出局部中心，平均平方距离为输入／输出分散度。输出沿全局单位方向的投影总体方差除以输出分散度／原生特征维数，得到方向方差比，再乘 `max(0, 1 - 输出分散度 / 输入分散度)`。
4. 正的小分母不截断；输入或输出分散度为零时分数为零；非有限和零范数特征报错。只依赖当前图像，不使用跨图校准、深度权重或拟合系数。
5. 选最高分 block，并列或全零均选原生顺序最早者；在其输入处 drop，全部步与视图保持边界。`peak_layer=block11` 对应 `checkpoint=block10`：后者是执行完前十个 block 后、第十一个 block 的输入。
6. `block0` 是首个 block 输入。PiT 的 `stageN_block0` 位于该 stage 原生 pooling 和 stage norm 之后；收缩不包含 pooling。
7. 同批各图可以选不同边界，按原生顺序遍历边界并集，各边界只作用于选择它的样本，其他样本 mask 为零。整批共同经过后续 block、pooling 和归一化；每张图、每个视图仅一个干预边界。

原 float32 cosine（eps1e-8、clamp[-1,1]）的 float64 总体方差保留为诊断字段，不提供其他选层策略。接口为 `select_contraction_checkpoints` 与 `select_contraction_peaks`。

## 攻击执行

每步使用一个扁平的独立视图序列。默认 K=20；每个视图、每张图独立以 1/2 概率取零平移，以 1/2 概率均匀选择默认集合 (4,4)/(8,8)/(12,12) 中的一个反射平移。零平移代表原始图像。一个 batch 内的不同样本也独立采样，不要求每步恰好各有十个原始／平移视图。相位通过 CPU PyTorch 全局随机函数采样；若候选集合包含 (0,0)，实际非零平移概率相应下降。

每个 step/view 根据所选边界原生网格，从该视图全部局部位置均匀、无放回抽取 k 个位置。各视图独立抽 mask，mask 不依赖 token 分数，不额外执行抽样前向。直接在当前视图的原生网格 hard-zero，不再跨视图共享、平移或用 occupancy top-k 映射 mask。所选边界、实际 drop 数与比例保持；prefix、token 数与网格不变。

每个可微视图构建一次初始状态，在全部初始局部位置注入一次独立采样的投影特征增强，包括后续会被清零的位置。平均所有视图关于当前像素的梯度，以系数 1 累积动量，再 sign 更新、相对干净图像的 L∞ 投影与 [0,1] 截断。选层仍只在攻击前的干净图像上执行一次。

默认 seed 42、10 steps、epsilon=16/255、步长 epsilon/steps，每图 200 次独立 mask 抽样。CLI 的 `input-diversity-groups × input-diversity-views-per-group` 只计算总预算，上限 20；默认 10×2 在内部展开为 20 个独立视图。保留 `--input-diversity-views-per-group 1` 作为原始视图消融，此时不抽相位，总预算仍为上述乘积；20×1 对应每步 20 个无 phase 视图。无 drop 消融跳过选层与 mask，但仍保留相同的视图采样、增强和优化。

2026-11-05 前的配对视图配置每步只有十个独立 mask，另十个是空间对应版本。新流程改变了视图与 mask 分布，已有配对结果和无 phase 测量保持其实际配置标签；本次代码替换不产生新的千图 ASR。

## 共同特征增强（补充设置）

代码默认 `opponent`，三个 RGB 正交方向方差为 0.5、1.25、1.25。显式 `--noise-structure isotropic_rgb` 将三个方向方差设为 1，等价于投影前 RGB 空间的 i.i.d. 标准高斯采样。用真实初始投影权重 W 映射，再逐样本 RMS 匹配：

`η = W ξ_RGB`，`η_hat = β RMS(H0)/RMS(η) · η`，`H0_aug = H0 + η_hat`。

RMS 覆盖全部初始局部位置和通道，特征 RMS 与投影权重 detach，prefix 不变。投影后的特征可以相关。`--opponent-noise-strength` 控制两种模式的 β，0 关闭增强。论文规划采用显式 i.i.d. 设置，构造与参数放补充材料，已完成 opponent 记录保持实际标签。

## 参数与评估

`attack_params.json` 保存 `input_diversity_sampling=independent_identity_or_phase_views`、phase 候选分支概率、公式、精度、总比例、并列规则、逐图图像名、峰值 block、输入 checkpoint、原始余弦方差、方向比、收缩比例、乘积分数、输入／输出分散度、网格、实际 drop 数和比例及边界频次。CaiT 保存 `patch_score_reference`。

唯一评估协议为 FPR13，ASR 使用全部对抗样本的 `1 - adversarial accuracy`；strict 排除完全相同源目标，四源等权平均。同一选择器的历史配对视图/opponent 千图证据为 `results/prd_direction_contraction_s42_1000_fpr/`。其他历史选层与 token 分析只保留证据，不恢复执行入口。`results/` 仅保存 FPR13 ASR、参数、直接差值与来源。
