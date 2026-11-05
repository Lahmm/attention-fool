# PRD 方法与执行契约

更新日期：2026-11-04。唯一可执行攻击为方向方差比 × 正收缩比例选层与单边界随机 token hard-zero。固定单层、固定双／多 checkpoint 路径和覆盖入口已移除。四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled；PiT-B 与 Visformer 保留为评估目标。全部模型权重保留。

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

每个 step/group 根据已知原生网格从全部局部位置均匀、无放回抽取 k 个位置。mask 不依赖 token 分数，不为抽样额外前向。清零所选 token 全部通道，prefix、token 数和原生网格保持。

每个可微视图构建一次初始状态，在全部初始局部位置注入一次该视图独立采样的投影特征增强，包括后续会被清零的位置。原始视图与反射平移视图共享空间对应的 mask；将 mask 映射到像素空间，反射平移后以原生网格 occupancy top-k 保持实际数量。相位在 CPU 用 PyTorch 全局随机函数采样。

默认 seed 42、10 steps、10 groups × 2 views、epsilon=16/255、步长 epsilon/steps。每图 100 次原始 mask 抽样。直接平均视图像素梯度，以系数 1 累积动量，再 sign 更新、相对干净输入的 L∞ 投影和 [0,1] 截断。

`--input-diversity-views-per-group` 保留 1/2 消融，单视图不采样相位。默认平移为 (4,4)/(8,8)/(12,12)，每步最多 20 视图。无 drop 消融跳过干净选层，保留增强与优化。

## 共同特征增强（补充设置）

命令行与攻击类仅支持直接 RGB 空间的 i.i.d. 标准高斯采样。结构选择参数、颜色方向基变换和旧专属参数名已删除。用真实初始投影权重 W 映射，再逐样本 RMS 匹配：

`η = W ξ_RGB`，`η_hat = β RMS(H0)/RMS(η) · η`，`H0_aug = H0 + η_hat`。

RMS 覆盖全部初始局部位置和通道，特征 RMS 与投影权重 detach，prefix 不变。投影后的特征可以相关。`--noise-strength` 控制 β，0 关闭增强。当前默认采用 i.i.d. 设置，构造与参数放补充材料，已完成 opponent 记录保持实际标签。

## 参数与评估

`attack_params.json` 保存公式、精度、总比例、并列规则、逐图图像名、峰值 block、输入 checkpoint、原始余弦方差、方向比、收缩比例、乘积分数、输入／输出分散度、网格、实际 drop 数和比例及边界频次。CaiT 保存 `patch_score_reference`。

唯一评估协议为 FPR13，ASR 使用全部对抗样本的 `1 - adversarial accuracy`；strict 排除完全相同源目标，四源等权平均。历史 opponent 千图证据为 `results/prd_direction_contraction_s42_1000_fpr/`。其他历史选层与 token 分析只保留证据，不恢复执行入口。`results/` 仅保存 FPR13 ASR、参数、直接差值与来源。

清理颜色基变换后，直接 RGB 高斯采样与原 i.i.d. 模式的采样分布和抽样形状相同，但相同 seed 的具体噪声值发生变化。旧测量保留原源码哈希；按项目约定，不因这一随机实现差异重跑既有结果。新增记录使用 `noise_enabled`、`noise_strength`，并保留实际 `noise_structure=isotropic_rgb` 标签，关闭增强时为 `none`。
