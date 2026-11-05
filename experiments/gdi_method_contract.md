# GDI 方法与执行契约

更新日期：2026-11-05。GDI（Geometry-Guided Depth Intervention）通过逐样本选层和所选层输入随机干扰增强迁移攻击。当前执行为 `main.py` 调用 `attack.py` 的 `GDIAttacker`，以方向方差比 × 正收缩比例定位单个原生 block 输入边界。四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled；PiT-B 与 Visformer 保留为评估目标。全部模型权重保留。

论文围绕有效干预层的发现及模型无关、样本相关的定位展开，见 [论文故事](gdi_paper_story.md)。本契约集中记录数值、架构和基础增强／优化设置。

## 源与随机设置

| Source | 观察范围 | 总 drop 比例 | 当前有效默认比例 | β |
| --- | --- | ---: | ---: | ---: |
| ViT-B/16 | 12 blocks | 20/196 | 0.102040816326 | 0.2 |
| CaiT-S24 | 24 个局部 blocks | 30/196 | 0.15306122449 | 0.2 |
| PiT-T | 三 stage 的 2+6+4 blocks | 30/196 | 0.15306122449 | 0.4 |
| DeiT-B distilled | 12 blocks | 20/196 | 0.102040816326 | 0.2 |

CLI 使用单个 `--drop-ratio`，有效值保留 12 位小数，满足 `0 < r <= 0.5`。所选边界的数量为 `max(1, round(N*r))`，ViT/CaiT/DeiT 为 20/30/20，PiT-T 原生 N=729/196/49 时为 112/30/8。

所有新执行在入口设置 PyTorch 全局 seed 42；固定 batch 布局和流程时可复现采样。历史 randomness.py seed 42 与当前 PyTorch seed 42 按用户约定统一报告，不因 RNG 实现变化要求补跑。改变 checkpoint、drop 参数或实际增强分布需要对应配置的测量。

## 逐图定位与架构定义

1. 每张干净图像在攻击前做一次无 drop、无特征增强的完整前向，采集所有候选 block 的原生局部输入与完整输出，不额外施加最终 LayerNorm。
2. ViT/DeiT 的参考为最终 `model.norm` 后的分类 CLS；DeiT 排除 CLS 和 distillation 两个 prefix。CaiT 使用全部 class-attention 后、最终 LayerNorm 前 CLS。PiT 使用各 stage 末端原始 CLS，最后 stage 使用最终归一化 CLS。
3. 将 float32 特征转为 float64，逐局部向量与参考分别 L2 归一化。局部特征各自减去输入／输出局部中心，平均平方距离为输入／输出分散度。输出沿全局单位方向的投影总体方差除以输出分散度／原生特征维数，得到方向方差比，再乘 `max(0, 1 - 输出分散度 / 输入分散度)`。
4. 正的小分母不截断；输入或输出分散度为零时分数为零；非有限和零范数特征报错。只依赖当前图像，不使用跨图校准、深度权重或拟合系数。
5. 选最高分 block，并列或全零均选原生顺序最早者；在其输入处 drop，全部步与视图保持边界。`peak_layer=block11` 对应 `checkpoint=block10`：后者是执行完前十个 block 后、第十一个 block 的输入。
6. `block0` 是首个 block 输入。PiT 的 `stageN_block0` 位于该 stage 原生 pooling 和 stage norm 之后；收缩不包含 pooling。
7. 同批各图可以选不同边界，按原生顺序遍历边界并集，各边界只作用于选择它的样本，其他样本 mask 为零。整批共同经过后续 block、pooling 和归一化；每张图、每个视图仅一个干预边界。

原 float32 cosine（eps1e-8、clamp[-1,1]）的 float64 总体方差保留为诊断字段。接口为 `select_contraction_checkpoints` 与 `select_contraction_peaks`。

## 攻击执行

每个 step/group 根据已知原生网格从全部局部位置均匀、无放回抽取 k 个位置构建 mask。清零所选 token 全部通道，prefix、token 数和原生网格保持。

每个可微视图构建一次初始状态，在全部初始局部位置注入一次该视图独立采样的投影特征增强，包括后续会被清零的位置。原始视图与反射平移视图共享空间对应的 mask；将 mask 映射到像素空间，反射平移后以原生网格 occupancy top-k 保持实际数量。相位在 CPU 用 PyTorch 全局随机函数采样。

默认 seed 42、10 steps、10 groups × 2 views、epsilon=16/255、步长 epsilon/steps。每图 100 次原始 mask 抽样。直接平均视图像素梯度，以系数 1 累积动量，再 sign 更新、相对干净输入的 L∞ 投影和 [0,1] 截断。

`--input-diversity-views-per-group` 保留 1/2 消融，单视图不采样相位。默认平移为 (4,4)/(8,8)/(12,12)，每步最多 20 视图。`--no-gdi-drop` 无 drop 消融跳过干净选层，保留增强与优化。

## 基础特征增强（补充实现设置）

直接在 RGB 空间采样 i.i.d. 标准高斯，以真实初始投影权重 W 映射，再逐样本 RMS 匹配：

`η = W ξ_RGB`，`η_hat = β RMS(H0)/RMS(η) · η`，`H0_aug = H0 + η_hat`。

RMS 覆盖全部初始局部位置和通道，特征 RMS 与投影权重 detach，prefix 不变。投影后的特征可以相关。`--noise-strength` 控制 β，0 关闭增强。构造与参数放补充实现设置，历史测量按实际增强标签索引。

## 参数与评估

`attack_params.json` 保存公式、精度、总比例、并列规则、逐图图像名、峰值 block、输入 checkpoint、原始余弦方差、方向比、收缩比例、乘积分数、输入／输出分散度、网格、实际 drop 数和比例及边界频次。CaiT 保存 `patch_score_reference`。

评估使用 FPR13，ASR 为全部对抗样本的 `1 - adversarial accuracy`；strict 排除完全相同源目标，四源等权平均。当前 i.i.d. 主结果见 [归档](../results/gdi_iid_paired_s42_1000_fpr/README.md)，历史选层与 token 分析见 [结果索引](../results/README.md)。`results/` 保存 FPR13 ASR、参数、直接差值与来源。

当前参数使用 `gdi_drop_enabled`、`noise_enabled` 和 `noise_strength`，增强分布标签为 `noise_structure=isotropic_rgb`，关闭增强时为 `none`。历史执行字段、实际配置和源码哈希由各归档 provenance 保存。
