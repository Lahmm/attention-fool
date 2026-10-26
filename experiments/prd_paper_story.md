# PRD 论文叙事与主实验记录

当前主实验使用 seed 42、四源、1000 图像和 FPR13；2026-10-23 更新了 PiT-T 与 distilled DeiT-B 的 drop/full 四个单元。当前结果的数值入口为 [FPR 实验记录](../results/prd_s42_1000_fpr_ablation.md)。当前论文源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，额外保留 PiT-B。

当前执行顺序见 [PRD 方法记录](prd_method_contract.md)。

本文的中心贡献是 **Progressive Route Disruption（PRD）**：一种面向跨模型迁移的攻击方法。主线聚焦完整方法的设计与迁移效果，核心机制只有两个：沿模型深度的 checkpoint-wise progressive random token drop，以及施加在全部初始局部特征上的 RGB opponent-channel noise。

## 核心故事

1. **观察。** 同一个局部 token 与全局表示的余弦相似度会随网络深度发生明显变化，高 patch-score 的 patch 位置也会随层数变化而迁移与更替。用同一样本的多层 patch-score 可视化展示这一动态过程，让读者看到局部证据如何在前向传播中演化。Patch score 用于呈现发现，PRD 的执行由随机 checkpoint schedule 驱动。
2. **方法。** PRD 在多个架构专属 checkpoint 顺序地随机选择局部 token 并 hard-zero；每次 drop 后，模型都在已改变的 hidden state 上继续前向传播。每个攻击 step、每个 augmentation group 都重新构造随机 schedule。原始视图使用该 schedule，相位视图使用对应的空间变换 schedule。
3. **互补扰动。** PRD 在全部初始局部特征上加入噪声，包括未来会被 drop 的位置；噪声从亮度、红绿、黄蓝三个 RGB 对手方向采样，经源模型的初始 RGB projection 映射到特征空间并进行 RMS 匹配。
4. **结果。** 四种源架构各攻击 1000 张图像，在相同的 FPR13 的 13 个目标模型上评估。PRD 的四源平均 Overall ASR 为 **89.72%**，平均严格黑盒 ASR 为 **89.29%**。

适合摘要或引言的核心表述：

> 随着图像向网络深处传播，同一个局部 token 与全局表示的关联会发生明显变化，高 patch-score 的区域也不断更替。这一动态发现启发我们沿着表示形成的过程发起攻击。我们提出 Progressive Route Disruption，在多个 checkpoint 对演化中的局部表示施加重新采样的随机中断，同时用 RGB 对手通道噪声扰动全部初始局部表示。在四种源架构、每源 1000 图像和 FPR13 的 13 个目标模型的评估中，PRD 取得 89.29% 的平均严格黑盒 ASR。

## 方法事实与配置

PRD 默认使用 10 个迭代 step、每步 10 个 augmentation group、每组 2 个视图。一个 step/group 构造一条新 schedule，因此每张图像有 100 条随机 schedule；ViT 的两个 checkpoint 对应 200 次 mask selection。不同 checkpoint 独立采样，允许重复选中同一空间位置。相位视图共享原始 schedule 的空间变换结果，而非另抽一套 mask。

当前各源的 checkpoint、实际网格和 drop 数见 [方法契约](prd_method_contract.md)。PiT-T 使用 14/2/2 的 checkpoint 丢弃数，PiT-B 额外保留自身的 5/2/6 配置；DeiT-B distilled 保留两个 prefix。每组视图数仍支持 1/2 消融。

相位配对、20-view 梯度平均、动量和投影更新属于攻击的实现流程；论文贡献陈述保持在上述两项核心机制上。

## 已完成的 1000 图像主实验

ASR 定义为 `1 - adversarial accuracy`，分母为全部 1000 张评估样本。目标集由 7 个 Transformer 和 6 个 CNN 组成；Overall 对 13 个目标取平均。严格黑盒只排除完全同名源目标，因此 ViT-B/16 和 CaiT-S24 各为 12 个目标，PiT-T 和 distilled DeiT-B 各为 13 个目标。

| 源模型 | Overall ASR | Transformer 平均 | CNN 平均 | 严格黑盒 Overall |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 88.02% | 91.17% | 84.35% | 87.06% |
| CaiT-S24 | 90.24% | 92.63% | 87.45% | 89.50% |
| PiT-T | 85.81% | 81.90% | 90.37% | 85.81% |
| DeiT-B distilled | 94.80% | 96.90% | 92.35% | 94.80% |
| **四源平均** | 89.72% | 90.65% | 88.63% | 89.29% |

主结果 52 项 source-target 评估均覆盖 1000 张图像。数值依据为 [FPR 逐目标表](../results/prd_s42_1000_fpr_per_target.csv)、[汇总表](../results/prd_s42_1000_fpr_summary.csv) 和 [各单元原始记录](../results/prd_s42_fpr_run_artifacts/)。

## 论文组织建议

1. **Introduction：** 用动态 patch 排序引出“在表示演化过程中施加干预”的设计问题；概括 PRD 的两个机制与四架构迁移结果。
2. **Observation and motivation：** 以同一样本的多层 patch-score 图和 token 分数变化曲线展示动态发现。可视化从实际样本和层输出准备；当前 results 仅保留 FPR13 迁移结果，辅助观察统计需另行准备。
3. **Method：** 用一张流程图呈现“当前对抗样本 → 新随机 schedule → 原始/相位配对 → 全部初始局部特征处的 RGB opponent-channel noise → 顺序 checkpoint drop → 梯度聚合与投影更新”。给出四个 adapter 的原生网格及默认预算。
4. **Experiments：** 主表展示四种源模型对 FPR13 的 13 个目标模型的迁移；报告 Overall、Transformer、CNN 和严格黑盒 ASR。公开方法比较统一数据、`16/255` 扰动预算、源/目标模型和 ASR 分母。
5. **Ablation and analysis：** 通过 ASR 对照呈现 progressive drop、opponent noise 及其组合的效果；围绕 checkpoint、drop budget、噪声强度和实际样本展开。梯度诊断只用于内部实验分析。

## 表达约定与结论主线

从动态发现进入方法：局部 token 与全局表示的关联沿深度变化，高分区域随之更替；PRD 顺着这一演化过程施加逐 checkpoint 随机中断，并扰动初始局部表示。以四源平均严格黑盒 ASR 89.29% 呈现完整方法的迁移表现。

表达直接、鲜明，围绕发现、设计和结果展开。省去对未提出主张的澄清，把篇幅用于描绘动态过程和解释方法。ASR 是论文的迁移性表征；effective rank 等梯度诊断保留在内部过程记录。完整约定见 `AGENTS.md` 的 Paper-writing objective and voice。
