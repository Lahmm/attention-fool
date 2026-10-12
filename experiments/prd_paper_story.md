# PRD 论文叙事与主实验记录

本文方法与主实验统一采用 2026-10-12 完成的 PRD 运行，实现为 `bf092246`。准确执行顺序见 [PRD 方法记录](prd_method_contract.md)。

本文的中心贡献是 **Progressive Route Disruption（PRD）**：一种面向跨模型迁移的攻击方法。主线聚焦完整方法的设计与迁移效果，核心机制只有两个：沿模型深度的 checkpoint-wise progressive random token drop，以及施加在全部初始局部特征上的 RGB opponent-channel noise。

## 核心故事

1. **观察。** 同一个局部 token 与全局表示的余弦相似度会随网络深度发生明显变化，高 patch-score 的 patch 位置也会随层数变化而迁移与更替。用同一样本的多层 patch-score 可视化展示这一动态过程，让读者看到局部证据如何在前向传播中演化。Patch score 用于呈现发现，PRD 的执行由随机 checkpoint schedule 驱动。
2. **方法。** PRD 在多个架构专属 checkpoint 顺序地随机选择局部 token 并 hard-zero；每次 drop 后，模型都在已改变的 hidden state 上继续前向传播。每个攻击 step、每个 augmentation group 都重新构造随机 schedule。原始视图使用该 schedule，相位视图使用对应的空间变换 schedule。
3. **互补扰动。** PRD 在全部初始局部特征上加入噪声，包括未来会被 drop 的位置；噪声从亮度、红绿、黄蓝三个 RGB 对手方向采样，经源模型的初始 RGB projection 映射到特征空间并进行 RMS 匹配。
4. **结果。** 四种源架构各攻击 1000 张图像，在相同的 14 个目标模型上评估。PRD 的四源平均 Overall ASR 为 **84.55%**，平均严格黑盒 ASR 为 **83.49%**。

适合摘要或引言的核心表述：

> 随着图像向网络深处传播，同一个局部 token 与全局表示的关联会发生明显变化，高 patch-score 的区域也不断更替。这一动态发现启发我们沿着表示形成的过程发起攻击。我们提出 Progressive Route Disruption，在多个 checkpoint 对演化中的局部表示施加重新采样的随机中断，同时用 RGB 对手通道噪声扰动全部初始局部表示。在四种源架构、每源 1000 图像和 14 个目标模型的评估中，PRD 取得 83.49% 的平均严格黑盒 ASR。

## 方法事实与配置

PRD 默认使用 10 个迭代 step、每步 10 个 augmentation group、每组 2 个视图。一个 step/group 构造一条新 schedule，因此每张图像有 100 条随机 schedule；ViT 的两个 checkpoint 对应 200 次 mask selection。不同 checkpoint 独立采样，允许重复选中同一空间位置。相位视图共享原始 schedule 的空间变换结果，而非另抽一套 mask。

| 源模型 | 默认 checkpoint | 原生 token 网格 | 每个 checkpoint 的 drop 数 | Opponent strength |
| --- | --- | --- | --- | ---: |
| ViT-B/16 | `block3,block10` | 14×14、14×14 | 10、10 | 0.2 |
| CaiT-S24 | `block17,block23` | 14×14、14×14 | 2、28 | 0.2 |
| PiT-B | `stage2_block1,stage3_block2,stage3_block3` | 16×16、8×8、8×8 | 5、2、6 | 0.4 |
| Visformer-S | `stage2_block1,stage3_block1` | 14×14、7×7 | 41、10 | 0.4 |

相位配对、20-view 梯度平均、Gaussian residual、动量和投影更新属于攻击的实现流程；论文贡献陈述保持在上述两项核心机制上。

## 已完成的 1000 图像主实验

ASR 定义为 `1 - adversarial accuracy`，分母为送入目标模型评估的全部对抗样本，不筛选 target-clean-correct 子集。目标集由 8 个 Transformer 和 6 个 CNN 组成。Overall 是 14 个目标的均值；严格黑盒均值排除与源架构相同的目标。

| 源模型 | Overall ASR | Transformer 平均 | CNN 平均 | 严格黑盒 Overall |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 85.06% | 89.96% | 78.52% | 84.18% |
| CaiT-S24 | 87.64% | 90.89% | 83.32% | 86.79% |
| PiT-B | 84.34% | 89.55% | 77.38% | 83.24% |
| Visformer-S | 81.18% | 83.16% | 78.53% | 79.75% |
| **四源平均** | 84.55% | 88.39% | 79.44% | 83.49% |

四组攻击各评估 1000 张图像，56 项 source-target 迁移评估均覆盖全部样本，记录的 `skipped=0`。逐目标 ASR 与正式配置见 [主实验报告](progressive_cross_arch_mainline_s1000.md)；数值依据见 [逐目标表](../results/prd_cross_arch_s1000.csv)、[汇总表](../results/prd_summary_s1000.csv) 和 `results/prd_run_artifacts/{vit,cait,pit,visformer}/transfer.csv`。

## 论文组织建议

1. **Introduction：** 用动态 patch 排序引出“在表示演化过程中施加干预”的设计问题；概括 PRD 的两个机制与四架构迁移结果。
2. **Observation and motivation：** 以同一样本的多层 patch-score 图和 token 分数变化曲线展示动态发现。已有统计使用 64 张图像和统一的 7×7 空间网格，四种架构的早晚层排序 Spearman 约为 −0.07、0.18、0.06、0.12，作为观察的辅助信息。统计快照位于 `results/patch_rank_observation_summary.json`；可视化从实际样本和层输出准备。
3. **Method：** 用一张流程图呈现“当前对抗样本 → 新随机 schedule → 原始/相位配对 → 全部初始局部特征处的 RGB opponent-channel noise → 顺序 checkpoint drop → 梯度聚合与投影更新”。给出四个 adapter 的原生网格及默认预算。
4. **Experiments：** 主表展示四种源模型对 14 个目标模型的迁移；报告 Overall、Transformer、CNN 和严格黑盒 ASR。公开方法比较统一数据、`16/255` 扰动预算、源/目标模型和 ASR 分母。
5. **Ablation and analysis：** 通过 ASR 对照呈现 progressive drop、opponent noise 及其组合的效果；围绕 checkpoint、drop budget、噪声强度和实际样本展开。梯度诊断只用于内部实验分析。

## 表达约定与结论主线

从动态发现进入方法：局部 token 与全局表示的关联沿深度变化，高分区域随之更替；PRD 顺着这一演化过程施加逐 checkpoint 随机中断，并扰动初始局部表示。以四源平均严格黑盒 ASR 83.49% 呈现完整方法的迁移表现。

表达直接、鲜明，围绕发现、设计和结果展开。省去对未提出主张的澄清，把篇幅用于描绘动态过程和解释方法。ASR 是论文的迁移性表征；effective rank 等梯度诊断保留在内部过程记录。完整约定见 `AGENTS.md` 的 Paper-writing objective and voice。
