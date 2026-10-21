# PRD：1000 图像跨架构主实验

主实验完成日期：2026-10-12。实际运行实现：`bf09224675093cd6a4fcccee2db10acae0eb5f4e`。本文结果保留原四源、1000 图像、14 目标的历史测量。当前迁移测试以 FPR 13-target 协议为主，本文 project 14-target 数值作为历史对照单独报告。当前源支持已调整，所有新执行固定 seed 42；不将历史结果视为新配置重测。

## 方法与配置

PRD 的核心机制是 checkpoint-wise progressive random token drop 与 RGB opponent-channel random noise。每个可微视图在初始 RGB 投影后的全部局部特征上加入一次 RMS 匹配的对手通道噪声，随后沿同一轨迹到达各 checkpoint，将选中的局部 token 所有通道清零并继续前向。CLS/prefix 保持原值。

每个 attack step/group 重新生成随机 schedule。构建过程无梯度地依次遍历 checkpoint，从各自原生网格的全部局部位置均匀抽样并立即清零；正式视图重新计算初始特征并注入噪声，再使用记录的 mask。不同 checkpoint 独立采样，允许重复位置。原始视图使用该 schedule，相位视图使用其空间变换并保持各 checkpoint 的 drop 数量。

| Source | Checkpoints | 原生网格 | Drop 数量 | Opponent β | 生成 batch size |
| --- | --- | --- | --- | ---: | ---: |
| ViT-B/16 | block3, block10 | 14×14, 14×14 | 10, 10 | 0.2 | 96 |
| CaiT-S24 | block17, block23 | 14×14, 14×14 | 2, 28 | 0.2 | 48 |
| PiT-B | stage2_block1, stage3_block2, stage3_block3 | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 | 96 |
| Visformer-S | stage2_block1, stage3_block1 | 14×14, 7×7 | 41, 10 | 0.4 | 48 |

共同约束：`epsilon=16/255`、10 steps、step size `epsilon/10`、动量系数 1、10 groups × 2 views、相位平移集合 `(4,4)/(8,8)/(12,12)`。每张图像生成 100 套 schedule；ViT/CaiT/Visformer 进行 200 次 checkpoint mask selection，PiT 进行 300 次。四源 drop ratios 与保留参数见 `results/prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json`。

## 迁移结果

ASR = `1 - adversarial accuracy`，分母是全部 1000 张评估的对抗图像。目标集包含 8 个 Transformer 和 6 个 CNN。Overall 对全部 14 个目标取平均；严格黑盒对排除 source-matched target 后的 13 个目标取平均。四源宏平均对四个源模型等权平均。

| Source | Overall ASR | Transformer（8） | CNN（6） | 严格黑盒（13） |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 85.06% | 89.96% | 78.52% | 84.18% |
| CaiT-S24 | 87.64% | 90.89% | 83.32% | 86.79% |
| PiT-B | 84.34% | 89.55% | 77.38% | 83.24% |
| Visformer-S | 81.18% | 83.16% | 78.53% | 79.75% |
| **四源平均** | 84.55% | 88.39% | 79.44% | 83.49% |

全部 56 项 source-target 评估均使用 1000 张图像，记录的 skipped 数为零。四源平均严格黑盒 ASR 为 **83.49%**。Transformer 的 88.39% 包含 source-matched target；排除这些目标后的 Transformer 平均 ASR 为 **86.97%**。CNN 平均 ASR 为 **79.44%**。

## 逐目标 ASR

| Target | ViT source | CaiT source | PiT source | Visformer source |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 96.40% | 86.20% | 81.80% | 64.70% |
| LeViT-256 | 87.00% | 90.40% | 87.00% | 88.00% |
| PiT-B/224 | 87.30% | 89.50% | 98.60% | 86.60% |
| DeiT-B/16 | 89.60% | 91.40% | 91.40% | 79.80% |
| TNT-S/16 | 89.70% | 91.40% | 90.10% | 86.00% |
| ConViT-B | 87.30% | 90.20% | 89.70% | 78.50% |
| Visformer-S | 87.80% | 89.30% | 90.20% | 99.70% |
| CaiT-S24 | 94.60% | 98.70% | 87.60% | 82.00% |
| Inception-v3 | 79.90% | 85.20% | 82.80% | 86.00% |
| Inception-v4 | 78.20% | 83.70% | 79.30% | 85.50% |
| Inception-ResNet-v2 | 78.90% | 84.30% | 77.40% | 81.00% |
| ResNet-101 | 80.90% | 85.10% | 80.80% | 83.90% |
| Inception-v3-adv | 78.90% | 83.70% | 75.40% | 74.60% |
| Inception-ResNet-v2-adv | 74.30% | 77.90% | 68.60% | 60.20% |

## 原始记录与环境

- [逐目标规范化表](../results/prd_cross_arch_s1000.csv)：56 条结果。
- [四源汇总](../results/prd_summary_s1000.csv)：各源及四源宏平均。
- [运行环境与来源](../results/prd_run_artifacts/provenance.json)：实际实现、环境、源配置和目标集。
- `results/prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json`：保留的 PRD 参数。
- `results/prd_run_artifacts/{vit,cait,pit,visformer}/transfer.csv`：迁移 CSV，含时间戳、保留参数与逐目标 ASR。

实际环境：Python 3.10.19、PyTorch 2.10.0+cu128、timm 1.0.24、CUDA 12.8、NVIDIA RTX 4090。目标评估启用 AMP 与 TF32，使用各目标模型自己的 transform。模型权重缓存位于 `data/huggingface/hub`。

本地归档参数已于 2026-10-14 精简为保留的 PRD 内容，不再是完整运行时参数快照；逐目标 ASR、时间戳、实现提交和服务器输出目录名称保持原记录。现有结果来自上述历史运行，尚未用当前代码重新测量。主表与汇总由这些记录计算；迁移性论证统一使用 ASR。当前运行的对抗图像、replay manifests、逐样本预测和日志保存在服务器，下载位置见 [写作交接](server_handoff_inventory.md)。
