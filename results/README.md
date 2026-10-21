# PRD 迁移结果

当前迁移测试以 FPR 13-target 协议为主，按 FPR 对齐的模型与预处理评估。project 14-target 协议保留为历史对照，结果单独呈现，不与 FPR 数值合并。ASR 定义为 `1 - adversarial accuracy`，分母为全部评估样本。

## 当前主协议结果

seed 42、1000 张图像、完整 PRD 的 FPR 13-target 汇总见 [FPR 汇总 CSV](prd_s42_1000_fpr_summary.csv) 和 [逐目标结果](prd_s42_1000_fpr_per_target.csv)。该批完整 PRD 的四源均值为 **89.16% Overall ASR**、**88.73% 严格黑盒 ASR**。PiT-T 随后在相同 seed、图像数及 FPR 协议下使用新选的 14/2/2 丢弃配置复核，Overall 与严格黑盒 ASR 均为 **85.81%**；逐候选记录见 [PiT-T 比例探索复核表](../outputs/pit_ratio_distribution_s42_20261021/confirmation_summary.csv)。2026-10-19 FPR 消融表中的 PiT-T full 值 85.34% 对应旧 4/2/5 配置，作为历史测量保留。

当前可用的 seed-42 FPR13 full-PRD 结果如下。ViT、CaiT 与 distilled DeiT 数值来自 2026-10-19 的主运行；PiT-T 使用 2026-10-21 复核的 14/2/2 配置。

| 源模型 | FPR13 测量时的 checkpoint | Overall ASR | Strict black-box ASR | Transformer ASR | CNN ASR |
| --- | --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | block3、block10 | 88.02% | 87.06% | 91.17% | 84.35% |
| CaiT-S24 | block17、block23 | 90.24% | 89.50% | 92.63% | 87.45% |
| PiT-T | S2-B1、S3-B2、S3-B3（14/2/2） | 85.81% | 85.81% | 81.90% | 90.37% |
| DeiT-B distilled | block3、block10 | 93.04% | 93.04% | 95.23% | 90.45% |

PiT-T 更新值来自本轮选层／比例实验；其余三行沿用各自 FPR13 完整 PRD 测量。注意，DeiT 当前默认选层为 block2、block9，而这份 seed-42 FPR13 记录使用 block3、block10，因此它还不是当前默认配置的测量；四个源模型完全对齐当前默认配置的 FPR13 汇总需要补测 DeiT。逐次运行的实际配置保存在各自 `attack_params.json` 中，读数跨运行汇总时不覆盖原始记录。

## 历史 project 14-target 结果

以下结果保留 2026-10-12 完成的 PRD 主实验记录：全部初始局部特征注入 RGB 对手通道投影噪声，随后按随机 schedule 在多个 checkpoint 逐次清零局部 token。该历史测量对四种源架构各攻击 1000 张图像，并在相同的 14 个目标模型上评估。

四源平均 Overall ASR 为 **84.55%**，严格黑盒 ASR 为 **83.49%**；Transformer 平均为 **88.39%**，CNN 平均为 **79.44%**。

| 文件 | 内容 |
| --- | --- |
| [prd_cross_arch_s1000.csv](prd_cross_arch_s1000.csv) | 56 项 source-target ASR、目标类别、source-match 标记和样本数 |
| [prd_summary_s1000.csv](prd_summary_s1000.csv) | 四个源模型及四源宏平均的 Overall、Transformer、CNN 和严格黑盒 ASR |
| [prd_run_artifacts/provenance.json](prd_run_artifacts/provenance.json) | 实际运行环境、实现提交、源配置与目标集 |
| `prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json` | 四次运行保留的 PRD 参数 |
| `prd_run_artifacts/{vit,cait,pit,visformer}/transfer.csv` | 四份迁移 CSV，包含实际运行时间、保留参数和逐目标结果 |
| [patch_rank_observation_summary.json](patch_rank_observation_summary.json) | 动机观察统计，呈现 token/global 关系随模型深度的变化 |

正式配置、结果表与计算规则见 [主实验报告](../experiments/progressive_cross_arch_mainline_s1000.md)。方法执行契约见 [PRD 方法记录](../experiments/prd_method_contract.md)。服务器上当前主实验图像、manifest 和日志的位置见 [写作交接](../experiments/server_handoff_inventory.md)。

本地归档参数已于 2026-10-14 精简为保留的 PRD 内容；历史 ASR 与实现提交保持原记录。

2026-10-19 另完成 project 14-target 的 seed-42 机制消融。该协议结果作为历史对照，记录、逐目标结果和因子效应见 [PRD seed-42 主运行与消融](prd_s42_1000_ablation.md)、[汇总 CSV](prd_s42_1000_summary.csv)、[效应 CSV](prd_s42_1000_effects.csv) 和 [逐目标 CSV](prd_s42_1000_per_target.csv)。

FPR 主运行与消融记录见 [FPR 主运行与消融记录](prd_s42_1000_fpr_ablation.md)、[FPR 汇总 CSV](prd_s42_1000_fpr_summary.csv)、[FPR 效应 CSV](prd_s42_1000_fpr_effects.csv) 和 [FPR 逐目标 CSV](prd_s42_1000_fpr_per_target.csv)。

论文统一通过 transfer ASR 呈现迁移性；patch-score 用于动机观察，梯度诊断用于实验内部分析。

当前论文四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，另保留 PiT-B 源；新执行固定 seed 42。历史原始 CSV 与参数文件不改写为新设置的测量。
