# PRD 迁移结果

当前迁移测试以 FPR 13-target 协议为主，按 FPR 对齐的模型与预处理评估。project 14-target 协议保留为历史对照，结果单独呈现，不与 FPR 数值合并。ASR 定义为 `1 - adversarial accuracy`，分母为全部评估样本。

## 当前主协议结果

当前四源设置已完成 seed 42、1000 张图像的 FPR13 完整 PRD 与 2×2 机制消融。2026-10-23 对 PiT-T 和 distilled DeiT-B 的 `drop/full` 四个单元完成重跑；其余 12 个单元沿用参数未变的 2026-10-19 测量。按照用户约定，历史 `randomness.py` seed 42 与当前 PyTorch seed 42 统一作为 seed-42 标准，随机实现差异不要求补跑。

完整 PRD 四源平均 **Overall ASR 为 89.72%**，**严格黑盒 ASR 为 89.29%**。Transformer 平均为 **90.65%**，CNN 平均为 **88.63%**。

| 源模型 | 当前 checkpoint / drop 数量 | Overall ASR | Strict black-box ASR | Transformer ASR | CNN ASR |
| --- | --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | block3、block10 / 10、10 | 88.02% | 87.06% | 91.17% | 84.35% |
| CaiT-S24 | block17、block23 / 2、28 | 90.24% | 89.50% | 92.63% | 87.45% |
| PiT-T | S2-B1、S3-B2、S3-B3 / 14、2、2 | 85.81% | 85.81% | 81.90% | 90.37% |
| DeiT-B distilled | block2、block9 / 10、10 | 94.80% | 94.80% | 96.90% | 92.35% |

| 源模型 | base（均关闭） | drop（仅丢弃） | noise（仅噪声） | full（完整 PRD） |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 64.50% | 72.92% | 79.32% | 88.02% |
| CaiT-S24 | 78.24% | 81.75% | 85.14% | 90.24% |
| PiT-T | 70.00% | 72.32% | 83.99% | 85.81% |
| DeiT-B distilled | 88.18% | 90.01% | 94.46% | 94.80% |
| 四源平均 | 75.23% | 79.25% | 85.73% | 89.72% |

上述数值和逐目标结果、效应及置信区间均已同步到 [FPR 完整实验记录](prd_s42_1000_fpr_ablation.md)、[汇总 CSV](prd_s42_1000_fpr_summary.csv)、[逐目标 CSV](prd_s42_1000_fpr_per_target.csv) 与 [效应 CSV](prd_s42_1000_fpr_effects.csv)。完整方法在 DeiT 上比仅噪声高 0.34 个百分点，配对 95% CI 为 [-0.10, +0.80]，区间跨零。

这批记录受 Git 管理。[轻量原始记录目录](prd_s42_fpr_run_artifacts/) 包含全部 16 个单元的真实参数 JSON、迁移 CSV/JSON；[provenance](prd_s42_1000_fpr_provenance.json) 保存代码 commit、测量日期、原始路径和文件哈希。[压缩逐图像成功标记](prd_s42_fpr_run_artifacts/paired_outcomes.npz) 同时保存所有单元的逐样本成功与否、样本标识及标签，足以在其他机器复算 ASR、配对区间与 McNemar 统计。另一台机器拉取仓库即可查看和复核这些结果；攻击图像、原始预测类别和日志继续保存在被忽略的服务器 `outputs/` 目录。

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
