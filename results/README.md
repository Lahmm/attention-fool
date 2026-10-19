# PRD 主实验结果

本目录保留 2026-10-12 完成的 PRD 主实验记录：全部初始局部特征注入 RGB 对手通道投影噪声，随后按随机 schedule 在多个 checkpoint 逐次清零局部 token。四种源架构分别攻击 1000 张图像，在相同的 14 个目标模型上评估。

四源平均 Overall ASR 为 **84.55%**，严格黑盒 ASR 为 **83.49%**；Transformer 平均为 **88.39%**，CNN 平均为 **79.44%**。ASR 定义为 `1 - adversarial accuracy`，分母为全部评估样本。

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

2026-10-19 已用当前代码和固定 seed 42 重新完成当前四源（ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled）的 1000 图像 14 目标测量，并完成完整的 2×2 机制消融。四源 Overall ASR 为 **84.91%**，严格黑盒 ASR 为 **84.46%**。记录、逐目标结果和因子效应见 [PRD seed-42 主运行与消融](prd_s42_1000_ablation.md)、[汇总 CSV](prd_s42_1000_summary.csv)、[效应 CSV](prd_s42_1000_effects.csv) 和 [逐目标 CSV](prd_s42_1000_per_target.csv)。

论文统一通过 transfer ASR 呈现迁移性；patch-score 用于动机观察，梯度诊断用于实验内部分析。

当前论文四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，另保留 PiT-B 源；新执行固定 seed 42。历史原始 CSV 与参数文件不改写为新设置的测量。
