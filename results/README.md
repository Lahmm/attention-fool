# PRD 迁移结果

本目录仅保留 FPR 13-target 迁移结果及其参数、消融、统计与来源记录，使用 FPR 对齐的模型和预处理。非 FPR13 结果文件已清理，历史内容可从 Git 历史查询。ASR 定义为 `1 - adversarial accuracy`，分母为全部评估样本。

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

## 文件索引

| 文件或目录 | 内容 |
| --- | --- |
| [FPR 完整实验记录](prd_s42_1000_fpr_ablation.md) | 当前配置、四源主结果、逐目标矩阵与完整机制消融 |
| [汇总 CSV](prd_s42_1000_fpr_summary.csv) | 16 个源/消融单元与四源宏平均 |
| [逐目标 CSV](prd_s42_1000_fpr_per_target.csv) | 208 个源/消融单元/目标组合的 ASR |
| [效应 CSV](prd_s42_1000_fpr_effects.csv) | 40 个配对效应及 95% bootstrap 区间 |
| [McNemar CSV](prd_s42_1000_fpr_mcnemar.csv) | 156 项探索性逐目标配对统计 |
| [Provenance JSON](prd_s42_1000_fpr_provenance.json) | seed 约定、运行参数、日期、代码提交和校验哈希 |
| [原始记录目录](prd_s42_fpr_run_artifacts/) | 全部 16 个单元的参数与原始迁移 CSV/JSON，以及可复算的压缩逐图像成功标记 |

方法执行契约见 [PRD 方法记录](../experiments/prd_method_contract.md)，服务器产物位置见 [写作交接](../experiments/server_handoff_inventory.md)。论文以 ASR 呈现迁移性；patch-score 用于动机可视化，观察材料从实际样本与层输出准备。
