# PRD 迁移结果

本目录仅保留 FPR 13-target 迁移结果及其参数、消融、ASR 差值与来源记录，使用 FPR 对齐的模型和预处理。非 FPR13 结果文件已清理，历史内容可从 Git 历史查询。ASR 定义为 `1 - adversarial accuracy`，分母为全部评估样本。

## 当前唯一主线结果

[方向方差比 × 正收缩比例](prd_direction_contraction_s42_1000_fpr/README.md) 已完成四源各 1000 图、seed 42、10 steps、10 groups × 2 views 的 opponent/FPR13 测试。Overall 89.1673%、strict 88.7234%；相对固定单层为 −0.3750 和 −0.3897 个百分点。该配置最初在外部快照执行，2026-11-04 迁入主线，原归档的 experiment_only 标签和哈希仍如实描述测量时状态。归档同时保留历史方差、二阶矩、标准差加均值和原方向方差比对照。

## 已归档固定配置结果

本页已归档数值属于固定多 checkpoint＋opponent 配置。当前唯一主线为方向方差比 × 正收缩比例选层与随机 token drop，共同增强采用 i.i.d. RGB 采样；叙事及动态运行索引见 [故事文档](../experiments/prd_paper_story.md)。已有数值与参数保持原标签。

该固定四源设置已完成 seed 42、1000 张图像的 FPR13 完整攻击与 2×2 消融。2026-10-23 对 PiT-T 和 distilled DeiT-B 的 `drop/full` 四个单元完成重跑；其余 12 个单元沿用参数未变的 2026-10-19 测量。按照用户约定，历史 `randomness.py` seed 42 与当前 PyTorch seed 42 统一作为 seed-42 标准，随机实现差异不要求补跑。

完整 PRD 四源平均 **Overall ASR 为 89.72%**，**严格黑盒 ASR 为 89.29%**。Transformer 平均为 **90.65%**，CNN 平均为 **88.63%**。

| 源模型 | 固定配置 checkpoint / drop 数量 | Overall ASR | Strict black-box ASR | Transformer ASR | CNN ASR |
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

上述 ASR、逐目标结果和直接差值已同步到 [FPR 完整实验记录](prd_s42_1000_fpr_ablation.md)、[汇总 CSV](prd_s42_1000_fpr_summary.csv)、[逐目标 CSV](prd_s42_1000_fpr_per_target.csv) 与 [效应 CSV](prd_s42_1000_fpr_effects.csv)。完整方法在 DeiT 上比仅噪声高 0.34 个百分点。

这批记录受 Git 管理。[轻量原始记录目录](prd_s42_fpr_run_artifacts/) 包含全部 16 个单元的真实参数 JSON、迁移 CSV/JSON；[provenance](prd_s42_1000_fpr_provenance.json) 保存代码 commit、测量日期、原始路径和文件哈希。另一台机器拉取仓库即可查看和复核 ASR 与直接差值；攻击图像、逐样本预测和日志继续保存在被忽略的服务器 `outputs/` 目录。

## 文件索引

| 文件或目录 | 内容 |
| --- | --- |
| [FPR 完整实验记录](prd_s42_1000_fpr_ablation.md) | 当前配置、四源主结果、逐目标矩阵与完整机制消融 |
| [汇总 CSV](prd_s42_1000_fpr_summary.csv) | 16 个源/消融单元与四源宏平均 |
| [逐目标 CSV](prd_s42_1000_fpr_per_target.csv) | 208 个源/消融单元/目标组合的 ASR |
| [效应 CSV](prd_s42_1000_fpr_effects.csv) | 40 个直接 ASR 效应点估计，单位为百分点 |
| [Provenance JSON](prd_s42_1000_fpr_provenance.json) | seed 约定、运行参数、日期、代码提交和校验哈希 |
| [原始记录目录](prd_s42_fpr_run_artifacts/) | 全部 16 个单元的参数与原始迁移 CSV/JSON |

方法执行契约见 [PRD 方法记录](../experiments/prd_method_contract.md)，服务器产物位置见 [写作交接](../experiments/server_handoff_inventory.md)。论文以 ASR 呈现迁移性；当前 patch score 用于动态观察和方向方差比组成量，选层还使用 block 前后的正收缩比例，层内 drop 位置仍均匀随机抽样。

## 二阶原点矩选层独立测试

[四源 FPR13 测试记录](prd_second_raw_moment_s42_1000_fpr/README.md) 保存逐图最大二阶原点矩选层的结果，以及相对现有方差选层运行的直接 ASR 差值。每源 1000 图、seed 42、10 steps、10 groups × 2 views，沿用对应基线的 opponent 增强及 drop 比例。此项仅为独立测试，该测量时的攻击主线为最大方差选层；2026-11-04 起唯一主线为方向方差比 × 正收缩比例；选层改动位于仓库外的执行快照，源码差异与参数、逐图选层和逐目标结果一并归档。
