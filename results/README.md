# PRD FPR13 结果

本目录保存 FPR13 ASR、参数、直接 ASR 差值及来源。ASR 为全部对抗样本的 `1 - adversarial accuracy`。当前唯一可执行方法是方向方差比 × 正收缩比例逐图选层与单边界随机 token drop。

## 当前方法

[方向方差比 × 正收缩比例](prd_direction_contraction_s42_1000_fpr/README.md) 已完成 seed 42、四源各 1000 图、10 steps、10 groups × 2 views、opponent/FPR13 测量。Overall **89.1673%**、strict **88.7234%**。相对历史最大方差选层，分别高 1.8346、1.8543 个百分点。

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| vit | 87.95% | 86.95% |
| cait | 89.14% | 88.37% |
| pit_t | 84.28% | 84.28% |
| deit_b | 95.30% | 95.30% |

当前归档含 `summary.csv`、52 项逐目标 `per_target.csv`、全部候选频次 `selection_counts.csv`、四源 `runs/` 参数与迁移表、对照原件、源码差异及 provenance。当前方法最初在外部快照执行，2026-11-04 晋升主线，保留实际 `experiment_only` 标签和执行来源，不重标为 i.i.d.。

## 保留的历史选层证据

| 目录 | 配置身份 |
| --- | --- |
| [direction_variance_ratio](prd_direction_variance_ratio_s42_1000_fpr/README.md) | 仅方向方差比，无正收缩因子 |
| [second_raw_moment](prd_second_raw_moment_s42_1000_fpr/README.md) | 最大二阶原点矩 |
| [std_abs_mean](prd_std_abs_mean_s42_1000_fpr/README.md) | 标准差加绝对均值 |
| [normalized_std_abs_mean](prd_normalized_std_abs_mean_s42_1000_fpr/README.md) | 按图层间最大值分别归一化后相加 |

历史最大方差对照的原始参数和迁移表保留在上述归档的 `variance_references/`，以及当前方法归档中。原始报告中的 mainline 指测量当时状态；这些记录和 `.patch` 仅用于来源核验，不提供可执行的历史策略。

## 清理与归档完整性

2026-11-04 按用户要求移除强固定单层基线、其引用与汇总差值，以及旧固定双／多 checkpoint 方法的主表、消融和专属分析。涉及删除固定层引用的两个动态归档更新了 `archive_sha256`，在 `archive_maintenance` 保存之前 provenance、summary 和 README 的哈希。保留的原始攻击参数、迁移数据及源码快照哈希不变。

固定 block10 的 token 选择分析作为历史证据单独保留，见 [token 证据](../experiments/prd_token_selection_evidence.md)，不等同于固定单层基线或当前动态消融。原始预测、图像与日志位于服务器 `outputs/`，来源见 [handoff](../experiments/server_handoff_inventory.md)。

评估口径见 [FPR13 协议](../experiments/fpr_target_protocol.md)。最终 i.i.d. 四源主表和匹配设置的选层对照仍需对应实际运行记录。
