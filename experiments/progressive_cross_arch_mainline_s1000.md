# 四源 FPR13：当前论文主线与历史结果入口

更新日期：2026-11-04。论文聚焦逐图方向方差比 × 正收缩比例选层与最高分 block 输入处的随机 token drop；采用 i.i.d. RGB 投影随机特征增强作为共同设置。PRD 暂为仓库工作名称。

## 当前方法与证据

四源为 ViT-B/16、CaiT-S24、PiT-T 和 distilled DeiT-B；每图只在干净前向选层一次，所有攻击步保持边界固定，位置按 step/group 均匀重采样。方法事实见 [执行契约](prd_method_contract.md)，论文结构见 [蓝图](cvpr2027_writing_blueprint.md)。

当前方法的 opponent 四源 FPR13 结果已归档 [结果](../results/prd_direction_contraction_s42_1000_fpr/README.md)：Overall 89.1673%、strict 88.7234%。独立实验于 2026-11-04 被确认为唯一主线。

历史最大方差动态选层的 seed-42、1000 图像、13 目标评估已在本地完成，仍使用 opponent 增强；包括当前 CaiT 归一化前 CLS 参照的后续运行。原始目录及 ASR 见 [故事与证据记录](prd_paper_story.md)。最终 i.i.d. 配置的主表和匹配选层对照须对应实际运行记录。

## 已归档的固定配置

`results/prd_s42_1000_fpr_*` 保存固定多 checkpoint＋opponent 配置的完整主表、2×2 消融和直接差值。该配置四源 Overall ASR 为 89.72%、strict 为 89.29%，Transformer 为 90.65%、CNN 为 88.63%。这些数值保留原方法和参数含义，作为历史证据。

2026-10-23 使用 `2d2625b` 更新 PiT-T、distilled DeiT-B 的 drop/full 四个单元，其余十二单元保留参数匹配的 2026-10-19 测量。历史 randomness.py seed 42 与当前 PyTorch seed 42 按用户约定统一报告。

| 受版本控制的材料 | 内容 |
| --- | --- |
| [固定配置实验记录](../results/prd_s42_1000_fpr_ablation.md) | 固定配置参数、主表、2×2 消融 |
| [汇总](../results/prd_s42_1000_fpr_summary.csv)、[逐目标](../results/prd_s42_1000_fpr_per_target.csv) | 原配置 ASR；full 为 52 项，完整消融为 208 项 |
| [差值](../results/prd_s42_1000_fpr_effects.csv)、[来源](../results/prd_s42_1000_fpr_provenance.json) | 直接差值、日期、参数和校验记录 |
| [轻量原始记录](../results/prd_s42_fpr_run_artifacts/) | 十六单元参数与迁移 CSV/JSON |

FPR13 为 7 Transformer＋6 CNN，ASR 使用全部对抗样本，strict 排除完全相同源目标。非 FPR13 旧结果从 Git 历史查询。服务器原件位置与移交状态见 [材料清单](server_handoff_inventory.md)。
