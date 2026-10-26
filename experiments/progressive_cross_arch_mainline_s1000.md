# PRD：FPR13 四源 1000 图像主实验

当前主结果使用 seed 42、四个源模型、1000 张图像和 FPR13 的 7 个 Transformer / 6 个 CNN 目标。完整主表、逐目标矩阵、2×2 机制消融和配对区间统一维护在 [FPR13 实验记录](../results/prd_s42_1000_fpr_ablation.md)。

完整 PRD 的四源 Overall ASR 为 **89.72%**，严格黑盒 ASR 为 **89.29%**；Transformer 平均为 **90.65%**，CNN 平均为 **88.63%**。源模型为 ViT-B/16、CaiT-S24、PiT-T 和 distilled DeiT-B。所有 ASR 使用全部评估样本，不筛选 target-clean-correct 子集。

## 方法与当前配置

PRD 在全部初始局部特征上注入 RGB opponent-channel 投影噪声，随后沿同一前向轨迹按 fresh、uniform random checkpoint schedule 清零局部 token。原始视图与空间平移视图使用对应的 schedule。具体 checkpoint、drop ratio、噪声强度与执行顺序见 [方法契约](prd_method_contract.md)；目标权重及预处理见 [FPR 协议](fpr_target_protocol.md)。

2026-10-23 使用 `2d2625b` 完成 PiT-T 与 distilled DeiT-B 的 drop/full 四个配置更新单元；另 12 个单元保留 2026-10-19 参数未变的测量。按照用户约定，两种 seed-42 实现属于同一实验报告标准。各单元采用同样的 1000 张图像标识和标签。

## 受版本控制的结果

- [汇总 CSV](../results/prd_s42_1000_fpr_summary.csv)：各源、各消融单元与宏平均。
- [逐目标 CSV](../results/prd_s42_1000_fpr_per_target.csv)：完整 208 项 source/cell/target ASR；其中 full 主结果为 4×13 项。
- [效应 CSV](../results/prd_s42_1000_fpr_effects.csv)：主效应、交互项、配对差值及 95% CI。
- [来源记录](../results/prd_s42_1000_fpr_provenance.json)：日期、代码提交、seed 约定、参数与哈希。
- [轻量原始记录](../results/prd_s42_fpr_run_artifacts/)：全部 16 个单元的参数 JSON、迁移 CSV/JSON 和逐图像成功标记。

非 FPR13 的旧结果已从当前 `results/` 清理；历史版本可通过 Git 查询。服务器图像与日志位置见 [材料清单](server_handoff_inventory.md)。
