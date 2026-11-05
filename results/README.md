# GDI FPR13 结果索引

更新日期：2026-11-05。本目录保存 FPR13 ASR、参数、直接 ASR 差值和 provenance。ASR 为全部对抗样本的 `1 - adversarial accuracy`；strict 排除完全相同源目标，四源等权汇总。

## 当前 i.i.d. 主结果

[GDI i.i.d. 配对四源归档](gdi_iid_paired_s42_1000_fpr/README.md) 对应 seed 42、每源 1000 图、10 steps、10 groups × 2 views。GDI 逐图以方向方差比 × 正收缩比例定位，在所选 block 输入处均匀随机清零局部 token。

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| ViT-B/16 | 87.41% | 86.36% |
| CaiT-S24 | 89.75% | 88.98% |
| PiT-T | 84.85% | 84.85% |
| DeiT-B distilled | 95.37% | 95.37% |
| 四源等权平均 | **89.3423%** | **88.8893%** |

归档保存原始参数和迁移表、52 项逐目标结果、选层频次、执行 manifest 与文件哈希。论文整体迁移效果以该实际 i.i.d. 配置组织，匹配选层对照按 blueprint 继续准备。

## 历史选层比较

以下均为各自实际执行配置的历史证据。

| 目录 | 配置身份 |
| --- | --- |
| [direction_contraction](gdi_direction_contraction_s42_1000_fpr/README.md) | GDI 乘积定位准则，历史 opponent 设置；Overall 89.1673%、strict 88.7234% |
| [direction_variance_ratio](gdi_direction_variance_ratio_s42_1000_fpr/README.md) | 仅方向方差比 |
| [second_raw_moment](gdi_second_raw_moment_s42_1000_fpr/README.md) | 最大二阶原点矩 |
| [std_abs_mean](gdi_std_abs_mean_s42_1000_fpr/README.md) | 标准差加绝对均值 |
| [normalized_std_abs_mean](gdi_normalized_std_abs_mean_s42_1000_fpr/README.md) | 按图层间最大值分别归一化后相加 |

历史乘积准则相对同配置最大方差选择器，Overall、strict 的直接差值为 +1.8346、+1.8543 个百分点。最大方差原始参数与迁移表保存在各归档的 `variance_references/`。完整频次按 `selection_counts.csv` 展示。

固定 block10 的 token 选择与固定／重采样分析见 [历史 token 证据](../experiments/gdi_token_selection_evidence.md)，对应原记录的增强和视图预算。

## 命名与来源

文档、报告标题及归档目录统一采用 GDI。原始测量参数、迁移数据、执行标识和源码差异按原字节保留，原始服务器位置由相应 `provenance.json` 与 `execution.json` 提供。每个历史归档记录此次目录迁移和说明更新的维护前哈希，`archive_sha256` 覆盖全部保留文件。

2026-11-04 清理强固定单层基线及旧固定双／多 checkpoint 结果的维护历史继续保留。权重、原始对抗图像、预测和日志由服务器保存，见 [交接记录](../experiments/server_handoff_inventory.md)。
