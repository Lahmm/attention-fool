# GDI i.i.d. 配对四源 FPR13 主结果

归档日期：2026-11-05。该归档整理已经完成的四源运行。GDI 对每张干净图像计算方向方差比 × 正收缩比例，在最高分 block 的输入处均匀随机清零局部 token，优化可迁移扰动。

## 实际设置

每源 1000 图，seed 42、epsilon=16/255、10 steps、步长 epsilon/10、动量系数 1、10 groups × 2 views。每图选层一次、原始 mask 抽样 100 次；原始／反射平移视图使用空间对应 mask。增强为投影前 i.i.d. RGB 标准高斯，经初始投影和逐样本 RMS 匹配。

| Source | drop 比例 | β | 攻击 batch |
| --- | ---: | ---: | ---: |
| ViT-B/16 | 20/196 | 0.2 | 48 |
| CaiT-S24 | 30/196 | 0.2 | 16 |
| PiT-T | 30/196 | 0.4 | 32 |
| DeiT-B distilled | 20/196 | 0.2 | 48 |

评估为 FPR13 的 13 个目标，batch 32、关闭 TF32、无 AMP。ASR 为全部对抗样本的 `1 - adversarial accuracy`；strict 排除完全相同源目标，四源等权汇总。

## 迁移结果

| Source | Overall ASR | Strict ASR |
| --- | ---: | ---: |
| ViT-B/16 | 87.41% | 86.36% |
| CaiT-S24 | 89.75% | 88.98% |
| PiT-T | 84.85% | 84.85% |
| DeiT-B distilled | 95.37% | 95.37% |
| 四源等权平均 | **89.3423%** | **88.8893%** |

`summary.csv` 保存四源及等权平均的原始精度 ASR；`per_target.csv` 保存全部 52 项逐目标结果；`selection_counts.csv` 保存实际选层记录中出现的边界与频次。全部 52 项 ASR 已按服务器原始逐样本预测与标签核对。

## 参数与来源

`runs/{vit,cait,pit_t,deit_b}/` 中的 `attack_params.json`、`transfer.csv`、`transfer.json` 按原字节复制。测量时的参数字段和方法标识保留其执行身份；增强实际分布由 `noise_structure=isotropic_rgb` 和执行 manifest 中的强度记录确定。

`execution_manifest.json` 保存原始运行命令、commit、源码哈希和目录；`execution_source.sha256` 保存原始源码清单；`provenance.json` 记录来源文件哈希、逐源服务器位置、ASR 核对及完整归档清单。原始对抗图像、逐样本预测和日志保存在这些来源目录。

当前方法说明与代码接口使用 GDI。论文写作见 [故事](../../experiments/gdi_paper_story.md) 和 [blueprint](../../experiments/cvpr2027_writing_blueprint.md)。
