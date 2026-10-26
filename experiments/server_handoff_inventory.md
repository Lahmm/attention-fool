# PRD 写作交接与材料清单

更新日期：2026-10-26。本地分支为 `prd`；当前主结果为 seed-42、四源、1000 图像、FPR13。更新的四个攻击单元运行代码为 `2d2625b`。相对路径均以仓库根目录为起点。

## 主线与主实验

PRD 以随深度演化的局部证据为动机，在全部初始局部特征中加入 RGB opponent-channel 投影噪声，随后在同一前向轨迹的多个 checkpoint 随机 hard-zero 局部 token。执行顺序与当前源参数见 [方法契约](prd_method_contract.md)。

四源为 ViT-B/16、CaiT-S24、PiT-T 和 distilled DeiT-B。完整 PRD Overall ASR 为 **89.72%**，严格黑盒 ASR 为 **89.29%**。全部 16 个消融单元使用相同 1000 张图像；每个单元在 FPR13 的 13 个目标上评估。完整数值、配对统计与来源见 [FPR 实验记录](../results/prd_s42_1000_fpr_ablation.md)。

## 随 Git 提交的材料

| 文件或目录 | 用途 |
| --- | --- |
| `AGENTS.md`、`README.md` | 方法主线、操作约定与结果入口 |
| `progressive_attack.py`、`main.py`、`nets/` | 当前攻击实现与架构遍历 |
| `transfer_eval.py`、`experiments/fpr_target_protocol.md` | FPR13 目标模型、权重及预处理 |
| `results/prd_s42_1000_fpr_ablation.md` | 主表、逐目标矩阵、完整 2×2 消融及配对统计 |
| `results/prd_s42_1000_fpr_summary.csv` | 各源、各单元及四源宏平均 |
| `results/prd_s42_1000_fpr_per_target.csv` | 208 项 source/cell/target ASR |
| `results/prd_s42_1000_fpr_effects.csv` | 消融效应与配对区间 |
| `results/prd_s42_1000_fpr_mcnemar.csv` | 探索性逐目标配对检验 |
| `results/prd_s42_1000_fpr_provenance.json` | 参数、来源路径、代码版本与哈希 |
| `results/prd_s42_fpr_run_artifacts/` | 16 个单元的原始参数、迁移表与可复算的逐图像成功标记 |
| `experiments/prd_paper_story.md`、`experiments/cvpr2027_writing_blueprint.md` | 论文故事与写作顺序 |

当前 `results/` 只保留 FPR13 实验数值与来源材料。旧协议及已清理的观察统计可从 Git 历史查询；动机可视化从实际样本与层输出另行准备。

## 服务器原件

| 路径 | 用途 |
| --- | --- |
| `outputs/reruns/prd_s42_adjusted_20261023T120255Z/` | 四个更新单元的日志、迁移结果与原始预测 |
| `outputs/attack/prd_s42_adjusted_20261023T120255Z_{drop,full}_{pit_t,deit_b}/` | 更新单元的攻击图像与参数 |
| `outputs/fpr_eval/prd_s42_{base,drop,noise,full}_{vit,cait,pit_t,deit_b}/` | 早期 FPR13 评估原件；哪些单元继续使用以 provenance 为准 |
| `outputs/predictions_fpr/` | 保留单元的逐图像预测 |
| `data/clean_resized_images/`、标签 JSON | 攻击输入与标注 |
| `data/huggingface/`、`data/torch/` | 权重缓存 |

更新单元环境为 PyTorch 2.10.0+cu128、timm 1.0.24、RTX 3080。FPR13 评估关闭 TF32，batch size 32。`outputs/` 与 `data/` 不随 Git 提交；已归档的轻量数值足够在其他机器重算 ASR 和配对统计。
