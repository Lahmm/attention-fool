# PRD 写作交接与材料清单

记录日期：2026-10-12。本地分支为 `prd`；主实验实现为 `bf092246`。此文档保存历史主实验材料位置；当前源支持与执行配置以方法契约为准。相对路径均以仓库根目录为起点。

## 主线与主实验

同一个局部 token 与全局表示的余弦相似度随深度发生明显变化，高 patch-score 区域随之迁移与更替。PRD 在全部初始局部特征中注入 RGB 对手通道噪声，让扰动随表示传播；到达各 checkpoint 时随机 hard-zero 局部 token，继续同一次前向。方法契约见 [PRD 方法记录](prd_method_contract.md)。

历史主实验的四种源架构分别攻击 1000 张输入，在相同的 14 个目标模型上评估。主实验完成于 2026-10-12，四源平均 Overall ASR 为 **84.55%**，严格黑盒 ASR 为 **83.49%**。Transformer 平均为 **88.39%**，CNN 平均为 **79.44%**。ASR 按全部评估样本计算，56 项评估各为 1000 样本且记录 skipped 为零。详细表与配置见 [主实验报告](progressive_cross_arch_mainline_s1000.md)。

| 历史 Source | Checkpoints | 原生网格 | Drop 数量 | Opponent β | 生成 batch size |
| --- | --- | --- | --- | ---: | ---: |
| ViT-B/16 | block3, block10 | 14×14, 14×14 | 10, 10 | 0.2 | 96 |
| CaiT-S24 | block17, block23 | 14×14, 14×14 | 2, 28 | 0.2 | 48 |
| PiT-B | stage2_block1, stage3_block2, stage3_block3 | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 | 96 |
| Visformer-S | stage2_block1, stage3_block1 | 14×14, 7×7 | 41, 10 | 0.4 | 48 |

历史运行的共同预算：`epsilon=16/255`、10 steps、step size `epsilon/10`、动量系数 1、10 groups × 2 views、相位集合 `(4,4)/(8,8)/(12,12)`。当前源配置见方法契约；新执行使用 `main.py`，固定 seed 42；具体主实验的保留参数和输出位置见归档迁移 CSV。

## 本地材料

| 文件或目录 | 用途 |
| --- | --- |
| `AGENTS.md` | 单一方法主线、表达与操作约定 |
| `README.md` | 方法、配置和主结果入口 |
| `progressive_attack.py`, `main.py`, `nets/` | PRD 实现和当前源遍历 |
| `randomness.py`, `utils.py`, `transfer_eval.py` | 固定随机流、输入输出与 ASR 评估 |
| `tests/` | 当前执行契约验证 |
| `data/clean_resized_images/` 与标签 JSON | 1000 张干净输入及标注 |
| `experiments/prd_method_contract.md` | 噪声注入与 progressive drop 的准确顺序 |
| `experiments/prd_paper_story.md` | 论文故事和章节建议 |
| `experiments/cvpr2027_writing_blueprint.md` | 全文纲要、图表计划和写作顺序 |
| `experiments/progressive_cross_arch_mainline_s1000.md` | 当前主实验协议、汇总和逐目标结果 |
| `results/prd_cross_arch_s1000.csv` | 56 项 source-target ASR |
| `results/prd_summary_s1000.csv` | 各源及四源汇总 |
| `results/prd_run_artifacts/provenance.json` | 环境、实现提交、源配置和目标集 |
| `results/prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json` | 保留的 PRD 参数 |
| `results/prd_run_artifacts/{vit,cait,pit,visformer}/transfer.csv` | 迁移 CSV、时间戳和保留参数 |
| `results/patch_rank_observation_summary.json` | 64 图像、统一 7×7 网格的动机观察统计 |

动机统计中的四架构早晚层 Spearman 约为 −0.071、0.182、0.063、0.117。写作优先准备同一样本的多层 patch-score 图和固定位置 token 的分数曲线。

## 服务器当前运行材料

本地保存主结果 CSV、参数与环境信息。

本地归档参数已于 2026-10-14 精简为保留的 PRD 内容；历史 ASR 与实现提交保持原记录，尚未用当前代码重新测量。

当前运行的对抗图像、replay manifests、逐样本预测和日志保存在服务器；使用原始记录中的真实目录名称定位。

| 服务器路径 | 用途 |
| --- | --- |
| `outputs/attack/prd1000_full_local_{vit,cait,pit,visformer}_*/` | 每个源的 1000 张当前主实验对抗图像及运行 metadata |
| `outputs/runs/prd1000_full_local_*/` | 当前运行日志和评估相关材料 |
| `data/huggingface/hub/` | source/target 模型权重缓存 |

正式运行环境为 Python 3.10.19、PyTorch 2.10.0+cu128、timm 1.0.24、CUDA 12.8、RTX 4090。目标评估启用 AMP/TF32，按各目标 transform 执行。

只写论文时，本地材料足以整理方法、配置和 ASR 表。准备定性对抗图像、逐样本分析或精确重放时，从上述当前运行目录取得对应原件；使用下载后实际获得的 manifest 和图像校验信息进行核对。
