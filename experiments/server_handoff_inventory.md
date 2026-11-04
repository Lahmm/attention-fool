# 论文写作交接与材料清单

更新日期：2026-11-04。本地分支为 `prd`。相对路径均以仓库根目录为起点。论文主线为方向方差比 × 正收缩比例选层与随机 token drop，i.i.d. RGB 投影噪声归入共同增强，正式方法名待定。

## 主线与实验状态

一次干净前向采集各 block 局部输入／输出，以输出方向方差比乘正收缩比例确定最高分 block，在其输入处实施随机 token drop；边界固定，位置按 step/group 重采样。方法见 [执行契约](prd_method_contract.md)，叙事、章节和图表见 [故事](prd_paper_story.md) 与 [蓝图](cvpr2027_writing_blueprint.md)。

当前唯一方法的四源 opponent 结果已归档 `results/prd_direction_contraction_s42_1000_fpr/`，Overall 89.1673%、strict 88.7234%，原始目录 `outputs/attack/prd_direction_contraction_s42_20261104T044138Z_{vit,cait,pit_t,deit_b}/`。该独立实验于 2026-11-04 迁入主线，原归档标签保持实际执行状态。

历史最大方差动态选层四源各 1000 图像、13 目标评估，仍为 opponent 增强。当前 CaiT 定义对应后续 preln 运行。最终 i.i.d. 配置与匹配选层矩阵需要实际记录支持。代码默认 opponent 未在此次文档任务中修改。

已归档固定多 checkpoint/opponent 配置的 Overall 89.72%、strict 89.29% 和完整 2×2 消融保留原含义。新增动态记录不从旧表改标签生成，所有 ASR 使用全部对抗样本。

## 随 Git 提交的材料

| 文件或目录 | 用途 |
| --- | --- |
| `AGENTS.md`、`README.md` | 方法主线、操作约定与结果入口 |
| `progressive_attack.py`、`main.py`、`nets/` | 当前攻击实现与架构遍历 |
| `transfer_eval.py`、`experiments/fpr_target_protocol.md` | FPR13 目标模型、权重及预处理 |
| `results/prd_s42_1000_fpr_ablation.md` | 历史固定配置主表、逐目标矩阵、2×2 消融及直接差值 |
| `results/prd_s42_1000_fpr_summary.csv` | 各源、各单元及四源宏平均 |
| `results/prd_s42_1000_fpr_per_target.csv` | 208 项 source/cell/target ASR |
| `results/prd_s42_1000_fpr_effects.csv` | 消融 ASR 效应点估计 |
| `results/prd_s42_1000_fpr_provenance.json` | 参数、来源路径、代码版本与哈希 |
| `results/prd_s42_fpr_run_artifacts/` | 16 个单元的原始参数和迁移表 |
| `experiments/prd_paper_story.md`、`experiments/cvpr2027_writing_blueprint.md` | 论文故事与写作顺序 |

当前 `results/` 只保留 FPR13 实验数值与来源材料。旧协议及已清理的观察统计可从 Git 历史查询；动机可视化从实际样本与层输出另行准备。

## 历史最大方差动态与 token 选择原件

下列目录均含参数与评估记录，位于 `outputs/attack/`，不随 Git 提交。各动态目录含 1000 条选层记录和 13 目标迁移 CSV/JSON。

| 目录 | 内容 |
| --- | --- |
| `prd_dynamic_variance_s42_20261029T031203Z_vit/` | 历史 ViT 最大方差选层＋opponent |
| `prd_dynamic_variance_s42_20261029T031203Z_pit_t/` | PiT-T 动态选层＋opponent |
| `prd_dynamic_variance_s42_20261029T031203Z_deit_b/` | distilled DeiT-B 动态选层＋opponent |
| `prd_cait_preln_variance_s42_20261029T070204Z/` | 当前 CaiT 参照定义的动态选层＋opponent |
| `prd_dynamic_variance_s42_20261029T031203Z_cait/` | 早期 CaiT 参照运行，另行保留，不替代后续 preln 记录 |
| `vit_drop_selection_s42_20261029T125456Z/` | 固定 block10 六组 token 策略，10 groups×2 views，含 asr_comparison.csv |
| `vit_drop_selection_g1v2_s42_20261101T131251Z/` | 固定 block10 六组 token 策略，1 group×2 views，含 asr_comparison.csv |

最终 i.i.d. 主表确定后另行归档必要的轻量 FPR13 参数与原始迁移记录。`figure1.py` 已支持 ViT 原始分数导出，完整方差曲线与新版 Fig. 2 尚需准备；`figure2.py` 当前仍生成旧固定双 checkpoint 素材。

## 历史固定配置与通用服务器原件

| 路径 | 用途 |
| --- | --- |
| `outputs/reruns/prd_s42_adjusted_20261023T120255Z/` | 四个更新单元的日志、迁移结果与原始预测 |
| `outputs/attack/prd_s42_adjusted_20261023T120255Z_{drop,full}_{pit_t,deit_b}/` | 更新单元的攻击图像与参数 |
| `outputs/fpr_eval/prd_s42_{base,drop,noise,full}_{vit,cait,pit_t,deit_b}/` | 早期 FPR13 评估原件；哪些单元继续使用以 provenance 为准 |
| `outputs/predictions_fpr/` | 保留单元的逐图像预测 |
| `data/clean_resized_images/`、标签 JSON | 攻击输入与标注 |
| `data/huggingface/`、`data/torch/` | 权重缓存 |

更新单元环境为 PyTorch 2.10.0+cu128、timm 1.0.24、RTX 3080。FPR13 评估关闭 TF32，batch size 32。`outputs/` 与 `data/` 不随 Git 提交；已归档的轻量数值足够在其他机器重算 ASR 和 ASR 差值。
