# Progressive Route Disruption

本项目研究面向黑盒迁移的 **Progressive Route Disruption（PRD）**。我们观察到：同一个局部 token 与全局表示的余弦相似度会随网络深度发生明显变化，高 patch-score 的区域也随之迁移与更替。局部证据参与全局表示形成的过程是动态的。PRD 以 RGB 对手通道噪声扰动全部初始局部特征，并在多个 checkpoint 反复随机中断局部 token 的传播。

Patch score 只用于得到上述研究观察；PRD 的执行完全由 checkpoint-wise random schedules 驱动。

## 方法主线

PRD 只有两个核心机制：

1. **checkpoint-wise progressive random token drop**：在 architecture-specific checkpoints 从全部当前 local-token 位置中均匀随机采样并 hard-zero，随后在已经修改的 hidden state 上继续同一次前向传播；
2. **RGB opponent-channel random noise**：在亮度、红绿和黄蓝方向采样，经模型真实初始 RGB projection 映射并在特征空间 RMS matching，注入全部初始局部特征，包括之后会被 drop 的位置；CLS/prefix 不加噪声。

每个 attack step、每个 augmentation group 都生成新的随机 schedule。Original view 使用该 schedule，phase view 使用其空间变换后的对应 schedule。默认 10 steps × 10 groups × 2 views；ViT 的两个 checkpoint 因而对应每张图 100 条 schedule 和 200 次 mask selection。

```text
current adversarial pixels
→ fresh uniformly random checkpoint schedule
→ original schedule / spatially transformed phase schedule
→ RGB opponent-channel noise on all initial local features
→ sequential checkpoint traversal and local-token hard zeroing
→ raw 20-view gradient mean
→ MI projected update
```

## 当前配置

论文当前四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled；PiT-B 额外保留源支持。Visformer 仅作为评估目标和历史结果中的模型保留。

| Source | Checkpoints | Native grids | Drop counts | Opponent strength |
| --- | --- | --- | --- | ---: |
| ViT-B/16 | `block3,block10` | 14×14, 14×14 | 10, 10 | 0.2 |
| CaiT-S24 | `block17,block23` | 14×14, 14×14 | 2, 28 | 0.2 |
| PiT-T | `stage2_block1,stage3_block2,stage3_block3` | 14×14, 7×7, 7×7 | 4, 2, 5 | 0.4 |
| DeiT-B distilled | `block3,block10` | 14×14, 14×14 | 10, 10 | 0.2 |
| PiT-B（额外保留） | `stage2_block1,stage3_block2,stage3_block3` | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 |

PiT-T 与 PiT-B 共享 drop ratios，但网格不同，上表数量按实际 timm 网格计算。DeiT-B distilled 的 CLS 和 distillation token 均不加噪声、不清零，输出使用模型原生双 head。

所有新执行统一使用 seed **42**，不可通过命令行改成其他值。checkpoint 由源适配器固定；保留 drop ratios、噪声强度、步数、步长和增强组数作为方法参数。

```bash
python main.py --whitebox-model vit_base_patch16_224 --output-dir outputs/attack/prd_vit
python main.py --whitebox-model pit_ti_224 --output-dir outputs/attack/prd_pit_t
python main.py --whitebox-model deit_base_distilled_patch16_224 --output-dir outputs/attack/prd_deit_b
```

`--input-diversity-views-per-group 1` 使用原始视图，`2` 使用原始／平移配对（默认）；`--input-diversity-groups` 控制每步组数，每步总视图不超过 20。初始 opponent noise 的特征 RMS 和投影权重固定停止梯度。像素梯度平均后直接进行系数为 1 的动量累积与投影更新。

`main.py` 保存对抗 PNG 与 `attack_params.json`；`randomness.py` 只负责按样本与攻击上下文生成随机数，不输出梯度诊断或 replay manifest。现存实验运行脚本已移除，使用上述入口执行方法。

## 1000-image transfer results

以下历史主实验于 2026-10-12 完成，实际实现为 `bf092246`，源模型为 ViT-B/16、CaiT-S24、PiT-B、Visformer-S。结果不代表新四源或 seed 42 的重测。

本地归档参数已于 2026-10-14 精简为保留的 PRD 内容；历史 ASR 与实现提交保持原记录，尚未用当前代码重新测量。

下表使用该次四源运行的记录结果。

ASR 定义为 `1 - adversarial accuracy`，分母是送入目标模型评估的全部对抗样本，不筛选 target-clean-correct 子集。四个 PRD source 均在相同的 14 个目标模型（8 Transformer、6 CNN）上完成评估：

| Source | Overall ASR | Transformer avg | CNN avg | Strict black-box overall |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | 85.06% | 89.96% | 78.52% | 84.18% |
| CaiT-S24 | 87.64% | 90.89% | 83.32% | 86.79% |
| PiT-B | 84.34% | 89.55% | 77.38% | 83.24% |
| Visformer-S | 81.18% | 83.16% | 78.53% | 79.75% |
| **Four-source mean** | 84.55% | 88.39% | 79.44% | 83.49% |

Strict black-box 指标排除与 source 架构相同的 target。56 个 source-target evaluation 均使用完整 1000 张对抗样本且没有 skipped images。逐目标结果和规范化 CSV 记录见 `experiments/progressive_cross_arch_mainline_s1000.md`。

方法执行顺序见 [PRD 方法记录](experiments/prd_method_contract.md)。论文叙事、主实验记录与建议写作结构见 [PRD 论文叙事文档](experiments/prd_paper_story.md)。

异地写作所需的当前主实验参数、原始记录和服务器材料位置见 [写作交接与文件清单](experiments/server_handoff_inventory.md)。

论文以 ASR 呈现迁移效果。写作结构、图表计划和表达约定见 [CVPR 2027 写作蓝图](experiments/cvpr2027_writing_blueprint.md)。

## 评估与验证

图像位于 `data/clean_resized_images`，标签为 `data/image_name_to_class_id_and_name.json`，源与 Transformer 目标从 `data/huggingface` 离线缓存读取。

```bash
python transfer_eval.py --image-dir outputs/attack/prd_vit
python transfer_eval.py --image-dir outputs/attack/prd_vit --protocol fpr --no-tf32
python -m unittest discover -s tests
RUN_REAL_PRD_ADAPTERS=1 python -m unittest tests.test_progressive_adapters
```

评估保留 project 的 14 目标和 FPR 的 13 目标协议，使用各自模型加载和预处理；目标集的选择独立于源支持。结果默认写入图像目录的 `transfer.csv` 与 `transfer.json`，可用 `--predictions-output` 保存逐样本预测。ASR 使用全部输入样本；严格黑盒排除与源模型完全相同的目标。

权重准备工具 `experiments/download_fpr_models.py` 保留，负责缓存 FPR 目标和额外的 PiT-T／DeiT-B distilled 源权重，不执行攻击或评估。

```bash
pip install -r requirements.txt
```
