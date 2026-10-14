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

当前代码默认关闭 Gaussian residual（`gaussian_alpha=0`）；显式设置 `--gaussian-alpha 0.75` 可开启。2026-10-12 主实验实际使用 `sigma=4, alpha=0.75`，其历史参数与结果保持原记录。

## 当前配置

| Source | Checkpoints | Native grids | Drop counts | Opponent strength |
| --- | --- | --- | --- | ---: |
| ViT-B/16 | `block3,block10` | 14×14, 14×14 | 10, 10 | 0.2 |
| CaiT-S24 | `block17,block23` | 14×14, 14×14 | 2, 28 | 0.2 |
| PiT-B | `stage2_block1,stage3_block2,stage3_block3` | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 |
| Visformer-S | `stage2_block1,stage3_block1` | 14×14, 7×7 | 41, 10 | 0.4 |

ViT-B/16 的唯一当前执行示例：

```bash
python main.py \
  --attack-method progressive \
  --whitebox-model vit_base_patch16_224 \
  --checkpoints block3,block10 \
  --drop-ratios 0.051020408163,0.051020408163
```

`main.py` 只调度 `progressive_attack.py` 中的 `ProgressiveRouteDisruptionAttacker`。四个 architecture adapter 只负责初始 RGB projection 元数据、checkpoint traversal、local-token mask application 和原生 forward completion。

## 1000-image transfer results

主实验于 2026-10-12 完成，实际实现为 `bf092246`。下表统一使用当前全局初始噪声与 progressive drop 的四源结果。

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

论文以 ASR 呈现迁移效果。梯度诊断保留为实验过程中的内部分析工具。写作结构、图表计划和表达约定见 [CVPR 2027 写作蓝图](experiments/cvpr2027_writing_blueprint.md)。

## 运行与验证

默认图像位于 `data/clean_resized_images`，标签为 `data/image_name_to_class_id_and_name.json`，模型从 `data/huggingface` 离线缓存读取。

```bash
/root/miniconda3/envs/att-atk/bin/python transfer_eval.py \
  --image-dir outputs/attack/progressive_route_disruption \
  --prefix adv_

/root/miniconda3/envs/att-atk/bin/python -m unittest discover -s tests
```

安装依赖：

```bash
pip install -r requirements.txt
```
