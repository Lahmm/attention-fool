# PRD 方法与执行契约

PRD 的两项核心机制是全部初始局部特征上的 RGB opponent-channel random noise，以及 checkpoint-wise progressive random token drop。

## 源与随机设置

当前论文四源为 ViT-B/16、CaiT-S24、PiT-T、DeiT-B distilled，另保留 PiT-B；Visformer 源实现已移除。所有新执行固定 seed 42。随机流按样本标识、step、group、view 与事件组织，不依赖 batch 位置。

| Source | Checkpoints | Native grids | Drop counts | Opponent strength |
| --- | --- | --- | --- | ---: |
| ViT-B/16 | `block3,block10` | 14×14, 14×14 | 10, 10 | 0.2 |
| CaiT-S24 | `block17,block23` | 14×14, 14×14 | 2, 28 | 0.2 |
| PiT-T | `stage2_block1,stage3_block2,stage3_block3` | 14×14, 7×7, 7×7 | 4, 2, 5 | 0.4 |
| DeiT-B distilled | `block2,block9` | 14×14, 14×14 | 10, 10 | 0.2 |
| PiT-B（额外保留） | `stage2_block1,stage3_block2,stage3_block3` | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.4 |

PiT-T 的初始网格为 27×27，PiT-B 为 31×31；上表列出各 checkpoint 的实际网格。两者 drop ratios 均为 `0.02081165,0.03125,0.09375`，按各站 token 数四舍五入得到不同 drop 数。DeiT-B distilled 保留两个 prefix 并使用原生分类输出。

## 执行顺序

1. 每个 step/group 从当前对抗像素构建 fresh schedule。无梯度地顺序到达 checkpoint，在全部原生局部位置中均匀、无放回抽样，立即 hard-zero 并继续遍历，保存 mask、数量和网格。
2. 每个可微视图只构建一次初始状态。将本视图独立采样的 opponent noise 加到全部初始局部特征，包括未来会被清零的位置。prefix 保持原值。
3. 按 schedule 顺序到达 checkpoint，清零选中 token 的全部通道，从改变后的状态继续传播。不同 checkpoint 独立抽样，允许重复位置；网格与 token 数不因清零改变。
4. 默认原始／平移两个视图。平移使用反射填充裁剪，mask 同步映射到图像空间、平移后按原生网格 occupancy top-k 保持 drop 数量。
5. 默认 10 groups×2 views。各视图对当前像素求梯度，直接平均后以系数 1 累积动量，执行 sign 更新、相对干净输入的 L∞ 投影和 [0,1] 截断；默认 10 steps、epsilon=16/255、步长 epsilon/steps。

`--input-diversity-views-per-group` 保留 1、2 两种设置用于消融。单视图时不采样平移，不构造平移图像或变换 masks，每组仍独立生成 schedule 和初始噪声。默认平移集合为 (4,4)/(8,8)/(12,12)，每步总视图不超过 20。

## 噪声

亮度、红绿、黄蓝方向为 `(1,1,1)/√3`、`(1,-1,0)/√2`、`(1,1,-2)/√6`，采样方差为 `0.5,1.25,1.25`。每个初始局部位置按 RGB 投影核采样，使用真实投影权重 W 映射：

`η = W ξ_RGB`，`η_hat = β RMS(H0)/RMS(η) · η`，`H0_noise = H0 + η_hat`。

RMS 在单样本全部初始局部位置和通道上计算。特征 RMS 与投影权重固定 detach；不提供可导尺度变体。噪声每个可微视图只注入一次。

## 记录与评估

`attack_params.json` 保存实际配置，包括 seed 42、源模型、checkpoint、drop 数量／网格、噪声强度和视图设置。攻击路径不计算或输出梯度诊断。迁移评估保留 project 与 FPR 目标协议，ASR 为全部对抗样本上的 `1 - adversarial accuracy`。

2026-10-12 的历史主实验实现为 bf092246；结果保留在 `results/prd_cross_arch_s1000.csv`、`results/prd_summary_s1000.csv` 和 `results/prd_run_artifacts/`。这些记录不因源支持调整或随机设置更新而改写数值，不作为当前代码的新测量。
