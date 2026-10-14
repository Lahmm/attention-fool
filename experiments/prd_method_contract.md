# PRD 方法与执行契约

PRD 仅包含两个核心机制：全部初始局部特征上的 RGB opponent-channel random noise，以及沿模型深度的 checkpoint-wise progressive random token drop。

## 执行顺序

1. 每个 attack step/group 从当前对抗像素构建一套随机 schedule。无梯度地顺序到达 checkpoint，在各 checkpoint 的全部原生局部位置中均匀、无放回抽取规定数量，立即 hard-zero，再继续遍历并保存 mask、数量与网格。
2. 原始可微视图从当前像素重新开始，在初始 RGB 投影后的全部局部特征上加入本视图独立采样的对手通道噪声。每个视图仅在这个位置注入一次，CLS/prefix 保持原值。
3. 顺序到达 checkpoint，使用记录的 mask 清零局部 token 的全部通道，沿已改变的状态继续前向至 loss。不同 checkpoint 独立抽样，允许重复选中同一位置。
4. 相位视图使用原始 schedule 的空间变换。图像空间反射平移后，按各 checkpoint 原生网格 occupancy top-k 保持 drop 数量；该视图独立采样初始噪声。
5. 默认每个 step 有 10 groups × 2 views。对各视图求像素梯度并平均，进行动量、sign 更新、相对干净输入的 L∞ 投影和 `[0,1]` 截断；重复 10 steps。

单视角消融接口：`--input-diversity-views-per-group 1` 仅使用当前对抗像素的原始视图，不采样 phase、不构造相位图像或变换 mask。每个 group 仍独立生成随机 schedule 并采样初始对手通道噪声。配合 `--input-diversity-groups 10`，每步平均 10 次原始视图梯度；配合 `--input-diversity-groups 1`，每步仅计算一次梯度。默认双视角配置保持不变。

当前代码默认 `gaussian_alpha=0`，不添加 Gaussian residual；显式设置正值时在梯度平均后、动量累积前添加。2026-10-12 四源主实验实际使用 `sigma=4, alpha=0.75`，历史记录保持不变。

## 噪声定义

全部局部位置在初始特征处接收噪声，包括 schedule 中之后会被 drop 的位置。亮度、红绿、黄蓝方向分别为 `(1,1,1)/√3`、`(1,-1,0)/√2`、`(1,1,-2)/√6`，采样方差为 `0.5, 1.25, 1.25`。每个初始局部位置按 RGB 投影核采样系数，经源模型真实初始投影权重映射到特征空间。

`η = W ξ_RGB`，`η_hat = β · RMS(H0) / RMS(η) · η`，`H0_noise = H0 + η_hat`。

RMS 在单样本的全部初始局部位置及通道上计算。特征 RMS 与投影权重使用 detach。ViT/CaiT 默认 `β=0.2`，PiT/Visformer 默认 `β=0.4`。噪声在特征空间注入，随后续计算传播。

噪声梯度消融开关：`--opponent-noise-grad`（构造参数 `opponent_noise_grad=True`）允许特征 RMS 分支向输入反传；默认关闭，可显式使用 `--no-opponent-noise-grad`。该开关保持噪声前向构造和投影权重的 detach 不变，记录于 `attack_params.json` 的 `opponent_noise_grad` 字段；上述默认配置及主实验保持特征 RMS 停止梯度。

## 跨架构与运行约定

四架构使用各自 adapter 的初始 RGB 投影和 checkpoint 遍历。ViT 当前配置为 `block3,block10`，每站独立 drop 10/196 个局部 token；其余配置见 [主实验报告](progressive_cross_arch_mainline_s1000.md)。token 数量和网格在清零操作中保持不变。

噪声 metadata：`opponent_noise=initial_rgb_projection_all_local_tokens`；强度为零时为 `disabled`。相位 mask 的空间映射与 RGB 投影元数据均为当前流程所需。

主实验完成于 2026-10-12，实现为 `bf092246`。结果唯一入口为 `results/prd_cross_arch_s1000.csv` 和 `results/prd_summary_s1000.csv`；实际参数、原始 CSV 和环境来源保存在 `results/prd_run_artifacts/`。transfer ASR 按全部评估样本计算。

## 写作约定

论文围绕动态局部证据、初始对手通道扰动、逐 checkpoint 随机中断及迁移 ASR 展开。Patch-score 可视化呈现动机；schedule 驱动攻击。梯度诊断作为实验内部工具。
