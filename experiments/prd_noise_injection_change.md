# PRD 初始噪声方式变更：kept-only → full-local

决策与实现日期：2026-10-12。本记录明确用户提出的新方式，供后续 agent 实现、验证和写作时遵循。

## 讨论结论

未来会被 drop 的位置也可以在初始阶段加入噪声。噪声随表示经过卷积或 attention 传播，checkpoint drop 随后清零指定位置当时的表示。早期传播出去的噪声影响可以继续参与后续计算，这是新方式允许的传播过程。

旧方式提前排除未来 drop 区域，是“噪声扰动保留证据”的空间分工。覆盖比例 `>0.5` 用于在跨尺度、重叠卷积窗口上落实该分工。新方式采用全部初始局部特征加噪声，再沿深度执行随机 drop。

## 旧方式与新方式

| 项目 | 旧方式：kept-only | 新方式：full-local |
| --- | --- | --- |
| 注入时点 | 初始 RGB 投影后的特征处、checkpoint 遍历前 | 同一时点 |
| 注入范围 | 由所有 checkpoint drop 区域并集映射出的保留位置 | 全部初始局部位置，包括未来 drop 的位置 |
| 初始特征更新 | `H0 + K(schedule) ⊙ noise` | `H0 + noise` |
| CLS/prefix | 不加噪声 | 不加噪声 |
| 空间门控 | 图像空间并集、初始卷积覆盖比例、`>0.5` 阈值 | 移除 |
| 噪声构造 | RGB 对手方向采样、真实初始 RGB 权重投影、单样本 feature RMS 匹配 | 保持该定义和强度 |
| checkpoint drop | 按记录的随机 mask 顺序 hard-zero | 保持该流程 |

“全域加噪声”具体指初始特征张量的全部 local tokens，并在所有通道上添加投影噪声。输入 RGB 像素通过现有投影生成初始特征；噪声由 RGB 方向构造后投影到特征空间再注入。

## 当前执行顺序

1. 每个 attack step/group，从当前对抗像素开始一遍无梯度的 schedule 构建：依次到达各 checkpoint，均匀随机抽样并立即 hard-zero，保存 checkpoint、原生网格、数量和布尔 mask。该遍不注入 opponent noise。
2. 正式可微前向从当前像素重新开始，重新计算初始特征；独立生成本视图的 opponent noise，将其加到全部初始局部特征。
3. 沿同一次可微前向依次到达 checkpoint，使用记录的 mask 清零局部 token，再继续前向至 loss。
4. 相位视图使用原始 schedule 的空间变换版本，并独立采样该视图的初始噪声。下一 step/group 重新生成 schedule 和噪声。
5. 保持多视图梯度聚合、Gaussian residual、动量与像素投影更新。

每个可微视图仅在初始特征处注入一次噪声。此变更保留现有 schedule 预演过程。

## Agent 修改约定

- `_forward_with_schedule` 中采用 `local = local + self._strict_opponent_feature_noise(initial)`；噪声范围不依赖 schedule。
- 删除仅用于噪声门控的 `_schedule_image_union`、`_image_mask_to_projection_drop_mask` 及覆盖比例阈值。
- 保留 `_mask_to_image` 和 `_phase_mask_schedule`：相位 mask 的跨尺度变换仍需要它们。
- 保留 RGB 投影权重、kernel 等元数据：对手通道噪声的构造仍需要它们。
- 保持噪声方差 `0.5, 1.25, 1.25`、RMS 计算范围、detach、各架构默认噪声强度。直接施加原定义的匹配后噪声；本次变更不增加按旧保留比例缩放的规则。
- 保持 ViT `block3,block10`、各 10/196 的当前配置；保持四架构原生遍历、均匀独立抽样、跨 checkpoint 重复位置和相位配对。
- 当前 metadata 的启用值为 `opponent_noise=initial_rgb_projection_all_local_tokens`；强度为零时为 `disabled`。
- 验证未来 drop 位置在第一层前已收到噪声、CLS/prefix 保持原值、checkpoint 仍清零预定位置，以及两视图梯度与扰动预算。

## 实验与写作版本

已有四源、每源 1000 图像、14 目标的正式结果来自 `98cb7cf1` 的 kept-only 方式，四源平均 Overall ASR 为 84.20%，严格黑盒 ASR 为 83.10%。原始 CSV、attack parameters、replay manifests 与图像哈希保持原记录。

full-local 方式尚无新的 transfer ASR。后续评估使用新 metadata 和独立输出目录记录结果；写作中明确方法版本与测量对应关系。移除门控后，保持现有噪声强度会使更多位置收到噪声；区域策略的比较据此说明其实际施加范围。

## 实现验证

2026-10-12 在本地 `attion-fool` 环境（PyTorch 2.9.1、timm 1.0.22）运行全部 unittest：37 项，35 项通过，2 项需要显式启用的真实模型测试跳过。

另以随机初始化权重运行四种真实架构的 CPU 检查：224×224 输入、各自默认 checkpoint 和噪声强度、1 step/group、原始与 `(4,4)` 相位两视图；四者均生成全部初始局部位置的非零投影噪声，并得到有限且非零的图像梯度。该检查验证结构与反向传播；transfer ASR 使用训练后的权重和正式图像集另行评估。
