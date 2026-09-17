# Patch-Score Routing Attack

本项目研究一种面向黑盒迁移的“语义路由 + 颜色结构随机扰动”攻击。当前代码和主实验记录聚焦完整攻击主线与必要对照。

## 当前主线

主线由两个核心机制组成：

1. **patch-score-guided patch drop**：用 adapter-specific global/local 表示关系（CLS 架构为 cosine，GAP 架构为 projection）提供 label-free、gradient-independent 的语义坐标，决定在哪里扰动；
2. **RGB opponent-channel noise**：在亮度、红绿和黄蓝方向采样，再经过模型首层 RGB projection，决定如何扰动保留证据。

当前研究主线是 progressive high-score attack。ViT-B/16 统一使用 `block3,block10`
两个 checkpoint boundary，在每个 checkpoint 从 high-score half 随机抽取 10/196 个
local tokens（`drop_ratio=0.051020408163`）并 hard-zero；后续 checkpoint 在已经过
前序 drop 的 token 状态上继续计算，允许重复选择同一位置。

每个 attack step、每个 augmentation group 都从当前对抗像素构造新的两级 ViT schedule。
original view 使用该 schedule，phase view 使用空间变换后的对应 schedule。默认 10 steps ×
10 groups，因此每张 ViT 图构造 100 个 schedule、执行 200 次 checkpoint mask selection。

```text
current adversarial pixels
→ sequential global/local scores at block3 and block10
→ 10/196 high-score-window local-token drop at each checkpoint
→ original schedule / spatially transformed phase schedule
→ kept-only opponent noise at the initial RGB projection
→ raw 20-view gradient mean
→ Gaussian residual (sigma=4, alpha=0.75)
→ MI update
```

当前 ViT 主线只通过 `main.py` 执行：

```bash
python main.py \
  --whitebox-model vit_base_patch16_224 \
  --checkpoints block3,block10 \
  --drop-ratios 0.051020408163,0.051020408163 \
  --progressive-patch-selector high
```

`main.py` 的 ViT 默认、文档参考和测试基准均为上述 `block3,block10` 配置，不再保留
另一套 ViT reference schedule。实验接口支持任意非零数量的、严格递增的 checkpoint，并要求
`--drop-ratios` 提供完全相同数量的逐层比例；不对这些比例的总和施加额外限制。自定义
`N` 个 checkpoint 时，默认 10 steps × 10 groups 对应每张图 `100 × N` 次 checkpoint
mask selection。

`main.py` 已通过 architecture adapters 承载 ViT、CaiT、PiT 和 Visformer 的
progressive 主线。ViT-B/16 默认使用 `block3,block10`，drop ratios 为
`0.051020408163,0.051020408163`（实际 drop 数 `10,10`）。CaiT-S24 默认使用
`block17_gap,block23_gap`，drop ratios 为
`0.010204081633,0.142857142857`（实际 drop 数 `2,28`），score 为
`gap_projection`。PiT-B 当前默认使用筛选出的 L2 配置
`stage2_block1,stage3_block2,stage3_block3`，对应 drop ratios
`0.02081165,0.03125,0.09375`（实际 drop 数 `5,2,6`），opponent strength 为
`0.4`。Visformer-S 默认使用
`stage2_block1,stage3_block1`，drop ratios 为
`0.209183673469,0.204081632653`（实际 drop 数 `41,10`），score 为
`gap_projection`，opponent strength 为 `0.4`。ViT/CaiT opponent strength 为
`0.2`。四个默认配置均已完成
1000 图验证。

默认数据位于 `data/clean_resized_images`，标签为 `data/image_name_to_class_id_and_name.json`，模型从 `data/huggingface` 离线缓存读取。

## 保留的攻击接口

项目只保留由 `main.py` 调用的 progressive attack。四个架构 adapter 只实现
progressive checkpoint traversal、score feature、mask application、forward completion
以及初始 RGB projection 元数据，不再提供 final-layer legacy score、token hook 或旧式
resumable-forward API。

`high`/`low` 分别从 patch-score 高/低半区随机抽取当前层预算；`extreme-high`/
`extreme-low` 直接按 score 排序选取最高/最低的当前层预算；`random` 从全部 local
tokens 均匀随机抽取。

默认 progressive phase-pair 每步为 20 个实际 model views。Score-global noise 与
Gaussian residual 是已完成控制变量的支撑因素，不作为新增论文核心机制。

## 主线结果

`progressive_attack.py` 是 `main.py` 唯一调用的攻击实现。ViT、CaiT、PiT、Visformer
四个源模型均已完成 1000 图攻击和完整的
14 目标复评（8 Transformer，包括 ViT-B/16；6 CNN）；Overall ASR 分别为
**85.26%、86.09%、84.79% 和 78.52%**；Transformer/CNN 均值分别为
90.34/78.48、89.68/81.32、90.25/77.50 和 80.54/75.83。四个源模型对自身架构的
ASR 分别为 **97.0%、98.0%、98.8% 和 99.4%**。每图均动态
生成 100 个 schedule；当前 ViT K=2 配置执行 200 次 checkpoint mask 选择。
完整逐迁移模型结果见 `experiments/progressive_cross_arch_mainline_s1000.md`。

ViT 当前正式配置为 `block3,block10`、10/10 drops、`selector=high`、
`score-window-ratio=0.5` 和 opponent strength 0.2；其 1000 图完整 14-target Overall
ASR 为 85.26%。旧 checkpoint schedule 上得到的 selector/noise 数值不再作为当前 ViT
攻击设置陈述。

完整的架构契约、测试门禁、逐源结果、控制变量和梯度诊断见
`experiments/progressive_cross_arch_mainline_s1000.md`。

## 迁移评估与测试

```bash
python transfer_eval.py --image-dir outputs/attack/vit_mainline --prefix adv_
python -m unittest discover -s tests
```

迁移评估覆盖项目注册的 Transformer/CNN target，并写入 `outputs/csv`。项目统一定义
`ASR = 1 - adversarial accuracy`：分母是送入目标模型评估的全部对抗样本，不筛选
target-clean-correct 子集。后续 CSV 会在 `asr_definition` 字段中保存该定义。除 ASR 外，
同时关注 cross-model gradient cosine、sign agreement、held-out one-step response 与 full
iterative transfer；source clean-logit suppression 不能替代这些指标。

安装依赖：

```bash
pip install -r requirements.txt
```
