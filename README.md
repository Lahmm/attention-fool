# Patch-Score Routing Attack

本项目研究一种面向黑盒迁移的“语义路由 + 颜色结构随机扰动”攻击。当前代码和主实验记录聚焦完整攻击主线与必要对照。

## 当前主线

主线由两个核心机制组成：

1. **patch-score-guided patch drop**：用 global/local 表示余弦关系提供 label-free、gradient-independent 的语义坐标，决定在哪里扰动；
2. **RGB opponent-channel noise**：在亮度、红绿和黄蓝方向采样，再经过模型首层 RGB projection，决定如何扰动保留证据。

当前研究主线已经晋升为 progressive high-score attack。已验证的 ViT-B/16 配置在
`(3, 7, 11)` 三个 checkpoint boundary 上依次计算当前 global/local patch score，每次从
high-score half 随机抽取 5% local tokens 并 hard-zero；后续 checkpoint 在已经过前序
drop 的 token 状态上继续计算，允许不同 checkpoint 重复选择同一位置。

每个 attack step、每个 augmentation group 都从当前对抗像素构造新的三级 schedule。
original view 使用该 schedule，phase view 使用空间变换后的对应 schedule。默认 10 steps ×
10 groups，因此每张图构造 100 个 schedule、执行 300 次 checkpoint mask selection。

```text
current adversarial pixels
→ sequential global/local scores at checkpoint boundaries 3, 7, 11
→ 5% high-tail local-token drop at each checkpoint
→ original schedule / spatially transformed phase schedule
→ kept-only opponent noise at the initial RGB projection
→ raw 20-view gradient mean
→ Gaussian residual (sigma=4, alpha=0.75)
→ MI update
```

当前主线的可执行参考仍位于独立的 ViT 文件中：

```bash
python vit_progressive_patch_score_attack.py \
  --checkpoints 3,7,11 \
  --drop-ratios 0.05,0.05,0.05 \
  --patch-selector high
```

`main.py` 暂时仍承载旧的 final-layer cross-architecture 实现；在 progressive 逻辑完成
跨架构适配并迁入之前，不应把 `main.py` 的默认行为称为当前研究主线。

默认数据位于 `data/clean_resized_images`，标签为 `data/image_name_to_class_id_and_name.json`，模型从 `data/huggingface` 离线缓存读取。

## 保留的攻击接口

| 类别 | 当前保留接口 | 定位 |
| --- | --- | --- |
| 当前研究主线 | `vit_progressive_patch_score_attack.py` | `3,7,11` progressive high-score schedule；当前仅 ViT |
| 历史跨架构基线 | `original_score_postdrop_phase_pair` | final-layer 动态 pixel drop；暂由 `main.py` 保留 |
| 基础路径 | `none` | 无 patch drop 的优化基线 |
| 像素对照 | `patch_dropout` | 通用 pixel patch dropout |
| token 对照 | `token_patch_dropout` | ViT token patch dropout |
| 优化与增强 | MI、NI、DIM、TI | 支撑机制和受控消融，不是新的论文主机制 |
| Progressive selector | `high`、`low`、`random` | 主线 high 与两个受控路由对照 |

`none`、pixel `patch_dropout`、token `patch_dropout` 与 NI/DIM/TI 的示例：

```bash
python main.py --attack-method none --dim --ni --ti-sigma 1.0 \
  --gaussian-alpha 0 --output-dir outputs/attack/dim_ti_ni

python main.py --attack-method patch_dropout --guide-aug-copies 20 \
  --feature-layer -1 --gaussian-alpha 0 \
  --output-dir outputs/attack/pixel_patch_dropout

python main.py --attack-method token_patch_dropout \
  --input-diversity-groups 20 --input-diversity-views-per-group 1 \
  --gaussian-alpha 0 --output-dir outputs/attack/token_patch_dropout
```

默认 progressive phase-pair 不与 DIM 组合，每步为 20 个实际 model views。CLS score
noise 与 Gaussian residual 是已完成控制变量的支撑因素，不作为新增论文核心机制。

## 主线结果

当前 ViT progressive 主线 `3,7,11 + high` 的 1000 图 Overall ASR 为 79.75%，高于
同 seed 的 final-layer 基线 78.45%。high/low/random、checkpoint schedule 和 CLS score
noise × Gaussian residual 控制结果位于 `outputs/csv/`。

旧四白盒 final-layer 实验现作为跨架构基线保留，报告见
`experiments/mainline_data_aug_gaussian_story_s1000.md`。下一阶段目标是将 progressive
语义迁入 `main.py`，并通过 `nets/` adapter 恢复 ViT、CaiT、PiT、Visformer 四源支持。

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
