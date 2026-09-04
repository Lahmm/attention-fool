# Patch-Score Routing Attack

本项目研究一种面向黑盒迁移的“语义路由 + 颜色结构随机扰动”攻击。当前代码和主实验记录聚焦完整攻击主线与必要对照。

## 当前主线

主线由两个核心机制组成：

1. **patch-score-guided patch drop**：用 global/local 表示余弦关系提供 label-free、gradient-independent 的语义坐标，决定在哪里扰动；
2. **RGB opponent-channel noise**：在亮度、红绿和黄蓝方向采样，再经过模型首层 RGB projection，决定如何扰动保留证据。

生产默认是动态 mask 的 `original_score_postdrop_phase_pair`：每个攻击 step、每个 augmentation group 都在当前对抗像素上重算 final-layer patch score，并从 high-score half 随机采样新 mask；仅同组 original/phase 两个视图共享 mask。默认 10 steps × 10 groups，因此每张图选择 100 次 mask。

```text
current adversarial pixels
→ final-layer global/local patch score
→ random drop from high-score half
→ original / phase-shift pair with a shared group mask
→ kept-only opponent noise at the initial RGB projection
→ raw 20-view gradient mean
→ Gaussian residual (sigma=4, alpha=0.75)
→ MI update
```

默认运行：

```bash
python main.py \
  --whitebox-model vit_base_patch16_224 \
  --seed 20260716 \
  --output-dir outputs/attack/vit_mainline
```

默认数据位于 `data/clean_resized_images`，标签为 `data/image_name_to_class_id_and_name.json`，模型从 `data/huggingface` 离线缓存读取。

## 保留的攻击接口

| 类别 | 当前保留接口 | 定位 |
| --- | --- | --- |
| 生产主线 | `original_score_postdrop_phase_pair` | 动态 patch-score mask + original/phase + kept-only feature noise |
| 基础路径 | `none` | 无 patch drop 的优化基线 |
| 像素对照 | `patch_dropout` | 通用 pixel patch dropout |
| token 对照 | `token_patch_dropout` | ViT token patch dropout |
| 优化与增强 | MI、NI、DIM、TI | 支撑机制和受控消融，不是新的论文主机制 |
| 主线路由选择 | `patch_score`、`random`、`no_drop` | 语义路由、随机路由、无 drop 对照 |

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

默认 phase-pair 主线不与 DIM 组合。每步实际 model views 上限为 20。主线还保留 `gaussian` feature noise 作为 `opponent_projected` 的结构对照，并可用 `--gaussian-alpha 0` 关闭梯度 Gaussian residual。

## 主线结果

正式主实验只保留四个白盒模型的 1000 图结果。实验报告见
`experiments/mainline_data_aug_gaussian_story_s1000.md`，迁移评估 CSV 位于
`outputs/csv/`，对应对抗样本和复现元数据位于 `outputs/attack/`。

当前最高 ViT 主线中 CLS score noise 与 Gaussian gradient residual 的 2×2 控制变量
补实验及可直接执行的攻击/迁移命令见
`experiments/vit_cls_score_noise_gradient_postprocess_s1000.md`。

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
