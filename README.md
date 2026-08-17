# Patch-Score Routing Attack

本项目研究一种面向黑盒迁移的“语义路由 + 颜色结构随机扰动”攻击。当前代码只保留两类内容：完整攻击主线与必要对照，以及用于发现跨模型前向语义共性的五条可复现实验链。已经证伪或退出主线的探索仅保留正式结果，不再暴露可执行入口。

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

## 保留的前向语义实验

| 探索方向 | 可执行脚本 | 正式产物 | 已支持的结论 | 对应主线参数 |
| --- | --- | --- | --- | --- |
| 语义成熟度 | `experiments/patch_score_layer_semantic_maturity_experiment.py` | `outputs/research/patch_score_layer_semantic_maturity/` | 四种架构均存在随深度发展的 global/local 语义结构；clean-output 敏感度不用于选攻击层 | `--patch-score-layer`，生产默认仍为 `final` |
| 跨层语义晋升 | `experiments/patch_score_promotion_observation.py` | `outputs/research/patch_score_promotion_e1_e2/` | patch 排名会从 early 到 final 系统性重排；共性是功能过程，不是固定空间模板 | final-layer `patch_score` 路由的表示依据 |
| 末层混合来源 | `experiments/patch_score_promotion_mixing_experiment.py` | `outputs/research/patch_score_promotion_e3_mixing/` | 晋升主要发生在末层 mixing 边界，可跨架构用同一 global/local 语言描述 | `patch_score_layer=final` 的机制解释 |
| 同激活 Grad-CAM | `experiments/patch_score_same_activation_experiment.py` | `outputs/research/patch_score_same_activation64/summary.json` | patch-score 与 Grad-CAM 在同一激活上部分重合但判据不同；不证明任一 selector 普遍更优 | Grad-CAM 仅作语义标准对照，不是主线 selector |
| 语义等价视图 | `experiments/semantic_equivalent_route_experiment.py` | `outputs/research/semantic_equivalent_route_e5_forward/` | phase/noise 视图可保持 global 语义，同时显著改变 local route；不同架构共享这一现象 | phase set、opponent/feature Gaussian、view groups |

五个脚本共享 `experiments/semantic_forward_utils.py`，负责数据切片、归一化、公共网格、rank/Spearman、bootstrap、结果写出和同激活捕获。它们不再依赖旧 selector calibration 或语义梯度实验。

从 semantic 角度，目前证据只支持以下边界内共性：

- global/local 语义关系会随网络深度成熟并发生 patch 排名重组；
- 末层 token mixing 是这一重组的重要共同边界；
- global 语义近似保持时，local routing 仍可被 phase/noise 系统性改变；
- 跨模型共性是“语义路由的功能形式”，不是每张图、每个模型共享同一张空间 mask；
- patch-score 与 class-conditioned Grad-CAM 是不同但可能部分重叠的语义标准。

## 主线与历史实验边界

当前可执行研究代码只有：

- `main.py`、`attack.py`、`gradient_replay.py`、`transfer_eval.py`；
- `nets/` 中四种白盒模型 adapter；
- 上述五个 forward-semantic 实验及公共工具；
- 与这些路径对应的测试。

clean-output causality/polarity/layer/route-response、固定 mask calibration、selector suite、Grad-CAM Protocol B/transfer/stability、E4 与 E5-gradient/E6-E10、frequency pair、patch shuffle 的可执行代码已经移除。它们的正式结果和结论仍保存在 [结果归档](results/README.md)；这些结果是边界证据，不得覆盖当前动态 mask 生产策略。

## 迁移评估与测试

```bash
python transfer_eval.py --image-dir outputs/attack/vit_mainline --prefix adv_
python -m unittest discover -s tests
```

迁移评估覆盖项目注册的 Transformer/CNN target，并写入 `outputs/csv`。最终有效性以 target-clean-correct transfer ASR 为准，同时关注 cross-model gradient cosine、sign agreement、held-out one-step response 与 full iterative transfer；source clean-logit suppression 不能替代这些指标。

安装依赖：

```bash
pip install -r requirements.txt
```
