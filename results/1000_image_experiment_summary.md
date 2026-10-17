# 1000 图实验结果汇总

记录整理日期：2026-10-17  
仓库分支：`prd`  
汇总范围：当前工作树 `results/` 与 `outputs/` 中可读取的全部 1000 图实验记录；附录 B 补充 Git 历史中仍可恢复的 1000 图 CSV 记录。

未纳入本表的内容：

- 192 图、500 图调参实验；
- 64 图 patch-rank 观察统计；
- 梯度诊断、effective rank 等内部指标；
- 只有服务器目录名、没有本地结果 CSV/日志的 1000 图运行（见附录 C 说明）。

> 所有 ASR 均为百分比。ASR 定义为 `1 - adversarial accuracy`，分母是送入目标模型评估的全部 1000 张对抗图像，不做 target-clean-correct 过滤。  
> `*` 标记与源模型完全相同的 target；`Strict BB` 表示排除 source-matched target 后的严格黑盒平均。  
> 1000 图上的单张样本对应 0.1 个百分点，表中小于约 0.2 pp 的差异不应单独解释为方法效应。

---

## 0. 速览

| 实验组 | 源模型 | Target 集 | 方法/组件 | Overall ASR | 严格黑盒 ASR |
| --- | --- | --- | --- | ---: | ---: |
| E1 主实验（full-local） | ViT-B/16、CaiT-S24、PiT-B、Visformer-S | 14（8 TF + 6 CNN） | PRD：progressive random drop + full-local RGB opponent noise + phase-pair + MI + Gaussian residual | 四源平均 84.55% | 四源平均 83.49% |
| E2 kept-only 归档基线 | 同上 | 同上 | PRD，但 opponent noise 只施加在 kept positions | 四源平均 84.20% | 四源平均 83.10% |
| E3 FPR 对齐实验 | ViT-B/16、CaiT-S24、PiT-Ti、DeiT-B distilled | 13（7 TF + 6 CNN） | PRD，full-local opponent noise，Gaussian 关闭 | 四源平均 89.12% | 四源平均 88.69%* |
| E4 输入多样性消融 | ViT-B/16 | 14 | 改变 groups × views；其余同 E1 | 57.82%–82.63% | 56.10%–81.64% |
| E5 Gaussian / noise-grad 消融 | ViT-B/16 | 14 | Gaussian α ∈ {0, 0.75}，noise-grad ∈ {off, on}，groups ∈ {5, 10} × 2 views | 82.03%–85.08% | 81.05%–84.21% |
| H1/H2 历史记录 | 四种源架构或单源 | 13（+ViT 扩展） | 旧版 score-based / score-guided progressive 方法，非当前 PRD | 74.25%–87.58% | — |

\* E3 的 PiT-Ti 与 DeiT-B distilled 在 13 个 target 中没有完全同名模型，因此其 black-box 平均等于 overall；只有 ViT 与 CaiT 两项排除了同名 target。

---

## 1. 统一评价口径

### 1.1 图像与样本数

- 当前 E1–E5 均使用仓库内 `data/clean_resized_images` 的 1000 张 ImageNet validation 图像，标签来自 `data/image_name_to_class_id_and_name.json`。
- E1、E2、E4、E5 使用 seed `20260907`。
- E3 使用 seed `20260907`，并由 runner 校验四个 source 的 `sample_ids` 与 `labels` 顺序完全一致；每个 source 均生成并评估 1000 张。
- 历史 H1/H2 中，`original_score_postdrop_phase_pair` 使用 seed `20260716`；`progressive_patch_score` 使用 seed `20260907`。

### 1.2 Target 集

**14-target 主实验集（E1、E2、E4、E5）：8 Transformer + 6 CNN**

| 类别 | Target |
| --- | --- |
| Transformer | `vit_base_patch16_224`、`levit_256`、`pit_b_224`、`deit_base_patch16_224`、`tnt_s_patch16_224`、`convit_base`、`visformer_small`、`cait_s24_224` |
| CNN | `inception_v3`、`inception_v4`、`inception_resnet_v2`、`resnet101`、`inception_v3_adv`、`inception_resnet_v2_adv` |

**13-target FPR 集（E3）：7 Transformer + 6 CNN**

| 类别 | Target |
| --- | --- |
| Transformer | `vit_base_patch16_224`、`cait_s24_224`、`pit_b_224`、`visformer_small`、`swin_tiny_patch4_window7_224`、`deit_tiny_patch16_224`、`coat_tiny` |
| CNN | `resnet18`、`vgg16`、`densenet121`、`efficientnet_b0`、`mobilenet_v3_small`、`resnext50_32x4d` |

**13-target 历史集（H1）**

`levit_256`、`pit_b_224`、`deit_base_patch16_224`、`tnt_s_patch16_224`、`convit_base`、`visformer_small`、`cait_s24_224`、`inception_v3`、`inception_v4`、`inception_resnet_v2`、`resnet101`、`inception_v3_adv`、`inception_resnet_v2_adv`。

H2 的 `newconfig1000_*` CSV 在上述 13 个 target 之外额外记录了 `vit_base_patch16_224`，因此同时有 13-target 平均和 14-target overall 两行。

### 1.3 评估精度

| 实验组 | 攻击生成 batch | 评估 batch | 评估精度 |
| --- | ---: | ---: | --- |
| E1、E2 | ViT 96 / CaiT 48 / PiT 96 / Visformer 48 | 512 | AMP + TF32 |
| E3 | 8（10 GB RTX 3080） | 32 | FP32，TF32 关闭 |
| E4、E5 | 96 | 96 | AMP 关闭，TF32 开启 |

因此，E1/E2 与 E3 的数值不能直接拼成同一张对比表；E4/E5 与 E1 的评估精度相同，可以直接比较。

E1、E2 的汇总值来自逐样本精确统计（CSV/汇总表）；E3 的 `overall` 与 `average` 来自逐样本精确统计，其 TF/CNN 平均由保留一位小数的逐 target ASR 重新平均；E4、E5 的汇总由日志中的逐 target ASR（保留一位小数）重新平均，均可能存在 ±0.01 pp 的舍入差。

---

## 2. 攻击组件与缩写

| 缩写 | 组件 | 含义 |
| --- | --- | --- |
| PRD-R | Progressive random token drop | 每个 attack step/group 重新生成随机 checkpoint schedule；在每个 checkpoint 从全部当前局部 token 中均匀随机采样并 hard-zero，随后在已修改的 hidden state 上继续前向。不同 checkpoint 独立采样，允许重复位置。 |
| OCN | RGB opponent-channel noise | 在亮度 `(1,1,1)/√3`、红绿 `(1,-1,0)/√2`、黄蓝 `(1,1,-2)/√6` 方向采样，经源模型真实初始 RGB projection 映射到特征空间，并按 RMS 匹配；CLS/prefix 不加噪声。 |
| FL / KL | Full-local / Kept-only placement | FL：噪声施加到全部初始局部特征，包括之后会被 drop 的位置；KL：只施加到所有 schedule 的 kept-position union。 |
| PP | Phase-pair input diversity | 每个 group 使用 original view 与空间反射平移后的 phase view；phase view 使用原 schedule 的空间变换结果。默认 phase shift 集合 `(4,4)/(8,8)/(12,12)`。 |
| MI | Momentum iterative update | 梯度动量累积、sign 更新、相对干净图像的 L∞ 投影与 `[0,1]` 截断；`mi_decay=1.0`。 |
| GR | Gaussian residual | 在梯度平均后、动量累积前加入 `sigma=4, alpha=0.75` 的 Gaussian residual。 |
| NG | Opponent-noise gradient | 允许 opponent noise 的 RMS 分支向输入反传；默认关闭。该开关在 E1–E4 对应实现中尚不存在或未记录。 |
| SGH | Score-high-half-random drop map | 历史 `progressive_patch_score` 的 drop-map 策略：在 patch score 较高的一半中随机选择 drop 位置。不是当前 PRD 的均匀随机 schedule。 |
| SPD | Score-based patch dropout | 更早的 `original_score_postdrop_phase_pair` 方法：按 patch score 选择并 drop 约 29/196 个 token，随后执行 phase-pair 与 feature noise；不是 progressive checkpoint 方法。 |

### 2.1 攻击预算（共同部分）

除非单独注明，所有实验均使用：

| 项目 | 值 |
| --- | ---: |
| 扰动预算 ε | `16/255 ≈ 0.0627451` |
| 单步大小 | `1.6/255 ≈ 0.00627451` |
| 迭代步数 | 10 |
| 每步 group 数 | 10 |
| 每 group 视图数 | 2（phase-pair） |
| 每步视图数 | 20 |
| 每图总视图梯度 | 200（10 steps × 20 views） |
| 每图随机 schedule 数 | 100（10 steps × 10 groups） |
| 动量衰减 | 1.0 |
| 更新方式 | MI + sign + L∞ projection + `[0,1]` clamp |

**Checkpoint mask selection 次数**（每图）：

| 源架构 | Checkpoint 数 | 每图 selection 次数 |
| --- | ---: | ---: |
| ViT-B/16、DeiT-B distilled | 2 | 200 |
| CaiT-S24 | 2 | 200 |
| Visformer-S | 2 | 200 |
| PiT-B、PiT-Ti | 3 | 300 |

---

## 3. 源模型与 checkpoint 配置

| 源模型 | Adapter 模板 | Checkpoints | 原生网格 | Drop 数 | Drop ratios | Opponent β | Prefix token | 使用实验 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |
| `vit_base_patch16_224` | ViT | `block3, block10` | 14×14, 14×14 | 10, 10 | 0.051020408163, 0.051020408163 | 0.2 | 1 | E1、E2、E3、E4、E5、H1、H2 |
| `cait_s24_224` | CaiT | `block17, block23` | 14×14, 14×14 | 2, 28 | 0.010204081633, 0.142857142857 | 0.2 | 1 | E1、E2、E3、H1、H2 |
| `pit_b_224` | PiT | `stage2_block1, stage3_block2, stage3_block3` | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.02081165, 0.03125, 0.09375 | 0.4 | 1 | E1、E2、H1、H2 |
| `visformer_small` | Visformer | `stage2_block1, stage3_block1` | 14×14, 7×7 | 41, 10 | 0.209183673469, 0.204081632653 | 0.4 | 1 | E1、E2、H1、H2 |
| `pit_ti_224` | PiT（沿用 PiT-B schedule） | `stage2_block1, stage3_block2, stage3_block3` | 16×16, 8×8, 8×8 | 5, 2, 6 | 0.02081165, 0.03125, 0.09375 | 0.4 | 1 | E3 |
| `deit_base_distilled_patch16_224` | ViT（保留两个 prefix token） | `block3, block10` | 14×14, 14×14 | 10, 10 | 0.051020408163, 0.051020408163 | 0.2 | 2（CLS + distillation） | E3 |

说明：

- PiT-Ti 的 checkpoint 配置是从 PiT-B 迁移的；E3 的 config 明确标记为 `configuration_status: transferred`。
- DeiT-B distilled 使用 ViT adapter，但 `num_prefix_tokens=2`；CLS 与 distillation token 都不参与 local-token noise 和 drop，`forward_head` 在 eval 模式下平均两个 head。
- 当前代码默认 `gaussian_alpha=0`；E1/E2/E4 的实际运行使用历史参数 `alpha=0.75`，E3 显式关闭 Gaussian，E5 对 Gaussian 做消融。

---

## 4. 实验清单与预算总表

| ID | 日期 | 实验 | 方法/组件 | 源模型 | Target 集 | Seed | 实现/记录 |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| E1 | 2026-10-12 | PRD 四源主实验（full-local） | PRD-R + FL OCN + PP + MI + GR | ViT-B/16、CaiT-S24、PiT-B、Visformer-S | 14 | 20260907 | `bf092246`；`results/`、`outputs/runs/prd1000_full_local_seed20260907/` |
| E2 | 2026-09-22（归档提交） | kept-only PRD 归档基线 | PRD-R + KL OCN + PP + MI + GR | 同上 | 14 | 20260907 | `98cb7cf1`；归档于 `f44be25`；当前可读对比在 `comparison_per_target.csv` |
| E3 | 2026-10-15 | FPR Table 2 对齐实验 | PRD-R + FL OCN + PP + MI，Gaussian 关闭 | ViT-B/16、CaiT-S24、PiT-Ti、DeiT-B distilled | 13 | 20260907 | `8277a38`；`outputs/fpr_aligned_no_gaussian/` |
| E4 | 2026-10-14 | ViT 输入多样性消融 | PRD-R + FL OCN + MI + GR；改变 groups × views | ViT-B/16 | 14 | 20260907 | `cc86c44`；`outputs/attack/prd_vit_single_view_ablation_20261014_094728/` |
| E5 | 2026-10-14 | ViT Gaussian / noise-grad 消融 | PRD-R + FL OCN + PP + MI；Gaussian / NG 开关 | ViT-B/16 | 14 | 20260907 | `8277a38`；`outputs/attack/prd_vit_gaussian_noisegrad_ablation_20261014_143903/` |
| H1 | 2026-07-16 / 2026-09 | 历史 13-target 1000 图记录 | SPD 或 SGH progressive | ViT/CaiT/PiT/Visformer（单源） | 13 | 20260716 / 20260907 | Git 历史 CSV；见附录 B |
| H2 | 2026-09-16 | `newconfig1000_*` 14-target 扩展 | SGH progressive + KL OCN + PP + MI + GR | 四种源架构 | 14 | 20260907 | Git 历史 CSV；见附录 B |

### 4.1 逐实验预算

| ID | 图像数 | ε | Steps | Step size | Groups × views | 每步视图 | 每图 schedule | 每图 mask selections | Gaussian | Noise-grad | 评估精度 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- | --- |
| E1 | 1000 | 16/255 | 10 | 1.6/255 | 10 × 2 | 20 | 100 | ViT/CaiT/Vis 200；PiT 300 | σ=4, α=0.75 | 未记录/关闭 | AMP + TF32 |
| E2 | 1000 | 16/255 | 10 | 1.6/255 | 10 × 2 | 20 | 100 | 同上 | σ=4, α=0.75 | 未记录/关闭 | AMP + TF32 |
| E3 | 1000 | 16/255 | 10 | 1.6/255 | 10 × 2 | 20 | 100 | ViT/CaiT/DeiT 200；PiT-Ti 300 | 关闭（α=0） | false | FP32，TF32 off |
| E4 | 1000 | 16/255 | 10 | 1.6/255 | 见 9.1 | 1 / 10 / 10 / 20 | 10 / 50 / 100 / 200 | 20 / 100 / 200 / 400 | σ=4, α=0.75 | 不适用 | AMP off + TF32 |
| E5 | 1000 | 16/255 | 10 | 1.6/255 | 5×2 或 10×2 | 10 / 20 | 50 / 100 | 100 / 200 | α=0 或 0.75 | false / true | AMP off + TF32 |
| H1/H2 | 1000 | 16/255 | 10 | 1.6/255 | 10 × 2 | 20 | 100 | 依 checkpoint 数 | 部分有 α=0.75 | 未记录 | 依各原始记录 |

---

## 5. E1：PRD 四源 1000 图主实验（full-local，2026-10-12）

### 5.1 方法与配置

- 方法：PRD；checkpoint-wise progressive random token drop + 全部初始局部特征上的 RGB opponent-channel noise。
- 噪声位置：`initial_rgb_projection_all_local_tokens`；CLS/prefix 不变。
- 输入多样性：10 groups × 2 views；phase shift 集合 `(4,4)/(8,8)/(12,12)`。
- 更新：MI decay 1.0，10 steps，ε=16/255，step size=1.6/255。
- Gaussian residual：σ=4，α=0.75。
- 源模型配置见第 3 节；每个源模型攻击 1000 张图像。
- 目标：14 个（8 Transformer + 6 CNN）；56 个 source-target 对全部使用 1000 张样本，`skipped=0`。
- 评估：AMP + TF32；每个 target 使用自己的 transform。

### 5.2 汇总 ASR

| Source | Overall | Transformer（8） | CNN（6） | Transformer 排除 source-matched（7） | Strict BB（13） |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | 85.06 | 89.96 | 78.52 | 89.04 | 84.18 |
| CaiT-S24 | 87.64 | 90.89 | 83.32 | 89.77 | 86.79 |
| PiT-B | 84.34 | 89.55 | 77.38 | 88.26 | 83.24 |
| Visformer-S | 81.18 | 83.16 | 78.53 | 80.80 | 79.75 |
| **四源平均** | **84.55** | **88.39** | **79.44** | **86.97** | **83.49** |

### 5.3 逐 target ASR（%）

| Source | ViT-B/16 | LeViT-256 | PiT-B | DeiT-B/16 | TNT-S/16 | ConViT-B | Visformer-S | CaiT-S24 | Inc-v3 | Inc-v4 | Inc-Res-v2 | ResNet-101 | Inc-v3-adv | Inc-Res-v2-adv | Overall | TF avg | CNN avg | Strict BB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | **96.4\*** | 87.0 | 87.3 | 89.6 | 89.7 | 87.3 | 87.8 | 94.6 | 79.9 | 78.2 | 78.9 | 80.9 | 78.9 | 74.3 | 85.06 | 89.96 | 78.52 | 84.18 |
| CaiT-S24 | 86.2 | 90.4 | 89.5 | 91.4 | 91.4 | 90.2 | 89.3 | **98.7\*** | 85.2 | 83.7 | 84.3 | 85.1 | 83.7 | 77.9 | 87.64 | 90.89 | 83.32 | 86.79 |
| PiT-B | 81.8 | 87.0 | **98.6\*** | 91.4 | 90.1 | 89.7 | 90.2 | 87.6 | 82.8 | 79.3 | 77.4 | 80.8 | 75.4 | 68.6 | 84.34 | 89.55 | 77.38 | 83.24 |
| Visformer-S | 64.7 | 88.0 | 86.6 | 79.8 | 86.0 | 78.5 | **99.7\*** | 82.0 | 86.0 | 85.5 | 81.0 | 83.9 | 74.6 | 60.2 | 81.18 | 83.16 | 78.53 | 79.75 |

### 5.4 文件与 provenance

| 内容 | 路径 |
| --- | --- |
| 逐 target 规范化表 | `results/prd_cross_arch_s1000.csv` |
| 四源汇总 | `results/prd_summary_s1000.csv` |
| 运行环境/提交/target 列表 | `results/prd_run_artifacts/provenance.json` |
| 四份实际攻击参数 | `results/prd_run_artifacts/{vit,cait,pit,visformer}/attack_params.json` |
| 四份原始 transfer CSV | `results/prd_run_artifacts/{vit,cait,pit,visformer}/transfer.csv` |
| 评估日志与逐样本预测 | `outputs/runs/prd1000_full_local_seed20260907/` |
| 1000 张对抗图像 | `outputs/attack/prd1000_full_local_{vit,cait,pit,visformer}_seed20260907/` |

实际环境：Python 3.10.19、PyTorch 2.10.0+cu128、timm 1.0.24、CUDA 12.8、NVIDIA RTX 4090。

---

## 6. E2：kept-only PRD 归档基线

### 6.1 与 E1 的差别

E2 是 `98cb7cf1` 实现下完成的四源 1000 图运行，使用同一组 source 配置、seed、14 个 target 和评估口径，但 opponent noise 的位置不同：

- E2：`initial_rgb_projection_kept_image_union_only`；
- E1：`initial_rgb_projection_all_local_tokens`。

两者都使用 progressive random drop、phase-pair、MI 和 Gaussian residual `σ=4, α=0.75`。E2 的原始 CSV 与完整报告保存在 `f44be25` 的 `results/` 归档中；当前工作树的 `outputs/runs/.../comparison_per_target.csv` 保留了逐 target 的 kept-only 数值。

### 6.2 汇总 ASR

| Source | Overall | Transformer（8） | CNN（6） | Transformer 排除 source-matched（7） | Strict BB（13） |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | 85.20 | 90.45 | 78.20 | 89.51 | 84.29 |
| CaiT-S24 | 86.58 | 90.06 | 81.93 | 88.86 | 85.66 |
| PiT-B | 84.36 | 89.91 | 76.95 | 88.69 | 83.27 |
| Visformer-S | 80.65 | 83.19 | 77.27 | 80.83 | 79.18 |
| **四源平均** | **84.20** | **88.40** | **78.59** | **86.97** | **83.10** |

### 6.3 逐 target ASR（%）

| Source | ViT-B/16 | LeViT-256 | PiT-B | DeiT-B/16 | TNT-S/16 | ConViT-B | Visformer-S | CaiT-S24 | Inc-v3 | Inc-v4 | Inc-Res-v2 | ResNet-101 | Inc-v3-adv | Inc-Res-v2-adv | Overall | TF avg | CNN avg | Strict BB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | **97.0\*** | 87.0 | 88.0 | 90.1 | 90.5 | 88.2 | 88.1 | 94.7 | 80.5 | 77.7 | 78.2 | 81.3 | 77.8 | 73.7 | 85.20 | 90.45 | 78.20 | 84.29 |
| CaiT-S24 | 85.4 | 89.4 | 89.2 | 90.5 | 90.0 | 89.3 | 88.2 | **98.5\*** | 84.9 | 82.2 | 82.2 | 85.0 | 81.3 | 76.0 | 86.58 | 90.06 | 81.93 | 85.66 |
| PiT-B | 83.3 | 87.4 | **98.5\*** | 91.3 | 90.8 | 89.7 | 90.2 | 88.1 | 82.4 | 79.8 | 78.0 | 80.1 | 75.1 | 66.3 | 84.36 | 89.91 | 76.95 | 83.27 |
| Visformer-S | 64.9 | 88.6 | 87.1 | 80.2 | 86.7 | 77.7 | **99.7\*** | 80.6 | 85.4 | 85.0 | 78.9 | 84.6 | 73.0 | 56.7 | 80.65 | 83.19 | 77.27 | 79.18 |

### 6.4 E1 相对 E2 的差值（pp）

| Source | Overall | Transformer | CNN | Strict BB |
| --- | ---: | ---: | ---: | ---: |
| ViT-B/16 | -0.14 | -0.49 | +0.32 | -0.11 |
| CaiT-S24 | +1.06 | +0.82 | +1.38 | +1.13 |
| PiT-B | -0.02 | -0.36 | +0.43 | -0.03 |
| Visformer-S | +0.53 | -0.02 | +1.27 | +0.57 |
| **四源平均** | **+0.36** | **-0.01** | **+0.85** | **+0.39** |

56 个 source-target 对中，31 对提升、4 对不变、21 对下降。full-local 的主要收益来自 CNN target 平均 +0.85 pp；Transformer 平均基本不变。

---

## 7. E3：FPR Table 2 对齐实验（2026-10-15，无 Gaussian）

### 7.1 定位

E3 使用 FPR Table 2 的四个 source（ViT-B/16、CaiT-S24、PiT-Ti、DeiT-B distilled）和十三个 target，攻击生成遵循当前 PRD，但：

- opponent noise 使用 full-local placement；
- **Gaussian residual 关闭**（`gaussian_alpha=0`，`gradient_postprocess=raw_mean`）；
- `opponent_noise_grad=false`；
- 评估使用 FPR 协议：直接 resize 224×224、[0,1] 转换、各模型自己的 normalization；FP32，TF32 关闭；
- source 攻击 batch size 为 8，评估 batch size 为 32。

因此 E3 是 FPR target 集上的独立测量，不应与 E1 的 14-target 主表逐项比较。

### 7.2 源模型配置

| Source | Checkpoints | Drop 数 | Drop ratios | Opponent β | 生成 batch |
| --- | --- | --- | --- | ---: | ---: |
| `vit_base_patch16_224` | `block3, block10` | 10, 10 | 0.051020408163, 0.051020408163 | 0.2 | 8 |
| `cait_s24_224` | `block17, block23` | 2, 28 | 0.010204081633, 0.142857142857 | 0.2 | 8 |
| `pit_ti_224` | `stage2_block1, stage3_block2, stage3_block3` | 5, 2, 6 | 0.02081165, 0.03125, 0.09375 | 0.4 | 8 |
| `deit_base_distilled_patch16_224` | `block3, block10` | 10, 10 | 0.051020408163, 0.051020408163 | 0.2 | 8 |

### 7.3 汇总 ASR

| Source | Overall（13） | Black-box avg | Transformer（7） | CNN（6） | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `vit_base_patch16_224` | 88.18 | 87.22 | 91.37 | 84.45 | 排除同名 ViT |
| `cait_s24_224` | 90.08 | 89.32 | 92.46 | 87.30 | 排除同名 CaiT |
| `pit_ti_224` | 85.12 | 85.12 | 81.37 | 89.50 | target 中没有 `pit_ti_224`，不排除 |
| `deit_base_distilled_patch16_224` | 93.08 | 93.08 | 95.34 | 90.45 | target 中没有同名 distilled DeiT，不排除 |
| **四源平均** | **89.12** | **88.69** | **90.14** | **87.92** | — |

### 7.4 逐 target ASR（%）

| Source | ViT-B/16 | CaiT-S24 | PiT-B | Visformer-S | Swin-T | DeiT-T | CoaT-T | ResNet-18 | VGG-16 | DenseNet-121 | EffNet-B0 | MobileNetV3-S | ResNeXt-50 | Overall | Black-box avg | TF avg | CNN avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT-B/16 | **99.7\*** | 94.7 | 89.2 | 88.9 | 89.4 | 93.4 | 84.3 | 82.7 | 84.1 | 85.0 | 89.7 | 84.3 | 80.9 | 88.18 | 87.22 | 91.37 | 84.45 |
| CaiT-S24 | 92.3 | **99.1\*** | 90.2 | 90.5 | 91.7 | 94.7 | 88.7 | 87.3 | 88.4 | 89.5 | 90.9 | 82.8 | 84.9 | 90.08 | 89.32 | 92.46 | 87.30 |
| PiT-Ti | 65.6 | 82.7 | 74.2 | 80.4 | 83.5 | 97.4 | 85.8 | 91.6 | 91.4 | 89.4 | 93.4 | 95.2 | 76.0 | 85.12 | 85.12 | 81.37 | 89.50 |
| DeiT-B distilled | 95.9 | 98.6 | 94.1 | 94.3 | 95.9 | 97.7 | 90.9 | 89.7 | 89.6 | 91.7 | 94.8 | 88.0 | 88.9 | 93.08 | 93.08 | 95.34 | 90.45 |

### 7.5 文件与说明

| 内容 | 路径 |
| --- | --- |
| 运行配置（四个 source） | `outputs/fpr_aligned_no_gaussian/run_config.json` |
| 汇总 ASR | `outputs/fpr_aligned_no_gaussian/transfer_asr.csv` |
| 逐 target 评估日志 | `outputs/fpr_aligned_no_gaussian/{source}_transfer.log` |
| 逐样本预测 | `outputs/fpr_aligned_no_gaussian/{source}_predictions.json` |
| 1000 张对抗图像 | `outputs/attack/prd_fpr_aligned_no_gaussian_{source}_seed20260907/` |
| runner | `experiments/run_prd_fpr_targets.py` |

首次 ViT 评估因离线权重加载失败中断，runner 在离线权重校验通过后从已有图像继续；最终四个 source 均完成 1000 张图像和 13 个 target。E3 的汇总 `overall` 与 `average` 由逐样本预测精确计算；本表 TF avg / CNN avg 由 CSV 中保留一位小数的逐 target ASR 重新平均，可能有 ±0.01 pp 的舍入差。

---

## 8. 补充说明：主实验与 kept-only 的差异来源

E1 与 E2 的唯一方法差异是 opponent noise 的位置：

- kept-only：只对所有随机 schedule 的 kept-position union 加噪声；
- full-local：对全部初始 local tokens 加噪声，包括之后会被 drop 的位置。

两者的 source 配置、seed、1000 张图像、14 个 target、Gaussian residual 和评估精度一致。因此 E1 − E2 的差值可以解释为 noise placement 的影响，而不是新的攻击模块。该对比属于内部实验记录；论文主线仍以完整 PRD 的 ASR 表述。

---

## 9. E4：ViT 输入多样性 / phase-pair 消融（2026-10-14）

### 9.1 配置

源模型固定为 `vit_base_patch16_224`；checkpoints `block3, block10`；drop 10/10；opponent β=0.2；Gaussian σ=4, α=0.75；MI decay 1.0；10 steps；ε=16/255；step size=1.6/255；1000 张图像；14 个 target。

| Arm | Groups | Views/group | 每步视图 | 每图 schedule | 每图 mask selections | Phase-pair | Gaussian | Noise-grad |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `groups1_view1` | 1 | 1 | 1 | 10 | 20 | 否 | α=0.75 | 不适用 |
| `groups5_view2` | 5 | 2 | 10 | 50 | 100 | 是 | α=0.75 | 不适用 |
| `groups10_view1` | 10 | 1 | 10 | 100 | 200 | 否 | α=0.75 | 不适用 |
| `groups20_view1` | 20 | 1 | 20 | 200 | 400 | 否 | α=0.75 | 不适用 |
| E1 ViT 参考 | 10 | 2 | 20 | 100 | 200 | 是 | α=0.75 | 未记录/关闭 |

### 9.2 逐 target ASR（%）

| Arm | ViT-B/16 | LeViT-256 | PiT-B | DeiT-B/16 | TNT-S/16 | ConViT-B | Visformer-S | CaiT-S24 | Inc-v3 | Inc-v4 | Inc-Res-v2 | ResNet-101 | Inc-v3-adv | Inc-Res-v2-adv | Overall | TF avg | CNN avg | Strict BB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `groups1_view1` | **80.2\*** | 60.1 | 57.4 | 61.3 | 62.3 | 59.4 | 56.4 | 86.5 | 51.7 | 49.9 | 45.7 | 48.9 | 48.5 | 41.2 | 57.82 | 65.45 | 47.65 | 56.10 |
| `groups5_view2` | **95.5\*** | 84.9 | 84.9 | 87.9 | 87.4 | 85.7 | 84.8 | 93.0 | 77.6 | 75.4 | 75.8 | 78.5 | 75.4 | 70.0 | 82.63 | 88.01 | 75.45 | 81.64 |
| `groups10_view1` | **90.6\*** | 73.4 | 70.9 | 75.3 | 75.5 | 72.6 | 71.7 | 94.0 | 66.2 | 62.9 | 62.9 | 65.1 | 64.0 | 54.3 | 71.39 | 78.00 | 62.57 | 69.91 |
| `groups20_view1` | **91.0\*** | 75.3 | 73.6 | 76.5 | 77.1 | 74.7 | 72.9 | 94.4 | 68.4 | 65.0 | 64.4 | 66.5 | 66.1 | 56.1 | 73.00 | 79.44 | 64.42 | 71.62 |
| E1 ViT 参考（10×2） | **96.4\*** | 87.0 | 87.3 | 89.6 | 89.7 | 87.3 | 87.8 | 94.6 | 79.9 | 78.2 | 78.9 | 80.9 | 78.9 | 74.3 | 85.06 | 89.96 | 78.52 | 84.18 |

### 9.3 结果解读

- 只使用 original view、每步仅 1 个视图时（`groups1_view1`），Overall 仅 57.82%；增加到 10 或 20 个 original views 后提高到 71.39% / 73.00%，但仍明显低于 phase-pair 的 85.06%。
- 同样每步 10 个视图，`groups5_view2`（5 groups × 2 views）Overall 82.63%，远高于 `groups10_view1`（10 groups × 1 view）的 71.39%；说明 phase view 的空间变换本身是主要收益来源，而不只是视图数量。
- `groups20_view1` 比 `groups10_view1` 仅高 1.61 pp，继续增加 original-only 视图的边际收益有限。

---

## 10. E5：ViT Gaussian residual / opponent-noise-grad 消融（2026-10-14）

### 10.1 配置

源模型固定为 `vit_base_patch16_224`；checkpoints `block3, block10`；drop 10/10；opponent β=0.2；MI decay 1.0；10 steps；ε=16/255；step size=1.6/255；1000 张图像；14 个 target；所有 arm 都使用 phase-pair（2 views/group）。

| Arm | Groups | Views/group | 每步视图 | 每图 schedule | 每图 mask selections | Gaussian | Noise-grad |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `gaussian0_noisegrad0_groups10_view2` | 10 | 2 | 20 | 100 | 200 | α=0 | false |
| `gaussian0_noisegrad0_groups5_view2` | 5 | 2 | 10 | 50 | 100 | α=0 | false |
| `gaussian0_noisegrad1_groups10_view2` | 10 | 2 | 20 | 100 | 200 | α=0 | true |
| `gaussian0_noisegrad1_groups5_view2` | 5 | 2 | 10 | 50 | 100 | α=0 | true |
| `gaussian1_noisegrad1_groups10_view2` | 10 | 2 | 20 | 100 | 200 | α=0.75 | true |
| `gaussian1_noisegrad1_groups5_view2` | 5 | 2 | 10 | 50 | 100 | α=0.75 | true |
| E1 ViT 参考（g1, n0, 10×2） | 10 | 2 | 20 | 100 | 200 | α=0.75 | 未记录/关闭 |

E5 没有单独的 `gaussian1_noisegrad0_groups10_view2` arm；该配置正是 E1 的 ViT 行，故以 E1 ViT 作为 Gaussian-on/no-noise-grad 的 10×2 参考。

### 10.2 逐 target ASR（%）

| Arm | ViT-B/16 | LeViT-256 | PiT-B | DeiT-B/16 | TNT-S/16 | ConViT-B | Visformer-S | CaiT-S24 | Inc-v3 | Inc-v4 | Inc-Res-v2 | ResNet-101 | Inc-v3-adv | Inc-Res-v2-adv | Overall | TF avg | CNN avg | Strict BB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `g0_n0_g10` | **95.9\*** | 86.4 | 87.0 | 89.3 | 89.3 | 87.0 | 86.9 | 94.6 | 80.9 | 78.8 | 79.2 | 80.7 | 79.2 | 73.2 | 84.89 | 89.55 | 78.67 | 84.04 |
| `g0_n0_g5` | **94.7\*** | 83.9 | 84.2 | 87.0 | 86.0 | 85.0 | 84.4 | 93.5 | 78.4 | 75.5 | 75.4 | 78.0 | 75.2 | 68.4 | 82.11 | 87.34 | 75.15 | 81.15 |
| `g0_n1_g10` | **96.1\*** | 86.7 | 86.8 | 89.2 | 89.6 | 86.8 | 86.7 | 94.7 | 81.1 | 79.2 | 79.0 | 81.2 | 78.5 | 73.4 | 84.93 | 89.58 | 78.73 | 84.07 |
| `g0_n1_g5` | **94.8\*** | 83.8 | 84.3 | 87.1 | 86.3 | 84.8 | 84.0 | 93.3 | 78.1 | 75.5 | 75.0 | 77.8 | 74.8 | 68.8 | 82.03 | 87.30 | 75.00 | 81.05 |
| `g1_n1_g10` | **96.4\*** | 86.8 | 87.4 | 89.6 | 89.8 | 87.6 | 87.5 | 94.9 | 79.9 | 77.9 | 78.4 | 82.0 | 78.8 | 74.1 | 85.08 | 90.00 | 78.52 | 84.21 |
| `g1_n1_g5` | **95.6\*** | 84.0 | 85.2 | 87.5 | 87.5 | 85.7 | 85.1 | 93.5 | 78.0 | 75.6 | 75.7 | 78.6 | 75.7 | 70.1 | 82.70 | 88.01 | 75.62 | 81.71 |
| E1 ViT 参考 `g1_n0_g10` | **96.4\*** | 87.0 | 87.3 | 89.6 | 89.7 | 87.3 | 87.8 | 94.6 | 79.9 | 78.2 | 78.9 | 80.9 | 78.9 | 74.3 | 85.06 | 89.96 | 78.52 | 84.18 |

### 10.3 结果解读

以 E1 ViT（g1, n0, 10×2）的 Overall 85.06% 为参考：

| 对照 | Overall | 差值 vs E1 ViT |
| --- | ---: | ---: |
| g0_n0_g10（Gaussian off, NG off） | 84.89 | -0.17 pp |
| g0_n1_g10（Gaussian off, NG on） | 84.93 | -0.13 pp |
| g1_n1_g10（Gaussian on, NG on） | 85.08 | +0.02 pp |
| g0_n0_g5 | 82.11 | -2.95 pp |
| g0_n1_g5 | 82.03 | -3.03 pp |
| g1_n1_g5 | 82.70 | -2.36 pp |
| E1 ViT（g1, n0, g10） | 85.06 | 0.00 |

- 在 10 groups × 2 views 下，Gaussian residual 和 opponent-noise-grad 的影响都小于 0.2 pp；这两个开关不是该 target 集上 ASR 的主要来源。
- groups 10 → 5（每步 20 → 10 个视图）会损失约 2.4–3.0 pp，远大于 Gaussian/NG 的差异。
- 由于 1000 图单样本粒度为 0.1 pp，上述 Gaussian/NG 的小差值不应作强结论；输入视图数量与 phase-pair 的效应更稳定。

---

## 11. 跨实验观察与使用建议

1. **主实验结论**：E1 四源平均 Overall 84.55%、严格黑盒 83.49%；Transformer 88.39%、CNN 79.44%。56 项评估均为 1000 样本、`skipped=0`。
2. **Noise placement**：full-local 相对 kept-only 四源平均 +0.36 pp Overall、+0.39 pp 严格黑盒；主要来自 CNN（+0.85 pp），Transformer 基本不变。分源差异较大（CaiT +1.06 pp，ViT −0.14 pp）。
3. **输入多样性**：phase-pair 是强组件。仅 original view 时，10 或 20 个视图的 Overall 为 71.39% / 73.00%；5 groups × 2 views 已达 82.63%，10 groups × 2 views 为 85.06%。比较时应固定总视图数或 group 数，避免把“视图数量”和“phase-pair”混在一起。
4. **Gaussian residual 与 noise-grad**：在 ViT/14-target 上影响很小（<0.2 pp）。当前代码默认关闭 Gaussian；E1 的 α=0.75 是历史主实验参数，E3 与当前默认均为 α=0。
5. **FPR 对齐实验**：E3 的 13-target 平均 Overall 为 89.12%，但这组 source（PiT-Ti、DeiT-B distilled）与 target 集都与 E1 不同，不能把 89.12% 与 84.55% 直接比较。E3 的价值是验证 PRD 在 FPR Table 2 模型集上的可复现迁移表现。
6. **历史记录**：H1 的 `original_score_postdrop_phase_pair` 是 score-based patch dropout，不是 progressive checkpoint 方法；H2 的 `progressive_patch_score` 使用 `score_high_half_random` drop map，也不是当前 PRD 的均匀随机 schedule。它们只能作为方法演进和调参背景，不应与当前 PRD 主表混合汇报。
7. **统计解释**：1000 图上的 0.1 pp 对应 1 张图像；小于约 0.2 pp 的差异接近单 seed 的记录粒度。若需主张组件效应，应使用多 seed 或更大样本，并报告置信区间。

---

## 12. 文件索引

| 内容 | 路径 |
| --- | --- |
| E1 主结果表 | `results/prd_cross_arch_s1000.csv` |
| E1 四源汇总 | `results/prd_summary_s1000.csv` |
| E1 参数与环境 | `results/prd_run_artifacts/` |
| E1 评估、对比、预测 | `outputs/runs/prd1000_full_local_seed20260907/` |
| E1/E2 逐 target 对比 | `outputs/runs/prd1000_full_local_seed20260907/comparison_per_target.csv` |
| E1/E2 汇总对比 | `outputs/runs/prd1000_full_local_seed20260907/comparison_summary.csv` |
| E2 说明 | `outputs/runs/prd1000_full_local_seed20260907/comparison.md` |
| E3 配置 | `outputs/fpr_aligned_no_gaussian/run_config.json` |
| E3 ASR | `outputs/fpr_aligned_no_gaussian/transfer_asr.csv` |
| E3 日志/预测 | `outputs/fpr_aligned_no_gaussian/{source}_{transfer.log,predictions.json}` |
| E4 manifest 与日志 | `outputs/attack/prd_vit_single_view_ablation_20261014_094728/` |
| E5 manifest 与日志 | `outputs/attack/prd_vit_gaussian_noisegrad_ablation_20261014_143903/` |
| E4/E5 逐 target 日志 | 上述目录下 `*_transfer.log` |
| 方法契约 | `experiments/prd_method_contract.md` |
| 主实验报告 | `experiments/progressive_cross_arch_mainline_s1000.md` |
| FPR 协议 | `experiments/fpr_target_protocol.md` |
| DeiT distilled 协议 | `experiments/deit_distilled_s1000_protocol.md` |

---

## 附录 A：target 名称对照

| 短名 | 完整模型名 | 类别 |
| --- | --- | --- |
| ViT-B/16 | `vit_base_patch16_224` | Transformer |
| LeViT-256 | `levit_256` | Transformer |
| PiT-B | `pit_b_224` | Transformer |
| DeiT-B/16 | `deit_base_patch16_224` | Transformer |
| TNT-S/16 | `tnt_s_patch16_224` | Transformer |
| ConViT-B | `convit_base` | Transformer |
| Visformer-S | `visformer_small` | Transformer |
| CaiT-S24 | `cait_s24_224` | Transformer |
| Inc-v3 | `inception_v3` | CNN |
| Inc-v4 | `inception_v4` | CNN |
| Inc-Res-v2 | `inception_resnet_v2` | CNN |
| ResNet-101 | `resnet101` | CNN |
| Inc-v3-adv | `inception_v3_adv` | CNN |
| Inc-Res-v2-adv | `inception_resnet_v2_adv` | CNN |
| Swin-T | `swin_tiny_patch4_window7_224` | Transformer（FPR） |
| DeiT-T | `deit_tiny_patch16_224` | Transformer（FPR） |
| CoaT-T | `coat_tiny` | Transformer（FPR） |
| ResNet-18 | `resnet18` | CNN（FPR） |
| VGG-16 | `vgg16` | CNN（FPR） |
| DenseNet-121 | `densenet121` | CNN（FPR） |
| EffNet-B0 | `efficientnet_b0` | CNN（FPR） |
| MobileNetV3-S | `mobilenet_v3_small` | CNN（FPR） |
| ResNeXt-50 | `resnext50_32x4d` | CNN（FPR） |

---

## 附录 B：Git 历史中的 1000 图记录（非当前工作树）

以下 CSV 已从当前工作树移除，但仍可从 Git 历史恢复。它们不是当前 PRD 主线，放在附录中仅用于完整记录和组件演进参考。

### B.1 方法族说明

| 方法族 | 记录数 | 方法特征 | 与当前 PRD 的关系 |
| --- | ---: | --- | --- |
| `original_score_postdrop_phase_pair` | 8 | score-based patch dropout（记录的实际丢弃约 29/196）、post-drop opponent-projected feature noise、phase-pair、MI；Gaussian residual 有/无两组 | 非 progressive checkpoint；更早的 score-based 路线 |
| `progressive_patch_score` | 10 | progressive checkpoint drops + `score_high_half_random` drop map（部分记录未写 drop map）、kept-only opponent noise、phase-pair、MI；Gaussian α=0.75 | PRD 的直接前身；drop 位置由 patch score 引导，不是当前均匀随机 |

### B.2 历史配置

| Experiment | Method | Source | Seed | Checkpoints | Drop ratios | Drop-map | Opponent noise | Gaussian |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `vit_..._opponent_projected_mainline` | SPD | `vit_base_patch16_224` | 20260716 | — | — | — | — | 否 |
| `vit_..._opponent_projected_gaussian_s40a075_mainline` | SPD | `vit_base_patch16_224` | 20260716 | — | — | — | — | 是（probe） |
| `cait_..._opponent_projected_mainline` | SPD | `cait_s24_224` | 20260716 | — | — | — | — | 否 |
| `cait_..._opponent_projected_gaussian_s40a075_mainline` | SPD | `cait_s24_224` | 20260716 | — | — | — | — | 是（α=0.75） |
| `pit_..._opponent_projected_mainline` | SPD | `pit_b_224` | 20260716 | — | — | — | — | 否 |
| `pit_..._opponent_projected_gaussian_s40a075_mainline` | SPD | `pit_b_224` | 20260716 | — | — | — | — | 是（α=0.75） |
| `visformer_..._opponent_projected_mainline` | SPD | `visformer_small` | 20260716 | — | — | — | — | 否 |
| `visformer_..._opponent_projected_gaussian_s40a075_mainline` | SPD | `visformer_small` | 20260716 | — | — | — | — | 是（α=0.75） |
| `newconfig1000_vit_b3_b10_c10_10` | SGH | `vit_base_patch16_224` | 20260907 | block3, block10 | 0.05102, 0.05102 | score_high_half_random | kept-only | α=0.75 |
| `newconfig1000_cait_b17_b23_c02_28_projection` | SGH | `cait_s24_224` | 20260907 | block17, block23 | 0.0102, 0.1429 | score_high_half_random | kept-only | α=0.75 |
| `newconfig1000_pit_s2b1_s3b2_s3b3_c05_02_06_opp04` | SGH | `pit_b_224` | 20260907 | stage2_block1, stage3_block2, stage3_block3 | 0.02081, 0.03125, 0.09375 | score_high_half_random | kept-only | α=0.75 |
| `newconfig1000_vis_s2b1_s3b1_c41_10_projection_opp04` | SGH | `visformer_small` | 20260907 | stage2_block1, stage3_block1 | 0.2092, 0.2041 | score_high_half_random | kept-only | α=0.75 |
| `caitproj1000_k2_b17_b23_c15_15` | SGH | `cait_s24_224` | 20260907 | block17_gap, block23_gap | 0.07653, 0.07653 | score_high_half_random | kept-only | α=0.75 |
| `caitproj1000_k3_b5_b17_b23_c10_10_10` | SGH | `cait_s24_224` | 20260907 | block5_gap, block17_gap, block23_gap | 0.05102 ×3 | score_high_half_random | kept-only | α=0.75 |
| `progressive_cait_b5_b17_b23` | SGH | `cait_s24_224` | 20260907 | block5_gap, block17_gap, block23_gap | 0.05 ×3 | — | kept-only | α=0.75 |
| `progressive_pit_l2` | SGH | `pit_b_224` | 20260907 | 未记录（adapter 默认） | 0.02081, 0.03125, 0.09375 | — | kept-only | α=0.75 |
| `scanD_visformer_k2eqr_s2b1_s3b1_confirm` | SGH | `visformer_small` | 20260907 | stage2_block1, stage3_block1 | 0.2092, 0.2041 | score_high_half_random | kept-only | α=0.75 |
| `scanD_vit_k2_budget20_confirm` | SGH | `vit_base_patch16_224` | 20260907 | block3, block11 | 0.05102, 0.05102 | score_high_half_random | kept-only | α=0.75 |

共同预算：ε=16/255、10 steps、step size=1.6/255、10 groups × 2 views、MI decay 1.0、1000 张图像。SPD 记录的 `patch_dropout_ratio=0.3`、`target_patch_drop_ratio=0.15`；按 source 的实际 drop 记录为 ViT/CaiT 29/196（14.8%）、PiT 10/64（15.6%）、Visformer 7/49（14.3%）。`post_dropout_feature_noise_type=opponent_projected`，强度 0.2。

### B.3 H1：13-target 逐 target ASR（%）

| Experiment | LeViT | PiT-B | DeiT-B | TNT-S | ConViT-B | Visformer-S | CaiT-S24 | Inc-v3 | Inc-v4 | Inc-Res-v2 | ResNet-101 | Inc-v3-adv | Inc-Res-v2-adv | Avg | TF avg | CNN avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cait + Gaussian | 90.2 | 88.9 | 90.0 | 91.0 | 89.9 | 89.5 | 97.2 | 85.8 | 84.3 | 84.5 | 85.4 | 82.9 | 78.9 | 87.58 | 90.96 | 83.63 |
| cait 无 Gaussian | 89.3 | 88.2 | 90.0 | 90.7 | 89.6 | 88.0 | 97.5 | 85.1 | 84.1 | 84.2 | 84.7 | 83.0 | 76.6 | 87.00 | 90.47 | 82.95 |
| caitproj K=2 | 85.7 | 85.1 | 86.9 | 87.3 | 86.1 | 85.4 | 98.3 | 82.8 | 79.9 | 79.7 | 81.7 | 79.3 | 72.5 | 83.90 | 87.83 | 79.32 |
| caitproj K=3 | 87.5 | 86.8 | 88.2 | 88.6 | 86.5 | 86.2 | 97.9 | 83.3 | 81.2 | 80.9 | 81.8 | 79.7 | 73.5 | 84.78 | 88.81 | 80.07 |
| pit + Gaussian | 87.9 | 97.2 | 90.2 | 91.3 | 89.1 | 91.3 | 86.7 | 79.1 | 77.3 | 77.4 | 80.2 | 71.2 | 62.4 | 83.18 | 90.53 | 74.60 |
| pit 无 Gaussian | 87.5 | 97.0 | 90.2 | 90.4 | 88.6 | 91.0 | 86.0 | 80.4 | 78.1 | 76.3 | 79.6 | 69.8 | 61.2 | 82.78 | 90.10 | 74.23 |
| progressive_cait_b5_b17_b23 | 86.3 | 85.7 | 87.2 | 87.9 | 85.5 | 85.5 | 97.7 | 82.8 | 79.8 | 80.7 | 82.0 | 79.9 | 72.9 | 84.15 | 87.97 | 79.68 |
| progressive_pit_l2 | 84.4 | 99.4 | 88.5 | 88.2 | 87.7 | 89.1 | 84.8 | 77.2 | 75.4 | 71.7 | 74.7 | 68.0 | 56.6 | 80.44 | 88.87 | 70.60 |
| scanD_visformer | 80.7 | 79.9 | 72.2 | 79.0 | 69.3 | 99.8 | 72.0 | 76.9 | 76.0 | 70.8 | 75.7 | 64.4 | 48.6 | 74.25 | 78.99 | 68.73 |
| scanD_vit | 82.9 | 84.6 | 85.4 | 85.2 | 84.4 | 84.3 | 91.2 | 76.7 | 75.0 | 74.3 | 77.9 | 73.2 | 68.6 | 80.28 | 85.43 | 74.28 |
| visformer + Gaussian | 84.9 | 83.1 | 76.9 | 81.7 | 72.8 | 99.8 | 76.6 | 79.7 | 78.5 | 72.3 | 78.1 | 66.8 | 49.2 | 76.95 | 82.26 | 70.77 |
| visformer 无 Gaussian | 83.2 | 82.4 | 73.8 | 80.0 | 69.8 | 99.7 | 74.9 | 78.0 | 77.8 | 70.8 | 76.6 | 64.6 | 46.3 | 75.22 | 80.54 | 69.02 |
| vit + Gaussian | 82.4 | 81.0 | 83.4 | 84.1 | 82.1 | 81.4 | 88.3 | 75.7 | 72.5 | 72.7 | 73.2 | 71.7 | 66.5 | 78.08 | 83.24 | 72.05 |
| vit 无 Gaussian | 80.5 | 79.6 | 82.6 | 82.5 | 80.7 | 79.4 | 88.0 | 75.4 | 73.5 | 71.6 | 72.2 | 71.2 | 65.3 | 77.12 | 81.90 | 71.53 |

注：`Avg` 是 13-target 平均；`TF avg` 与 `CNN avg` 分别是 7 个 Transformer 和 6 个 CNN target 的平均。`gaussian` 列对 ViT 记录以 `gradient_study_probe=gaussian_blend_s40_a75` 标识，对 CaiT/PiT/Visformer 记录为 `gaussian_alpha=0.75`。

### B.4 H2：`newconfig1000_*` 的 14-target 扩展（%）

这些记录的同一批对抗图像先按 13-target 评估，后补充 `vit_base_patch16_224` 形成 14-target 行。表中 `13T avg` 来自原 13-target 行，`14T overall` 来自加入 ViT 后的行。

| Experiment | ViT-B/16 | LeViT | PiT-B | DeiT-B | TNT-S | ConViT-B | Visformer-S | CaiT-S24 | Inc-v3 | Inc-v4 | Inc-Res-v2 | ResNet-101 | Inc-v3-adv | Inc-Res-v2-adv | 13T avg | 14T overall | TF avg（8） | CNN avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `newconfig1000_vit_b3_b10_c10_10` | 97.0 | 86.5 | 87.7 | 90.1 | 90.2 | 88.4 | 87.9 | 94.9 | 80.1 | 78.0 | 78.3 | 82.7 | 78.3 | 73.5 | 84.35 | 85.26 | 90.34 | 78.48 |
| `newconfig1000_cait_b17_b23_c02_28_projection` | 84.7 | 88.5 | 88.5 | 90.2 | 90.0 | 89.2 | 88.3 | 98.0 | 84.1 | 81.7 | 81.4 | 83.9 | 80.8 | 76.0 | 86.20 | 86.09 | 89.68 | 81.32 |
| `newconfig1000_pit_s2b1_s3b2_s3b3_c05_02_06_opp04` | 83.5 | 87.2 | 98.8 | 91.6 | 90.9 | 90.9 | 90.8 | 88.3 | 83.0 | 80.0 | 78.5 | 81.2 | 75.1 | 67.2 | 84.88 | 84.79 | 90.25 | 77.50 |
| `newconfig1000_vis_s2b1_s3b1_c41_10_projection_opp04` | 62.2 | 87.2 | 83.2 | 76.5 | 84.9 | 73.7 | 99.4 | 77.2 | 83.8 | 83.9 | 76.9 | 82.9 | 72.5 | 55.0 | 79.77 | 78.52 | 80.54 | 75.83 |

### B.5 历史记录的 Git 位置

| 记录组 | 最后包含该 CSV 的提交 |
| --- | --- |
| `original_score_postdrop_phase_pair`（8 个 CSV） | `d1ac809e` |
| `newconfig1000_*`（4 个 CSV） | `6cad5483` |
| `caitproj1000_*`（2 个 CSV） | `12db3d5d` |
| `progressive_cait_b5_b17_b23`、`progressive_pit_l2`、`scanD_*` | `92e8a8bb` |
| kept-only PRD 原始 CSV（4 个 source） | `f44be25` 归档；实现 `98cb7cf1` |

当前工作树在 `fd7ba31` 中清理了这些旧 CSV；如需复现，可从对应提交的 `outputs/csv/` 或 `results/` 取回。

---

## 附录 C：未纳入的 1000 图服务器运行

`f44be25:results/server_untracked_files.tsv.gz` 记录了服务器上大量未提交的 1000 图攻击目录，例如：

- `randomselector1000_{vit,cait,pit,visformer}_adapter_defaults_s1000_seed20260907`；
- `ranktransition1000_{vit,cait,pit,visformer}_defaults_s1000_seed20260920`；
- `selectorgnoise1000_*`（4 个 source × 9 个 selector/noise 组合，共 36 个目录）；
- `progressive_mainline_{vit,cait,pit,visformer}_s1000_seed20260907`；
- `progressive_cait_b6/b16/b18_*`、`progressive_visformer_*`、`progressive_pit_*`、`scanA2_*`、`scanB2/B3_*` 等调参运行；
- `vit_progressive_*_s1000_seed20260903` 的 13 个早期 ViT 运行。

这些目录的对抗图像和日志在服务器上，但当前工作树没有对应的可读结果 CSV/ASR 汇总；本文件只汇总本地可读取的记录，不对这些运行给出 ASR 数值。若需要把它们纳入，需要先从服务器取回对应的 `outputs/csv/*.csv` 或 `outputs/logs/*_transfer.log`。
