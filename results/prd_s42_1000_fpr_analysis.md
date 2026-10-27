# PRD seed-42 FPR13 ViT-B/16 设计分析（§4.4）

日期：2026-10-27。本记录只使用 ViT-B/16（`vit_base_patch16_224`），复用当前 seed-42、1000 图像、FPR13 主实验的数据与权重。13 个新增配置全部完成 1,000 张攻击，并完成 13 目标评估；3 个参考单元按用户约定直接复用，不重跑。新增部分共 169 个 source-target 评估单元。论文分析主指标为严格黑盒平均 ASR，即排除 `vit_base_patch16_224` 后其余 12 个目标等权平均；差值统一为百分点。按项目约定，不计算置信区间、bootstrap/resampling 误差条、显著性检验、McNemar 检验或 p 值。未修改论文 LaTeX、现有主实验、主表或消融表。

本分析覆盖 checkpoint 位置、checkpoint 数量、drop 总预算、噪声结构、空间配对和 groups 数量六个参数轴。任务中的“空间配对与 groups 数量”拆成两个独立实验，避免把视图配对与采样数量混为同一因素。

## 1. 固定设置

| 项目 | 设置 |
| --- | --- |
| 源模型 | `vit_base_patch16_224` |
| 源权重 | `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k`，snapshot `063c6c38a5d8510b2e57df480445e94b231dad2c` |
| 数据 | `data/clean_resized_images` 的 1,000 张图像；顺序、标签与当前主实验一致 |
| 输入尺寸 | 224×224 |
| 随机设置 | PyTorch 全局 seed 42 |
| 攻击 batch size | 48 |
| 扰动预算 | 16/255 |
| 迭代次数 | 10 |
| 步长 | 1.6/255 |
| 动量系数 | 1 |
| 默认 checkpoint | `block3,block10` |
| 默认丢弃数量 | 10、10 |
| 默认噪声 | RGB opponent-channel noise，β=0.2 |
| 默认 groups | 10 |
| 每组视图 | 2（原始／平移） |
| 默认平移集合 | `(4,4),(8,8),(12,12)` |
| 目标评估 | FPR13，batch size 32，关闭 TF32，不启用 AMP |
| 梯度处理 | 20 个视图的像素梯度直接平均；动量累积；sign 更新与 L∞ 投影 |

`blockk` 表示执行完前 k 个 Transformer blocks 之后的状态，不是零索引模块 k。ViT-B 的 12 个 block 均有 196 个局部 token；所有分析配置的 checkpoint 网格均为 14×14。

数据顺序哈希为 `b182c97d353f8b2b7aaa05e57e55467485cbda4943005256e7bc6f50f54c1e94`，标签顺序哈希为 `7f220ccf352c1597c0736bf8d8105dbc14a295279036de88e90bec53e51b181e`，与现有 FPR13 provenance 完全一致。源权重、数据、参考参数和评估协议在复用前已核对；参考单元的生成 batch size 为 48，评估 batch size 为 32。

## 2. 最小代码支持

新增行为仅由显式 CLI 参数启用，当前默认配置未改变：

- `--vit-analysis-checkpoints`：仅允许 `vit_base_patch16_224`。参数校验 `block<k>` 格式、`1 ≤ k ≤ 12`、严格递增、无重复，并要求 checkpoint 数量与 `--drop-ratios` 数量一致。未指定时 ViT-B 仍为 `block3,block10` 和 10/10。该入口不适用于 CaiT、PiT 或 DeiT。
- `--noise-structure opponent|isotropic_rgb`：默认仍为 `opponent`。各向同性对照复用当前正交基，将三个方向的采样方差设为 1、1、1；仍经真实初始 RGB 投影映射，并按逐样本特征 RMS 匹配，β=0.2。没有改成在特征空间直接采样高斯噪声。
- `attack_params.json` 现在记录实际 checkpoint、实际丢弃数量、噪声结构、方向方差、β、groups、视图数量和平移集合。默认 `opponent` 的方向方差记录为 `0.5,1.25,1.25`，`isotropic_rgb` 为 `1,1,1`。

运行代码为 commit `3940e6fe8360c5f7f57d353d2a107746e88fc3e6`。新增测试覆盖 checkpoint 遍历与整数预算、零平移的图像与 mask 恒等性、两种噪声的 RMS 与方向方差、以及原始／原始配对的实际视图数量。默认测试集 `python -m unittest discover -s tests` 通过（30 项，1 项 optional real-adapter 跳过）。

可选的真实 adapter 测试 `RUN_REAL_PRD_ADAPTERS=1 python -m unittest tests.test_progressive_adapters` 仍包含 PiT-T 4/2/5 的旧期望值；当前 adapter 与实验记录为 14/2/2（README 的旧配置行仍写 4/2/5）。该过期期望在本次改动前已存在，不涉及本分析使用的 ViT-B 路径，本次未修改 PiT 或该测试。

## 3. 执行与完成状态

每组攻击使用独立目录 `outputs/attack/prd_s42_analysis_20261027_<实验ID>/`，由 `main.py` 清空后重新生成；每组恰好 1,000 张 `adv_*.png`。评估统一使用：

```bash
python transfer_eval.py --image-dir outputs/attack/prd_s42_analysis_20261027_<dir> \
  --protocol fpr --batch-size 32 --no-tf32
```

未传 `--amp`，未修改目标权重或预处理。

| 实验 ID | 攻击 | 图像数 | FPR13 行数 | 评估 |
| --- | --- | ---: | ---: | --- |
| `POS_EARLY` | 完成 | 1,000 | 13 | 完成 |
| `POS_MIDDLE` | 完成 | 1,000 | 13 | 完成 |
| `POS_LATE` | 完成 | 1,000 | 13 | 完成 |
| `COUNT_ONE_EARLY` | 完成 | 1,000 | 13 | 完成 |
| `COUNT_ONE_LATE` | 完成 | 1,000 | 13 | 完成 |
| `COUNT_THREE` | 完成 | 1,000 | 13 | 完成 |
| `BUDGET_10` | 完成 | 1,000 | 13 | 完成 |
| `BUDGET_40` | 完成 | 1,000 | 13 | 完成 |
| `NOISE_ISOTROPIC_RGB` | 完成 | 1,000 | 13 | 完成 |
| `PAIR_ORIGINAL_ORIGINAL` | 完成 | 1,000 | 13 | 完成 |
| `GROUPS_1` | 完成 | 1,000 | 13 | 完成 |
| `GROUPS_2` | 完成 | 1,000 | 13 | 完成 |
| `GROUPS_5` | 完成 | 1,000 | 13 | 完成 |

新增完成 13 组，失败 0 组，未运行 0 组。加上 `REF_FULL`、`REF_DROP`、`REF_NOISE` 三个复用参考，共 16 个唯一配置；新增 13×13=169 个 source-target 评估。

完整攻击命令保存在 provenance 的 `attack_commands` 字段。共同部分为：

```bash
python main.py --whitebox-model vit_base_patch16_224 --batch-size 48 --num-workers 4 \
  --max-attacked-samples 1000 --steps 10 --epsilon 0.06274509803921569 \
  --step-size 0.006274509803921568 --opponent-noise-strength 0.2 \
  --input-diversity-groups 10 --input-diversity-views-per-group 2 \
  --input-diversity-phase-shift-set '4,4;8,8;12,12' \
  --noise-structure opponent --drop-ratios 0.051020408163,0.051020408163 \
  <实验额外参数> --output-dir outputs/attack/prd_s42_analysis_20261027_<dir>
```

| 实验 | 额外参数 |
| --- | --- |
| `POS_EARLY` | `--vit-analysis-checkpoints block1,block4` |
| `POS_MIDDLE` | `--vit-analysis-checkpoints block5,block8` |
| `POS_LATE` | `--vit-analysis-checkpoints block8,block11` |
| `COUNT_ONE_EARLY` | `--vit-analysis-checkpoints block3 --drop-ratios 0.102040816327` |
| `COUNT_ONE_LATE` | `--vit-analysis-checkpoints block10 --drop-ratios 0.102040816327` |
| `COUNT_THREE` | `--vit-analysis-checkpoints block3,block6,block10 --drop-ratios 0.035714285714,0.030612244898,0.035714285714` |
| `BUDGET_10` | `--drop-ratios 0.025510204082,0.025510204082` |
| `BUDGET_40` | `--drop-ratios 0.102040816327,0.102040816327` |
| `NOISE_ISOTROPIC_RGB` | `--noise-structure isotropic_rgb` |
| `PAIR_ORIGINAL_ORIGINAL` | `--input-diversity-phase-shift-set '0,0'` |
| `GROUPS_1` | `--input-diversity-groups 1` |
| `GROUPS_2` | `--input-diversity-groups 2` |
| `GROUPS_5` | `--input-diversity-groups 5` |

GROUPS 实验的额外参数在共同参数之后覆盖 `--input-diversity-groups 10`（argparse 取最后一次值）；provenance 保留实际执行的完整字符串。`BUDGET_40` 以及 `COUNT_ONE_*` 的 20/196 比例运行记录均为实际 `[20,20]` 或 `[20]`；`BUDGET_10` 为 `[5,5]`；`COUNT_THREE` 为 `[7,6,7]`，没有舍入偏差。

## 4. 配置矩阵与 ASR

`*` 标记的 3 个参考单元直接复用归档结果，没有重跑。严格黑盒 ASR 为排除源目标后的 12 目标平均；Overall ASR 为 13 目标平均。表中显示到两位小数；未舍入结果、Transformer 黑盒（6 目标）和 CNN（6 目标）平均值见 summary CSV。

| Experiment | Checkpoints | Drops | Budget | Noise | Groups | Views/step | Shifts | Strict ASR | Overall ASR |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| REF_FULL* | block3, block10 | 10, 10 | 20 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 87.06 | 88.02 |
| REF_DROP* | block3, block10 | 10, 10 | 20 | none | 10 | 20 | 4,4; 8,8; 12,12 | 70.66 | 72.92 |
| REF_NOISE* | block3, block10 | — | 0 (drop off) | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 77.78 | 79.32 |
| POS_EARLY | block1, block4 | 10, 10 | 20 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 77.35 | 78.73 |
| POS_MIDDLE | block5, block8 | 10, 10 | 20 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 75.93 | 77.45 |
| POS_LATE | block8, block11 | 10, 10 | 20 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 80.94 | 82.35 |
| COUNT_ONE_EARLY | block3 | 20 | 20 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 76.88 | 78.26 |
| COUNT_ONE_LATE | block10 | 20 | 20 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 87.53 | 88.48 |
| COUNT_THREE | block3, block6, block10 | 7, 6, 7 | 20 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 86.54 | 87.54 |
| BUDGET_10 | block3, block10 | 5, 5 | 10 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 86.11 | 87.12 |
| BUDGET_40 | block3, block10 | 20, 20 | 40 | opponent | 10 | 20 | 4,4; 8,8; 12,12 | 87.50 | 88.44 |
| NOISE_ISOTROPIC_RGB | block3, block10 | 10, 10 | 20 | isotropic RGB | 10 | 20 | 4,4; 8,8; 12,12 | 86.93 | 87.90 |
| PAIR_ORIGINAL_ORIGINAL | block3, block10 | 10, 10 | 20 | opponent | 10 | 20 | 0,0 | 80.02 | 81.55 |
| GROUPS_1 | block3, block10 | 10, 10 | 20 | opponent | 1 | 2 | 4,4; 8,8; 12,12 | 73.73 | 75.57 |
| GROUPS_2 | block3, block10 | 10, 10 | 20 | opponent | 2 | 4 | 4,4; 8,8; 12,12 | 79.77 | 81.22 |
| GROUPS_5 | block3, block10 | 10, 10 | 20 | opponent | 5 | 10 | 4,4; 8,8; 12,12 | 84.57 | 85.71 |

`REF_NOISE` 的 0 表示关闭 progressive drop，走现有无 drop 路径，不是向随机 mask 函数传零比例。`REF_NOISE` 的 checkpoint 字段保留归档参数但在该单元中不执行；`REF_DROP` 执行 checkpoint drop，仅关闭噪声。

## 5. 预先指定的直接差值

差值为百分点，使用未舍入的逐目标 ASR 计算；表中显示到两位小数，精确值见 differences CSV。严格黑盒差值为主要分析量，Overall 差值仅作补充。

| Comparison | Strict Δ (pp) | Overall Δ (pp) |
| --- | ---: | ---: |
| POS_EARLY_vs_REF_FULL | -9.71 | -9.29 |
| POS_MIDDLE_vs_REF_FULL | -11.13 | -10.58 |
| POS_LATE_vs_REF_FULL | -6.12 | -5.68 |
| COUNT_ONE_EARLY_vs_REF_FULL | -10.18 | -9.76 |
| COUNT_ONE_LATE_vs_REF_FULL | +0.48 | +0.46 |
| COUNT_THREE_vs_REF_FULL | -0.52 | -0.48 |
| BUDGET_10_vs_REF_FULL | -0.95 | -0.90 |
| BUDGET_40_vs_REF_FULL | +0.44 | +0.42 |
| NOISE_ISOTROPIC_RGB_vs_REF_FULL | -0.13 | -0.12 |
| NOISE_ISOTROPIC_RGB_vs_REF_DROP | +16.28 | +14.98 |
| REF_FULL_vs_REF_DROP | +16.40 | +15.11 |
| PAIR_ORIGINAL_ORIGINAL_vs_REF_FULL | -7.04 | -6.48 |
| GROUPS_1_vs_GROUPS_2 | -6.04 | -5.65 |
| GROUPS_2_vs_GROUPS_5 | -4.80 | -4.48 |
| GROUPS_5_vs_REF_FULL | -2.49 | -2.32 |
| GROUPS_1_vs_REF_FULL | -13.33 | -12.45 |
| GROUPS_2_vs_REF_FULL | -7.29 | -6.80 |

## 6. 观察与可进入 §4.4 的内容

以下均为 seed-42、1,000 图像的单次描述性比较，不附带区间或显著性结论。小于约 1 pp 的差值不宜单独作强结论。

### Checkpoint 位置

在固定 10+10、总预算 20 的条件下，`REF_FULL`（block3,block10）严格黑盒 ASR 为 87.06%。`POS_EARLY`（block1,block4）为 77.35%（−9.71 pp），`POS_MIDDLE`（block5,block8）为 75.93%（−11.13 pp），`POS_LATE`（block8,block11）为 80.94%（−6.12 pp）。三个替代位置均低于当前默认。

`POS_MIDDLE` 与 `REF_FULL` 的平均深度同为 6.5（(5+8)/2=(3+10)/2），但前者跨度 3、后者跨度 7；在这一对匹配平均深度的比较中，跨深度分布的默认设置高 11.13 pp。三个替代配置同时改变了绝对深度与跨度，因此本矩阵不能把位置效应单独归因于“深度”或“跨度”；能直接说明的是默认的跨深度位置优于所测的早、中、晚集中配置。

### Checkpoint 数量

固定总预算 20：单 checkpoint 早期（block3，20）为 76.88%（−10.18 pp），单 checkpoint 晚期（block10，20）为 87.53%（+0.48 pp），双 checkpoint 默认（block3,block10；10+10）为 87.06%，三 checkpoint（block3,block6,block10；7+6+7）为 86.54%（−0.52 pp）。

单晚期与三 checkpoint 都与默认值相差约 0.5 pp；单早期低约 10 pp。该矩阵不支持“checkpoint 越多越好”的简单结论；在固定预算下，干预放置位置比数量更关键。三 checkpoint 增加中间干预没有带来可辨识的增益；单晚期几乎复现双 checkpoint 的结果。

### Token 丢弃总预算

在 block3,block10 与 opponent noise 下：关闭 drop（0）为 77.78%；预算 10（5+5）为 86.11%（−0.95 pp vs 20）；预算 20（10+10）为 87.06%；预算 40（20+20）为 87.50%（+0.44 pp vs 20）。10 到 40 之间仅相差 1.39 pp，默认 20 接近该区间的平台；从无 drop 到任意已测预算的差距约为 8.3–9.7 pp。

可直接写：drop 的“有／无”是这一组的主要差异来源，10–40 的预算大小影响较浅；10、20、40 三点数值单调上升但幅度很小，40 仅比 20 高 0.44 pp。

### 噪声结构

`REF_DROP`（无噪声）为 70.66%；当前 opponent noise 为 87.06%（相对无噪声 +16.40 pp）；`NOISE_ISOTROPIC_RGB` 为 86.93%（相对无噪声 +16.28 pp，相对 opponent −0.13 pp）。

两种噪声都带来约 16 pp 的提升，而各向同性 RGB 对照与 opponent 在当前尺度上仅差 0.13 pp。这组结果不支持“opponent 协方差本身带来额外迁移收益”的强结论；可写的是：在相同 RGB 投影、逐样本 RMS 匹配、β 和视图设置下，加入 RMS 匹配的 RGB 噪声带来约 16 pp 的提升，opponent 与各向同性方向在本次比较中未显示可辨识差异。该结论不改变默认 opponent 配置，也不排除其他数据或设置下方向结构的作用。

### 空间配对

固定 10 groups×2 views、100 条 schedule、200 次 mask selection，每视图独立采样噪声：原始／平移为 87.06%，原始／原始为 80.02%（−7.04 pp）。零平移时第二视图的图像与 mask 均与原视图一致，因此该对照隔离的是第二视图的空间变换，而不是额外的 schedule 或视图数量。平移配对在该矩阵中带来约 7 pp 的描述性增益。

### Groups 数量

保留原始／平移配对、固定 10 steps：1、2、5、10 groups 的严格黑盒 ASR 分别为 73.73%、79.77%、84.57%、87.06%。相邻差值为：1→2 +6.04 pp，2→5 +4.80 pp，5→10 +2.49 pp；相对 10 groups 分别为 −13.33、−7.29、−2.49 pp。ASR 随 groups（同时增加每步视图与每图 schedule 数）单调上升，但增益递减；相邻设置中 1→2 的增益最大。没有通过增加迭代次数补偿较少 groups。

### 对 §4.4 的适合性

可进入 §4.4 的描述性观察：

1. 默认跨深度 checkpoint 位置优于所测的早、中、晚集中位置；匹配平均深度的集中对照低 11.13 pp。
2. 固定 20 token 预算时，单个晚期 checkpoint 与双 checkpoint 基本持平（+0.48 pp），单个早期 checkpoint 低 10.18 pp；三 checkpoint 未超过双 checkpoint（−0.52 pp）。
3. 预算 10–40 的 ASR 较平坦（极差 1.39 pp）；关闭 drop 明显更低。
4. 各向同性 RGB 与 opponent noise 在 0.13 pp 内接近，二者相对无噪声都高约 16 pp；不应把 opponent 方向写成已证实的必需因素。
5. 原始／平移配对相对原始／原始高 7.04 pp，且该对照保持视图数、schedule 数和噪声独立采样不变。
6. groups 从 1 到 10 单调提升，相邻增益递减；10 groups 是本 groups 矩阵中的最高值。

需要保留的限定：这些是单 seed、1,000 图像的点估计；项目约定不报告区间或显著性，因此不应把小于约 1 pp 的差异解释为稳定优势或劣势。默认 PRD 配置和其他源模型均未改变。

## 7. 文件与来源

| 文件 | 内容 |
| --- | --- |
| `results/prd_s42_1000_fpr_analysis_summary.csv` | 16 个唯一配置的 ASR 与配置元数据；含严格黑盒（12 目标）、Transformer 黑盒（6 目标）、CNN（6 目标）和 Overall（13 目标） |
| `results/prd_s42_1000_fpr_analysis_per_target.csv` | 16×13=208 个逐目标 correct/total/ASR；新增部分 169 个单元 |
| `results/prd_s42_1000_fpr_analysis_differences.csv` | 17 个预先指定的直接差值（Overall、严格黑盒、Transformer 黑盒、CNN） |
| `results/prd_s42_1000_fpr_analysis_provenance.json` | 实际参数、完整命令、代码 commit、环境、数据与样本顺序、权重文件哈希、参考文件哈希、评估时间与完成状态 |
| `results/prd_s42_fpr_analysis_run_artifacts/<实验ID>/` | 13 个新增单元的 `attack_params.json`、`transfer.csv`、`transfer.json` |
| `results/prd_s42_fpr_run_artifacts/{full_vit,drop_vit,noise_vit}/` | 3 个复用参考的原始轻量记录，未覆盖 |

参考 ASR 与归档主表核对一致：`REF_FULL` 87.058333%、`REF_DROP` 70.658333%、`REF_NOISE` 77.783333%（严格黑盒）。评估协议为 FPR13、全部 1,000 张、batch size 32、`--no-tf32`、无 AMP；ASR 不筛选原本分类正确的样本。

本分析不包含噪声强度扫描、运行时间或显存分析，也不包含攻击样本的定性展示。攻击图像和运行日志保留在被忽略的服务器 `outputs/` 下，未进入 Git。
