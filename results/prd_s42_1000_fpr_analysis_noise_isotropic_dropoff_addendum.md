# PRD §4.4 噪声结构 follow-up：isotropic RGB + drop-off

日期：2026-10-27。这是一个 **post-hoc 诊断单元**，在看完主分析后按用户要求补跑，用于判断 opponent 与 isotropic RGB 的接近是否由 drop-on 基线造成。它不属于原先预注册的 13 组新增实验，也不并入原有 16 配置 summary/differences/provenance；原有文件保持不变。

## 1. 配置

| 项目 | 设置 |
| --- | --- |
| 源模型 | `vit_base_patch16_224` |
| 数据 | 与主分析相同的 1,000 张图像、标签和顺序 |
| Progressive drop | **关闭**（`--no-progressive-drop`） |
| Checkpoint | `block3,block10`（仅保留归档字段，不执行） |
| 噪声 | **isotropic RGB**，三个方向方差 `1,1,1` |
| β | 0.2 |
| 噪声注入 | 全部初始局部特征；同一初始 RGB 投影；逐样本特征 RMS 匹配；每个可微视图独立采样一次 |
| Groups / views | 10 groups × 2 views = 20 views/step |
| 平移集合 | `(4,4),(8,8),(12,12)` |
| Steps / ε / 步长 / momentum | 10；16/255；1.6/255；1 |
| 攻击 batch size | 48 |
| Seed | 42 |
| 评估 | FPR13；batch size 32；`--no-tf32`；无 AMP |

该单元与 `REF_NOISE`（opponent + drop-off）只在噪声结构上不同；与 `NOISE_ISOTROPIC_RGB`（isotropic + drop-on）只在 drop 开关上不同。两种噪声的总方向方差均为 3，且使用同一投影和 RMS 匹配，因此比较的是协方差/方向分配，不是噪声强度。

## 2. 完成状态

- 新增攻击：**1 / 1 完成**；失败 0；未运行 0。
- 攻击图像：恰好 **1,000** 张。
- FPR13 逐目标结果：**13 行**，每个目标 total=1,000。
- 新增 source-target 评估：13 项（单个配置）。
- 攻击目录：`outputs/attack/prd_s42_analysis_20261027_noise_isotropic_rgb_drop_off/`
- 轻量归档：`results/prd_s42_fpr_analysis_run_artifacts/NOISE_ISOTROPIC_RGB_DROP_OFF/`

## 3. 结果

严格黑盒 ASR（%，排除 `vit_base_patch16_224` 后 12 目标平均）：

| Progressive drop | 无噪声 | Opponent | Isotropic RGB |
| --- | ---: | ---: | ---: |
| 关闭 | 61.54（主 2×2 base，context） | 77.78（`REF_NOISE`） | **77.39（本次新增）** |
| 开启 | 70.66（`REF_DROP`） | 87.06（`REF_FULL`） | 86.93（`NOISE_ISOTROPIC_RGB`） |

Overall ASR（13 目标）对应为：

| Progressive drop | 无噪声 | Opponent | Isotropic RGB |
| --- | ---: | ---: | ---: |
| 关闭 | 64.50 | 79.32 | **78.93** |
| 开启 | 72.92 | 88.02 | 87.90 |

关键直接差值（百分点；负值表示 isotropic 低于 opponent / 参考）：

| 比较 | Strict Δ | Overall Δ |
| --- | ---: | ---: |
| isotropic drop-off − opponent drop-off（结构效应，无 drop） | **−0.39** | **−0.39** |
| isotropic drop-on − opponent drop-on（结构效应，有 drop） | **−0.13** | **−0.12** |
| opponent drop-off − base（噪声效应，无 drop） | +16.24 | +14.82 |
| isotropic drop-off − base（噪声效应，无 drop） | +15.85 | +14.43 |
| opponent drop-on − drop-only（噪声效应，有 drop） | +16.40 | +15.11 |
| isotropic drop-on − drop-only（噪声效应，有 drop） | +16.28 | +14.98 |
| opponent full − opponent drop-off（drop 效应） | +9.28 | +8.70 |
| isotropic drop-on − isotropic drop-off（drop 效应） | +9.54 | +8.97 |

Transformer 黑盒（6 目标）与 CNN（6 目标）也呈相同量级：本次 isotropic drop-off 的 Transformer 黑盒均值为 80.53%，CNN 均值为 74.25%；`REF_NOISE` 分别为 80.88% 和 74.68%。结构差分别为 −0.35 pp 和 −0.43 pp。

逐目标看 isotropic drop-off 相对 `REF_NOISE`：12 个黑盒目标中 9 个略降、2 个略升、1 个持平；范围为 −1.9～+0.9 pp，宏平均 −0.39 pp。目标间仍有反向变化，但没有出现 opponent 一致占优的模式。

## 4. 解读

1. **drop-on 基线没有掩盖一个大的 opponent 优势。** 去掉 drop 后，opponent 仍只比 isotropic 高 0.39 pp（77.78% vs 77.39%）；drop-on 时差 0.13 pp（87.06% vs 86.93%）。两个差异都很小，且方向一致（opponent 略高），但没有量级上的变化。
2. **噪声主效应在两种基线下都很大且相近。** 无 drop 时，opponent 和 isotropic 分别相对 base 提升 +16.24 / +15.85 pp；有 drop 时，分别相对 drop-only 提升 +16.40 / +16.28 pp。主要 ASR 增益出现在加入 RMS 匹配的 RGB 噪声时，而不是在两种协方差之间切换时。
3. **drop 效应也相近。** opponent 下 drop 带来 +9.28 pp，isotropic 下 +9.54 pp；没有表现出明显的噪声结构 × drop 交互。
4. **仍不能证明 opponent 方向结构有额外优势。** 在 β=0.2、总方差匹配、同一投影/RMS、同一视图设置下，有 drop 和无 drop 两个基线都给出相近结论。

需要注意：这是单 seed、1,000 图像的单次 post-hoc 比较；按项目约定不计算区间或显著性。0.39 pp 和 0.13 pp 都不宜单独作为强优势或强等价结论。结论仍只覆盖 ViT-B/16、β=0.2；没有噪声强度扫描，也没有其他源模型。

## 5. 可写入 §4.4 的表述

可增加一句：

> The opponent-versus-isotropic near-tie is not specific to the drop-on baseline. With progressive drop disabled, opponent and isotropic RGB noise reach 77.78% and 77.39% strict black-box ASR (Δ = −0.39 pp); with drop enabled they reach 87.06% and 86.93% (Δ = −0.13 pp). Both noise structures improve over their respective no-noise baselines by about 16 pp. These results do not indicate a large opponent-specific effect hidden by the drop-on baseline.

不要写成 opponent 显著优于 isotropic，也不要写成二者严格等价。若论文要主张 opponent 方向本身是关键，需要更强证据；当前两个基线都不支持这个强主张。

## 6. 文件与命令

| 文件 | 内容 |
| --- | --- |
| `results/prd_s42_1000_fpr_analysis_noise_isotropic_dropoff_summary.csv` | 本次单元与相关参考的 strict/overall/Transformer 黑盒/CNN ASR |
| `results/prd_s42_1000_fpr_analysis_noise_isotropic_dropoff_differences.csv` | 上述直接差值 |
| `results/prd_s42_1000_fpr_analysis_noise_isotropic_dropoff_provenance.json` | 实际参数、命令、文件哈希、时间、完成状态与参考来源 |
| `results/prd_s42_fpr_analysis_run_artifacts/NOISE_ISOTROPIC_RGB_DROP_OFF/` | 本次 `attack_params.json`、`transfer.csv`、`transfer.json` |
| `outputs/attack/prd_s42_analysis_20261027_noise_isotropic_rgb_drop_off/` | 1,000 张攻击图像（服务器本地，被 Git 忽略） |

实际攻击命令：

```bash
python main.py --whitebox-model vit_base_patch16_224 --batch-size 48 --num-workers 4 \
  --max-attacked-samples 1000 --steps 10 --epsilon 0.06274509803921569 \
  --step-size 0.006274509803921568 --opponent-noise-strength 0.2 \
  --input-diversity-groups 10 --input-diversity-views-per-group 2 \
  --input-diversity-phase-shift-set '4,4;8,8;12,12' \
  --noise-structure isotropic_rgb --no-progressive-drop \
  --output-dir outputs/attack/prd_s42_analysis_20261027_noise_isotropic_rgb_drop_off
```

实际评估命令：

```bash
python transfer_eval.py \
  --image-dir outputs/attack/prd_s42_analysis_20261027_noise_isotropic_rgb_drop_off \
  --protocol fpr --batch-size 32 --no-tf32
```

本次未修改原 16 配置 summary、逐目标表、differences、provenance 或主实验/消融表；该单元只在上述独立 addendum 中记录。
